"""A compact, self-contained PPO implementation (clipped surrogate, GAE).

Written from scratch rather than pulled from a framework so that the report can
point at every line of the algorithm.  Runs comfortably on CPU for the network
sizes used here (a 2x160 MLP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PPOConfig


# --------------------------------------------------------------------------- #
#  Network
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    """Shared-trunk Gaussian policy with a state-value head.

    A learnable ``log_std`` parameterisation is used for the action
    distribution; the action is the bounded residual acceleration and is
    squashed with ``tanh`` so the policy output is always admissible.
    """

    def __init__(self, obs_dim: int, act_dim: int = 3, hidden: int = 160,
                 log_std_init: float = -1.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    # ------------------------------------------------------------------ #
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, v = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, v

    # ------------------------------------------------------------------ #
    def act(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mu, std, v = self.distribution(o)
            if deterministic:
                z = mu
            else:
                z = mu + std * torch.randn_like(mu)
            a = torch.tanh(z)
            logp = _gaussian_logprob(z, mu, std) - _tanh_correction(a)
            return (a.squeeze(0).numpy().astype(np.float32),
                    float(logp.item()), float(v.item()))

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor):
        """Recompute log-prob, entropy and value for a batch of actions.

        ``act`` are the squashed actions actually stored in the buffer; the
        inverse ``atanh`` recovers the pre-squash sample for the Gaussian
        density.
        """
        mu, std, v = self.distribution(obs)
        z = torch.atanh(torch.clamp(act, -1.0 + 1e-6, 1.0 - 1e-6))
        logp = _gaussian_logprob(z, mu, std) - _tanh_correction(act)
        ent = (0.5 * math.log(2 * math.pi * math.e)
               + torch.log(std)).sum(-1)
        return logp, ent, v


def _gaussian_logprob(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = std * std
    return (-0.5 * ((z - mu) ** 2 / var + torch.log(2 * math.pi * var))).sum(-1)


def _tanh_correction(a: torch.Tensor) -> torch.Tensor:
    # log(1 - a^2) summed over the action dimensions
    return torch.log(torch.clamp(1.0 - a * a, min=1e-6)).sum(-1)


# --------------------------------------------------------------------------- #
#  Rollout buffer
# --------------------------------------------------------------------------- #
class RolloutBuffer:
    def __init__(self, steps: int, obs_dim: int, act_dim: int):
        self.steps = steps
        self.obs = np.zeros((steps, obs_dim), dtype=np.float32)
        self.act = np.zeros((steps, act_dim), dtype=np.float32)
        self.logp = np.zeros(steps, dtype=np.float32)
        self.val = np.zeros(steps, dtype=np.float32)
        self.rew = np.zeros(steps, dtype=np.float32)
        self.done = np.zeros(steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, act, logp, val, rew, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.logp[i] = logp
        self.val[i] = val
        self.rew[i] = rew
        self.done[i] = float(done)
        self.ptr = i + 1

    def full(self) -> bool:
        return self.ptr >= self.steps

    def reset(self) -> None:
        self.ptr = 0


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
@dataclass
class TrainStats:
    step: int = 0
    episodes: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    lr: float = 0.0


class PPOTrainer:
    def __init__(self, env, cfg: PPOConfig, seed: int = 0,
                 device: str = "cpu", curriculum_schedule=None):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = env.OBS_DIM
        self.net = ActorCritic(obs_dim, 3, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.buf = RolloutBuffer(cfg.rollout_steps, obs_dim, 3)
        self.curriculum_schedule = curriculum_schedule  # fn(progress)->level

        self.step_count = 0
        self.episode_count = 0
        self.updates = 0
        self.history: List[Dict[str, float]] = []
        self._ep_rewards: List[float] = []
        self._ep_success: List[float] = []
        self._ep_lateral: List[float] = []
        self._ep_speed: List[float] = []
        self._ep_fuel: List[float] = []
        self._ep_tilt: List[float] = []
        self._ep_lengths: List[int] = []
        self._last_obs, _ = env.reset(seed=seed)
        self._ep_ret = 0.0
        self._ep_len_cur = 0

    # ------------------------------------------------------------------ #
    def _lr_at(self, progress: float) -> float:
        frac = self.cfg.lr_final_frac
        return self.cfg.lr * (1.0 - (1.0 - frac) * float(np.clip(progress, 0, 1)))

    # ------------------------------------------------------------------ #
    def collect(self) -> None:
        cfg = self.cfg
        self.buf.reset()
        while not self.buf.full():
            obs = self._last_obs
            act, logp, val = self.net.act(obs)
            nobs, rew, done, trunc, info = self.env.step(act)
            self.buf.add(obs, act, logp, val, rew, done or trunc)
            self._ep_ret += rew
            self._ep_len_cur += 1
            self.step_count += 1
            self._last_obs = nobs

            if done or trunc:
                self.episode_count += 1
                self._ep_rewards.append(self._ep_ret)
                self._ep_success.append(1.0 if info.get("success") else 0.0)
                self._ep_lateral.append(float(info.get("lateral", np.nan)))
                self._ep_speed.append(float(info.get("speed", np.nan)))
                self._ep_fuel.append(float(info.get("fuel_used", np.nan)))
                self._ep_tilt.append(float(info.get("tilt_deg", np.nan)))
                self._ep_lengths.append(self._ep_len_cur)
                self._ep_ret = 0.0
                self._ep_len_cur = 0
                if self.curriculum_schedule is not None:
                    prog = self.step_count / max(cfg.total_steps, 1)
                    self.env.curriculum = self.curriculum_schedule(prog)
                self._last_obs, _ = self.env.reset()

    # ------------------------------------------------------------------ #
    def update(self, progress: float) -> TrainStats:
        cfg = self.cfg
        dev = self.device
        obs = torch.as_tensor(self.buf.obs, device=dev)
        act = torch.as_tensor(self.buf.act, device=dev)
        old_logp = torch.as_tensor(self.buf.logp, device=dev)
        old_val = torch.as_tensor(self.buf.val, device=dev)
        rew = torch.as_tensor(self.buf.rew, device=dev)
        done = torch.as_tensor(self.buf.done, device=dev)

        # ---- GAE --------------------------------------------------------
        with torch.no_grad():
            _, _, last_v = self.net.distribution(
                torch.as_tensor(self._last_obs, dtype=torch.float32, device=dev).unsqueeze(0))
            adv = torch.zeros_like(rew)
            last_gae = 0.0
            for t in reversed(range(cfg.rollout_steps)):
                if t == cfg.rollout_steps - 1:
                    next_v = last_v.squeeze(0)
                    next_nonterm = 1.0 - done[t]
                else:
                    next_v = old_val[t + 1]
                    next_nonterm = 1.0 - done[t]
                delta = rew[t] + cfg.gamma * next_v * next_nonterm - old_val[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterm * last_gae
                adv[t] = last_gae
            ret = adv + old_val

        n = cfg.rollout_steps
        idx = np.arange(n)
        lr = self._lr_at(progress)
        for pg in self.opt.param_groups:
            pg["lr"] = lr

        stats = TrainStats(lr=lr)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch):
                mb = idx[start:start + cfg.minibatch]
                mbt = torch.as_tensor(mb, device=dev)
                b_adv = adv[mbt]
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                logp, ent, val = self.net.evaluate(obs[mbt], act[mbt])
                ratio = torch.exp(logp - old_logp[mbt])
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(val, ret[mbt])
                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * ent.mean())

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats.policy_loss += float(policy_loss.item())
                    stats.value_loss += float(value_loss.item())
                    stats.entropy += float(ent.mean().item())
                    stats.approx_kl += float(((ratio - 1) - (logp - old_logp[mbt])).mean().item())
                    stats.clip_frac += float(((ratio - 1.0).abs() > cfg.clip).float().mean().item())

        nb = max(cfg.epochs * math.ceil(n / cfg.minibatch), 1)
        stats.policy_loss /= nb
        stats.value_loss /= nb
        stats.entropy /= nb
        stats.approx_kl /= nb
        stats.clip_frac /= nb
        stats.step = self.step_count
        stats.episodes = self.episode_count
        self.updates += 1
        return stats

    # ------------------------------------------------------------------ #
    def log(self) -> Dict[str, float]:
        def tail(xs, k=40):
            xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
            return xs[-k:] if xs else [float("nan")]

        def tail_int(xs, k=40):
            return xs[-k:] if xs else [0]

        row = {
            "step": float(self.step_count),
            "episodes": float(self.episode_count),
            "curriculum": float(self.env.curriculum),
            "reward": float(np.mean(tail(self._ep_rewards))),
            "success": float(np.mean(tail(self._ep_success))),
            "lateral": float(np.mean(tail(self._ep_lateral))),
            "speed": float(np.mean(tail(self._ep_speed))),
            "fuel": float(np.mean(tail(self._ep_fuel))),
            "tilt": float(np.mean(tail(self._ep_tilt))),
            "ep_len": float(np.mean(tail_int(self._ep_lengths))),
        }
        self.history.append(row)
        return row

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({
            "model": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step_count,
            "episodes": self.episode_count,
            "history": self.history,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["model"])
        self.step_count = int(ck.get("step", 0))
        self.episode_count = int(ck.get("episodes", 0))
        self.history = ck.get("history", [])
