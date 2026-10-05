"""Train the residual-recovery policy with PPO and a four-stage curriculum.

The curriculum is the key to making this train in minutes rather than hours.
Stage boundaries are expressed as fractions of the total step budget:

===== ============ =========================================================
stage fraction     what changes
===== ============ =========================================================
1     0 - 0.20     near-vertical drop, light wind: learn not to overshoot
2     0.20 - 0.45  wider offsets, moderate wind: learn the lateral cascade
3     0.45 - 0.72  full offset envelope: learn to arrest large crossrange
4     0.72 - 1.00  full wind + shear + turbulence: learn disturbance rejection
===== ============ =========================================================

The curriculum level is a single scalar in [0, 1] that scales the initial
condition envelope and the wind strength, so a policy never has to un-learn a
skill: the difficulty only ever increases.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# allow "python scripts/train.py" without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zq3rl.config import Config, PPOConfig          # noqa: E402
from zq3rl.env import RecoveryEnv                   # noqa: E402
from zq3rl.ppo import PPOTrainer                    # noqa: E402
from zq3rl.utils import (                           # noqa: E402
    Tee, banner, environment_info, make_run_dir, save_csv, save_json,
    set_global_seed,
)


def curriculum_schedule(progress: float) -> float:
    """Map training progress in [0, 1] to a curriculum level in [0, 1].

    The curve must leave a substantial *tail* at full difficulty.  An earlier
    version reached level 1.0 only at ~99% of the budget, so the policy spent
    essentially no time training on the disturbance level it is evaluated at --
    the resulting policy was indistinguishable from the baseline.  Full
    difficulty is therefore reached at 55% of the budget, leaving 45% of the
    steps to refine the residual at the evaluation envelope.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    if p < 0.12:
        return 0.05 + (p / 0.12) * 0.25           # 0.05 -> 0.30
    if p < 0.28:
        return 0.30 + ((p - 0.12) / 0.16) * 0.30  # 0.30 -> 0.60
    if p < 0.55:
        return 0.60 + ((p - 0.28) / 0.27) * 0.40  # 0.60 -> 1.00
    return 1.00                                    # full difficulty tail


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the ZQ3 residual recovery policy.")
    ap.add_argument("--steps", type=int, default=400_000,
                    help="total environment steps (default 400k, ~15 min on CPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rollout", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--no-prediction", action="store_true",
                    help="ablation: disable the predictive-perception term")
    ap.add_argument("--no-residual", action="store_true",
                    help="ablation: run the pure analytic baseline")
    ap.add_argument("--no-sensors", action="store_true",
                    help="ablation: perfect state knowledge")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()
    if args.lr is not None:
        cfg.ppo.lr = args.lr
    if args.rollout is not None:
        cfg.ppo.rollout_steps = args.rollout
    if args.hidden is not None:
        cfg.ppo.hidden = args.hidden
    cfg.ppo.total_steps = args.steps
    cfg.ppo.device = args.device

    name = args.run_name or f"ppo_s{args.seed}"
    rd = make_run_dir(args.out, name)
    tee = Tee(rd.logs / "train.log")
    if not args.quiet:
        sys.stdout = tee

    set_global_seed(args.seed)

    use_pred = not args.no_prediction
    use_res = not args.no_residual
    use_sen = not args.no_sensors

    env = RecoveryEnv(cfg, seed=args.seed, curriculum_level=curriculum_schedule(0.0),
                      use_prediction=use_pred, use_residual=use_res,
                      use_sensors=use_sen)

    print(banner("ZQ3-Recovery-RL  ::  residual PPO training"))
    print(f"run dir        : {rd.root}")
    print(f"total steps    : {cfg.ppo.total_steps:,}")
    print(f"rollout steps  : {cfg.ppo.rollout_steps}")
    print(f"prediction     : {use_pred}")
    print(f"residual RL    : {use_res}")
    print(f"real sensors   : {use_sen}")
    print(f"obs dim        : {env.OBS_DIM}")
    print(banner("environment", ch="-"))
    for k, v in environment_info().items():
        print(f"  {k:14s}: {v}")
    print(banner("", ch="-"))

    save_json(rd.data / "config.json", {
        "args": vars(args),
        "config": cfg.to_dict(),
        "environment": environment_info(),
    })

    trainer = PPOTrainer(env, cfg.ppo, seed=args.seed, device=args.device,
                         curriculum_schedule=curriculum_schedule)

    t0 = time.time()
    updates_per_epoch = max(cfg.ppo.total_steps // cfg.ppo.rollout_steps, 1)
    print(f"{'upd':>5} {'step':>9} {'eps':>6} {'curric':>6} {'rew':>9} "
          f"{'succ':>6} {'lat_m':>7} {'spd':>6} {'fuel_kg':>8} {'kl':>7} {'t_s':>6}")
    print("-" * 88)

    while trainer.step_count < cfg.ppo.total_steps:
        progress = trainer.step_count / max(cfg.ppo.total_steps, 1)
        env.curriculum = curriculum_schedule(progress)
        trainer.collect()
        stats = trainer.update(progress)
        if trainer.updates % cfg.ppo.log_every == 0 or trainer.updates == 1:
            row = trainer.log()
            row["approx_kl"] = stats.approx_kl
            row["policy_loss"] = stats.policy_loss
            row["value_loss"] = stats.value_loss
            row["entropy"] = stats.entropy
            row["lr"] = stats.lr
            print(f"{trainer.updates:5d} {int(row['step']):9d} {int(row['episodes']):6d} "
                  f"{row['curriculum']:6.3f} {row['reward']:9.1f} {row['success']:6.3f} "
                  f"{row['lateral']:7.2f} {row['speed']:6.2f} {row['fuel']:8.0f} "
                  f"{stats.approx_kl:7.4f} {time.time() - t0:6.0f}")
        if trainer.updates % cfg.ppo.save_every == 0:
            trainer.save(str(rd.models / f"ckpt_{trainer.step_count:08d}.pt"))
            save_csv(rd.data / "training_history.csv", trainer.history)

    trainer.save(str(rd.models / "final.pt"))
    save_csv(rd.data / "training_history.csv", trainer.history)
    save_json(rd.data / "summary.json", {
        "steps": trainer.step_count,
        "episodes": trainer.episode_count,
        "updates": trainer.updates,
        "wall_clock_s": time.time() - t0,
        "final_history": trainer.history[-1] if trainer.history else {},
    })

    print(banner(f"done: {trainer.step_count:,} steps / {trainer.episode_count} episodes "
                 f"in {time.time() - t0:.0f}s"))
    if not args.quiet:
        sys.stdout = tee._stdout
    tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
