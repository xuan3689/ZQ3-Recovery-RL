"""Evaluate a trained policy and export everything the report and the browser
demo need.

Produces, under ``runs/<run>/``:

``data/eval_summary.json``
    Monte-Carlo statistics per controller (success rate with a Wilson interval,
    lateral accuracy, touchdown speed, tilt, fuel, shield-clip rate).
``data/eval_episodes.csv``
    One row per episode, for the report's result tables.
``data/traces.json``
    A handful of full trajectories (baseline vs. residual RL) for the browser
    demo and the trajectory figures.
``data/training_curve.csv``
    The training history, copied out for plotting.

Controllers compared
--------------------
``baseline``
    Layered analytic guidance, no prediction, no RL.
``baseline+pred``
    Adds the predictive-perception feed-forward only.
``residual_rl``
    The full method: analytic baseline + prediction + bounded PPO residual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zq3rl.config import Config                          # noqa: E402
from zq3rl.env import RecoveryEnv                        # noqa: E402
from zq3rl.ppo import PPOTrainer                         # noqa: E402
from zq3rl.utils import (                                # noqa: E402
    banner, load_json, save_csv, save_json, set_global_seed, summarise,
    wilson_interval,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate recovery controllers.")
    ap.add_argument("--run", type=str, default="runs/ppo_main2",
                    help="run directory containing models/final.pt")
    ap.add_argument("--model", type=str, default=None, help="override checkpoint path")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--traces", type=int, default=4,
                    help="how many trajectories to export per controller")
    ap.add_argument("--no-sensors", action="store_true",
                    help="evaluate with perfect state knowledge")
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
def make_policy_fn(trainer, cfg, use_prediction: bool):
    """Wrap the trained network as a deterministic policy over observations."""
    def fn(obs: np.ndarray) -> np.ndarray:
        a, _, _ = trainer.net.act(obs, deterministic=True)
        return a
    return fn


def run_episodes(env: RecoveryEnv, policy_fn, n: int, seed0: int) -> dict:
    rows = []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done = trunc = False
        while not (done or trunc):
            a = policy_fn(obs) if policy_fn is not None else np.zeros(3)
            obs, r, done, trunc, info = env.step(a)
        rows.append(info)
    succ = sum(1 for r in rows if r["success"])
    lo, hi = wilson_interval(succ, n)
    out = {
        "n": n,
        "successes": succ,
        "success_rate": succ / n if n else 0.0,
        "success_ci95": [lo, hi],
        "lateral": summarise([r["lateral"] for r in rows]),
        "speed": summarise([r["speed"] for r in rows]),
        "tilt": summarise([r["tilt_deg"] for r in rows]),
        "fuel": summarise([r["fuel_used"] for r in rows]),
        "shield_clip_rate": float(np.mean([1.0 if r["shield_clip"] else 0.0
                                           for r in rows])),
        "est_pos_err": summarise([r["est_pos_err"] for r in rows]),
        "est_vel_err": summarise([r["est_vel_err"] for r in rows]),
    }
    return out, rows


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()
    set_global_seed(args.seed)
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    print(banner("ZQ3-Recovery-RL  ::  evaluation"))
    print(f"run dir     : {run_dir}")
    print(f"episodes    : {args.episodes}")
    print(f"real sensors: {not args.no_sensors}")

    use_sensors = not args.no_sensors
    results = {}
    episode_rows = []

    controllers = [
        ("baseline", False, False, None),
        ("baseline_pred", True, False, None),
        ("residual_rl", True, True, "trained"),
    ]

    # load the trained network once (needed for the residual controller)
    env_rl = RecoveryEnv(cfg, seed=args.seed, curriculum_level=1.0,
                         use_prediction=True, use_residual=True,
                         use_sensors=use_sensors)
    trainer = PPOTrainer(env_rl, cfg.ppo, seed=args.seed, device="cpu")
    ckpt = Path(args.model) if args.model else run_dir / "models" / "final.pt"
    if not ckpt.exists():
        print(f"!! checkpoint not found: {ckpt}")
        print("   falling back to an untrained network (results will be poor)")
    else:
        trainer.load(str(ckpt))
        print(f"loaded checkpoint: {ckpt} (step {trainer.step_count:,})")
    trainer.net.eval()

    for name, use_pred, use_res, _ in controllers:
        env = RecoveryEnv(cfg, seed=args.seed, curriculum_level=1.0,
                          use_prediction=use_pred, use_residual=use_res,
                          use_sensors=use_sensors)
        pol = make_policy_fn(trainer, cfg, use_pred) if use_res else None
        res, rows = run_episodes(env, pol, args.episodes, args.seed)
        res["prediction"] = use_pred
        res["residual_rl"] = use_res
        results[name] = res
        for i, r in enumerate(rows):
            episode_rows.append({
                "controller": name, "episode": i, "seed": args.seed + i,
                "success": int(r["success"]), "lateral": r["lateral"],
                "speed": r["speed"], "tilt_deg": r["tilt_deg"],
                "fuel_used": r["fuel_used"], "shield_clip": int(r["shield_clip"]),
                "est_pos_err": r["est_pos_err"], "est_vel_err": r["est_vel_err"],
            })
        print(f"  {name:16s} success {res['success_rate']*100:5.1f}% "
              f"[{res['success_ci95'][0]*100:.1f}, {res['success_ci95'][1]*100:.1f}]  "
              f"lat {res['lateral']['mean']:6.2f} m  "
              f"speed {res['speed']['mean']:5.2f} m/s  "
              f"clip {res['shield_clip_rate']*100:4.1f}%")

    # ---- export trajectories for the figures and the web demo -----------
    traces = {}
    for name, use_pred, use_res, _ in controllers:
        env = RecoveryEnv(cfg, seed=args.seed, curriculum_level=1.0,
                          use_prediction=use_pred, use_residual=use_res,
                          use_sensors=use_sensors)
        pol = make_policy_fn(trainer, cfg, use_pred) if use_res else None
        eps = []
        for i in range(args.traces):
            # pick seeds from a spread of difficulties (identical across
            # controllers, so the comparison is paired)
            seed_i = args.seed + i * 7
            env.reset(seed=seed_i)
            tr = env.rollout_trace(policy=pol)
            eps.append(_trace_to_json(tr))
        traces[name] = eps
    save_json(out_dir / "data" / "traces.json", {
        "meta": {
            "run": str(run_dir),
            "checkpoint": str(ckpt),
            "seed": args.seed,
            "n_episodes": args.episodes,
            "use_sensors": use_sensors,
        },
        "controllers": traces,
    })

    save_json(out_dir / "data" / "eval_summary.json", {
        "run": str(run_dir), "checkpoint": str(ckpt),
        "episodes": args.episodes, "seed": args.seed,
        "use_sensors": use_sensors,
        "controllers": results,
    })
    save_csv(out_dir / "data" / "eval_episodes.csv", episode_rows)

    # copy the training curve next to the evaluation results
    hist_path = run_dir / "data" / "training_history.csv"
    if hist_path.exists():
        (out_dir / "data" / "training_curve.csv").write_text(
            hist_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(banner("evaluation complete"))
    return 0


def _trace_to_json(tr: dict) -> dict:
    """Convert a rollout trace into compact JSON for the browser demo."""
    def arr(x, nd=3):
        a = np.asarray(x, dtype=float)
        return [None if not np.isfinite(v) else round(float(v), nd)
                for v in a.ravel()]

    t = np.asarray(tr["t"], dtype=float)
    pos = np.asarray(tr["pos"], dtype=float)
    vel = np.asarray(tr["vel"], dtype=float)
    est = np.asarray(tr["est_pos"], dtype=float)
    dist = np.asarray(tr["dist_est"], dtype=float)
    pred = np.asarray(tr["dist_pred"], dtype=float)
    true_dist = np.asarray(tr["true_dist"], dtype=float)
    return {
        "t": [round(float(v), 3) for v in t],
        "pos": [round(float(v), 3) for v in pos.ravel()],
        "vel": [round(float(v), 3) for v in vel.ravel()],
        "est_pos": [round(float(v), 3) for v in est.ravel()],
        "tilt_deg": [round(float(v), 3) for v in np.asarray(tr["tilt_deg"])],
        "throttle": [round(float(v), 4) for v in np.asarray(tr["throttle"])],
        "dist_est": [round(float(v), 4) for v in dist.ravel()],
        "dist_pred": [round(float(v), 4) for v in pred.ravel()],
        "true_dist": [round(float(v), 4) for v in true_dist.ravel()],
        "n": int(len(t)),
        "success": bool(tr["success"][0]),
        "lateral": round(float(tr["lateral"][0]), 3),
        "speed": round(float(tr["speed"][0]), 3),
        "tilt_final": round(float(tr["tilt_deg_final"][0]), 3),
        "fuel_used": round(float(tr["fuel_used"][0]), 1),
        "total_reward": round(float(tr["total_reward"][0]), 2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
