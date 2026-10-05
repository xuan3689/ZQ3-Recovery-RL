"""Ablation and Monte-Carlo study.

Runs a controlled set of controller variants over identical disturbance
realisations so that every difference is attributable to the component under
test.  Produces the tables used in the report's 有益效果 section.

Variants
--------
===================== ============ ========== ===========
variant               prediction   residual   sensors
===================== ============ ========== ===========
``A_baseline``        no           no         real
``B_baseline_pred``   yes          no         real
``C_residual_only``   no           yes        real
``D_full``            yes          yes        real
``E_ideal_sensing``   yes          yes        perfect
===================== ============ ========== ===========

``C`` isolates the value of the RL residual, ``B`` the value of prediction,
and ``D`` the combination.  ``E`` bounds how much of the remaining error is
attributable to state estimation rather than to control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zq3rl.config import Config                        # noqa: E402
from zq3rl.env import RecoveryEnv                      # noqa: E402
from zq3rl.ppo import PPOTrainer                       # noqa: E402
from zq3rl.utils import (                              # noqa: E402
    banner, save_csv, save_json, set_global_seed, summarise, wilson_interval,
)

VARIANTS = [
    ("A_baseline",      False, False, True,  "基线（无预测/无RL）"),
    ("B_baseline_pred", True,  False, True,  "基线 + 预测感知前馈"),
    ("C_residual_only", False, True,  True,  "基线 + 残差RL（无预测）"),
    ("D_full",          True,  True,  True,  "本发明（预测感知 + 残差RL）"),
    ("E_ideal_sensing", True,  True,  False, "本发明 + 理想感知（上界）"),
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ablation / Monte-Carlo study.")
    ap.add_argument("--run", type=str, default="runs/ppo_main")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()
    set_global_seed(args.seed)
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    print(banner("ZQ3-Recovery-RL  ::  ablation study"))
    print(f"episodes per variant: {args.episodes}")

    # one trainer instance; the network is shared across the residual variants
    env0 = RecoveryEnv(cfg, seed=args.seed)
    trainer = PPOTrainer(env0, cfg.ppo, seed=args.seed, device="cpu")
    ckpt = Path(args.model) if args.model else run_dir / "models" / "final.pt"
    if ckpt.exists():
        trainer.load(str(ckpt))
        print(f"loaded {ckpt} (step {trainer.step_count:,})")
    else:
        print(f"!! {ckpt} missing -- residual variants will use an untrained net")
    trainer.net.eval()

    summary = {}
    rows = []
    print(f"\n{'variant':18s} {'succ%':>7} {'CI95':>16} {'lat_m':>8} "
          f"{'spd':>7} {'tilt°':>7} {'fuel_kg':>8} {'clip%':>7}")
    print("-" * 88)

    for name, pred, res, sens, label in VARIANTS:
        env = RecoveryEnv(cfg, seed=args.seed, curriculum_level=1.0,
                          use_prediction=pred, use_residual=res,
                          use_sensors=sens)
        succ = 0
        lats, spds, tilts, fuels, clips = [], [], [], [], []
        for i in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + i)
            done = trunc = False
            while not (done or trunc):
                if res:
                    a, _, _ = trainer.net.act(obs, deterministic=True)
                else:
                    a = np.zeros(3)
                obs, r, done, trunc, info = env.step(a)
            succ += int(info["success"])
            lats.append(info["lateral"]); spds.append(info["speed"])
            tilts.append(info["tilt_deg"]); fuels.append(info["fuel_used"])
            clips.append(1.0 if info["shield_clip"] else 0.0)
            rows.append({
                "variant": name, "label": label, "episode": i,
                "seed": args.seed + i, "success": int(info["success"]),
                "lateral": info["lateral"], "speed": info["speed"],
                "tilt_deg": info["tilt_deg"], "fuel_used": info["fuel_used"],
                "shield_clip": int(info["shield_clip"]),
            })
        lo, hi = wilson_interval(succ, args.episodes)
        summary[name] = {
            "label": label,
            "prediction": pred, "residual_rl": res, "real_sensors": sens,
            "n": args.episodes, "successes": succ,
            "success_rate": succ / args.episodes,
            "success_ci95": [lo, hi],
            "lateral": summarise(lats), "speed": summarise(spds),
            "tilt": summarise(tilts), "fuel": summarise(fuels),
            "shield_clip_rate": float(np.mean(clips)),
        }
        print(f"{name:18s} {100*succ/args.episodes:6.1f}% "
              f"[{lo*100:5.1f},{hi*100:5.1f}] {np.mean(lats):8.2f} "
              f"{np.mean(spds):7.2f} {np.mean(tilts):7.2f} "
              f"{np.mean(fuels):8.0f} {100*np.mean(clips):6.1f}%")

    # ---- deltas that the report quotes directly -------------------------
    d = summary["D_full"]; a = summary["A_baseline"]
    b = summary["B_baseline_pred"]; c = summary["C_residual_only"]
    e = summary["E_ideal_sensing"]
    deltas = {
        "success_gain_total_pp": 100 * (d["success_rate"] - a["success_rate"]),
        "success_gain_from_prediction_pp": 100 * (b["success_rate"] - a["success_rate"]),
        "success_gain_from_residual_pp": 100 * (c["success_rate"] - a["success_rate"]),
        "lateral_reduction_m": a["lateral"]["mean"] - d["lateral"]["mean"],
        "lateral_reduction_pct": 100 * (1 - d["lateral"]["mean"]
                                        / max(a["lateral"]["mean"], 1e-9)),
        "lateral_p95_reduction_m": a["lateral"]["p95"] - d["lateral"]["p95"],
        "speed_reduction_ms": a["speed"]["mean"] - d["speed"]["mean"],
        "estimation_headroom_pp": 100 * (e["success_rate"] - d["success_rate"]),
    }

    save_json(out_dir / "data" / "ablation_summary.json", {
        "run": str(run_dir), "checkpoint": str(ckpt),
        "episodes": args.episodes, "seed": args.seed,
        "variants": summary, "deltas": deltas,
    })
    save_csv(out_dir / "data" / "ablation_episodes.csv", rows)

    print(banner("key deltas"))
    for k, v in deltas.items():
        print(f"  {k:36s}: {v:+.3f}")
    print(banner("ablation complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
