"""Thrust-degradation stratified evaluation.

The headline Monte-Carlo result averages over the whole off-nominal envelope, so
a modest mean gain can hide where the residual policy actually pays off.  This
script instead evaluates the analytic baseline and the residual policy on
*narrow thrust-degradation bands*, on identical seeds (paired), and reports
success with a Wilson interval plus a McNemar-style discordant-pair count.

The degradation factor is drawn in ``RecoveryEnv.reset`` from
``rocket.thrust_scale_range``; at full curriculum it is simply
``uniform(lo, hi)``.  Narrowing that range to a band therefore stratifies the
evaluation without touching the simulator.  Because the draw consumes exactly
one RNG value regardless of ``lo``/``hi``, the wind and initial conditions are
identical across bands for a given seed -- the only thing that changes between
bands is the delivered thrust.

Outputs (under ``runs/<run>/``):
    data/degradation.json      per-band stats and deltas
    data/degradation.csv       one row per (band, controller, episode)
    figures/fig14_degradation.png

Usage:
    python scripts/degradation.py --run runs/ppo_v3 --episodes 100
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zq3rl.config import Config          # noqa: E402
from zq3rl.env import RecoveryEnv        # noqa: E402
from zq3rl.ppo import PPOTrainer         # noqa: E402
from zq3rl.utils import (                # noqa: E402
    banner, save_json, set_global_seed, wilson_interval,
)

#: (lo, hi, label) -- the top band is the nominal end of the envelope; the
#: bands below it probe progressively worse engine health.  A pilot run showed
#: both controllers reach 0% below ~0.80 (the descent can no longer be
#: arrested at all), so the informative region is 0.82-1.00 and the bands are
#: placed there; the single bottom band documents the infeasible floor.
BANDS = [
    (0.98, 1.00, "1.00"),
    (0.94, 0.98, "0.94-0.98"),
    (0.90, 0.94, "0.90-0.94"),
    (0.86, 0.90, "0.86-0.90"),
    (0.82, 0.86, "0.82-0.86"),
    (0.74, 0.82, "0.74-0.82"),
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Thrust-degradation stratified study.")
    ap.add_argument("--run", type=str, default="runs/ppo_v3")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=100,
                    help="episodes per band per controller")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args(argv)


def _binomial_p(w: int, l: int) -> float:
    """Two-sided exact binomial p-value on w wins in w+l discordant pairs."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run_band(cfg, env, trainer, lo, hi, n, seed0, use_rl):
    """Roll out ``n`` paired episodes with the thrust scale confined to [lo,hi]."""
    cfg.rocket.thrust_scale_range = (lo, hi)
    succ, lats, spds, scales, mags = [], [], [], [], []
    rmax = cfg.guidance.residual_max
    for i in range(n):
        env.reset(seed=seed0 + i)
        scales.append(float(env.thrust_scale))
        obs = env._observation()
        done = trunc = False
        ep_mag = []
        while not (done or trunc):
            if use_rl:
                a = trainer.net.act(obs, deterministic=True)[0]
                ep_mag.append(float(np.linalg.norm(np.clip(a, -1, 1) * rmax)))
            else:
                a = np.zeros(3)
            obs, r, done, trunc, info = env.step(a)
        succ.append(int(info["success"]))
        lats.append(info["lateral"])
        spds.append(info["speed"])
        mags.append(float(np.mean(ep_mag)) if ep_mag else 0.0)
    return {
        "success": np.array(succ),
        "lateral": np.array(lats),
        "speed": np.array(spds),
        "thrust_scale": np.array(scales),
        "residual": np.array(mags),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(8)
    cfg = Config()
    set_global_seed(args.seed)
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    env = RecoveryEnv(cfg, seed=args.seed, curriculum_level=1.0,
                      use_prediction=True, use_residual=True, use_sensors=True)
    trainer = PPOTrainer(env, cfg.ppo, seed=args.seed, device="cpu")
    ckpt = Path(args.model) if args.model else run_dir / "models" / "final.pt"
    if ckpt.exists():
        trainer.load(str(ckpt))
        print(f"loaded {ckpt} (step {trainer.step_count:,})")
    else:
        print(f"!! {ckpt} missing -- RL rows will use an untrained network")
    trainer.net.eval()

    print(banner("ZQ3-Recovery-RL  ::  thrust-degradation stratified study"))
    print(f"episodes per band per controller : {args.episodes}")
    print(f"bands                            : {len(BANDS)}")
    print()
    print(f"{'band':>11s} {'ts_mean':>8s} {'base%':>7s} {'RL%':>7s} "
          f"{'gain':>7s} {'wins':>5s} {'loss':>5s} {'p':>7s} {'|a_rl|':>7s}")
    print("-" * 82)

    rows = []
    bands_out = {}
    for lo, hi, label in BANDS:
        b = run_band(cfg, env, trainer, lo, hi, args.episodes, args.seed, False)
        r = run_band(cfg, env, trainer, lo, hi, args.episodes, args.seed, True)
        nb, nr = int(b["success"].sum()), int(r["success"].sum())
        n = args.episodes
        lo_b, hi_b = wilson_interval(nb, n)
        lo_r, hi_r = wilson_interval(nr, n)
        win = int(((r["success"] == 1) & (b["success"] == 0)).sum())
        loss = int(((r["success"] == 0) & (b["success"] == 1)).sum())
        p = _binomial_p(win, loss)
        bands_out[label] = {
            "band": [lo, hi],
            "thrust_scale_mean": float(b["thrust_scale"].mean()),
            "n": n,
            "baseline": {"successes": nb, "success_rate": nb / n,
                         "ci95": [lo_b, hi_b],
                         "lateral_mean": float(b["lateral"].mean()),
                         "speed_mean": float(b["speed"].mean())},
            "residual_rl": {"successes": nr, "success_rate": nr / n,
                            "ci95": [lo_r, hi_r],
                            "lateral_mean": float(r["lateral"].mean()),
                            "speed_mean": float(r["speed"].mean()),
                            "residual_mean": float(r["residual"].mean())},
            "gain_pp": 100.0 * (nr - nb) / n,
            "paired_wins": win, "paired_losses": loss, "mcnemar_p": p,
        }
        print(f"{label:>11s} {b['thrust_scale'].mean():8.3f} "
              f"{100*nb/n:6.1f}% {100*nr/n:6.1f}% {100*(nr-nb)/n:+6.1f}pp "
              f"{win:5d} {loss:5d} {p:7.4f} {r['residual'].mean():7.3f}")
        for i in range(n):
            rows.append({
                "band": label, "thrust_scale": round(float(b["thrust_scale"][i]), 4),
                "controller": "baseline", "success": int(b["success"][i]),
                "lateral": round(float(b["lateral"][i]), 3),
                "speed": round(float(b["speed"][i]), 3), "residual": 0.0,
            })
            rows.append({
                "band": label, "thrust_scale": round(float(r["thrust_scale"][i]), 4),
                "controller": "residual_rl", "success": int(r["success"][i]),
                "lateral": round(float(r["lateral"][i]), 3),
                "speed": round(float(r["speed"][i]), 3),
                "residual": round(float(r["residual"][i]), 3),
            })

    save_json(out_dir / "data" / "degradation.json", {
        "run": str(run_dir), "checkpoint": str(ckpt),
        "episodes_per_band": args.episodes, "seed": args.seed,
        "bands": bands_out,
    })
    csv_path = out_dir / "data" / "degradation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _figure(bands_out, out_dir / "figures" / "fig14_degradation.png")

    print()
    print(banner("degradation study complete"))
    print(f"  {out_dir / 'data' / 'degradation.json'}")
    print(f"  {csv_path}")
    print(f"  {out_dir / 'figures' / 'fig14_degradation.png'}")
    return 0


def _figure(bands: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reuse the project's CJK font setup so labels render
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_figures as mf
        mf._configure_cjk_font()
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": [mf._CJK, "DejaVu Sans"],
                             "axes.unicode_minus": False})
        c_base, c_prop = mf.C_BASE, mf.C_PROP
    except Exception:  # noqa: BLE001
        c_base, c_prop = "#e0a02a", "#2f7fd0"

    labels = list(bands.keys())
    ts = [bands[k]["thrust_scale_mean"] for k in labels]
    sb = [bands[k]["baseline"]["success_rate"] * 100 for k in labels]
    sr = [bands[k]["residual_rl"]["success_rate"] * 100 for k in labels]
    eb = [[sb[i] - bands[k]["baseline"]["ci95"][0] * 100,
           bands[k]["baseline"]["ci95"][1] * 100 - sb[i]]
          for i, k in enumerate(labels)]
    er = [[sr[i] - bands[k]["residual_rl"]["ci95"][0] * 100,
           bands[k]["residual_rl"]["ci95"][1] * 100 - sr[i]]
          for i, k in enumerate(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    x = np.arange(len(labels))
    ax = axes[0]
    ax.errorbar(x, sb, yerr=np.array(eb).T, marker="o", color=c_base, lw=1.6,
                capsize=3, label="基线（解析制导）")
    ax.errorbar(x, sr, yerr=np.array(er).T, marker="s", color=c_prop, lw=1.6,
                capsize=3, label="本发明（残差 RL）")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("交付推力 / 额定推力"); ax.set_ylabel("成功率 (%)")
    ax.set_title("(a) 成功率随推力退化")
    ax.set_ylim(-4, 104); ax.legend(fontsize=8)

    ax = axes[1]
    gain = [bands[k]["gain_pp"] for k in labels]
    colors = [c_prop if g > 0 else c_base for g in gain]
    b = ax.bar(x, gain, color=colors, edgecolor="#333", linewidth=0.7)
    for rect, g in zip(b, gain):
        ax.text(rect.get_x() + rect.get_width() / 2, g,
                f"{g:+.0f}", ha="center",
                va="bottom" if g >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("交付推力 / 额定推力"); ax.set_ylabel("成功率增益 (百分点)")
    ax.set_title("(b) 本发明相对基线的增益")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
