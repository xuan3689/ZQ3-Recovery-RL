"""Generate every figure used in the course report.

Figures are written to ``runs/<run>/figures/`` as both PNG (for the report) and
(where useful) SVG.  All figures are monochrome-friendly greyscale-safe where
the patent specification requires it: the *patent* figures (流程图, 结构图) are
rendered in pure black-and-white, while the *analysis* figures use colour
because the report itself is a colour PDF.

Usage
-----
    python scripts/make_figures.py --run runs/ppo_main
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zq3rl.config import Config                      # noqa: E402
from zq3rl.utils import banner                        # noqa: E402

# --------------------------------------------------------------------------- #
#  Style
# --------------------------------------------------------------------------- #
def _configure_cjk_font() -> str:
    """Pick an installed CJK font so the Chinese axis labels render.

    All figure text is Chinese, so without this every glyph would be a tofu box.
    ``SimHei`` (黑体) is present on every Windows install and is also the font the
    patent specification prefers for figures.
    """
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                      "Source Han Sans SC", "SimSun", "DengXian"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


_CJK = _configure_cjk_font()

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.edgecolor": "#333333",
    "font.family": "sans-serif",
    "font.sans-serif": [_CJK, "DejaVu Sans"],
    "axes.unicode_minus": False,   # the CJK fonts carry a proper minus sign
    # Log/symlog tick labels are rendered by mathtext, which uses its own font
    # set.  The default ('dejavusans') has U+2212, but when it is left to follow
    # the CJK font the exponent's minus sign becomes a tofu box; pinning the
    # font set keeps the tick labels intact.
    "mathtext.fontset": "dejavusans",
})

C_BASE = "#e0a02a"
C_PROP = "#2f7fd0"
C_TRUE = "#444444"
C_ACC = "#c0392b"
C_OK = "#2e9e5b"
C_GREY = "#8a8a8a"


def _plain_ticks(ax, which: str) -> None:
    """Label log/symlog ticks with plain decimal numbers instead of mathtext.

    Matplotlib's default log formatter renders exponents through mathtext, and
    the exponent's minus sign (U+2212) is missing from SimHei, so the tick
    labels come out with tofu boxes.  Formatting *both* the major and the minor
    ticks as ordinary decimal strings sidesteps mathtext completely; the minor
    ticks matter because the log locator labels them too.
    """
    import matplotlib.ticker as mticker
    axis = ax.yaxis if which == "y" else ax.xaxis
    fmt = mticker.FuncFormatter(lambda v, _p: f"{v:g}")
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(fmt)


def save(fig, out_dir: Path, name: str, svg: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    if svg:
        fig.savefig(out_dir / f"{name}.svg")
    plt.close(fig)
    print(f"  wrote {name}.png")


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_history(run: Path) -> dict:
    p = run / "data" / "training_history.csv"
    if not p.exists():
        return {}
    rows = []
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in r.items()})
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def load_eval(run: Path) -> dict:
    p = run / "data" / "eval_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_traces(run: Path) -> dict:
    p = run / "data" / "traces.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------------------------------------------------------- #
#  Patent-style flow charts (pure black & white)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fs=8.5, lw=1.1, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor="black", facecolor="white",
        linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.45, color="black")


def _arrow(ax, x0, y0, x1, y1, lw=1.0, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color="black",
        linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


def fig_flowchart(out: Path) -> None:
    """图1 —— 本发明方法总体流程图（摘要附图）。"""
    fig, ax = plt.subplots(figsize=(5.6, 8.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    steps = [
        "步骤一  多源传感器数据采集\n100 Hz IMU · 10 Hz GNSS · 20 Hz 气压高度",
        "步骤二  多源融合状态估计\n导航EKF（位置/速度/扰动加速度）\n姿态EKF（倾角/方位/陀螺零偏）",
        "步骤三  气动扰动在线辨识与预测\n由扰动估计反演风矢量 w\n沿预测轨迹外推气动加速度 a_pred",
        "步骤四  分层解析制导生成标称指令\na_nom = a_vert(v_ref) + a_lat(位置-速度级联)",
        "步骤五  预测感知前馈补偿\na_cmd = a_nom - K_ff · a_pred",
        "步骤六  残差强化学习策略输出有界修正\na_rl = π_θ(o) ,  ‖a_rl‖ ≤ a_max_res",
        "步骤七  安全盾投影至可行域\n推力幅值 · 垂直推力下限 · 倾角权限",
        "步骤八  执行机构动作\n变推力发动机节流 + 双向矢量摆角",
        "步骤九  闭环反馈：新状态返回步骤一\n直至触地判定（精度/速度/倾角）",
    ]
    n = len(steps)
    top, bot = 0.965, 0.035
    h = (top - bot) / n * 0.70
    gap = (top - bot) / n
    for i, s in enumerate(steps):
        y = top - (i + 1) * gap + (gap - h) / 2
        _box(ax, 0.045, y, 0.91, h, s, fs=8.0)
        if i < n - 1:
            _arrow(ax, 0.5, y, 0.5, y - (gap - h))
    # feedback loop
    _arrow(ax, 0.045, 0.5 * (top + bot), 0.018, 0.5 * (top + bot), lw=0.9)
    ax.plot([0.018, 0.018], [0.5 * (top + bot), top - 0.5 * gap],
            color="black", lw=0.9, linestyle="--")
    _arrow(ax, 0.018, top - 0.5 * gap, 0.045, top - 0.5 * gap, lw=0.9)
    ax.text(0.022, 0.5, "闭环反馈", rotation=90, va="center", ha="left",
            fontsize=7.5, color="black")

    save(fig, out, "fig01_flowchart", svg=True)


def fig_architecture(out: Path) -> None:
    """图2 —— 系统总体结构框图（分层，箭头不交叉）。"""
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    bands = [
        (0.795, 0.155, "感知层", [
            (0.105, 0.255, "传感器组\nIMU 100 Hz · GNSS 10 Hz\n气压高度计 20 Hz"),
            (0.385, 0.255, "状态估计\n导航 EKF（p, v, a_b）\n姿态 EKF（θ, θ_dot, ψ, b_g）"),
            (0.665, 0.255, "预测感知\n风矢量反演 w\n高度剖面拟合 w(h)\n前向预测 a_pred"),
        ]),
        (0.525, 0.155, "决策层", [
            (0.105, 0.255, "分层解析制导\na_vert：速度剖面跟踪\na_lat：位置-速度级联"),
            (0.385, 0.255, "有界残差强化学习\nPPO 策略 π_θ(o)\ntanh 压缩，‖a_rl‖ ≤ 2 m/s^2"),
            (0.665, 0.255, "指令合成\na_cmd = a_nom + a_rl"),
        ]),
        (0.255, 0.155, "安全层", [
            (0.105, 0.255, "安全盾（一）\n残差盒约束\n垂直推力分量下限"),
            (0.385, 0.255, "安全盾（二）\n推力幅值限幅\n倾角权限调度 25° → 8°"),
            (0.665, 0.255, "可行执行指令\n节流比 + 矢量摆角"),
        ]),
        (0.055, 0.115, "执行层", [
            (0.105, 0.255, "变推力发动机\n一阶惯性 0.15 s"),
            (0.385, 0.255, "双向矢量机构\n摆角速率 30°/s"),
            (0.665, 0.255, "箭体动力学\n3-DOF 平动 + 姿态"),
        ]),
    ]

    for y, h, label, boxes in bands:
        ax.text(0.030, y + h / 2, label, rotation=90, va="center", ha="center",
                fontsize=8.6, fontweight="bold", color="black")
        for (bx, bw, text) in boxes:
            _box(ax, bx, y, bw, h, text, fs=7.5)
        for i in range(len(boxes) - 1):
            _arrow(ax, boxes[i][0] + boxes[i][1], y + h / 2,
                   boxes[i + 1][0], y + h / 2)

    # vertical flow between bands, all on the right-hand column (no crossings)
    for (y0, y1) in [(0.795, 0.680), (0.525, 0.410), (0.255, 0.170)]:
        _arrow(ax, 0.792, y0, 0.792, y1)
    # execution returns to the sensors as the closed loop
    ax.plot([0.665, 0.075], [0.112, 0.112], color="black", lw=0.9, ls="--")
    ax.plot([0.075, 0.075], [0.112, 0.872], color="black", lw=0.9, ls="--")
    _arrow(ax, 0.075, 0.872, 0.105, 0.872, lw=0.9)
    ax.text(0.088, 0.49, "闭环反馈", rotation=90, va="center", ha="left",
            fontsize=7.2, color="black")

    save(fig, out, "fig02_architecture", svg=True)


def fig_guidance_detail(out: Path) -> None:
    """图3 —— 分层制导与残差修正的信号流。"""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _box(ax, 0.02, 0.60, 0.20, 0.28,
         "高度 h\n垂直速度 v_y", fs=8.0)
    _box(ax, 0.26, 0.60, 0.26, 0.28,
         "下降速度剖面\nv_ref(h) = -min(v_max,\n√(v_t^2 + 2 a_ref h))", fs=7.6)
    _box(ax, 0.56, 0.60, 0.20, 0.28,
         "垂直通道\nPD 跟踪 + 前馈\n dv_ref/dh · v_y", fs=7.6)
    _box(ax, 0.02, 0.20, 0.20, 0.28,
         "水平位置偏差\nΔx, Δz", fs=8.0)
    _box(ax, 0.26, 0.20, 0.26, 0.28,
         "位置→速度级联\nv_cmd = K_p Δx\n饱和限幅", fs=7.6)
    _box(ax, 0.56, 0.20, 0.20, 0.28,
         "水平通道\nPD 跟踪\nK_d (v_cmd - v)", fs=7.6)

    _box(ax, 0.80, 0.40, 0.18, 0.28,
         "合成\n a_nom\n(标称)", fs=8.2)

    _arrow(ax, 0.22, 0.74, 0.26, 0.74)
    _arrow(ax, 0.52, 0.74, 0.56, 0.74)
    _arrow(ax, 0.22, 0.34, 0.26, 0.34)
    _arrow(ax, 0.52, 0.34, 0.56, 0.34)
    _arrow(ax, 0.76, 0.74, 0.80, 0.60)
    _arrow(ax, 0.76, 0.34, 0.80, 0.48)
    ax.text(0.89, 0.30, "→ 安全盾", fontsize=8, ha="center")

    save(fig, out, "fig03_guidance_detail", svg=True)


# --------------------------------------------------------------------------- #
#  Training curves
# --------------------------------------------------------------------------- #
def fig_training(h: dict, out: Path) -> None:
    if not h:
        print("  [skip] no training history")
        return
    steps = h["step"] / 1000.0
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.6))

    ax = axes[0, 0]
    ax.plot(steps, h["reward"], color=C_PROP, lw=1.6)
    ax.set_title("(a) 回合回报（滑动平均）")
    ax.set_xlabel("训练步数 (×10^3)"); ax.set_ylabel("回报")
    ax.axhline(0, color=C_GREY, lw=0.7, ls=":")

    ax = axes[0, 1]
    ax.plot(steps, h["success"] * 100, color=C_OK, lw=1.6)
    ax.set_title("(b) 近 40 回合成功率")
    ax.set_xlabel("训练步数 (×10^3)"); ax.set_ylabel("成功率 (%)")
    ax.set_ylim(-3, 103)

    ax = axes[0, 2]
    ax.plot(steps, h["lateral"], color=C_ACC, lw=1.6, label="着陆偏差")
    ax.set_title("(c) 着陆精度")
    ax.set_xlabel("训练步数 (×10^3)"); ax.set_ylabel("偏差 (m)")
    ax.axhline(3.0, color=C_GREY, lw=0.8, ls="--", label="成功阈值 3 m")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.plot(steps, h["speed"], color="#8e44ad", lw=1.6)
    ax.set_title("(d) 触地速度")
    ax.set_xlabel("训练步数 (×10^3)"); ax.set_ylabel("速度 (m/s)")
    ax.axhline(2.5, color=C_GREY, lw=0.8, ls="--")

    ax = axes[1, 1]
    ax.plot(steps, h["curriculum"], color="#16a085", lw=1.8)
    ax.set_title("(e) 课程难度")
    ax.set_xlabel("训练步数 (×10^3)"); ax.set_ylabel("课程等级")
    ax.set_ylim(-0.02, 1.05)

    ax = axes[1, 2]
    ax.plot(steps, h["approx_kl"], color="#d35400", lw=1.2)
    ax.set_title("(f) 策略更新幅度 (KL)")
    ax.set_xlabel("训练步数 (×10^3)"); ax.set_ylabel("近似 KL")

    fig.tight_layout()
    save(fig, out, "fig04_training_curves")


# --------------------------------------------------------------------------- #
#  Trajectories
# --------------------------------------------------------------------------- #
def _trace_arr(t: dict, key: str, nd: int = 3) -> np.ndarray:
    return np.asarray(t[key], dtype=float).reshape(-1, nd)


def fig_trajectory(traces: dict, out: Path, idx: int = 0) -> None:
    if not traces:
        print("  [skip] no traces")
        return
    C = traces["controllers"]
    base = C["baseline"][idx]
    prop = C["residual_rl"][idx]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (a) vertical profile
    ax = axes[0]
    pb = _trace_arr(base, "pos"); pp = _trace_arr(prop, "pos")
    ax.plot(np.asarray(base["t"]), pb[:, 1], color=C_BASE, lw=1.8, label="基线")
    ax.plot(np.asarray(prop["t"]), pp[:, 1], color=C_PROP, lw=1.8, label="本发明")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("高度 (m)")
    ax.set_title("(a) 高度剖面")
    ax.legend()
    ax.axhline(0, color=C_GREY, lw=0.6)

    # (b) ground track
    ax = axes[1]
    ax.plot(pb[:, 0], pb[:, 2], color=C_BASE, lw=1.6, label="基线")
    ax.plot(pp[:, 0], pp[:, 2], color=C_PROP, lw=1.6, label="本发明")
    ax.plot(0, 0, marker="*", ms=13, color=C_OK, label="着陆坪")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("(b) 水平轨迹")
    ax.axis("equal"); ax.legend(loc="best")

    # (c) lateral error vs altitude
    ax = axes[2]
    eb = np.hypot(pb[:, 0], pb[:, 2])
    ep = np.hypot(pp[:, 0], pp[:, 2])
    ax.plot(pb[:, 1], eb, color=C_BASE, lw=1.6, label="基线")
    ax.plot(pp[:, 1], ep, color=C_PROP, lw=1.6, label="本发明")
    ax.set_xlabel("高度 (m)"); ax.set_ylabel("水平偏差 (m)")
    ax.set_title("(c) 偏差收敛")
    ax.legend()
    ax.set_xlim(left=0)

    fig.tight_layout()
    save(fig, out, f"fig05_trajectory_s{idx + 1}")


def fig_trajectory_multi(traces: dict, out: Path) -> None:
    """Small-multiples: every exported scenario, baseline vs proposed."""
    if not traces:
        return
    C = traces["controllers"]
    n = len(C["baseline"])
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), squeeze=False)
    for i in range(n):
        ax = axes[0][i]
        b = C["baseline"][i]; p = C["residual_rl"][i]
        pb = _trace_arr(b, "pos"); pp = _trace_arr(p, "pos")
        ax.plot(pb[:, 0], pb[:, 2], color=C_BASE, lw=1.5, label="基线")
        ax.plot(pp[:, 0], pp[:, 2], color=C_PROP, lw=1.5, label="本发明")
        ax.plot(0, 0, marker="*", ms=12, color=C_OK)
        ax.set_title(f"场景 {i + 1}\n基线 {b['lateral']:.1f} m / 本发明 {p['lateral']:.1f} m",
                     fontsize=8.5)
        ax.set_xlabel("x (m)")
        if i == 0:
            ax.set_ylabel("z (m)")
        ax.axis("equal")
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig06_trajectory_multi")


def fig_estimation(traces: dict, out: Path, idx: int = 0) -> None:
    """Estimator performance: truth vs estimate for position, velocity, disturbance."""
    if not traces:
        return
    t = traces["controllers"]["residual_rl"][idx]
    tt = np.asarray(t["t"])
    pos = _trace_arr(t, "pos"); est = _trace_arr(t, "est_pos")
    vel = _trace_arr(t, "vel")
    de = _trace_arr(t, "dist_est"); dp = _trace_arr(t, "dist_pred")
    dt_ = _trace_arr(t, "true_dist")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))

    ax = axes[0]
    ax.plot(tt, np.linalg.norm(pos - est, axis=1), color=C_PROP, lw=1.5)
    ax.set_title("(a) 位置估计误差")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("误差 (m)")
    ax.set_yscale("log")
    _plain_ticks(ax, "y")

    ax = axes[1]
    ax.plot(tt, np.linalg.norm(dt_, axis=1), color=C_TRUE, lw=1.5, label="真实扰动")
    ax.plot(tt, np.linalg.norm(de, axis=1), color=C_BASE, lw=1.4, ls="--",
            label="EKF 估计")
    ax.plot(tt, np.linalg.norm(dp, axis=1), color=C_PROP, lw=1.4, ls=":",
            label="预测感知前向预测")
    ax.set_title("(b) 气动扰动估计与预测")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("加速度 (m/s^2)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(tt, vel[:, 0], color="#7f8c8d", lw=1.4, label="v_x 真实")
    ax.plot(tt, np.asarray(t["vel"], dtype=float).reshape(-1, 3)[:, 1],
            color=C_ACC, lw=1.4, label="v_y 真实")
    ax.set_title("(c) 速度剖面")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("速度 (m/s)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, out, "fig07_estimation")


def fig_disturbance_forecast(traces: dict, out: Path, idx: int = 0) -> None:
    """Zoom on the wind shear: how far ahead the forecast sees."""
    if not traces:
        return
    t = traces["controllers"]["residual_rl"][idx]
    tt = np.asarray(t["t"])
    dt_ = _trace_arr(t, "true_dist")
    de = _trace_arr(t, "dist_est")
    dp = _trace_arr(t, "dist_pred")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for k, (lbl, col) in enumerate([("x", "#c0392b"), ("y", "#27ae60"), ("z", "#2980b9")]):
        ax.plot(tt, dt_[:, k], color=col, lw=1.4, label=f"真实 a_{lbl}")
        ax.plot(tt, de[:, k], color=col, lw=1.0, ls="--", alpha=0.8)
    ax.set_title("(a) 三分量真实扰动与估计")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("加速度 (m/s^2)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    err_inst = np.linalg.norm(de - dt_, axis=1)
    err_pred = np.linalg.norm(dp - dt_, axis=1)
    ax.plot(tt, err_inst, color=C_BASE, lw=1.5, label="瞬时估计误差")
    ax.plot(tt, err_pred, color=C_PROP, lw=1.5, label="前向预测误差")
    ax.set_title("(b) 估计与前向预测的误差对比")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("误差 (m/s^2)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, out, "fig08_disturbance_forecast")


# --------------------------------------------------------------------------- #
#  Ablation / Monte-Carlo
# --------------------------------------------------------------------------- #
def fig_ablation(eval_summary: dict, out: Path) -> None:
    if not eval_summary:
        print("  [skip] no eval summary")
        return
    C = eval_summary["controllers"]
    names = ["baseline", "baseline_pred", "residual_rl"]
    labels = ["基线\n(无预测/无RL)", "基线+预测感知", "本发明\n(预测+残差RL)"]
    colors = [C_GREY, C_BASE, C_PROP]

    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.5))

    def bars(ax, key, getter, title, ylabel, lower_better=True, fmt="{:.2f}"):
        vals = [getter(C[n]) for n in names]
        errs = [C[n][key].get("std", 0.0) if isinstance(C[n][key], dict) else 0.0
                for n in names] if False else [0, 0, 0]
        b = ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.7)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=7.5)
        return vals

    ax = axes[0]
    rates = [C[n]["success_rate"] * 100 for n in names]
    ci = [C[n]["success_ci95"] for n in names]
    yerr = [[r - c[0] * 100 for r, c in zip(rates, ci)],
            [c[1] * 100 - r for r, c in zip(rates, ci)]]
    b = ax.bar(labels, rates, color=colors, edgecolor="#333333", linewidth=0.7,
               yerr=yerr, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333"})
    for rect, v in zip(b, rates):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("(a) 着陆成功率"); ax.set_ylabel("成功率 (%)")
    ax.set_ylim(0, 108); ax.tick_params(axis="x", labelsize=7.5)

    ax = axes[1]
    vals = [C[n]["lateral"]["mean"] for n in names]
    b = ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.7)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("(b) 平均着陆偏差"); ax.set_ylabel("偏差 (m)")
    ax.tick_params(axis="x", labelsize=7.5)

    ax = axes[2]
    vals = [C[n]["speed"]["mean"] for n in names]
    b = ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.7)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("(c) 平均触地速度"); ax.set_ylabel("速度 (m/s)")
    ax.tick_params(axis="x", labelsize=7.5)

    ax = axes[3]
    vals = [C[n]["shield_clip_rate"] * 100 for n in names]
    b = ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.7)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("(d) 安全盾触发率"); ax.set_ylabel("触发率 (%)")
    ax.tick_params(axis="x", labelsize=7.5)

    fig.tight_layout()
    save(fig, out, "fig09_ablation")


def fig_lateral_cdf(run: Path, out: Path) -> None:
    """CDF of landing accuracy per controller (from the episode CSV)."""
    p = run / "data" / "eval_episodes.csv"
    if not p.exists():
        return
    groups: dict[str, list[float]] = {}
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            groups.setdefault(r["controller"], []).append(float(r["lateral"]))
    if not groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    styles = {"baseline": (C_BASE, "基线"), "baseline_pred": (C_GREY, "基线+预测"),
              "residual_rl": (C_PROP, "本发明")}

    ax = axes[0]
    for name, vals in groups.items():
        v = np.sort(np.asarray(vals))
        y = np.arange(1, len(v) + 1) / len(v)
        col, lbl = styles.get(name, ("#000", name))
        ax.plot(v, y * 100, color=col, lw=1.7, label=lbl)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("着陆偏差 (m)"); ax.set_ylabel("累积概率 (%)")
    ax.set_title("(a) 着陆偏差累积分布")
    _plain_ticks(ax, "x")
    ax.axvline(3.0, color=C_ACC, lw=0.9, ls="--")
    ax.text(3.2, 12, "成功阈值 3 m", fontsize=7, color=C_ACC, rotation=90)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 102)

    ax = axes[1]
    for name, vals in groups.items():
        col, lbl = styles.get(name, ("#000", name))
        ax.hist(vals, bins=28, color=col, alpha=0.55, label=lbl, edgecolor="none")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("着陆偏差 (m)"); ax.set_ylabel("频数")
    ax.set_title("(b) 着陆偏差分布")
    _plain_ticks(ax, "x")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, out, "fig10_lateral_cdf")


# --------------------------------------------------------------------------- #
#  Reward decomposition / safety
# --------------------------------------------------------------------------- #
def fig_safety(traces: dict, out: Path, idx: int = 0) -> None:
    if not traces:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for name, col, lbl in [("baseline", C_BASE, "基线"), ("residual_rl", C_PROP, "本发明")]:
        t = traces["controllers"][name][idx]
        tt = np.asarray(t["t"])
        ax = axes[0]
        ax.plot(tt, t["tilt_deg"], color=col, lw=1.5, label=lbl)
        ax2 = axes[1]
        ax2.plot(tt, np.asarray(t["throttle"]) * 100, color=col, lw=1.5, label=lbl)
    axes[0].axhline(25, color=C_ACC, lw=0.9, ls="--")
    axes[0].text(0.5, 25.6, "倾角权限上限 25°", fontsize=7, color=C_ACC)
    axes[0].set_title("(a) 姿态倾角"); axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("倾角 (°)"); axes[0].legend(fontsize=8)
    axes[1].axhline(35, color=C_ACC, lw=0.9, ls="--")
    axes[1].text(0.5, 36, "节流下限 35%", fontsize=7, color=C_ACC)
    axes[1].set_title("(b) 发动机节流"); axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("推力 (%)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig11_safety")


def fig_wind_field(out: Path) -> None:
    """Wind profile + the disturbance it produces (illustrative, from config)."""
    cfg = Config()
    h = np.linspace(0, 1800, 400)
    for speed, col, lbl in [(10, "#8e44ad", "10 m/s"), (18, C_PROP, "18 m/s"),
                            (26, C_ACC, "26 m/s")]:
        shear = 1.0 + cfg.wind.shear_gain * np.exp(
            -0.5 * ((h - cfg.wind.shear_alt) / 90.0) ** 2)
        bl = 1.0 + 0.25 * np.exp(-np.maximum(h, 0) / 260.0)
        plt.plot(speed * shear * bl, h, color=col, lw=1.8, label=f"基准风速 {lbl}")
    plt.axhline(cfg.wind.shear_alt, color=C_GREY, lw=0.8, ls="--")
    plt.text(2, cfg.wind.shear_alt + 22, "低空风切变层 (350 m)", fontsize=7.5, color="#444")
    plt.xlabel("风速 (m/s)"); plt.ylabel("高度 (m)")
    plt.title("图：随机风场平均剖面（含低空切变与边界层增长）")
    plt.legend(fontsize=8)
    save(plt.gcf(), out, "fig12_wind_profile")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate report figures.")
    ap.add_argument("--run", type=str, default="runs/ppo_main")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "figures"
    print(banner("ZQ3-Recovery-RL  ::  figures"))
    print(f"run: {run}")
    print(f"out: {out}")

    hist = load_history(run)
    ev = load_eval(run)
    tr = load_traces(run)

    print("patent-style diagrams ...")
    fig_flowchart(out)
    fig_architecture(out)
    fig_guidance_detail(out)

    print("analysis figures ...")
    fig_training(hist, out)
    fig_wind_field(out)
    if tr:
        n = len(tr["controllers"]["baseline"])
        for i in range(min(n, 2)):
            fig_trajectory(tr, out, idx=i)
        fig_trajectory_multi(tr, out)
        fig_estimation(tr, out, idx=0)
        fig_disturbance_forecast(tr, out, idx=0)
        fig_safety(tr, out, idx=0)
    fig_ablation(ev, out)
    fig_lateral_cdf(run, out)

    print(banner(f"figures written to {out}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
