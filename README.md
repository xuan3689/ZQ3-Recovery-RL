# ZQ3-Recovery-RL

[![CI](https://github.com/example/zq3-recovery-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/example/zq3-recovery-rl/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%2B%20torch-lightgrey)

**预测感知 + 残差强化学习** 的运载火箭一子级垂直回收制导方法。

> 一种基于预测感知与残差强化学习的可回收火箭垂直回收制导方法及系统
> —— 具身智能系统项目实践课程作业（专利格式报告）

---

## 1. 一句话概述

把强化学习**嵌进**一个有物理依据的控制器里，而不是用它替换控制器：
先用多源融合 EKF **在线辨识并前向预测**风与气动扰动（预测感知），
再由 PPO 学习一个**有界残差**叠加在分层解析制导基线上（强化学习），
最后由**安全盾**把总指令投影回物理可行域，从而保证约束满足与可解释性。

```
a_cmd = a_nom(状态估计)          ← 分层解析制导（下降速度剖面 + 位置速度级联）
      + a_ff(a_pred)             ← 预测感知前馈（风矢量反演 + 前向外推）
      + a_rl(o)                  ← PPO 残差策略，‖a_rl‖ ≤ 1 m/s²
      → 安全盾投影 → 变推力发动机 + 双向矢量摆角
```

## 2. 为什么这样设计

| 直接端到端 RL | 本方法（残差 RL） |
| --- | --- |
| 需要从零学会"如何下落"，奖励稀疏 | 基线已给出可行轨迹，奖励密集 |
| 可能输出发散指令，约束无法保证 | 残差有界 + 安全盾，约束由构造保证 |
| 策略不可解释，难以工程审查 | 基线部分可解释，RL 只做小幅修正 |
| CPU 上需数小时 | CPU 上数十分钟收敛 |
| 扰动只能靠反馈事后修正 | 预测感知让控制器**提前**对冲风切变 |

## 3. 安装

三种方式任选：

```bash
# ① 可编辑安装（推荐；无需手动设置 PYTHONPATH）
pip install -e ".[plot]"          # 核心 + 绘图；报告另需 ".[report]"

# ② 只装依赖
pip install -r requirements.txt

# ③ Docker（完全可复现，镜像构建时会自动跑测试）
docker build -t zq3-recovery-rl .
docker run --rm zq3-recovery-rl pytest -q
```

运行时依赖仅 `numpy` 与 `torch`（CPU 版即可）；`matplotlib`/`pandas` 仅用于绘图与
统计，`python-docx`/`PyMuPDF` 仅用于生成报告。**不依赖 Gym / Stable-Baselines3**
——PPO 与 RL 环境均为自研实现，便于在报告中逐行引用算法。
浏览器演示额外需要 Three.js（已随仓库内置在 `web/vendor/`，无需联网）。


## 4. 快速开始

> 仓库内已附带一次完整训练的产物 `runs/ppo_v3/`（含评估、消融、13 张附图与
> 专利格式报告），**可直接查看结果而不必重新训练**。下面是从零复现的完整流程。

```bash
# ① 训练（CPU，约 40 分钟，40 万步）
python scripts/train.py --steps 400000 --rollout 2048 --seed 0 --out runs --run-name ppo_v3 --quiet

# ② 评估 + 导出可视化数据（200 局配对蒙特卡洛）
python scripts/evaluate.py --run runs/ppo_v3 --episodes 200 --traces 4 --seed 12345

# ③ 消融实验（分离预测感知 / 残差 RL 各自的贡献）
python scripts/ablation.py --run runs/ppo_v3 --episodes 200

# ④ 生成报告全部附图
python scripts/make_figures.py --run runs/ppo_v3

# ⑤ 生成专利格式报告（docx + pdf）
python scripts/make_report.py --run runs/ppo_v3

# ⑥ 推力退化分层评估（关键实验：证明增益在整个包线上一致）
python scripts/degradation.py --run runs/ppo_v3 --episodes 100

# ⑦ 浏览器 3D 演示（同屏对比：基线 vs 本发明）
cp runs/ppo_v3/data/traces.json web/data/traces.json
python -m http.server 8099 --directory web
#   然后打开 http://localhost:8099
```

## 5. 仓库结构

```
zq3-recovery-rl/
├── src/zq3rl/                 核心库（无 Gym 依赖）
│   ├── config.py              全部物理/任务/奖励/PPO 超参数
│   ├── dynamics.py            3-DOF 平动 + 简化姿态动力学，指数大气，风场
│   ├── sensors.py             IMU / GNSS / 气压计，含零偏与随机游走
│   ├── estimator.py           导航EKF + 姿态EKF + 扰动观测器与风矢量反演
│   ├── guidance.py            分层解析制导 + 预测前馈 + 安全盾
│   ├── env.py                 残差 MDP 环境（29 维观测，全部来自估计量）
│   ├── ppo.py                 自研 PPO（clipped surrogate + GAE）
│   └── utils.py               种子、IO、Wilson 区间、日志
├── scripts/
│   ├── train.py               分四阶段课程训练
│   ├── evaluate.py            蒙特卡洛评估 + 导出 traces.json
│   ├── ablation.py            消融实验（5 个变体）
│   ├── degradation.py         推力退化分层评估（6 波段，配对 + McNemar）
│   ├── make_figures.py        生成报告全部附图
│   ├── make_report.py         生成专利格式 docx + pdf
│   └── finish_report_pdf.py   目标 PDF 被阅读器锁定时完成替换
├── web/                       浏览器 3D 演示（Three.js，离线可用）
├── tests/                     23 项冒烟与回归测试
├── pyproject.toml             打包与 pytest 配置
├── Dockerfile / .dockerignore 可复现的 CPU 环境
├── .github/workflows/ci.yml   CI（py3.10 + py3.13）
└── runs/                      训练与评估产物（checkpoint、曲线、图）
```

## 6. 方法要点

### 6.1 多源融合状态估计（`estimator.py`）

- **导航 EKF**：9 状态 `[p, v, a_b]`，`a_b` 为集总未建模加速度（气动 + 风）。
  100 Hz IMU 传播 + 10 Hz GNSS + 20 Hz 气压计更新，Joseph 形式协方差更新。
- **姿态 EKF**：4 状态 `[θ, θ̇, ψ, b_g]`。关键洞察是**陀螺零偏无法自观测**，
  因此把指令倾角经自动驾驶仪二阶模型前向传播，并以陀螺作为
  `θ̇ + b_g` 的量测——这是标准的互补滤波结构。实测倾角估计误差均值 0.06°。
- **扰动观测器 / 风矢量反演**：由 `a_b = -½ρ|v-w|(v-w)C_dA/m` 反解风矢量

  ```
  |rel|³ = |a_b| · 2m / (ρ C_d A)
  w      = v + (a_b/|a_b|) · |rel|
  ```

  反演风（而非拟合系数）让预测在**减速段自动衰减**，不会发散。

### 6.2 预测感知前馈（`guidance.py`）

沿预测轨迹重新计算气动阻力，得到 `a_pred`，并以 `-K_ff · a_pred` 前馈。
这让控制器在**进入**风切变层之前就开始对冲，而不是等轨迹漂移后再反馈修正。

### 6.3 分层解析制导（`guidance.py`）

单一多项式制导（Apollo E-guidance）对垂直回收是**过约束**的：
垂直通道需要 `T≈19 s` 排空 150 m/s 下沉率，水平通道需要 `T≈31 s` 消除 400 m 偏差，
不存在同时满足两者的单一 `T`。因此按课程手册的"分层控制"原则拆为：

- 垂直通道：跟踪单调下降速度剖面
  `v_ref(h) = -min(v_max, √(v_t² + 2 a_ref h))`，含解析前馈 `v_y · dv_ref/dh`；
- 水平通道：位置 → 速度 → 加速度级联，并受倾角锥约束；
- 姿态：由推力矢量方向反解，安全盾按高度调度权限（25° → 8°）。

### 6.4 残差 PPO（`ppo.py` / `env.py`）

- 动作 = 有界加速度修正，`‖a_rl‖ ≤ 1 m/s²`（`tanh` 压缩，天然可执行）；
- 观测 29 维，**全部来自估计量**（含扰动估计、前向预测、已辨识风速、EKF 不确定度、
  以及由加速度计反推的推力亏空）；
- 奖励 = **势函数塑形** `Φ(s') - Φ(s)` + 姿态/残差努力惩罚 + 终局成败项。
  势函数塑形在折扣因子 `γ = 1` 时可证明**不改变最优策略**，同时把回报从 `O(10⁶)`
  压到 `O(10²)`，显著改善价值函数条件数。
  （注意：`γ < 1` 会破坏势函数塑形的望远镜求和性质，并让终局奖励在长回合里被指数衰减，
  本项目因此固定取 `γ = 1`。）
- 四阶段课程：`0.05→0.30→0.60→1.00` 逐步放开初始条件包线与风速，55% 进度即到满难度。

### 6.5 安全盾（`guidance.py`）

依次强制：残差盒约束 → 垂直推力下限（发动机不能下拉箭体）→
推力幅值 `[a_min, a_max]` → 倾角权限。约束满足由**构造**保证，
不依赖策略是否学到。

## 7. 结果

见 `runs/ppo_v3/figures/`、`runs/ppo_v3/课程报告_专利格式.pdf` 与报告"有益效果"一节。
权威 run 为 **`runs/ppo_v3`**（其余 run 是调试中间产物，见 `runs/README_重要提示.md`）。

**200 局配对蒙特卡洛（相同扰动种子）**：

| 控制器 | 成功率 | 95% 置信区间 |
| --- | --- | --- |
| 纯解析基线 | 51.5% | [44.6, 58.3] |
| **本发明（预测感知 + 残差 RL）** | **60.5%** | [53.6, 67.0] |

**消融实验（各 200 局）**：A 基线 45.0% / B 基线+预测 45.0% /
C 基线+残差RL 53.5% / D 全量 53.5% / E 理想感知 75.5%。核心结论：

- 成功率增益主要来自**有界残差强化学习**（约 +8.5 个百分点）；
- **预测感知作为解析前馈项的独立增益有限（+0.0 个百分点）**——长下降过程中水平级联
  已能抑制缓变扰动，预测信息主要通过策略的**观测特征**发挥作用；
- 残余误差的主要来源是**状态估计而非控制**（"理想感知"上界变体再提升约 22 个百分点），
  可通过提升传感器精度进一步改善；
- 安全盾触发率随训练下降，说明策略学会在可行域内工作，而非依赖安全盾兜底。

> **诚实说明**：全局平均增益是**温和的**（+9.0pp，两个置信区间有少量重叠；150 局配对验证
> 给出 +5.3pp、McNemar p≈0.057）。但这掩盖了一个更干净的事实：把评估**按推力退化程度
> 分层**后，本发明在**每一档都优于基线**（见下）。报告中的定量结论均从
> `runs/ppo_v3/data/*.json` 读取，与实测一致。

### 7.1 推力退化分层评估（关键结果）

每回合抽取的交付推力/额定推力之比划分为若干窄带，在**相同随机种子**下配对评估
（退化系数无论取值区间如何都只消耗一个随机数，故各带的初始条件与风场完全一致，
唯一变化的是交付推力）。每带 100 局：

| 推力比区间 | 实际均值 | 基线 | 本发明 | 增益 | 配对胜负 | p 值 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.990 | 60.0% | **68.0%** | **+8.0pp** | 9:1 | **0.021** |
| 0.94–0.98 | 0.960 | 59.0% | 61.0% | +2.0pp | 5:3 | 0.727 |
| 0.90–0.94 | 0.920 | 50.0% | 57.0% | +7.0pp | 9:2 | 0.065 |
| 0.86–0.90 | 0.880 | 45.0% | 49.0% | +4.0pp | 4:0 | 0.125 |
| 0.82–0.86 | 0.840 | 37.0% | 43.0% | +6.0pp | 8:2 | 0.109 |
| 0.74–0.82 | 0.780 | 1.0% | 7.0% | +6.0pp | 7:1 | 0.070 |

**结论**：六个波段**全部为正增益**（合计 +5.5pp）；其中标称推力区间（推力比约 1.00）
的 +8.0pp、9:1、p≈0.02 达到统计显著。残差指令幅度在各带间基本恒定（0.24–0.26 m/s²），
说明策略学到的是**与工况无关的稳健修正**，而非对单一扰动的过拟合。推力比低于 0.82
时两者都趋近 0%（推力已不足以排空下沉速度，属物理不可达，而非控制器缺陷）。

复现：`python scripts/degradation.py --run runs/ppo_v3 --episodes 100`
→ 输出 `data/degradation.json`、`data/degradation.csv`、`figures/fig14_degradation.png`。

## 8. 可复现性

- 所有随机源统一由 `set_global_seed` 控制；
- `runs/<name>/data/config.json` 记录完整配置与运行环境；
- `scripts/ablation.py` 与 `scripts/degradation.py` 均用**同一批扰动种子**跑全部变体/波段，
  配对比较，并给出 Wilson 95% 置信区间与 McNemar 精确检验 p 值；
- `pyproject.toml` 声明依赖与 pytest 配置（`pip install -e ".[dev,plot]"` 后可直接 `pytest`）；
- `.github/workflows/ci.yml` 在 Python 3.10 与 3.13 上跑测试、CLI 冒烟与环境契约检查；
- `Dockerfile` 提供完全可复现的 CPU 环境（构建时即运行测试）。

## 9. 许可

MIT，见 `LICENSE`。仅用于课程教学与学术研究。

## 10. 参考

- 哈尔滨工业大学《具身智能理论与火箭回收技术》实验指导书（一）
- 哈尔滨工业大学《基于 Three.js 的火箭回收可视化仿真系统》实验指导书（二）
- 哈尔滨工业大学《基于 Ollama 的大语言模型本地部署》实验指导书（三）
- 张淼《专利写作规范》
