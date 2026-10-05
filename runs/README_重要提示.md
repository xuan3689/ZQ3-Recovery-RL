# ⚠️ runs/ 目录说明（2026-10-05 08:40 更新）

## ✅ `ppo_v3/` —— **权威结果，报告与演示均基于此**

最终交付的 run。配置：修复后的奖励函数（见下）、`w_quality=12`、`gamma=1.0`、
`residual_max=1.0 m/s²`。

实测（200 局配对蒙特卡洛，相同随机种子）：

| 控制器 | 成功率 | 95% CI |
|---|---|---|
| 纯解析基线 | 51.5% | [44.6, 58.3] |
| 本发明（预测感知 + 残差RL） | **60.5%** | [53.6, 67.0] |

产物：`课程报告_专利格式.docx` / `.pdf`、`figures/fig01..fig13.png`、
`data/eval_summary.json`、`data/ablation_summary.json`、`data/traces.json`。

## ⚠️ 其余 run 均为调试中间产物，**不要用于报告**

| run | 说明 |
|---|---|
| `ppo_main/` | 用**有缺陷的奖励函数**训练（残差惩罚在 100 Hz 子步里被放大 5 倍）→ 策略退化成纯解析基线，无参考价值 |
| `ppo_fixed/` | 只修了奖励缩放，但 `w_quality=60` + `gamma=0.995` 仍在 → 策略**有害**（60 局 0%） |
| `ppo_v2/` | 修了奖励缩放与折扣，但 `residual_max=4.0` 过大 → 策略方向对、幅度过大（60 局 45%） |

保留它们是为了让「为什么需要那三处修复」有据可查（见仓库同级 `交接文档.md` §2）。

## 三处关键修复（简述）

1. **奖励缩放在物理子步里被放大 5 倍** → 拆成 `_step_reward`（仅势函数，100 Hz 可调）
   与 `_effort_penalty`（20 Hz 控制步调一次）；残差按 `residual_max` 归一化。
2. **回报量级 O(600) + γ<1** → `w_quality` 60→12，`gamma` 0.995→1.0
   （γ=1 是势函数塑形「不改变最优策略」的前提）。
3. **残差权限过大** → `residual_max` 4.0→1.0 m/s²。

回归测试：`tests/test_smoke.py::test_effort_penalty_is_once_per_control_step`、
`test_reward_magnitude_is_bounded_per_step`。
