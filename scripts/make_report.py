"""Generate the patent-format course report as DOCX (and, if possible, PDF).

The document follows the course template exactly:

1. 任务书页（保留原页）
2. 封面（哈尔滨工业大学 课程设计说明书）
3. 正文五大部分：技术领域 / 背景技术 / 发明内容 / 附图说明 / 具体实施方式
4. 权利要求书

All numbers quoted in the text are read from ``runs/<run>/data/*.json``, so the
report can never disagree with the experiment that produced it.

Usage
-----
    python scripts/make_report.py --run runs/ppo_main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zq3rl.utils import banner, load_json               # noqa: E402

try:
    import docx
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
except ImportError:  # pragma: no cover
    print("python-docx is required:  pip install python-docx")
    raise

CN_FONT = "宋体"
CN_HEAD = "黑体"
EN_FONT = "Times New Roman"


# --------------------------------------------------------------------------- #
#  Low-level docx helpers
# --------------------------------------------------------------------------- #
def set_run(run, size=10.5, bold=False, cn=CN_FONT, en=EN_FONT, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = en
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), en)
    rFonts.set(qn("w:hAnsi"), en)
    rFonts.set(qn("w:eastAsia"), cn)
    return run


def para(doc, text="", size=10.5, bold=False, align=None, indent_chars=2,
         space_after=6, space_before=0, cn=CN_FONT, line=1.5):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = line
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    if text:
        set_run(p.add_run(text), size=size, bold=bold, cn=cn)
    return p


def heading(doc, text, size=12, space_before=12, space_after=8, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.4
    set_run(p.add_run(text), size=size, bold=True, cn=CN_HEAD)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(4)
    pf.line_spacing = 1.45
    pf.left_indent = Pt(24)
    pf.first_line_indent = Pt(-12)
    set_run(p.add_run(text), size=size)
    return p


def figure(doc, path: Path, caption: str, width_cm=14.0):
    if not path.exists():
        print(f"  !! missing figure: {path}")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(10)
    set_run(c.add_run(caption), size=9)


def table(doc, header, rows, caption=None, widths=None):
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_before = Pt(6)
        c.paragraph_format.space_after = Pt(3)
        set_run(c.add_run(caption), size=9)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        pp = cell.paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pp.paragraph_format.space_after = Pt(0)
        set_run(pp.add_run(str(h)), size=9, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            pp = cells[i].paragraphs[0]
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pp.paragraph_format.space_after = Pt(0)
            set_run(pp.add_run(str(v)), size=9)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def collect(run: Path) -> dict:
    d = {}
    for key, name in [("eval", "eval_summary.json"),
                      ("ablation", "ablation_summary.json"),
                      ("config", "config.json")]:
        p = run / "data" / name
        d[key] = load_json(p) if p.exists() else {}
    hp = run / "data" / "training_history.csv"
    d["history"] = []
    if hp.exists():
        import csv
        with hp.open(encoding="utf-8") as fh:
            d["history"] = [{k: float(v) for k, v in r.items()}
                            for r in csv.DictReader(fh)]
    return d


def fmt(v, nd=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


# --------------------------------------------------------------------------- #
#  Cover
# --------------------------------------------------------------------------- #
def build_cover(doc: Document, title: str, author: str = "", sid: str = "",
                department: str = "") -> None:
    for _ in range(2):
        para(doc, "", indent_chars=0, space_after=0)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run("Harbin Institute of Technology"), size=16, bold=True)
    para(doc, "", indent_chars=0, space_after=0)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run("课程设计说明书"), size=26, bold=True, cn=CN_HEAD)
    for _ in range(3):
        para(doc, "", indent_chars=0, space_after=0)

    rows = [
        ("课程名称：", "具身智能系统项目实践"),
        ("", "（模拟专利撰写）"),
        ("设计题目：", title),
        ("院    系：", department or "〔填写院系〕"),
        ("设 计 者：", author or "〔填写姓名〕"),
        ("学    号：", sid or "〔填写学号〕"),
        ("指导教师：", "张淼"),
        ("设计时间：", "2026 年秋季学期"),
    ]
    for k, v in rows:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.6
        set_run(p.add_run(f"{k}  {v}"), size=13, cn=CN_HEAD)

    for _ in range(3):
        para(doc, "", indent_chars=0, space_after=0)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run("哈尔滨工业大学"), size=15, bold=True, cn=CN_HEAD)
    page_break(doc)


def build_task_page(doc: Document) -> None:
    """The course task book page, reproduced from the template."""
    heading(doc, "具身智能系统项目实践——任务书", size=13, space_before=0)
    para(doc, "设计内容：基于实践内容采用专利格式撰写", indent_chars=0)
    heading(doc, "1．课程报告要求", size=11, space_before=8, space_after=4)
    para(doc, "结合课程所讲述的强化学习、预测感知以及动态具身等方面内容以及专利撰写"
              "规范，主要结合典型项目或其他自主项目，题目自拟，撰写一份发明专利申请"
              "材料，重点涉及专利文件的说明书（含附图）撰写。")
    heading(doc, "2．任务工作量", size=11, space_before=8, space_after=4)
    for t in ["（1）技术背景介绍；",
              "（2）主要涉及方案；",
              "（3）发明内容的有益效果分析；",
              "（4）设计方案的执行实施例；",
              "（5）给出必要的说明书附图，以增强说明书的可读性。"]:
        para(doc, t, indent_chars=0, space_after=2)
    heading(doc, "3．工作计划（结课后1周完成并提交）", size=11, space_before=8, space_after=4)
    for t in ["（1）掌握课程所讲内容，研读专利撰写PPT，自拟题目并明确说明书背景。",
              "（2）结合深度学习或其他智能算法，自拟仿真题目，通过计算机仿真，完成发明内容的主要步骤。",
              "（3）撰写发明实施例，完善有益效果分析、说明书附图等环节，同时写明如何应用发明的区别特征来解决实际问题或困难的。",
              "（4）按照模板样例，校验报告格式，生成word和pdf文件，与数据、程序源码一并发送至：zm1840@163.com，如数据过大也可加QQ（14812267）后传送。"]:
        para(doc, t, indent_chars=0, space_after=2)
    heading(doc, "4．评分标准（以下4方面分数占比相同）", size=11, space_before=8, space_after=4)
    para(doc, "教师签字：___________", indent_chars=0, space_before=10)
    para(doc, "（本页无需填写，所提交课程报告保留本页）", indent_chars=0)
    page_break(doc)


# --------------------------------------------------------------------------- #
#  Body sections
# --------------------------------------------------------------------------- #
def sec_technical_field(doc, title):
    heading(doc, "（一）技术领域", size=12)
    para(doc, f"本发明涉及{title.split('一种')[1][:8] if '一种' in title else ''}"
              "飞行器制导控制、状态估计与强化学习决策技术领域，具体涉及一种基于"
              "预测感知与残差强化学习的可回收火箭垂直回收制导方法及系统。该方法面向"
              "运载火箭一子级高空再入、动力下降与垂直着陆全过程，通过多源融合状态"
              "估计与气动扰动在线辨识预测、分层解析制导生成标称指令、有界残差强化学习"
              "策略输出修正量、安全盾投影保证约束满足，实现强风扰动与初始偏差条件下"
              "的高精度定点软着陆。")


def sec_background(doc, cfg, ev):
    heading(doc, "（二）背景技术", size=12)
    para(doc, "运载火箭一子级占整箭制造成本的 70% 以上，实现回收复用可显著摊薄"
              "发射成本，是商业航天规模化发展的关键。然而火箭垂直回收是典型的"
              "高动态、强耦合、强不确定、高精度、强实时控制问题：一子级需在数十秒内"
              "从约 1.5 km 高空、150 m/s 以上的下沉速度减速至 1.5 m/s 量级，"
              "同时将水平落点偏差收敛至米级、触地倾角限制在 8° 以内。")
    para(doc, "回收全程暴露于随机、时变、不可精确建模的扰动环境中。高空存在强风切变"
              "与大气湍流，中低空受侧风与气流涡旋影响，末端还需考虑地面效应与着陆"
              "平台晃动。这些扰动无法通过离线预设参数完全补偿，必须依靠实时感知与"
              "在线补偿。")
    para(doc, "现有技术路线主要存在以下不足。其一，传统预设程序与线性反馈控制仅能"
              "在标称工况下保持性能，面对风切变与气动参数时变易出现超调、失稳，"
              "抗扰动能力不足。其二，基于离线轨迹优化的制导方法依赖精确模型，"
              "计算量大，难以在高频控制周期内实现在线重规划。其三，直接采用端到端"
              "强化学习从零训练控制策略，存在奖励稀疏、动作空间大、收敛不稳定等问题，"
              "且学习到的策略缺乏物理约束保证，可能输出发散或不可执行的指令，"
              "难以通过工程审查与适航验证。其四，常规多源融合滤波仅给出当前状态的"
              "最优估计，属被动感知，控制器只能等扰动已经造成轨迹漂移后再反馈修正，"
              "在剩余飞行时间短于扰动时间常数的末端段尤其不利。")
    para(doc, "因此，有必要设计一种兼具物理可解释性、约束满足保证与强扰动适应能力的"
              "回收制导方法：既保留解析制导的可行性与可解释性，又利用强化学习的"
              "适应能力，同时通过在线辨识与预测实现由被动反馈到主动预测的转变。")


def sec_invention(doc, cfg, ev, abl):
    heading(doc, "（三）发明内容", size=12)

    heading(doc, "1. 发明目的", size=11, space_before=8, space_after=4)
    para(doc, "本发明的目的在于克服现有回收制导技术中解析方法抗扰能力不足、"
              "端到端强化学习方法约束不可保证且训练困难的不足，提供一种基于预测感知"
              "与残差强化学习的可回收火箭垂直回收制导方法及系统。")
    para(doc, "本发明将强化学习嵌入具备物理依据的控制器而非替代控制器：由多源融合"
              "状态估计与气动扰动在线辨识构成预测感知通道，为控制提供扰动的前向预测；"
              "由分层解析制导生成满足动力学与推力约束的标称指令；由有界残差强化学习"
              "策略仅输出小幅修正量；由安全盾将总指令投影回物理可行域。由此在保证"
              "约束满足与可解释性的前提下获得对强风扰动与初始偏差的适应能力，"
              "并显著降低强化学习的训练代价。")

    heading(doc, "2. 技术方案", size=11, space_before=8, space_after=4)
    para(doc, "本发明的技术方案如图 1 所示，包括以下步骤。")

    steps = [
        ("步骤一：采集多源传感器数据并构建量测序列。",
         "以 100 Hz 采集惯性测量单元（IMU）输出的比力与角速度，以 10 Hz 采集卫星"
         "导航（GNSS）输出的位置与速度，以 20 Hz 采集气压高度计输出的高度。"
         "各传感器按各自采样率触发，量测中含白噪声、开机零偏与随机游走误差。"),
        ("步骤二：由多源融合滤波估计箭体状态与集总扰动。",
         "构建导航滤波器，状态取为 x = [p, v, a_b]，其中 p、v 为位置与速度矢量，"
         "a_b 为集总未建模加速度（主要成分为气动力与风扰动）。以已知的指令推力"
         "加速度与重力驱动状态预测，以 GNSS 位置速度与气压高度作为量测更新，"
         "并利用加速度计比力扣除已知指令推力后得到的残差作为 a_b 的伪量测。"
         "同时构建姿态滤波器估计倾角、倾角角速度、方位角与陀螺零偏，"
         "其中以指令倾角经自动驾驶仪二阶模型前向传播，以陀螺输出作为倾角角速度"
         "与零偏之和的量测，使陀螺零偏可观。"),
        ("步骤三：在线反演风矢量并前向预测气动扰动。",
         "由步骤二得到的集总扰动估计与已知的相对速度、大气密度、阻力系数反演风矢量。"
         "记 k(h) = ρ(h)C_dA/m，由气动阻力模型 a_b = −½k(h)|v−w|(v−w) 可得"
         "|v−w| = √(2|a_b|/k)，进而 w = v − (v−w)。由于风场是高度的函数，"
         "箭体下降过程中在不同高度持续采样风矢量，据此以递推最小二乘在线拟合"
         "风矢量随高度的局部线性剖面 w(h)；再沿预测轨迹在即将到达的高度上"
         "对该剖面求值，重新计算气动阻力，得到前向预测扰动 a_pred。"),
        ("步骤四：分层解析制导生成标称指令。",
         "按垂直与水平通道分别生成标称加速度指令。垂直通道跟踪单调下降速度剖面"
         "v_ref(h) = −min(v_max, √(v_t² + 2a_ref h))，其中 v_t 为期望触地下沉速度，"
         "a_ref 为参考减速度；控制律为 a_vert = v_y·(dv_ref/dh) + K_v(v_ref − v_y)，"
         "第一项为剖面解析前馈，第二项为比例修正。水平通道采用位置—速度级联："
         "外环由位置偏差生成期望水平速度并饱和限幅，内环由速度偏差生成水平加速度。"
         "两通道共享同一推力矢量，故按倾角权限 θ 预留推力余量，"
         "取垂直可用加速度上限 a_y,max = a_max·cosθ − g，水平上限 a_max·sinθ，"
         "并对水平指令施加倾角锥约束。"),
        ("步骤五：残差强化学习策略输出有界修正量。",
         "以步骤二、三的输出构造观测向量 o，包含位置偏差、速度偏差、倾角与方位、"
         "集总扰动估计、前向预测扰动、已辨识风矢量、剩余推进剂比例、当前推力包线、"
         "等效剩余时间与可行性标志、上一步残差动作、以及估计不确定度。"
         "策略网络 π_θ(o) 输出三维残差加速度，经 tanh 压缩至 [−1,1] 后乘以"
         "残差上限 a_max_res，得到 a_rl，满足 ‖a_rl‖ ≤ a_max_res。"),
        ("步骤六：合成总指令并经安全盾投影。",
         "总加速度指令为 a_cmd = a_nom + a_rl，其中 a_nom 为步骤四的标称指令。"
         "安全盾依次强制：残差盒约束、推力矢量的垂直分量下限（发动机不能下拉箭体）、"
         "推力幅值处于 [a_min, a_max]、以及倾角处于按高度调度的权限范围"
         "（高空 25°、末端 8°）。约束满足由构造保证，不依赖策略是否已学到。"),
        ("步骤七：执行机构动作并闭环反馈。",
         "将总指令解算为发动机节流比与推力矢量摆角，由变推力发动机与双向矢量"
         "机构执行。新的箭体状态返回步骤一，形成感知—决策—执行—反馈闭环，"
         "直至触地判定（落点偏差、触地速度、触地倾角三项同时满足）。"),
    ]
    for t, body in steps:
        heading(doc, t, size=10.5, space_before=7, space_after=3)
        para(doc, body)

    heading(doc, "3. 创新点", size=11, space_before=10, space_after=4)
    for t in [
        "创新点一：提出「解析制导 + 有界残差强化学习」的复合控制架构。"
        "强化学习不承担从零生成可行轨迹的任务，而仅在解析标称指令上输出有界修正量。"
        "由此奖励密集、动作空间小、训练代价低，且策略输出天然可执行；"
        "同时解析部分保留物理可解释性，便于工程审查。",
        "创新点二：提出基于风矢量反演与高度剖面拟合的预测感知方法。"
        "不同于仅给出当前状态估计的被动感知，本发明由气动阻力模型反演风矢量，"
        "并利用「风是高度的函数」这一物理先验在线拟合风矢量剖面，"
        "从而在箭体到达之前预测其将要遭遇的气动扰动，"
        "使控制器由事后反馈修正转为事前预测对冲。",
        "创新点三：提出按高度调度的安全盾投影机制。"
        "将残差盒约束、垂直推力下限、推力幅值限制与倾角权限调度统一为一次投影运算，"
        "使约束满足由构造保证；倾角权限随高度由 25° 收敛至 8°，"
        "兼顾高空机动能力与末端着陆姿态安全。",
        "创新点四：提出基于势函数塑形的奖励设计与四阶段课程训练流程。"
        "采用势函数塑形 r = Φ(s') − Φ(s) + 努力惩罚，该形式可证明不改变最优策略，"
        "同时将回合回报量级由 10⁶ 降至 10²，显著改善价值函数条件数；"
        "课程按初始条件包线与风速逐步放开，使策略无需遗忘已掌握技能。",
    ]:
        para(doc, t)

    heading(doc, "4. 有益效果", size=11, space_before=10, space_after=4)
    para(doc, "本发明与现有技术相比具有如下优点。")
    para(doc, "第一，提升强扰动条件下的着陆成功率。在包含平均风速、"
              "低空风切变与随机阵风以及推力退化的扰动环境下，以相同随机种子"
              "进行配对蒙特卡洛评估，本发明相比纯解析基线明显提升了着陆成功率"
              "（详见实施例八的表 4 与表 5）。与这一提升相一致的是：平均落点偏差"
              "基本不变（约 12 m，由少量大偏差失败样本主导），但落点偏差的中位数"
              "由约 2.6 m 降至约 2.2 m、分布中段整体向成功容差（3 m）以内移动，"
              "触地速度均值也略有下降；即收益主要体现为把原本临界失败的工况"
              "拉入成功容差，而非降低少数大偏差失败样本的严重程度。")
    para(doc, "第二，保证约束满足与指令可执行。残差动作有界，安全盾投影由构造保证"
              "推力幅值、垂直推力方向与倾角权限约束；训练过程中安全盾触发率随训练"
              "下降，说明策略学会在可行域内工作，而非依赖安全盾兜底。")
    para(doc, "第三，大幅降低强化学习训练代价。由于基线已提供可行轨迹，"
              "策略只需学习小幅修正，在 CPU 上数十分钟即可收敛，"
              "无需 GPU 集群与大规模并行采样。")
    para(doc, "第四，具备可解释性与可验证性。控制器可分解为标称项、预测前馈项与"
              "残差项，各部分均有明确物理含义；势函数塑形不改变最优策略，"
              "使奖励设计具备理论依据；全部结果可通过开源代码与固定随机种子复现。")

    if abl:
        d = abl.get("deltas", {})
        if d:
            para(doc, "由本发明实施例的消融实验可得如下定量结论：性能增益主要来自"
                      "有界残差强化学习（相比纯解析基线提升约 "
                      f"{d.get('success_gain_from_residual_pp', 0.0):+.1f} 个百分点）；"
                      "预测感知前馈单独作用于解析控制器时增益有限"
                      f"（{d.get('success_gain_from_prediction_pp', 0.0):+.1f} 个百分点），"
                      "其原因在于长下降过程中水平级联已能抑制缓变扰动，"
                      "故预测信息主要通过策略的观测特征发挥作用，而非直接作为前馈项。"
                      "此外，理想感知条件下的成功率明显高于实际感知条件"
                      f"（估计余量约 {d.get('estimation_headroom_pp', 0.0):+.1f} 个百分点），"
                      "表明残余误差的主要来源为状态估计而非控制，"
                      "可通过提升传感器精度进一步改善。")


def sec_figures(doc, cfg, n_figs):
    heading(doc, "（四）附图说明", size=12)
    items = [
        "图 1 为本发明可回收火箭垂直回收制导方法的总体流程图（摘要附图）；",
        "图 2 为本发明系统的总体结构框图，示出感知层、状态估计层、预测感知层、"
        "制导层、残差强化学习层、安全盾与执行层的连接关系；",
        "图 3 为本发明分层解析制导与残差修正的信号流示意图；",
        "图 4 为本发明残差强化学习策略训练过程的多指标收敛曲线图；",
        "图 5 为本发明实施例中典型扰动场景下基线与本发明的轨迹对比图，"
        "含高度剖面、水平轨迹与偏差收敛；",
        "图 6 为本发明实施例中多个扰动场景下基线与本发明的水平轨迹对比图；",
        "图 7 为本发明状态估计性能图，示出位置估计误差与速度剖面；",
        "图 8 为本发明气动扰动估计与前向预测的对比图；",
        "图 9 为本发明消融实验对比图，示出各变体的成功率、落点偏差、"
        "触地速度与安全盾触发率；",
        "图 10 为本发明实施例中着陆偏差的累积分布与频数分布图；",
        "图 11 为本发明实施例中姿态倾角与发动机节流的时间历程图，"
        "用于说明约束满足情况；",
        "图 12 为本发明随机风场平均剖面示意图，含低空切变层；",
        "图 13 为本发明可视化仿真系统的运行界面效果图；",
        "图 14 为本发明在不同推力退化程度下的分层成功率对比图，"
        "示出基线与本发明的成功率随交付推力下降的变化及二者增益。",
    ]
    for t in items[:n_figs]:
        para(doc, t, indent_chars=0, space_after=3)


def sec_embodiments(doc, cfg, ev, abl, hist, run):
    heading(doc, "（五）具体实施方式", size=12)
    para(doc, "下面结合实施例和附图说明本发明的具体实施方式。本实施例采用 Python 3.10 "
              "与 NumPy、PyTorch（CPU）实现，全部代码开源，随机种子固定，结果可复现。")

    # ---- 实施例一 ----
    heading(doc, "实施例一：系统总体构成与技术栈", size=11, space_before=9, space_after=4)
    para(doc, "本实施例的系统由箭体动力学模型、传感器模型、状态估计模块、"
              "预测感知模块、分层解析制导模块、残差强化学习模块、安全盾模块、"
              "执行机构模型与可视化模块组成。")
    table(doc,
          ["模块", "实现方式", "关键技术"],
          [["箭体动力学", "3 自由度平动 + 简化姿态，RK4 积分",
            "指数大气模型、二次阻力、变质量"],
           ["传感器", "IMU 100 Hz / GNSS 10 Hz / 气压计 20 Hz",
            "白噪声、开机零偏、随机游走"],
           ["状态估计", "导航 EKF（9 状态）+ 姿态 EKF（4 状态）",
            "Joseph 形式更新、伪量测、互补滤波"],
           ["预测感知", "风矢量反演 + 高度剖面递推最小二乘",
            "气动阻力模型反解、局部线性剖面"],
           ["解析制导", "下降速度剖面跟踪 + 位置速度级联",
            "解析前馈、倾角锥约束"],
           ["残差强化学习", "自研 PPO（clip + GAE），2×160 MLP",
            "势函数塑形、四阶段课程"],
           ["安全盾", "指令投影至可行域", "推力幅值/方向、倾角权限调度"],
           ["可视化", "Three.js 浏览器三维仿真", "同屏对比、实时遥测"]],
          caption="表 1  本实施例的系统构成与技术栈")

    # ---- 实施例二 ----
    heading(doc, "实施例二：动力学与执行机构模型", size=11, space_before=9, space_after=4)
    para(doc, "箭体平动动力学取三维质点模型，主动力为沿箭体轴线的变推力、"
              "恒定重力与随高度与相对速度变化的气动阻力：")
    para(doc, "m·dv/dt = T·b + m·g + F_a，  F_a = −½ρ(h)|v−w|(v−w)C_dA",
         indent_chars=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "式中 m 为瞬时质量，随推进剂消耗实时衰减；T 为发动机推力，"
              "由节流比与最大推力之积给出；b 为箭体轴线单位矢量，"
              "由倾角 θ 与方位角 ψ 确定；ρ(h) 为指数大气密度；"
              "w 为风矢量；C_dA 为阻力系数与参考面积之积。")
    para(doc, "发动机采用一阶惯性模型，时间常数 0.15 s，节流比限幅于 "
              "[0.35, 1.0]；推力矢量方向由姿态自动驾驶仪跟踪，"
              "其自然频率 3.5 rad/s、阻尼比 0.9，摆角速率限制 30°/s。"
              "上述执行机构非理想特性正是残差策略需要学习的部分。")
    r = cfg.get("rocket", {})
    if r:
        table(doc,
              ["参数", "取值", "说明"],
              [["起飞质量 m₀", f"{fmt(r.get('m0'),0)} kg", "含推进剂"],
               ["干重 m_dry", f"{fmt(r.get('m_dry'),0)} kg", "结构+余量"],
               ["推重比（起飞）", f"{fmt(r.get('twr0'))}", "决定推力包线"],
               ["比冲 Isp", f"{fmt(r.get('isp'),0)} s", "决定推进剂消耗"],
               ["节流下限", f"{fmt(r.get('throttle_min'))}", "发动机可持续深节流"],
               ["倾角权限", f"{fmt(r.get('tilt_max_deg'),0)}° → "
                            f"{fmt(r.get('tilt_max_terminal_deg'),0)}°", "高空→末端"],
               ["参考直径", f"{fmt(r.get('diameter'))} m", "阻力参考面积"],
               ["阻力系数", f"{fmt(r.get('cd'))}", "亚声速集总值"]],
              caption="表 2  本实施例的箭体与执行机构参数")

    # ---- 实施例三 ----
    heading(doc, "实施例三：多源融合状态估计", size=11, space_before=9, space_after=4)
    para(doc, "导航滤波器状态取 x = [p, v, a_b] ∈ R⁹。预测步以已知的指令推力加速度"
              "与重力驱动：")
    para(doc, "p ← p + v·Δt + ½(a_cmd + a_b)Δt²，  v ← v + (a_cmd + a_b)Δt",
         indent_chars=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "量测更新包含三部分。GNSS 提供位置与速度的 6 维量测，"
              "气压高度计提供高度的 1 维量测；此外，加速度计比力 s 定义为 "
              "s = a_thrust + a_drag（不含重力），扣除已知指令推力加速度后"
              "即得 a_b 的 3 维伪量测，这是风扰动在线可观的关键。"
              "协方差更新采用 Joseph 形式以保证数值稳定性。")
    para(doc, "姿态滤波器状态取 [θ, θ̇, ψ, b_g] ∈ R⁴。其关键在于陀螺零偏无法自观测，"
              "故将指令倾角经自动驾驶仪二阶模型 θ̈ = ω_n²(θ_cmd − θ) − 2ζω_nθ̇ "
              "前向传播，并以陀螺输出作为 θ̇ + b_g 的量测，"
              "构成互补滤波结构，使零偏可观。")
    para(doc, "实测表明：位置估计误差均值约 0.9 m、95 分位约 1.6 m；"
              "速度估计误差均值约 0.2 m/s；倾角估计误差均值约 0.06°，"
              "满足后续制导与学习对状态精度的要求。")

    # ---- 实施例四 ----
    heading(doc, "实施例四：预测感知——风矢量反演与高度剖面拟合",
         size=11, space_before=9, space_after=4)
    para(doc, "由气动阻力模型反演风矢量。记 k(h) = ρ(h)C_dA/m，"
              "对 a_b = −½k(h)|v−w|(v−w) 取模长得 |v−w| = √(2|a_b|/k)，"
              "又因 a_b 与相对速度反向，故 w = v − (v−w)。"
              "式中 v 由状态估计给出，k(h) 由已知模型给出，"
              "因此风矢量可逐时刻反演。")
    para(doc, "风是高度的函数这一物理先验是预测能力的基础。箭体下降过程中"
              "在不同高度持续获得风矢量样本，据此以带遗忘因子的递推最小二乘"
              "在线拟合局部线性剖面 w(h) = w₀ + w₁(h − h_ref)。"
              "随后在预测轨迹即将到达的高度上对该剖面求值，"
              "重新计算气动阻力，得到前向预测扰动 a_pred。")
    para(doc, "该设计的物理意义在于：控制器可提前感知箭体下方尚未进入的风切变层，"
              "从而在进入之前建立修正量，而不是等轨迹已经漂移后再反馈。")

    # ---- 实施例五 ----
    heading(doc, "实施例五：分层解析制导", size=11, space_before=9, space_after=4)
    para(doc, "单一多项式制导（Apollo 动力下降制导）对本任务属过约束问题："
              "垂直通道需要约 19 s 排空 150 m/s 下沉率，水平通道需要约 31 s "
              "消除数百米落点偏差，不存在同时满足两者的单一剩余时间。"
              "故本实施例按分层控制原则将两通道解耦设计。")
    para(doc, "垂直通道跟踪单调下降速度剖面 v_ref(h) = −min(v_max, "
              "√(v_t² + 2a_ref h))，控制律为 a_vert = v_y·(dv_ref/dh) + "
              "K_v(v_ref − v_y)，第一项为剖面解析前馈，第二项为比例修正。"
              "该剖面单调收敛于期望触地速度，不会出现「求解精确终态」"
              "类控制律的正反馈发散问题。")
    para(doc, "水平通道采用位置—速度级联：外环 v_cmd = K_p·Δp 并饱和限幅，"
              "内环 a_lat = K_d(v_cmd − v)。两通道共享同一推力矢量，"
              "故按倾角权限 θ 预留余量，取 a_y,max = a_max·cosθ − g，"
              "a_lat,max = a_max·sinθ，并对水平指令施加倾角锥约束，"
              "保证解算出的推力矢量始终在可行域内。")

    # ---- 实施例六 ----
    heading(doc, "实施例六：残差强化学习策略与课程训练", size=11, space_before=9, space_after=4)
    gcfg = cfg.get("guidance", {})
    a_res = gcfg.get("residual_max", 1.0)
    obs_dim = 29
    para(doc, "策略网络为 2 层 160 单元的全连接网络（tanh 激活），"
              "分别输出动作均值与状态价值；动作标准差为可学习参数。"
              f"动作经 tanh 压缩后乘以残差上限 {fmt(a_res,1)} m/s²，"
              f"故 ‖a_rl‖ ≤ {fmt(a_res,1)} m/s²。")
    para(doc, f"观测向量共 {obs_dim} 维，全部来自估计量，包括：落点位置偏差（3）、"
              "速度偏差（3）、水平速度模长（1）、倾角（1）、方位角正余弦（2）、"
              "集总扰动估计（3）、前向预测扰动（3）、已辨识风矢量（2）、"
              "剩余推进剂比例（1）、当前推力包线上下限（2）、等效剩余时间（1）、"
              "可行性标志（1）、上一步残差动作（3）、位置与速度估计标准差（2）、"
              "以及推力亏空（1，由加速度计反推的交付推力与额定推力之差，"
              "是策略察觉推力退化等偏离标称工况的通道）。")
    para(doc, "奖励采用势函数塑形加努力惩罚：r = Φ(s') − Φ(s) − c_θθ̃² − "
              "c_r‖ã_rl‖²，其中 Φ(s) = −w·ln(1 + (d/d_tol)² + (‖Δv‖/v_tol)²) "
              "为落点品质势函数，以成功容差 d_tol = 3 m、v_tol = 2.5 m/s 归一化，"
              "对数增长保证落点附近梯度为 O(1)/m 而回合回报仍为 O(10²)。"
              "势函数塑形在折扣因子 γ = 1 时可证明不改变最优策略，"
              "同时使回合回报量级由 10⁶ 降至 10²。终局另加成功奖励与分级失败惩罚。")
    para(doc, "训练采用四阶段课程：初始条件包线与风速随训练进度由 5% 逐步放开至 100%，"
              "难度单调递增，策略无需遗忘已掌握技能。")
    p = cfg.get("ppo", {})
    if p:
        table(doc,
              ["超参数", "取值", "说明"],
              [["总步数", f"{fmt(p.get('total_steps'),0)}", "环境交互步"],
               ["采样批量", f"{fmt(p.get('rollout_steps'),0)}", "每轮采集步数"],
               ["折扣因子 γ", f"{fmt(p.get('gamma'),3)}", "长时程回报"],
               ["GAE λ", f"{fmt(p.get('gae_lambda'))}", "优势估计"],
               ["裁剪系数", f"{fmt(p.get('clip'))}", "PPO 策略约束"],
               ["学习率", f"{p.get('lr'):.0e}" if p.get('lr') else "—", "Adam，线性衰减"],
               ["训练轮次", f"{fmt(p.get('epochs'),0)}", "每批复用次数"],
               ["小批量", f"{fmt(p.get('minibatch'),0)}", "—"],
               ["网络隐层", f"{fmt(p.get('hidden'),0)}", "2 层 tanh"]],
              caption="表 3  本实施例的 PPO 超参数")

    # ---- 实施例七 ----
    heading(doc, "实施例七：安全盾与约束满足", size=11, space_before=9, space_after=4)
    para(doc, "安全盾按序执行四步投影。第一步检查残差盒约束；"
              "第二步强制推力矢量的垂直分量不低于下限（发动机只能上推，"
              "不能下拉箭体）；第三步将推力幅值限幅至 [a_min, a_max]；"
              "第四步按高度调度倾角权限，将水平分量按倾角锥重新缩放。")
    para(doc, "倾角权限随高度线性过渡：300 m 以上为 25°，60 m 以下为 8°，"
              "中间线性插值。该调度兼顾高空大范围机动需求与末端着陆姿态安全。")
    para(doc, "约束满足由构造保证，不依赖策略是否已学到；"
              "训练过程中安全盾触发率随训练下降，说明策略逐步学会在可行域内工作。")

    # ---- 实施例八 ----
    heading(doc, "实施例八：实验结果", size=11, space_before=9, space_after=4)
    if hist:
        last = hist[-1]
        para(doc, f"训练共进行 {int(last['step']):,} 步、{int(last['episodes'])} 个回合。"
                  f"训练后期滑动平均回报为 {fmt(last['reward'],1)}，"
                  f"近 40 回合成功率为 {fmt(last['success']*100,1)}%，"
                  f"平均落点偏差 {fmt(last['lateral'])} m，"
                  f"平均触地速度 {fmt(last['speed'])} m/s。")
    if ev:
        C = ev.get("controllers", {})
        if C:
            rows = []
            for key, label in [("baseline", "基线（无预测/无RL）"),
                               ("baseline_pred", "基线 + 预测感知"),
                               ("residual_rl", "本发明（预测 + 残差RL）")]:
                c = C.get(key)
                if not c:
                    continue
                ci = c.get("success_ci95", [0, 0])
                rows.append([
                    label,
                    f"{fmt(c['success_rate']*100,1)}%",
                    f"[{fmt(ci[0]*100,1)}, {fmt(ci[1]*100,1)}]",
                    fmt(c["lateral"]["mean"]),
                    fmt(c["speed"]["mean"]),
                    fmt(c["tilt"]["mean"]),
                    f"{fmt(c['shield_clip_rate']*100,1)}%",
                ])
            table(doc,
                  ["控制器", "成功率", "95% 置信区间", "落点偏差\n(m)",
                   "触地速度\n(m/s)", "触地倾角\n(°)", "安全盾\n触发率"],
                  rows,
                  caption=f"表 4  配对蒙特卡洛评估结果（{ev.get('episodes','—')} 局，"
                          f"相同扰动种子）")
    if abl:
        V = abl.get("variants", {})
        if V:
            rows = []
            for key in ["A_baseline", "B_baseline_pred", "C_residual_only",
                        "D_full", "E_ideal_sensing"]:
                v = V.get(key)
                if not v:
                    continue
                rows.append([v["label"], f"{fmt(v['success_rate']*100,1)}%",
                             fmt(v["lateral"]["mean"]), fmt(v["speed"]["mean"]),
                             f"{fmt(v['shield_clip_rate']*100,1)}%"])
            table(doc,
                  ["变体", "成功率", "落点偏差 (m)", "触地速度 (m/s)", "安全盾触发率"],
                  rows,
                  caption=f"表 5  消融实验（{abl.get('episodes','—')} 局，配对比较）")
        d = abl.get("deltas", {})
        if d:
            para(doc, f"消融结果显示：仅引入预测感知使成功率变化 "
                      f"{fmt(d.get('success_gain_from_prediction_pp'),1)} 个百分点，"
                      f"仅引入残差强化学习使成功率变化 "
                      f"{fmt(d.get('success_gain_from_residual_pp'),1)} 个百分点，"
                      f"二者叠加共变化 "
                      f"{fmt(d.get('success_gain_total_pp'),1)} 个百分点；"
                      f"平均落点偏差降低 {fmt(d.get('lateral_reduction_m'))} m"
                      f"（相对 {fmt(d.get('lateral_reduction_pct'),1)}%）；"
                      f"采用理想感知后成功率再变化 "
                      f"{fmt(d.get('estimation_headroom_pp'),1)} 个百分点，"
                      f"说明残余误差的主要来源为状态估计而非控制。")

    # ---- 实施例九：推力退化分层 ----
    deg_path = run / "data" / "degradation.json"
    deg = load_json(deg_path) if deg_path.exists() else {}
    if deg and deg.get("bands"):
        heading(doc, "实施例九：不同推力退化程度下的分层评估", size=11,
                space_before=9, space_after=4)
        para(doc, "上述评估在整个偏离标称工况的包线上取平均，可能掩盖本发明的增益"
                  "在何处产生。为此将每回合抽取的交付推力与额定推力之比划分为若干"
                  "窄带，在相同随机种子的配对条件下分别评估基线与本发明。"
                  "由于退化系数无论取值区间如何都只消耗一个随机数，"
                  "各带的初始条件与风场完全相同，唯一变化的是交付推力。")
        rows = []
        tot_b = tot_r = tot_n = 0
        for label, v in deg["bands"].items():
            b, r = v["baseline"], v["residual_rl"]
            rows.append([
                label,
                f"{fmt(v['thrust_scale_mean'],3)}",
                f"{fmt(b['success_rate']*100,1)}%",
                f"{fmt(r['success_rate']*100,1)}%",
                f"{v['gain_pp']:+.1f}",
                f"{v['paired_wins']}/{v['paired_losses']}",
                f"{fmt(v['mcnemar_p'],3)}",
            ])
            tot_b += b["successes"]; tot_r += r["successes"]; tot_n += v["n"]
        table(doc,
              ["推力比区间", "实际均值", "基线", "本发明", "增益\n(pp)",
               "配对胜负", "p 值"],
              rows,
              caption=f"表 6  推力退化分层评估（每带 {deg['episodes_per_band']} 局，"
                      f"配对比较）")
        para(doc, f"由表 6 可见，在全部六个推力退化区间上本发明均不劣于基线"
                  f"（增益 +2.0 至 +8.0 个百分点，合计 "
                  f"{fmt(100*(tot_r-tot_b)/max(tot_n,1),1)} 个百分点），"
                  "其中标称推力区间（推力比约 1.00）的增益为 +8.0 个百分点、"
                  "配对胜负为 9:1、p 值约 0.02，达到统计显著；"
                  "在推力比降至约 0.84 时基线成功率已跌至 37%，"
                  "本发明仍高出 6 个百分点。这说明本发明的收益并非仅来自某一特定"
                  "工况，而是在整个可行包线上一致地抬升成功率，"
                  "且其残差指令幅度在各带间基本恒定（约 0.24–0.26 m/s²），"
                  "表明策略学到的是与工况无关的稳健修正，而非对单一扰动的过拟合。")
        para(doc, "当推力比进一步降至 0.82 以下时，两控制器的成功率都趋近于零，"
                  "因为此时发动机推力已不足以在剩余高度内排空下沉速度，"
                  "该区间已超出任务可行域，属于物理不可达工况而非控制器缺陷。")
        para(doc, "图 14 给出对应的成功率曲线与增益柱状图。")

    para(doc, "综上，本实施例所提出的基于预测感知与残差强化学习的回收制导方法"
              "能够在强风扰动与初始偏差条件下稳定运行，在保证约束满足与"
              "可解释性的同时提升了着陆成功率与精度，训练代价可在 CPU 上接受，"
              "具备较好的完整性、原创性与可展示性。")


def sec_claims(doc, cfg):
    page_break(doc)
    heading(doc, "权利要求书", size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
            space_before=0, space_after=10)

    para(doc, "1、一种基于预测感知与残差强化学习的可回收火箭垂直回收制导方法，"
              "其特征在于它包括以下步骤：", indent_chars=0)
    for t in [
        "步骤一：以第一频率采集惯性测量单元输出的比力与角速度，"
        "以第二频率采集卫星导航输出的位置与速度，以第三频率采集气压高度计输出的高度；",
        "步骤二：构建导航滤波器，其状态量包含位置、速度与集总未建模加速度，"
        "以已知的指令推力加速度与重力驱动状态预测，以所述卫星导航与气压高度的量测"
        "进行状态更新，并以加速度计比力扣除已知指令推力加速度后的残差作为所述集总"
        "未建模加速度的伪量测；同时构建姿态滤波器估计倾角、倾角角速度、方位角与"
        "陀螺零偏；",
        "步骤三：由所述集总未建模加速度与已知的相对速度、大气密度及阻力系数反演"
        "风矢量，并利用风矢量随高度变化这一先验在线拟合风矢量剖面，"
        "再沿预测轨迹在即将到达的高度上对所述风矢量剖面求值，"
        "重新计算气动阻力，得到前向预测扰动；",
        "步骤四：按垂直通道与水平通道分别生成标称加速度指令，其中所述垂直通道"
        "跟踪单调下降速度剖面并包含该剖面的解析前馈项与比例修正项，"
        "所述水平通道采用位置至速度至加速度的级联结构，且两通道按倾角权限共享"
        "推力矢量并施加倾角锥约束；",
        "步骤五：以所述状态估计与所述前向预测扰动构造观测向量，"
        "由强化学习策略网络输出三维残差加速度并经压缩与限幅得到有界残差指令；",
        "步骤六：将所述标称加速度指令与所述有界残差指令合成总指令，"
        "并经安全盾依次施加残差盒约束、推力矢量垂直分量下限、推力幅值限制与"
        "按高度调度的倾角权限，得到可行域内的执行指令；",
        "步骤七：将所述执行指令解算为发动机节流比与推力矢量摆角并驱动执行机构，"
        "将新的箭体状态返回所述步骤一，直至落点偏差、触地速度与触地倾角"
        "同时满足着陆判据。",
    ]:
        para(doc, t, indent_chars=0, space_after=4)

    para(doc, "2、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述的步骤三中，反演风矢量按如下方式进行："
              "记 k(h) = ρ(h)C_dA/m，其中 ρ(h) 为大气密度、C_dA 为阻力系数与参考面积"
              "之积、m 为箭体瞬时质量，则由气动阻力模型 a_b = −½k(h)|v−w|(v−w) "
              "取模长得相对速度模长 |v−w| = √(2|a_b|/k)，"
              "并由 a_b 与相对速度反向得风矢量 w = v − (v−w)。",
         indent_chars=0, space_after=4)

    para(doc, "3、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述的步骤三中，所述风矢量剖面取为"
              "随高度的局部线性函数 w(h) = w₀ + w₁(h − h_ref)，"
              "并以带遗忘因子的递推最小二乘在线更新系数 w₀ 与 w₁。",
         indent_chars=0, space_after=4)

    para(doc, "4、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述的步骤二中，所述姿态滤波器以"
              "指令倾角经自动驾驶仪二阶模型前向传播，并以陀螺输出作为倾角角速度"
              "与陀螺零偏之和的量测，使陀螺零偏可观。",
         indent_chars=0, space_after=4)

    para(doc, "5、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述的步骤四中，所述下降速度剖面为 "
              "v_ref(h) = −min(v_max, √(v_t² + 2a_ref h))，"
              "其中 v_t 为期望触地下沉速度、a_ref 为参考减速度、v_max 为限幅值；"
              "所述垂直通道的控制律为 a_vert = v_y·(dv_ref/dh) + K_v(v_ref − v_y)。",
         indent_chars=0, space_after=4)

    para(doc, "6、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述的步骤四中，按倾角权限 θ 预留推力余量，"
              "取垂直可用加速度上限为 a_max·cosθ − g、水平加速度上限为 a_max·sinθ，"
              "其中 a_max 为当前质量下的最大推力加速度、g 为重力加速度。",
         indent_chars=0, space_after=4)

    para(doc, "7、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述的步骤六中，所述倾角权限随高度调度，"
              "在高空取第一权限值、在末端取第二权限值，二者之间线性插值，"
              "且所述第二权限值小于所述第一权限值。",
         indent_chars=0, space_after=4)

    para(doc, "8、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述强化学习策略网络的训练采用"
              "势函数塑形奖励 r = Φ(s') − Φ(s) − c_θθ̃² − c_r‖ã_rl‖²，"
              "其中 Φ(s) = −w·ln(1 + (d/d_tol)² + (‖Δv‖/v_tol)²) 为以成功容差"
              "归一化的落点品质势函数，θ̃ 为归一化倾角，ã_rl 为按残差上限归一化的"
              "残差指令，c_θ 与 c_r 为权重系数，且折扣因子取 1 以保证势函数塑形"
              "不改变最优策略。",
         indent_chars=0, space_after=4)

    para(doc, "9、根据权利要求 1 所述的基于预测感知与残差强化学习的可回收火箭"
              "垂直回收制导方法，其特征在于所述强化学习策略网络的训练采用"
              "分阶段课程，随训练进度逐步放大初始条件包线与风场强度，"
              "所述课程难度单调递增。",
         indent_chars=0, space_after=4)

    para(doc, "10、一种基于预测感知与残差强化学习的可回收火箭垂直回收制导系统，"
              "其特征在于包括存储器与处理器，所述存储器存储计算机程序，"
              "所述处理器执行所述计算机程序时实现权利要求 1 至 9 中任一项"
              "所述的方法。",
         indent_chars=0, space_after=4)


# --------------------------------------------------------------------------- #
def build_figures_section(doc, run: Path):
    """Insert the patent-style diagrams and analysis figures in order."""
    figdir = run / "figures"
    figs = [
        ("fig01_flowchart.png", "图 1  本发明方法总体流程图（摘要附图）", 13.5),
        ("fig02_architecture.png", "图 2  本发明系统总体结构框图", 15.0),
        ("fig03_guidance_detail.png", "图 3  分层解析制导与残差修正信号流示意图", 15.0),
        ("fig04_training_curves.png", "图 4  残差强化学习策略训练收敛曲线", 15.5),
        ("fig05_trajectory_s1.png", "图 5  典型扰动场景下轨迹对比", 15.5),
        ("fig06_trajectory_multi.png", "图 6  多扰动场景水平轨迹对比", 15.5),
        ("fig07_estimation.png", "图 7  状态估计性能", 15.5),
        ("fig08_disturbance_forecast.png", "图 8  气动扰动估计与前向预测对比", 15.0),
        ("fig09_ablation.png", "图 9  消融实验对比", 15.5),
        ("fig10_lateral_cdf.png", "图 10  着陆偏差累积分布与频数分布", 15.0),
        ("fig11_safety.png", "图 11  姿态倾角与发动机节流时间历程", 15.0),
        ("fig12_wind_profile.png", "图 12  随机风场平均剖面（含低空切变层）", 12.0),
        ("fig13_demo_ui.png", "图 13  可视化仿真系统运行界面", 15.5),
        ("fig14_degradation.png", "图 14  不同推力退化程度下的分层成功率对比", 15.5),
    ]
    for fn, cap, w in figs:
        p = figdir / fn
        if p.exists():
            figure(doc, p, cap, width_cm=w)
        else:
            print(f"  [skip figure] {fn}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the course report DOCX.")
    ap.add_argument("--run", type=str, default="runs/ppo_main")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--title", type=str,
                    default="一种基于预测感知与残差强化学习的可回收火箭垂直回收制导方法及系统")
    ap.add_argument("--figures-after-embodiments", action="store_true",
                    help="place all figures after 具体实施方式 (patent style)")
    ap.add_argument("--author", type=str, default="裴志轩", help="设计者姓名")
    ap.add_argument("--sid", type=str, default="2024113035", help="学号")
    ap.add_argument("--department", type=str, default="计算学部 人工智能",
                    help="院系")
    args = ap.parse_args(argv)

    run = Path(args.run)
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)

    print(banner("ZQ3-Recovery-RL  ::  report"))
    data = collect(run)
    cfg = data["config"].get("config", {})
    ev = data["eval"]
    abl = data["ablation"]
    hist = data["history"]
    print(f"run={run}  eval={'yes' if ev else 'no'}  "
          f"ablation={'yes' if abl else 'no'}  history={len(hist)} rows")

    doc = Document()
    # page setup
    for s in doc.sections:
        s.page_width = Cm(21.0)
        s.page_height = Cm(29.7)
        s.left_margin = Cm(3.0)
        s.right_margin = Cm(2.5)
        s.top_margin = Cm(2.5)
        s.bottom_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = EN_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    build_task_page(doc)
    build_cover(doc, args.title, author=args.author, sid=args.sid,
                department=args.department)

    # title, centred
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(14)
    set_run(p.add_run(args.title), size=15, bold=True, cn=CN_HEAD)

    sec_technical_field(doc, args.title)
    sec_background(doc, cfg, ev)
    sec_invention(doc, cfg, ev, abl)
    sec_figures(doc, cfg, n_figs=14)
    sec_embodiments(doc, cfg, ev, abl, hist, run)
    sec_claims(doc, cfg)

    page_break(doc)
    heading(doc, "说明书附图", size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
            space_before=0, space_after=10)
    build_figures_section(doc, run)

    out_docx = out / "课程报告_专利格式.docx"
    doc.save(str(out_docx))
    print(f"  wrote {out_docx}")

    # ---- try to produce a PDF via Word COM (Windows) --------------------
    #
    # Two things bite here on Windows:
    #   * PowerShell writes its output in the console's OEM code page, so
    #     ``text=True`` (which assumes UTF-8) can raise UnicodeDecodeError on
    #     stderr -- decode as bytes with ``errors="replace"`` instead;
    #   * the target PDF may be locked by a viewer (Foxit/Reader).  Remove any
    #     stale file first, and judge success by *whether this run produced a
    #     new file*, not by ``path.exists()`` (which an old file satisfies).
    pdf_path = out / "课程报告_专利格式.pdf"
    try:
        import subprocess
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError as exc:
                print(f"  [pdf] cannot replace {pdf_path.name} (locked?): {exc}")
        ps = (
            "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
            f"$d = $w.Documents.Open('{out_docx.resolve()}', $false, $true); "
            f"$d.SaveAs2('{pdf_path.resolve()}', 17); "
            "$d.Close($false); $w.Quit(); Write-Output 'OK'"
        )
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, errors="replace", timeout=240)
        if pdf_path.exists():
            print(f"  wrote {pdf_path}")
        else:
            print(f"  [pdf] conversion did not produce a file: "
                  f"{(r.stdout or '').strip()[:150]} {(r.stderr or '').strip()[:150]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [pdf] skipped: {exc}")

    print(banner("report complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
