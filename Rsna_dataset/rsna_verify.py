# -*- coding: utf-8 -*-
"""
RSNA 子集自检（缺前置产物时自动 SKIP 对应项）

重点覆盖本次重建要保证的不变量：
  A. 患者级划分互斥（旧版按 study 划分会在患者有多次检查时泄漏）
  B. 检查不跨集合
  C. 患者记账闭合（抽样 ∪ 未抽样 = 全库合格患者）
  D. 患者级标签 == 该患者切片标签的最大值
  E. 索引与 PNG 实际文件一致
  F. 预处理口径与 CQ500 一致（同一脑窗 WL/WW）
  G. 噪声/滤波/投票定义与 CQ500 是**同一份对象**（单一事实来源）
并输出 RSNA vs CQ500 的规模对比表，供论文描述两个队列。

用法: python rsna_verify.py
退出码 0 = 通过（含 SKIP），1 = 有失败
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        FAIL.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return
    if r is True:
        PASS.append(name); print(f"  [ok]   {name}")
    elif r is None:
        SKIP.append(name); print(f"  [skip] {name}")
    else:
        FAIL.append(f"{name}: {r}"); print(f"  [FAIL] {name}: {r}")


def _load(name):
    p = ROOT / name
    return pd.read_csv(p) if p.exists() else None


def _index():
    import rsna_commons as C
    return C.load_index()


# ---------------- A/B. 划分粒度 ----------------
def a_split_disjoint():
    sp = _load("rsna_split.csv")
    if sp is None:
        return None
    s = {k: set(sp[sp["split"] == k]["patient_id"]) for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if s[a] & s[b]:
            return (f"患者级泄漏！{a} ∩ {b} = {sorted(s[a] & s[b])[:5]}"
                    f"（共 {len(s[a] & s[b])} 例）")
    if not all(s.values()):
        return "存在空划分"
    return True


def b_no_study_across_splits():
    sp, pm = _load("rsna_split.csv"), _load("rsna_patient_map.csv")
    if sp is None or pm is None:
        return None
    d = pm.merge(sp, on="patient_id")
    bad = int((d.groupby("study_id")["split"].nunique() > 1).sum())
    if bad:
        return f"{bad} 个检查跨集合"
    return True


def b2_multistudy_patients_caught():
    """确认库里确实存在"多次检查的患者" —— 也就是旧版按 study 划分会泄漏的那批人。"""
    pm = _load("rsna_patient_map.csv")
    if pm is None:
        return None
    pm = pm[(pm["patient_id"].fillna("") != "")
            & (~pm["study_id"].astype(str).str.startswith("ERR:"))]
    per = pm.groupby("patient_id")["study_id"].nunique()
    multi = int((per > 1).sum())
    if multi == 0:
        return "库里没有多次检查的患者（与预期不符，请复核 DICOM 标签解析）"
    print(f"         （全库 {len(per)} 例中有 {multi} 例做过多次检查）")
    return True


# ---------------- C. 记账 ----------------
def c_patient_accounting():
    frame, sp = _load("rsna_sampling_frame.csv"), _load("rsna_split.csv")
    if frame is None or sp is None:
        return None
    sampled = set(sp["patient_id"])
    in_frame = set(frame[frame["sampled"] == True]["patient_id"])  # noqa: E712
    if sampled != in_frame:
        return (f"划分患者与采样清单不一致：差 {len(sampled ^ in_frame)} 例")
    # 合格池 = 未被排除的患者。注意 excluded_reason 读回后可能是 NaN（空串的 CSV 往返），
    # 所以既接受 sentinel "kept"，也接受空/NaN。
    exc = frame["excluded_reason"].fillna("").astype(str).str.strip()
    pool = set(frame[exc.isin(["", "kept"])]["patient_id"])
    if not sampled <= pool:
        return (f"划分里有 {len(sampled - pool)} 例不在合格池中："
                f"{sorted(sampled - pool)[:5]}")
    return True


# ---------------- D. 标签一致性 ----------------
def d_label_consistency():
    idx, sl = _index(), _load("rsna_slice_labels.csv")
    if sl is None:
        return None
    g = idx.groupby("patient_id").agg(
        lab=("label", "first"), anysl=("slice_label", "max"))
    wrong = g[(g["lab"] == "Hemorrhagic") != (g["anysl"] == 1)]
    if len(wrong):
        return (f"{len(wrong)} 例的患者级标签与切片标签最大值不一致，"
                f"例如 {list(wrong.index[:3])}")
    return True


# ---------------- E. 索引 vs 文件 ----------------
def e_index_files_exist():
    idx = _index()
    if idx.empty:
        return None
    need = {"patient_id", "study_id", "split", "label", "slice_label", "image_path"}
    miss = need - set(idx.columns)
    if miss:
        return f"索引缺列 {miss}"
    rng = np.random.default_rng(0)
    take = rng.choice(len(idx), size=min(300, len(idx)), replace=False)
    bad = [idx.iloc[i]["image_path"] for i in take
           if not (ROOT / "png" / idx.iloc[i]["image_path"]).exists()]
    if bad:
        return f"{len(bad)}/{len(take)} 抽样文件不存在，例如 {bad[:3]}"
    if idx["patient_id"].nunique() == 0:
        return "索引没有患者"
    return True


# ---------------- F. 与 CQ500 预处理口径一致 ----------------
def f_window_matches_cq500():
    """预处理口径：两个 DICOM->PNG 转换脚本必须用**同一组**脑窗常量。

    常量已上移到 ich_common（共享模块），所以这里检查的是"两边都取自共享常量、
    且与共享值一致"，而不是去 import 另一个数据集的脚本 —— RSNA 因此可以独立部署。
    """
    import ich_common as ic
    import rsna_dicom_to_png as rs
    same = (abs(rs.HU_LO - ic.HU_LO) < 1e-9 and abs(rs.HU_HI - ic.HU_HI) < 1e-9)
    if not same:
        return (f"脑窗与共享常量不一致！RSNA [{rs.HU_LO},{rs.HU_HI}] vs "
                f"ich_common [{ic.HU_LO},{ic.HU_HI}]")
    if rs.IMG_SIZE != ic.IMG_SIZE:
        return f"输出尺寸不一致 {rs.IMG_SIZE} vs {ic.IMG_SIZE}"
    # CQ500 侧若在同级目录，顺带确认它也用了同一组常量
    cq_png = ROOT.parent / "Cq500_dataset" / "cq500_dicom_to_png.py"
    if cq_png.exists():
        sys.path.insert(0, str(ROOT.parent / "Cq500_dataset"))
        import cq500_dicom_to_png as cq
        if abs(cq.HU_LO - ic.HU_LO) > 1e-9 or abs(cq.HU_HI - ic.HU_HI) > 1e-9:
            return (f"CQ500 脑窗 [{cq.HU_LO},{cq.HU_HI}] 与共享常量 "
                    f"[{ic.HU_LO},{ic.HU_HI}] 不一致")
        note = "（含 CQ500 侧）"
    else:
        note = "（未发现 Cq500_dataset，跳过其侧检查）"
    print(f"         （共享脑窗 WL{rs.WL:.0f}/WW{rs.WW:.0f} -> HU "
          f"[{rs.HU_LO:.0f},{rs.HU_HI:.0f}]，{rs.IMG_SIZE}x{rs.IMG_SIZE}）{note}")
    return True


# ---------------- G. 共享定义 ----------------
def g_shared_definitions():
    """RSNA 必须与共享模块用同一份计算定义，且**不得依赖别的数据集**。"""
    import ich_common as ic
    import rsna_commons as rs
    for attr in ("NOISE_FUNCS", "FILTER_FUNCS", "apply_noise", "apply_filter",
                 "patient_vote_metrics", "_to_tensor", "build_eval_transform",
                 "create_model_with_pretrained", "grad_cam",
                 # 边界校正 / TTA 消融引入的共享量：19 与 20 都靠它们，必须同源
                 "TTA_VARIANTS", "boot_ci", "mcnemar", "match_prevalence",
                 "match_prevalence_rate", "prior_correction_mode"):
        if getattr(rs, attr) is not getattr(ic, attr):
            return f"{attr} 不是同一对象 —— 与共享定义已分叉"
    # sweep_filters：RSNA 是绑定路径的薄包装，底层必须是 ich_common 的那一个
    if getattr(rs, "_sweep_filters", None) is not ic.sweep_filters:
        return "rsna.sweep_filters 的实现不是 ich_common.sweep_filters（疑似另写了一份）"
    import inspect
    src = inspect.getsource(rs.sweep_filters)
    for need in ("data_root=DATA_ROOT", "noisy_root=NOISY_ROOT", "classes=CLASSES"):
        if need not in src:
            return f"rsna.sweep_filters 未绑定 {need}，可能误读别的数据集路径"
    if list(rs.NOISE_NAMES) != list(ic.NOISE_NAMES):
        return "NOISE_NAMES 不一致"
    if list(rs.FILTER_NAMES) != list(ic.FILTER_NAMES):
        return "FILTER_NAMES 不一致"
    return True


def g2_no_cross_dataset_dependency():
    """RSNA 不得 import 任何别的数据集 —— 否则无法单独拷到新机器上跑。

    用 AST 查**真实的 import 语句**与路径字面量，而不是全文搜字符串
    （文档里解释"早期版本曾 import cq500_commons"不应算违规）。
    """
    import ast
    src_path = ROOT / "rsna_commons.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in ("cq500_commons", "cq500_dicom_to_png"):
                    return f"rsna_commons.py 第 {node.lineno} 行 import 了 {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in ("cq500_commons", "cq500_dicom_to_png"):
                return f"rsna_commons.py 第 {node.lineno} 行 from {node.module} import ..."
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "Cq500_dataset" in node.value:
                return f"rsna_commons.py 第 {node.lineno} 行出现 Cq500_dataset 路径字面量"
    # 排除自身后再确认没有任何 cq500* 模块被引用
    for name, mod in list(sys.modules.items()):
        if name.startswith("cq500") and mod is not None:
            origin = getattr(mod, "__file__", "") or ""
            if "Rsna_dataset" in origin:
                return f"rsna_commons 仍加载了 {name}"

    # 功能性验证：把 Cq500_dataset 从 sys.path 移除后仍能导入 rsna_commons
    real = ROOT.parent / "Cq500_dataset"
    if real.exists():
        while str(real) in sys.path:
            sys.path.remove(str(real))
    sys.modules.pop("rsna_commons", None)
    try:
        import rsna_commons as rs2  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"隐藏 Cq500_dataset 后 rsna_commons 无法导入: {type(e).__name__}: {e}"
    return True


# ---------------- 规模对比（可选） ----------------
def report_cohorts():
    """生成 RSNA vs CQ500 队列对比表；CQ500 不在同级目录时只报 RSNA。"""
    idx = _index()
    if idx.empty:
        return None
    rows = []
    for name, df in (("RSNA (subset)", idx),
                     ("CQ500", _load_cq500_index())):
        if df is None:
            continue
        pat = df.groupby("patient_id").size()
        pz = df.groupby("patient_id")["label"].first()
        rows.append({
            "cohort": name,
            "patients": int(df["patient_id"].nunique()),
            "slices": int(len(df)),
            "slices_per_patient_mean": round(float(pat.mean()), 1),
            "slices_per_patient_median": int(pat.median()),
            "patient_pos_rate": round(float((pz == "Hemorrhagic").mean()), 4),
        })
    rep = pd.DataFrame(rows)
    print("\n[队列对比]")
    print(rep.to_string(index=False))
    rep.to_csv(ROOT / "rsna_vs_cq500_cohort.csv", index=False)
    print(f"         -> {ROOT / 'rsna_vs_cq500_cohort.csv'}")
    if len(rows) == 1:
        print("         （未发现同级的 Cq500_dataset，仅统计 RSNA；"
              "把两者放同级即可生成对比行）")
    return True


def _load_cq500_index():
    p = ROOT.parent / "Cq500_dataset" / "cq500_slices_index.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["image_path"] = d["image_path"].astype(str).str.replace("\\", "/", regex=False)
    return d


def h2_noise_encoding_consistent():
    """09 的判别器数据集与 04 的噪声测试图必须用同一种编码（PNG 无损）。

    09 曾用 JPEG q95 写盘，实测把高斯噪声衰减 25.5%、斑点 23.4%，
    而椒盐/运动模糊几乎不受影响 —— 判别器学到的是"被压过的噪声"，
    推理时只在高斯/斑点上崩。训练与推理口径必须一致。
    """
    import rsna_commons as C
    src = (ROOT / "09_build_noise_detector_dataset_rsna.py").read_text(
        encoding="utf-8", errors="ignore")
    if "IMWRITE_JPEG" in src:
        return "09 仍用 JPEG 写判别器数据集（应与 04 的 PNG 无损一致）"
    if '".jpg"' in src or "'.jpg'" in src:
        return "09 的 out_name 仍带 .jpg 后缀"
    d = C.DETECTOR_DATASET_DIR
    if d.is_dir():
        jpgs = list(d.rglob("*.jpg"))
        if jpgs:
            return (f"判别器数据集含 {len(jpgs)} 个旧 JPEG 文件（与 04 的 PNG 不一致）。"
                    f"删掉整个目录后重跑 09: rm -rf {d.name}")
    return True


def l1_tta_variants_wired():
    """19 / 20 必须与共享层的 TTA 变体表同步（防版本错配）。

    背景：曾出现 "ich_common 已更新、19 没更新" 的错配。旧版 19 不认识 prior，
    会把它当未知变体交给 apply_tta；而 apply_tta 按路由表对 prior 返回**未改动的模型**，
    于是 prior 行与 none 行逐位相同、CSV 里也没有审计列 —— 看起来像"校正无效"，
    实际上是校正代码根本不存在。这个检查就是为了让这种情况在跑之前就报出来。
    """
    import ich_common as ic
    prior_variants = [v for v in ic.TTA_VARIANTS if ic.uses_prior_correction(v)]
    if not prior_variants:
        return True                       # 共享层没有校正变体，无需检查

    modes = {ic.prior_correction_mode(v) for v in prior_variants}
    # 两个脚本都必须真的调用校正；19 额外用 uses_prior_correction 做目标患病率诊断
    common = ["prior_correction_mode", "mcnemar", "boot_ci"]
    if "mean" in modes:
        common.append("match_prevalence")
    if "rate" in modes:
        common.append("match_prevalence_rate")

    problems = []
    # (文件, 该文件额外必须出现的符号)
    for fn, extra in (("19_tta_unknown_rsna.py", ["uses_prior_correction"]),
                      ("20_boundary_correction_rsna.py", [])):
        p = ROOT / fn
        if not p.exists():
            problems.append(f"{fn} 不存在")
            continue
        src = p.read_text(encoding="utf-8", errors="ignore")
        miss = [t for t in common + extra if t not in src]
        if miss:
            problems.append(f"{fn} 缺少 {miss}（旧版本，校正会被静默跳过）")
            continue
        for col in ("prior_target", "prior_bias", "pos_rate_before",
                    "pos_rate_after", "vs_none_p", "vs_none_delta"):
            if col not in src:
                problems.append(f"{fn} 缺少审计列 {col}")
    if problems:
        return "；".join(problems) + f"（共享层变体: {prior_variants}）"
    return True


def main():
    print("=" * 68)
    print("RSNA 子集自检")
    print("=" * 68)
    check("A1 患者级划分互斥（核心）", a_split_disjoint)
    check("B1 同一检查不跨集合", b_no_study_across_splits)
    check("B2 库中确有多检查患者（旧版会泄漏）", b2_multistudy_patients_caught)
    check("C1 抽样记账闭合", c_patient_accounting)
    check("D1 患者级标签 == 切片标签最大值", d_label_consistency)
    check("E1 索引与 PNG 文件一致（抽样）", e_index_files_exist)
    check("F1 预处理口径与共享脑窗一致", f_window_matches_cq500)
    check("G1 噪声/滤波/投票与共享模块同一份定义", g_shared_definitions)
    check("G2 不依赖任何其它数据集（可独立部署）", g2_no_cross_dataset_dependency)
    check("H1 生成队列规模对比表", report_cohorts)
    check("H2 判别器数据集与噪声测试图编码一致", h2_noise_encoding_consistent)
    check("L1 19/20 与共享层 TTA 变体表同步（防版本错配）", l1_tta_variants_wired)

    print("-" * 68)
    print(f"PASS={len(PASS)}  FAIL={len(FAIL)}  SKIP={len(SKIP)}")
    for f in FAIL:
        print(f"  FAIL: {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
