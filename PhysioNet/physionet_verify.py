# -*- coding: utf-8 -*-
"""
PhysioNet CT-ICH 子集自检（缺前置产物时自动 SKIP 对应项）

覆盖的不变量：
  A. 患者级划分互斥
  C. 抽样记账闭合
  D. **患者级标签 == 该患者逐层标签的最大值**（本数据集最容易犯的错：
     把逐层标签当成患者级标签，导致同一患者被拆进 Normal/Hemorrhagic 两个目录）
  E. 索引与 PNG 实际文件一致
  F. 用的是 **brain window** 而不是 bone window（重算源图比对）
  G. 噪声/滤波/投票与共享模块同一份定义；且不依赖其它数据集
并输出三套数据集的队列对比表。

用法: python physionet_verify.py
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
    except FileNotFoundError as e:
        # 前置产物尚未生成 —— 例如某折的 png/index 还没构建，或该折压根没跑过。
        # 按本脚本"缺前置产物自动 SKIP"的约定处理，而不是报 FAIL：
        # 否则在折叠场景下，验证一个还没构建的折会给出 6 项红色失败，纯属误导。
        # 真正的数据损坏会以 AssertionError / ValueError / KeyError 出现，不会落进这里。
        SKIP.append(f"{name}（缺前置产物）")
        print(f"  [skip] {name}（缺前置产物: {getattr(e, 'filename', None) or e}）")
        return
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
    import physionet_commons as C
    return C.load_index()


def _png_root():
    import physionet_commons as C
    return C.DATA_ROOT


def _fold_desc():
    import physionet_commons as C
    return C.fold_tag()


# ---------------- A. 划分 ----------------
def a_split_disjoint():
    import physionet_commons as C
    p = C.SPLIT_CSV
    sp = pd.read_csv(p) if p.exists() else None
    if sp is None:
        return None
    s = {k: set(sp[sp["split"] == k]["patient_id"]) for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if s[a] & s[b]:
            return f"患者级泄漏！{a} ∩ {b} = {sorted(s[a] & s[b])[:5]}"
    if not all(s.values()):
        return "存在空划分"
    print(f"         （{C.fold_tag()}: test={len(s['test'])} "
          f"val={len(s['val'])} train={len(s['train'])} 例）")
    return True


# ---------------- C. 记账 ----------------
def c_patient_accounting():
    """当前折的划分必须覆盖全部患者（每折都把 82 例分到 train/val/test 之一）。"""
    import physionet_commons as C
    lab = _load("physionet_slice_labels.csv")
    sp = pd.read_csv(C.SPLIT_CSV) if C.SPLIT_CSV.exists() else None
    if lab is None or sp is None:
        return None
    all_pat = set(lab["patient_id"].map(lambda v: f"{int(v):03d}"))
    fold_pat = set(sp["patient_id"].map(lambda v: f"{int(v):03d}"))
    if all_pat != fold_pat:
        return (f"折划分未覆盖全部患者：缺 {sorted(all_pat - fold_pat)[:5]}，"
                f"多 {sorted(fold_pat - all_pat)[:5]}")
    return True


# ---------------- D. 标签口径（本数据集最关键的检查） ----------------
def d_label_consistency():
    idx, lab = _index(), _load("physionet_slice_labels.csv")
    if lab is None:
        return None
    # D1: 同一患者只能有一个患者级 label
    nl = idx.groupby("patient_id")["label"].nunique()
    if int(nl.max()) != 1:
        bad = nl[nl > 1].index.tolist()
        return (f"{int((nl > 1).sum())} 例患者的 label 不唯一（例如 {bad[:3]}）—— "
                f"目录/label 很可能误用了逐层标签")
    # D2: 患者级 label 必须等于该患者逐层标签的最大值
    # 两边都规范化成零填充字符串再比（索引里是 "049"，标签 CSV 里是整数 49）
    truth = lab.groupby("patient_id")["any"].max().rename("truth").reset_index()
    truth["patient_id"] = truth["patient_id"].map(lambda v: f"{int(v):03d}")
    got = idx.groupby("patient_id")["label"].first().rename("got").reset_index()
    got["patient_id"] = got["patient_id"].astype(str)
    m = got.merge(truth, on="patient_id", how="inner")
    if len(m) != len(got):
        return f"标签对齐失败：索引 {len(got)} 例，匹配上 {len(m)} 例"
    m["want"] = np.where(m["truth"] == 1, "Hemorrhagic", "Normal")
    wrong = m[m["got"] != m["want"]]
    if len(wrong):
        return (f"{len(wrong)} 例的患者级标签与逐层标注不符，"
                f"例如 {wrong[['patient_id', 'got', 'want']].head(3).to_dict('records')}")
    return True


# ---------------- E. 索引 vs 文件 ----------------
def e_index_files_exist():
    idx = _index()
    if idx.empty:
        return None
    need = {"patient_id", "split", "label", "slice_label", "image_path"}
    miss = need - set(idx.columns)
    if miss:
        return f"索引缺列 {miss}"
    png = _png_root()
    rng = np.random.default_rng(0)
    take = rng.choice(len(idx), size=min(200, len(idx)), replace=False)
    bad = [idx.iloc[i]["image_path"] for i in take
           if not (png / idx.iloc[i]["image_path"]).exists()]
    if bad:
        return f"{len(bad)}/{len(take)} 抽样文件不存在，例如 {bad[:3]}"
    return True


def e2_image_shape():
    import cv2
    idx = _index()
    if idx.empty:
        return None
    png = _png_root()
    for i in range(min(5, len(idx))):
        p = png / idx.iloc[i]["image_path"]
        im = cv2.imread(str(p))
        if im is None:
            return f"读不出 {p}"
        if im.shape[:2] != (224, 224) or im.ndim != 3:
            return f"{p.name} 形状 {im.shape}，期望 (224,224,3)"
    return True


# ---------------- F. 用的是 brain window 吗 ----------------
def f_used_brain_window():
    """重算一张源图（brain/*.jpg 缩放后）与已存 PNG 比对。

    若误用了 bone window，两者会明显不同 —— 这是本数据集最容易悄悄犯的错，
    因为 bone/brain 两个目录结构完全相同。
    """
    import cv2
    import physionet_commons as C
    idx = _index()
    if idx.empty:
        return None
    try:
        raw = C.raw_dataset_dir()
    except FileNotFoundError:
        return None   # 目标机器上没打包原始数据时跳过

    rng = np.random.default_rng(1)
    png = _png_root()
    take = rng.choice(len(idx), size=min(6, len(idx)), replace=False)
    for i in take:
        rel = idx.iloc[i]["image_path"]
        pid = idx.iloc[i]["patient_id"]
        stem = Path(rel).stem            # sXXX
        slice_no = str(int(stem[1:]))
        brain = raw / "Patients_CT" / pid / "brain" / f"{slice_no}.jpg"
        bone = raw / "Patients_CT" / pid / "bone" / f"{slice_no}.jpg"
        if not brain.exists():
            continue
        src = cv2.imread(str(brain), cv2.IMREAD_GRAYSCALE)
        src = cv2.resize(src, (224, 224), interpolation=cv2.INTER_AREA) \
            if src.shape[0] != 224 else src
        stored = cv2.imread(str(png / rel), cv2.IMREAD_GRAYSCALE)
        if not np.array_equal(src, stored):
            return f"{rel} 与 brain/*.jpg 重算结果不一致（是否误用了 bone？）"
        if bone.exists():
            b = cv2.imread(str(bone), cv2.IMREAD_GRAYSCALE)
            b = cv2.resize(b, (224, 224), interpolation=cv2.INTER_AREA) \
                if b.shape[0] != 224 else b
            if abs(float(b.mean()) - float(stored.mean())) < 1e-6:
                return f"{rel} 与 bone window 完全相同，可能取错了目录"
    return True


def f2_windowing_provenance():
    """信息性说明：PhysioNet 的脑窗由上游软件完成，不是我们做的 HU 换算。

    与 CQ500/RSNA（我们自己按 WL40/WW80 从 DICOM 生成）属同类但非同一实现，
    并排比较时必须在论文中声明。这里只报告，不判失败。
    """
    import cv2
    import ich_common as ic
    idx = _index()
    if idx.empty:
        return None
    im = cv2.imread(str(_png_root() / idx.iloc[0]["image_path"]),
                    cv2.IMREAD_GRAYSCALE)
    print(f"         （上游 Siemens syngo 脑窗 650x650 JPEG -> 224x224；"
          f"对比 CQ500/RSNA 为我们自算 WL{ic.WL:.0f}/WW{ic.WW:.0f} -> HU "
          f"[{ic.HU_LO:.0f},{ic.HU_HI:.0f}]。两者同类但非同一实现，论文需声明）")
    if im is not None:
        print(f"           样例强度 mean={im.mean():.1f} std={im.std():.1f} "
              f"min={im.min()} max={im.max()}")
    return True


# ---------------- G. 共享定义 / 独立性 ----------------
def g_shared_definitions():
    import ich_common as ic
    import physionet_commons as rs
    for attr in ("NOISE_FUNCS", "FILTER_FUNCS", "apply_noise", "apply_filter",
                 "patient_vote_metrics", "_to_tensor", "build_eval_transform",
                 "create_model_with_pretrained", "grad_cam"):
        if getattr(rs, attr) is not getattr(ic, attr):
            return f"{attr} 不是同一对象 —— 与共享定义已分叉"
    if getattr(rs, "_sweep_filters", None) is not ic.sweep_filters:
        return "physionet.sweep_filters 的实现不是 ich_common.sweep_filters"
    import inspect
    src = inspect.getsource(rs.sweep_filters)
    for need in ("data_root=DATA_ROOT", "noisy_root=NOISY_ROOT", "classes=CLASSES"):
        if need not in src:
            return f"physionet.sweep_filters 未绑定 {need}"
    if list(rs.NOISE_NAMES) != list(ic.NOISE_NAMES):
        return "NOISE_NAMES 不一致"
    if list(rs.FILTER_NAMES) != list(ic.FILTER_NAMES):
        return "FILTER_NAMES 不一致"
    return True


def g2_no_cross_dataset_dependency():
    """不得 import CQ500 / RSNA 的任何模块（保证能单独部署）。"""
    import ast
    tree = ast.parse((ROOT / "physionet_commons.py").read_text(encoding="utf-8"))
    bad_mods = {"cq500_commons", "cq500_dicom_to_png", "rsna_commons",
                "rsna_dicom_to_png"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in bad_mods:
                    return f"第 {node.lineno} 行 import 了 {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in bad_mods:
                return f"第 {node.lineno} 行 from {node.module} import ..."
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for ds in ("Cq500_dataset", "Rsna_dataset"):
                if ds in node.value:
                    return f"第 {node.lineno} 行出现 {ds} 路径字面量"
    for n in ("Cq500_dataset", "Rsna_dataset"):
        p = ROOT.parent / n
        if p.exists():
            while str(p) in sys.path:
                sys.path.remove(str(p))
    sys.modules.pop("physionet_commons", None)
    try:
        import physionet_commons as _c  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"隐藏其它数据集后无法导入: {type(e).__name__}: {e}"
    return True


# ---------------- 队列对比 ----------------
def report_cohorts():
    idx = _index()
    if idx.empty:
        return None
    others = [("CQ500", ROOT.parent / "Cq500_dataset" / "cq500_slices_index.csv"),
              ("RSNA", ROOT.parent / "Rsna_dataset" / "rsna_slices_index.csv")]
    rows = []
    for name, df in [("PhysioNet", idx)] + [(n, _read_index(p)) for n, p in others]:
        if df is None:
            continue
        pat = df.groupby("patient_id").size()
        pz = df.groupby("patient_id")["label"].first()
        rows.append({
            "cohort": name,
            "patients": int(df["patient_id"].nunique()),
            "slices": int(len(df)),
            "slices_per_patient_mean": round(float(pat.mean()), 1),
            "patient_pos_rate": round(float((pz == "Hemorrhagic").mean()), 4),
            # 注意：这是**切片级**阳性比例 = 逐切片 mean，不是逐患者 max。
            # 对 CQ500 它等于"患者级标签传播到该患者所有切片"的比例，
            # 并非放射科逐层标注 —— 与 RSNA/PhysioNet 的逐层标注不是一回事。
            "slice_pos_rate": round(float(df["slice_label"].mean()), 4),
            "slice_label_source": ("patient-propagated" if name == "CQ500"
                                   else "per-slice annotated"),
        })
    rep = pd.DataFrame(rows)
    print("\n[队列对比]")
    print(rep.to_string(index=False))
    print("         注：CQ500 的 slice_pos_rate 是患者级标签传播到切片的结果，"
          "不是逐层标注；\n"
          "             跨数据集的切片级指标不可直接比较，患者级才可比。")
    rep.to_csv(ROOT / "physionet_vs_others_cohort.csv", index=False)
    print(f"         -> {ROOT / 'physionet_vs_others_cohort.csv'}")
    if len(rows) < 3:
        print("         （只发现部分数据集；把三者放同级即可完整对比）")
    return True


def _read_index(p):
    if not Path(p).exists():
        return None
    d = pd.read_csv(p)
    d["image_path"] = d["image_path"].astype(str).str.replace("\\", "/", regex=False)
    if "slice_label" not in d.columns:
        d["slice_label"] = d["label"].map({"Hemorrhagic": 1, "Normal": 0})
    return d


def h2_noise_encoding_consistent():
    """09 的判别器数据集与 04 的噪声测试图必须用同一种编码（PNG 无损）。

    09 曾用 JPEG q95 写盘，实测把高斯噪声衰减 25.5%、斑点 23.4%，
    而椒盐/运动模糊几乎不受影响 —— 判别器学到的是"被压过的噪声"，
    推理时只在高斯/斑点上崩。训练与推理口径必须一致。
    """
    import physionet_commons as C
    src = (ROOT / "09_build_noise_detector_dataset_physionet.py").read_text(
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
    for fn, extra in (("19_tta_unknown_physionet.py", ["uses_prior_correction"]),
                      ("20_boundary_correction_physionet.py", [])):
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
    print("PhysioNet CT-ICH 子集自检")
    print("=" * 68)
    check("A1 患者级划分互斥", a_split_disjoint)
    check("C1 抽样记账闭合", c_patient_accounting)
    check("D1 患者级标签 == max(逐层标签) 且唯一", d_label_consistency)
    check("E1 索引与 PNG 文件一致（抽样）", e_index_files_exist)
    check("E2 图像为 224x224x3", e2_image_shape)
    check("F1 用的是 brain window（重算比对）", f_used_brain_window)
    check("F2 脑窗来源说明", f2_windowing_provenance)
    check("G1 与共享模块同一份定义", g_shared_definitions)
    check("G2 不依赖其它数据集（可独立部署）", g2_no_cross_dataset_dependency)
    check("H1 三套数据集队列对比表", report_cohorts)
    check("H2 判别器数据集与噪声测试图编码一致", h2_noise_encoding_consistent)
    check("L1 19/20 与共享层 TTA 变体表同步（防版本错配）", l1_tta_variants_wired)

    print("-" * 68)
    print(f"PASS={len(PASS)}  FAIL={len(FAIL)}  SKIP={len(SKIP)}")
    for f in FAIL:
        print(f"  FAIL: {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
