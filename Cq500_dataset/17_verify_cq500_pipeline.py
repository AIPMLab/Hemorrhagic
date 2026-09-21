# -*- coding: utf-8 -*-
"""
CQ500 流水线自检（可在任何阶段运行，缺前置产物时自动 SKIP 对应项）

覆盖本次修复的回归点与数据完整性不变量：
  A. filter 互异性 / clahe_gamma 语义         —— 回归 B4
  B. low_light 方向 / 复合噪声                —— 回归 B5
  C. 噪声函数与 filter 函数单一事实来源        —— 回归 B6
  D. 增强图路径含噪声层级                      —— 回归 B1
  E. 判别器类别数 7 且为字母序（与 ImageFolder 一致）
  F. 患者级划分互不相交（train/val/test）
  G. 索引完整性（抽样检查 image_path 是否存在）
  H. patient_vote_metrics 行为正确性
  I. 判别器数据集源患者 ∩ test 患者 = ∅（需 09 产物）
  J. sweep_filters 与 apply_filter 口径一致（需 02 权重 + 04 产物）

用法: python 17_verify_cq500_pipeline.py
退出码 0 = 全部通过（含 SKIP），1 = 存在失败
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from cq500_commons import (DATA_ROOT, DETECTOR_CLASSES, DETECTOR_DATASET_DIR,
                           DETECTOR_VARIANTS, FILTER_FUNCS, FILTER_NAMES, NOISE_FUNCS,
                           NOISE_NAMES, TTA_VARIANTS, apply_filter, apply_noise,
                           detector_weight_path, enhanced_dir, load_index,
                           prior_correction_mode, uses_prior_correction)

HERE = Path(__file__).resolve().parent
PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    try:
        result = fn()
    except Exception as e:  # noqa: BLE001
        FAIL.append(f"{name}: 异常 {type(e).__name__}: {e}")
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return
    if result is True:
        PASS.append(name)
        print(f"  [ok]   {name}")
    elif result is None:
        SKIP.append(name)
        print(f"  [skip] {name}")
    else:
        FAIL.append(f"{name}: {result}")
        print(f"  [FAIL] {name}: {result}")


def sample_test_image():
    df = load_index()
    test = df[df["split"] == "test"]
    for rel in test["image_path"]:
        p = DATA_ROOT / rel
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    raise FileNotFoundError("找不到可读的 test 图像")


# ---------------- A. filter ----------------
def a_filters_distinct():
    img = sample_test_image()
    outs = {f: apply_filter(img, f) for f in FILTER_NAMES}
    for f, o in outs.items():
        if o.shape != img.shape:
            return f"{f} 输出形状 {o.shape} != 输入 {img.shape}"
        if o.dtype != np.uint8:
            return f"{f} 输出 dtype {o.dtype} != uint8"
    pairs = []
    for f in FILTER_NAMES:
        for g in FILTER_NAMES:
            if f < g and np.array_equal(outs[f], outs[g]):
                pairs.append((f, g))
    if pairs:
        return f"以下 filter 输出完全相同: {pairs}"
    return True


def a_clahe_gamma_has_clahe():
    img = sample_test_image()
    cg = apply_filter(img, "clahe_gamma")
    ga = apply_filter(img, "gamma")
    cl = apply_filter(img, "clahe")
    if np.array_equal(cg, ga):
        return "clahe_gamma == gamma（CLAHE 步骤丢失，回归 B4）"
    if np.array_equal(cg, cl):
        return "clahe_gamma == clahe（gamma 步骤丢失）"
    return True


def a_filter_names_match_funcs():
    if FILTER_NAMES != list(FILTER_FUNCS.keys()):
        return f"FILTER_NAMES 与 FILTER_FUNCS 键不一致"
    if len(FILTER_NAMES) != 10 or FILTER_NAMES[0] != "none":
        return f"期望 10 个 filter 且 none 在首位，实际 {FILTER_NAMES}"
    return True


# ---------------- B. noise ----------------
def b_low_light_darkens():
    img = sample_test_image()
    out = apply_noise(img, "low_light")
    if out.mean() >= img.mean():
        return (f"low_light 未变暗: {img.mean():.2f} -> {out.mean():.2f}（回归 B5）")
    return True


def b_compound_is_compounded():
    img = sample_test_image()
    mixed = apply_noise(img, "unknown_mixed")
    low = apply_noise(img, "low_light")
    blur = apply_noise(img, "motion_blur")
    if np.array_equal(mixed, img):
        return "unknown_mixed 未改变图像"
    if np.array_equal(mixed, low) or np.array_equal(mixed, blur):
        return "unknown_mixed 退化为单一噪声"
    return True


def b_noises_dtype_and_range():
    img = sample_test_image()
    for n in NOISE_NAMES:
        o = apply_noise(img, n)
        if o.dtype != np.uint8:
            return f"{n} dtype {o.dtype} != uint8"
        if o.min() < 0 or o.max() > 255:
            return f"{n} 越界 [{o.min()}, {o.max()}]"
    if NOISE_FUNCS["clean"] is not None:
        return "clean 应为 None（不加噪）"
    return True


def b_noise_seeded_reproducible():
    img = sample_test_image()
    np.random.seed(123)
    a = apply_noise(img, "unknown_mixed")
    np.random.seed(123)
    b = apply_noise(img, "unknown_mixed")
    if not np.array_equal(a, b):
        return "同一 seed 下 unknown_mixed 不可复现"
    return True


# ---------------- C. 单一事实来源 ----------------
NOISE_DEF = re.compile(
    r"^\s*def\s+\w*(gaussian|salt_pepper|speckle|motion_blur|low_light|unknown_mixed)",
    re.M)
FILTER_DEF = re.compile(
    r"^\s*def\s+\w*(median_filter|gaussian_filter|bilateral_filter|clahe_enhance|"
    r"gamma_correction|unsharp_mask|sharpen_filter|hist_equalization)", re.M)


# FILTER_DEF 的备选名（8 个 def 覆盖全部 filter；"clahe"/"none" 由这些函数组合而来）
FILTER_DEF_NAMES = {"median_filter", "gaussian_filter", "bilateral_filter", "clahe_enhance",
                    "gamma_correction", "unsharp_mask", "sharpen_filter", "hist_equalization"}


def c_single_source_of_truth():
    """噪声/filter 函数只能有一个定义处。

    判据从"共享模块实际在哪"出发（ich_common.__file__），**不写死文件名**：
    定义已从 <dataset>_commons 移进共享模块，写死 'cq500_commons.py' 会把
    共享模块自己误判成"重复定义"。共享模块放 code 根还是放数据集目录内都应当通过。
    同时反向验证共享模块确实定义了全套函数，否则这个检查会在"谁都没定义"时空过。
    """
    import ich_common
    canonical = Path(ich_common.__file__).resolve()
    src = canonical.read_text(encoding="utf-8", errors="ignore")

    miss_n = set(NOISE_NAMES) - set(NOISE_DEF.findall(src))
    if miss_n:
        return (f"共享模块 {canonical.name} 缺噪声函数定义: {sorted(miss_n)}"
                f"（单一事实来源已破坏）")
    miss_f = FILTER_DEF_NAMES - set(FILTER_DEF.findall(src))
    if miss_f:
        return (f"共享模块 {canonical.name} 缺 filter 定义: {sorted(miss_f)}"
                f"（单一事实来源已破坏）")

    me = Path(__file__).resolve()
    offenders = []
    for p in sorted(HERE.glob("*.py")):
        rp = p.resolve()
        if rp == canonical or rp == me:
            continue          # 共享模块自己、以及本检查脚本，都不算重复
        s = p.read_text(encoding="utf-8", errors="ignore")
        for label, pat in (("noise", NOISE_DEF), ("filter", FILTER_DEF)):
            hits = pat.findall(s)
            if hits:
                offenders.append(f"{p.name}({label}:{sorted(set(hits))})")
    if offenders:
        return (f"以下文件重复定义了噪声/filter 函数，应改为从共享模块 import: "
                f"{offenders}（共享模块 = {canonical}）")
    return True


# ---------------- D. 路径含噪声层级 ----------------
def d_enhanced_dir_has_noise_level():
    p = enhanced_dir("gaussian", "median")
    if "gaussian" not in p.parts:
        return f"enhanced_dir 缺噪声层级: {p}"
    p2 = enhanced_dir("unknown_mixed", "unsharp")
    if p2 == p or p2.parts[-2:] != ("unknown_mixed", "unsharp"):
        return f"enhanced_dir 结果异常: {p2}"
    return True


# ---------------- E. 判别器类别 ----------------
def e_detector_classes():
    if len(DETECTOR_CLASSES) != 7:
        return f"判别器类别数 {len(DETECTOR_CLASSES)} != 7"
    if DETECTOR_CLASSES != sorted(DETECTOR_CLASSES):
        return f"类别非字母序（与 ImageFolder sorted() 不一致）: {DETECTOR_CLASSES}"
    missing = [c for c in NOISE_NAMES if c not in DETECTOR_CLASSES]
    if missing or "clean" not in DETECTOR_CLASSES:
        return f"判别器类别未覆盖全部噪声: 缺 {missing}"
    # 两个变体都必须是全集里保持字母序的子集；6class 必须不含复合噪声，7class 必须含
    for v, cls in DETECTOR_VARIANTS.items():
        if cls != sorted(cls):
            return f"变体 {v} 类别非字母序: {cls}"
        if not set(cls) <= set(DETECTOR_CLASSES):
            return f"变体 {v} 含未知类别: {set(cls) - set(DETECTOR_CLASSES)}"
    if "unknown_mixed" in DETECTOR_VARIANTS["6class"]:
        return "6class 变体不应包含 unknown_mixed"
    if "unknown_mixed" not in DETECTOR_VARIANTS["7class"]:
        return "7class 变体应包含 unknown_mixed"
    if DETECTOR_VARIANTS["7class"] != DETECTOR_CLASSES:
        return "7class 变体应等于全集 DETECTOR_CLASSES"
    return True


def e2_detector_weights_selfconsistent():
    """已存在的判别器权重，其 class_names 必须与对应变体的类别一致。"""
    import torch
    checked = 0
    for v, cls in DETECTOR_VARIANTS.items():
        p = detector_weight_path(v)
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck.get("class_names") != cls:
            return (f"{p.name} 的 class_names={ck.get('class_names')} "
                    f"与变体 {v}={cls} 不一致")
        if ck.get("variant") != v:
            return f"{p.name} 的 variant 字段={ck.get('variant')} 与 {v} 不一致"
        checked += 1
    return True if checked else None


# ---------------- F. 划分互不相交 ----------------
def f_splits_disjoint():
    df = load_index()
    sets = {s: set(df[df["split"] == s]["patient_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets[a] & sets[b]
        if inter:
            return f"{a} ∩ {b} 非空 ({len(inter)} 例): {sorted(inter)[:5]}"
    if not all(sets.values()):
        return "存在空划分"
    return True


def f2_split_matches_index():
    """cq500_split.csv 覆盖的患者应都能在索引里找到；
    否则 split_summary 里的患者数（论文里的 N）与实际评估人数不符。"""
    sp_path = HERE / "cq500_split.csv"
    if not sp_path.exists():
        return None
    sp = pd.read_csv(sp_path)
    idx = set(load_index()["patient_id"])
    miss = sp[~sp["patient_id"].isin(idx)]
    if len(miss):
        by_split = miss.groupby("split").size().to_dict()
        return (f"{len(miss)} 例患者有划分但无任何切片（按划分 {by_split}），"
                f"例: {sorted(miss['patient_id'])[:5]}；"
                f"论文中的 N 会与 split_summary 不符，需补转换或重建划分")
    return True


def f3_patient_accounting():
    """患者记账必须闭合：划分患者 ∪ 排除患者 == reads.csv 标注患者。

    官方 CQ500 队列为 491 例，本地原始数据集只有 473 例有 DICOM；
    这 18 例被显式记录在 cq500_excluded_patients.csv，而不是无声消失。
    """
    exc_p = HERE / "cq500_excluded_patients.csv"
    sp_p = HERE / "cq500_split.csv"
    lab_p = HERE / "cq500_patient_labels.csv"
    if not all(p.exists() for p in (exc_p, sp_p, lab_p)):
        return None
    exc = set(pd.read_csv(exc_p)["patient_id"])
    sp = set(pd.read_csv(sp_p)["patient_id"])
    lab = set(pd.read_csv(lab_p)["patient_id"])
    if sp & exc:
        return f"排除名单与划分患者重叠 {len(sp & exc)} 例: {sorted(sp & exc)[:5]}"
    if sp | exc != lab:
        unassigned = sorted(lab - (sp | exc))
        extra = sorted((sp | exc) - lab)
        return (f"记账不闭合: 划分 {len(sp)} + 排除 {len(exc)} = {len(sp | exc)} "
                f"!= 标注 {len(lab)}；未归属 {unassigned[:5]}，多出 {extra[:5]}")
    return True


# ---------------- G. 索引完整性 ----------------
def g_index_integrity():
    df = load_index()
    if set(df.columns) < {"patient_id", "split", "label", "image_path"}:
        return f"索引列缺失: {list(df.columns)}"
    if df["label"].nunique() != 2:
        return f"标签数 != 2: {df['label'].unique()}"
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(200, len(df)), replace=False)
    bad = [df.iloc[i]["image_path"] for i in idx
           if not (DATA_ROOT / df.iloc[i]["image_path"]).exists()]
    if bad:
        return f"{len(bad)}/{len(idx)} 抽样 image_path 不存在，例如 {bad[:3]}"
    return True


# ---------------- H. 投票逻辑 ----------------
def h_patient_vote_behavior():
    from cq500_commons import patient_vote_metrics
    # 患者 A: 3 张切片中 2 张判出血 -> 均值 >0.5 -> 患者级出血；真值出血
    # 患者 B: 均值 <0.5 但真值出血 -> 患者级漏判
    sdf = pd.DataFrame({
        "patient_id": ["A"] * 3 + ["B"] * 3,
        "label": [1] * 3 + [1] * 3,
        "prob": [0.9, 0.8, 0.4, 0.6, 0.4, 0.3],
    })
    sdf["pred"] = (sdf["prob"] >= 0.5).astype(int)
    m = patient_vote_metrics(sdf, tag="self-test")
    if m["num_patients"] != 2:
        return f"患者数 {m['num_patients']} != 2"
    # A: mean=0.7 -> 出血(1) 正确; B: mean=0.433 -> 正常(0) 错误 => acc=0.5
    if abs(m["accuracy"] - 0.5) > 1e-9:
        return f"患者级 acc {m['accuracy']} != 0.5（投票逻辑异常）"
    if m["confusion"] != [[0, 0], [1, 1]]:
        return f"混淆矩阵 {m['confusion']} != [[0,0],[1,1]]"
    return True


# ---------------- I. 判别器数据无泄漏 ----------------
def i_detector_no_leak():
    mf = DETECTOR_DATASET_DIR / "manifest.csv"
    if not mf.exists():
        return None
    man = pd.read_csv(mf)
    df = load_index()
    test_pat = set(df[df["split"] == "test"]["patient_id"])
    inter = set(man["patient_id"]) & test_pat
    if inter:
        return f"判别器源患者与 test 重叠 ({len(inter)} 例): {sorted(inter)[:5]}"
    if set(man["source_split"]) - {"train", "val"}:
        return f"manifest 出现非 train/val 来源: {set(man['source_split'])}"
    if man["class"].nunique() != 7:
        return f"manifest 类别数 {man['class'].nunique()} != 7"
    return True


def k1_sweep_tensor_matches_transform():
    """_SweepDataset 产出的张量必须与 build_eval_transform 的口径逐元素一致。

    sweep 为了并行化把"解码+加噪+滤波+resize+normalize"整体挪进了 worker，
    这是最容易悄悄改变数值的地方（通道顺序、resize 插值、归一化顺序）。
    """
    import cq500_commons as C
    from cq500_commons import build_eval_transform, _SweepDataset
    # 只从 <dataset>_commons 取东西：共享模块 ich_common 放在哪里是它的内部实现，
    # 直接 import ich_common 会在"只拷了 Cq500_dataset 目录"的机器上炸掉。

    df = load_index()
    test = df[df["split"] == "test"].head(3)
    if test.empty:
        return None
    filt = ["none", "gamma", "clahe_gamma"]
    ds = _SweepDataset(test, C.DATA_ROOT, "clean", filt, True, None)
    tf = build_eval_transform()

    for k in range(len(ds)):
        t_new, i, j = ds[k]
        rel = test.iloc[i]["image_path"]
        arr = cv2.cvtColor(
            np.array(Image.open(C.DATA_ROOT / rel).convert("RGB")), cv2.COLOR_RGB2BGR)
        arr = apply_filter(arr, filt[j])
        ref = tf(Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)))
        if t_new.shape != ref.shape:
            return f"切片{i}/filter{filt[j]}: 形状 {tuple(t_new.shape)} != {tuple(ref.shape)}"
        d = (t_new - ref).abs().max().item()
        if d > 1e-6:
            return f"切片{i}/filter{filt[j]}: 张量不一致 max diff {d:.3e}"
    return True


# ---------------- J. sweep 与 apply_filter 口径一致 ----------------
def j_sweep_matches_apply():
    from cq500_commons import MODEL_DIR, NOISY_ROOT, load_model, sweep_filters
    w = MODEL_DIR / "resnet50_cq500_best.pth"
    if not w.exists():
        return None
    if not (NOISY_ROOT / "unknown_mixed").exists():
        return None
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("resnet50", w, device)

    limit = 24
    sweeps = sweep_filters(split="test", noise_name="unknown_mixed",
                           filter_names=FILTER_NAMES, model=model, device=device,
                           batch_size=1, num_workers=0, from_disk=True, limit=limit)

    # 手工复算：直接 apply_filter + 前向，与 sweep 的结果逐张比对。
    # batch_size 与手工路径对齐（都是 1）：这样这个检查只反映"预处理/滤波口径"，
    # 不掺 cuDNN 的 batch 数值噪声（实测 batch 1 vs 64 只差 ~5e-6，属预防性对齐）。
    from cq500_commons import build_eval_transform
    from PIL import Image
    tf = build_eval_transform()
    df = load_index()
    df = df[df["split"] == "test"].reset_index(drop=True).head(limit)
    model.eval()
    for f in ("none", "clahe_gamma", "unsharp"):
        manual, manual_t = [], []
        for rel in df["image_path"]:
            arr = cv2.imread(str(NOISY_ROOT / "unknown_mixed" / rel))
            t = tf(Image.fromarray(cv2.cvtColor(apply_filter(arr, f), cv2.COLOR_BGR2RGB)))
            manual_t.append(t)
            with torch.no_grad():
                p = torch.softmax(model(t.unsqueeze(0).to(device)), dim=1)[0, 1].item()
            manual.append(p)
        sw = sweeps[f].sort_values("slice_index")["prob"].values
        if not np.allclose(manual, sw, atol=1e-5):
            # 自诊断：分清是"预处理出的张量不同"还是"同一张量前向结果不同"。
            # 前者是 sweep 的滤波/resize/normalize 口径问题；
            # 后者是模型或前向方式（eval 模式、权重）问题。两者的修法完全不同。
            ds = _SweepDataset(df, NOISY_ROOT / "unknown_mixed", "unknown_mixed",
                               [f], True, None)
            t_sweep, i0, _ = ds[0]
            t_diff = (t_sweep - manual_t[i0]).abs().max().item()
            return (f"{f}: sweep 与手工前向不一致 "
                    f"(概率 max diff {np.max(np.abs(np.array(manual) - sw)):.2e}；"
                    f"预处理张量 max diff {t_diff:.2e} "
                    f"-> {'张量就不同，属预处理口径问题' if t_diff > 1e-6 else '张量相同，问题在前向/模型'})")
    return True


def d2_noise_encoding_consistent():
    """09 的判别器数据集与 04 的噪声测试图必须用同一种编码（PNG 无损）。

    09 曾用 JPEG q95 写盘，实测把高斯噪声衰减 25.5%、斑点 23.4%，
    而椒盐/运动模糊几乎不受影响 —— 判别器学到的是"被压过的噪声"，
    推理时只在高斯/斑点上崩（test acc 0.02 / 0.35）。训练与推理口径必须一致。
    """
    from cq500_commons import DETECTOR_DATASET_DIR
    src = (HERE / "09_build_noise_detector_dataset_cq500.py").read_text(encoding="utf-8")
    if "IMWRITE_JPEG" in src:
        return "09 仍用 JPEG 写判别器数据集（应与 04 的 PNG 无损一致）"
    if '".jpg"' in src or "'.jpg'" in src:
        return "09 的 out_name 仍带 .jpg 后缀"
    if DETECTOR_DATASET_DIR.is_dir():
        jpgs = list(DETECTOR_DATASET_DIR.rglob("*.jpg"))
        if jpgs:
            return (f"判别器数据集含 {len(jpgs)} 个旧 JPEG 文件（与 04 的 PNG 不一致）。"
                    f"这是修复前的残留，删掉整个目录后重跑 09 即可: "
                    f"rm -rf {DETECTOR_DATASET_DIR.name}")
    return True


def l1_tta_variants_wired():
    """19 必须与共享层的 TTA 变体表同步（防版本错配）。

    背景：曾出现 "ich_common 已更新、19 没更新" 的错配。旧版 19 不认识 prior，
    会把它当未知变体交给 apply_tta；而 apply_tta 按路由表对 prior 返回**未改动的模型**，
    于是 prior 行与 none 行逐位相同、CSV 里也没有审计列——看起来像"校正无效"，
    实际上是校正代码根本不存在。这个检查就是为了让这种情况在跑之前就报出来。
    """
    prior_variants = [v for v in TTA_VARIANTS if uses_prior_correction(v)]
    if not prior_variants:
        return True                       # 共享层没有校正变体，无需检查
    src = (HERE / "19_tta_unknown_cq500.py").read_text(encoding="utf-8",
                                                       errors="ignore")
    # 两种校正口径都必须真的被 19 调用：mean=prior/bn_prior，rate=priorq/bn_priorq
    modes = {prior_correction_mode(v) for v in prior_variants}
    need = ["uses_prior_correction"]
    if "mean" in modes:
        need.append("match_prevalence")
    if "rate" in modes:
        need.append("match_prevalence_rate")
    need += ["mcnemar", "boot_ci"]        # 配对检验
    missing = [tok for tok in need if tok not in src]
    if missing:
        return (f"共享层已声明先验校正变体 {prior_variants}，但 19 的源码里没有 "
                f"{missing} —— 19 是旧版本（版本错配），校正会被静默跳过。"
                f"请先覆盖 19_tta_unknown_cq500.py")
    for col in ("prior_target", "prior_bias", "pos_rate_before", "pos_rate_after",
                "vs_none_p", "vs_none_delta"):
        if col not in src:
            return f"19 缺少先验校正/配对检验的审计列 {col}"
    return True


def main():
    print("=" * 68)
    print("CQ500 流水线自检")
    print("=" * 68)
    check("A1 10 个 filter 输出两两不同", a_filters_distinct)
    check("A2 clahe_gamma 同时含 CLAHE 与 gamma (B4)", a_clahe_gamma_has_clahe)
    check("A3 FILTER_NAMES 与 FILTER_FUNCS 一致", a_filter_names_match_funcs)
    check("B1 low_light 确实变暗 (B5)", b_low_light_darkens)
    check("B2 unknown_mixed 确为复合", b_compound_is_compounded)
    check("B3 噪声输出 uint8 且在 [0,255]", b_noises_dtype_and_range)
    check("B4 同一 seed 下噪声可复现", b_noise_seeded_reproducible)
    check("C1 噪声/filter 函数单一事实来源 (B6)", c_single_source_of_truth)
    check("D1 增强图路径含噪声层级 (B1)", d_enhanced_dir_has_noise_level)
    check("D2 判别器数据集与噪声测试图编码一致", d2_noise_encoding_consistent)
    check("E1 判别器 7 类且字母序（含两变体）", e_detector_classes)
    check("E2 判别器权重与变体自洽 (需 10)", e2_detector_weights_selfconsistent)
    check("F1 train/val/test 患者互不相交", f_splits_disjoint)
    check("F2 划分患者均有切片（N 与 split_summary 一致）", f2_split_matches_index)
    check("F3 患者记账闭合（划分 ∪ 排除 = reads.csv）", f3_patient_accounting)
    check("G1 索引完整性抽样", g_index_integrity)
    check("H1 patient_vote_metrics 行为正确", h_patient_vote_behavior)
    check("I1 判别器数据源无 test 泄漏 (需 09)", i_detector_no_leak)
    check("K1 sweep 张量与 eval transform 数值一致", k1_sweep_tensor_matches_transform)
    check("J1 sweep_filters 与 apply_filter 口径一致 (需 02+04)", j_sweep_matches_apply)
    check("L1 19 与共享层 TTA 变体表同步", l1_tta_variants_wired)

    print("-" * 68)
    print(f"PASS={len(PASS)}  FAIL={len(FAIL)}  SKIP={len(SKIP)}")
    for f in FAIL:
        print(f"  FAIL: {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
