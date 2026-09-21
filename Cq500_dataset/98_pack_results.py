# -*- coding: utf-8 -*-
"""
打包 CQ500 关键结果，便于从服务器取回做分析。

按"对论文结论的贡献"分层，默认只打包前两层（很小，几 MB），
明细层与权重按需加。同时会报告**缺失**的文件 —— 一眼就能看出哪一步没跑完。

分层:
  tier 1  核心结论      约 1 MB    Q1/Q2、两个判别器对比、val 拟合的 filter 表、论文表 1-10
  tier 2  支撑表与图表  约 10 MB   各噪声基线、全网格、逐类准确率、全部 png/tex
  tier 3  逐切片明细    约 100 MB  per-slice 日志、逐患者指标、全量质量指标（复核/复算用）
  tier 4  模型权重      约 250 MB  4 个 .pth（要重新推理或核验权重时）

用法:
    python 98_pack_results.py --dest ../results_bundle
    python 98_pack_results.py --dest out --tier 1
    python 98_pack_results.py --dest out --tier all --with-weights
    python 98_pack_results.py --dest out --list          # 只列出会打包什么、缺什么
    python 98_pack_results.py --dest out --results-root 别的/results   # 换个结果根目录

**只打包 CQ500**：physionet / rsna 等其它数据集的结果一律排除，
判据是一级目录以 `_cq500` 结尾、或一级文件以 `cq500_` 开头。
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cq500_commons import DETECTOR_VARIANTS, MODEL_DIR, RESULT_ROOT

VARIANTS = sorted(DETECTOR_VARIANTS)
DS = "cq500"          # 只打包这个数据集的结果


def is_ds(rel):
    """判定一个相对路径属不属于目标数据集。"""
    first = rel.parts[0]
    return first.endswith("_" + DS) or first.startswith(DS + "_")

# (相对 RESULT_ROOT 的路径或 glob, 层级, 说明)
SPECS = []


def add(path, tier, note):
    SPECS.append((path, tier, note))


# ---------------- tier 1: 核心结论 ----------------
for v in VARIANTS:
    add(f"optimality_cq500/{v}/oracle_summary_cq500.csv", 1,
        f"[{v}] Q1 恢复性 + Q2 最优性 一行汇总：adaptive/noisy/三层 oracle/gap/一致性/McNemar")
    add(f"optimality_cq500/{v}/filter_sweep_unknown_mixed_cq500.csv", 1,
        f"[{v}] 复合噪声下 10 个固定 filter 的患者级指标 —— oracle 的依据")
    add(f"optimality_cq500/{v}/per_patient_oracle_cq500.csv", 1,
        f"[{v}] 逐患者：oracle 最优 filter vs adaptive 实际选择 vs 是否判对")
    add(f"optimality_cq500/{v}/detector_accuracy_cq500.csv", 1,
        f"[{v}] 判别器在 7 种条件下的表现（6class 在复合噪声上 acc=NaN 属正常）")
    add(f"adaptive_unknown_cq500/{v}/adaptive_unknown_results_cq500.csv", 1,
        f"[{v}] raw / unknown_noisy / adaptive 三条件患者级指标")
add("tta_unknown_cq500/tta_unknown_results_cq500.csv", 1,
    "★ TTA 消融：raw/unknown_noisy/adaptive x none/bn/tent/bn_tent 的患者级指标")
add("tta_unknown_cq500/norm_layers_cq500.csv", 1,
    "两个 backbone 的归一化层构成（解释 bn 对 swin 为何是空操作）")
add("tta_unknown_cq500/tta_adaptation_info_cq500.csv", 2,
    "TTA 自适应过程审计（BN 层数/步数/熵变化）——用于确认每个变体确实执行了，"
    "而非被静默跳过")
add("boundary_correction_cq500/boundary_all_noise_cq500.csv", 1,
    "★ 全部噪声条件下边界校正的增益 + 阈值上限（判断还剩多少没拿到）——"
    "'决策边界位移'主线的核心实测表")
add("detector_comparison_cq500/detector_comparison_cq500.csv", 1,
    "★ 6class vs 7class 判别器正面对比（含复合噪声上的行为）")
add("detector_comparison_cq500/detector_variant_delta_cq500.csv", 1,
    "★ 加入复合噪声类带来的增益 Δadaptive / Δgap")
add("enhanced_known_noise_cq500/best_filter_map_cq500.csv", 1,
    "★ val 上拟合的 类别->最优filter 表（11 用它路由，非 test 挑选）")
add("enhanced_known_noise_cq500/all_enhanced_known_noise_cq500.csv", 1,
    "已知噪声 × 10 filter 的 test 全网格（140 行）")
for n in range(1, 11):
    add(f"final_tables_cq500/table{n}_*.csv", 1, f"论文表 {n}")

# ---------------- tier 2: 支撑表与图表 ----------------
add("noisy_evaluation_cq500/all_noisy_results_cq500.csv", 2,
    "各噪声下未增强的患者级基线（噪声破坏程度）")
add("enhanced_known_noise_cq500/val_grid_cq500.csv", 2,
    "已知噪声 × 10 filter 的 val 全网格（filter 选择的依据）")
add("quality_metrics_cq500/quality_summary_cq500.csv", 2,
    "PSNR/SSIM 及相对加噪图的增益")
add("cq500_train_summary.csv", 2, "分类器训练汇总（含 batch/seed/最优 F1）")
for v in VARIANTS:
    add(f"noise_detector_cq500/{v}/all_condition_behavior.csv", 2,
        f"[{v}] 判别器在全部 7 种条件下的准确率/最常预测/平均置信度")
    add(f"noise_detector_cq500/{v}/per_class_accuracy.csv", 2, f"[{v}] 逐类准确率")
    add(f"noise_detector_cq500/{v}/confusion_matrix.csv", 2, f"[{v}] 验证集混淆矩阵")
    add(f"noise_detector_cq500/{v}/training_log.csv", 2, f"[{v}] 判别器训练曲线")
    add(f"final_unknown_comparison_cq500/{v}/*.csv", 2, f"[{v}] 三条件汇总表")
add("final_tables_cq500/*.tex", 2, "论文 LaTeX 表格")
add("detector_comparison_cq500/*.tex", 2, "判别器对比 LaTeX 表")
for v in VARIANTS:
    add(f"optimality_cq500/{v}/*.tex", 2, f"[{v}] Q1/Q2 LaTeX 表")
add("**/*.png", 2, "全部图（对比柱状图、热力图、混淆矩阵、Grad-CAM 等）")

# ---------------- tier 3: 逐切片明细 ----------------
for v in VARIANTS:
    add(f"adaptive_unknown_cq500/{v}/unknown_noise_detection_log_cq500.csv", 3,
        f"[{v}] 逐切片判别器预测 + 置信度（做预测分布/置信度分析的必要输入）")
    add(f"adaptive_unknown_cq500/{v}/unknown_filter_selection_log_cq500.csv", 3,
        f"[{v}] 逐切片实际选用 filter（复核 adaptive 与 oracle 是否同口径）")
    add(f"optimality_cq500/{v}/detector_confusion_*.csv", 3, f"[{v}] 各条件混淆矩阵")
add("noisy_evaluation_cq500/*/*_patient_metrics.csv", 3, "各噪声逐患者指标")
add("noisy_evaluation_cq500/*/*_slice_preds.csv", 3,
    "各噪声逐切片预测（做配对/置信区间复算用）")
add("quality_metrics_cq500/all_quality_metrics_cq500.csv", 3,
    "全量 PSNR/SSIM（很大，只在要重算质量统计时需要）")

WEIGHT_SPECS = ["resnet50_cq500_best.pth", "swin_tiny_patch4_window7_224_cq500_best.pth",
                "noise_detector_efficientnetb0_cq500_6class_best.pth",
                "noise_detector_efficientnetb0_cq500_7class_best.pth"]

TIER_NAME = {1: "核心结论", 2: "支撑表与图表", 3: "逐切片明细", 4: "模型权重"}


def expand(spec, root):
    """把 spec 展开成实际存在的文件列表；结果一律限定在 DS 数据集内。"""
    if spec.startswith("**/"):
        pat = spec[3:]
        hits = []
        for d in sorted(root.glob("*_" + DS)):
            if d.is_dir():
                hits.extend(d.rglob(pat))
        return sorted(p for p in hits
                      if p.is_file() and is_ds(p.relative_to(root)))
    if any(ch in spec for ch in "*?"):
        return sorted(p for p in root.glob(spec)
                      if p.is_file() and is_ds(p.relative_to(root)))
    p = root / spec
    if p.exists() and p.is_file() and is_ds(p.relative_to(root)):
        return [p]
    return []


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="打包目标目录")
    ap.add_argument("--tier", default="2",
                    help="打包到哪一层：1 / 2 / 3 / all（默认 2）")
    ap.add_argument("--with-weights", action="store_true", help="额外打包 4 个 .pth（约 250 MB）")
    ap.add_argument("--list", action="store_true", help="只列出会打包/缺失什么，不实际拷贝")
    ap.add_argument("--results-root", default=None,
                    help="结果根目录（默认用 cq500_commons.RESULT_ROOT）")
    args = ap.parse_args()

    max_tier = 3 if args.tier == "all" else int(args.tier)
    if max_tier not in (1, 2, 3):
        raise SystemExit("--tier 只能是 1 / 2 / 3 / all")

    root = Path(args.results_root).expanduser().resolve() if args.results_root else RESULT_ROOT
    if not root.is_dir():
        raise SystemExit(f"结果根目录不存在: {root}")
    dest = Path(args.dest).expanduser()
    excluded = sorted(d.name for d in root.iterdir()
                      if d.is_dir() and not is_ds(Path(d.name)))
    print(f"结果根目录: {root}")
    print(f"目标目录  : {dest}")
    print(f"打包层级  : 1..{max_tier}（{' + 权重' if args.with_weights else ''}）")
    print(f"仅打包    : {DS}（已排除其它数据集目录 {len(excluded)} 个"
          f"{': ' + ', '.join(excluded) if excluded else ''}）\n")

    found, missing = [], []
    for spec, tier, note in SPECS:
        if tier > max_tier:
            continue
        hits = expand(spec, root)
        if hits:
            found.extend((p, tier, note) for p in hits)
        else:
            missing.append((spec, tier, note))

    weights = []
    if args.with_weights:
        for w in WEIGHT_SPECS:
            p = MODEL_DIR / w
            if p.exists():
                weights.append(p)
            else:
                missing.append((f"models/{w}", 4, "模型权重"))

    # 去重（**/*.png 与其它通配可能重叠）
    seen, uniq = set(), []
    for p, t, n in found:
        if p in seen:
            continue
        seen.add(p)
        uniq.append((p, t, n))
    found = uniq

    for tier in range(1, max_tier + 1):
        sub = [f for f in found if f[1] == tier]
        if not sub:
            continue
        size = sum(p.stat().st_size for p, _, _ in sub)
        print(f"--- tier {tier} {TIER_NAME[tier]}：{len(sub)} 个文件, {size/1048576:.2f} MB ---")
        for p, _, note in sub[:6]:
            print(f"    {p.relative_to(root).as_posix()}")
        if len(sub) > 6:
            print(f"    ...（另有 {len(sub)-6} 个）")

    tot = sum(p.stat().st_size for p, _, _ in found) + sum(p.stat().st_size for p in weights)
    print(f"\n合计 {len(found) + len(weights)} 个文件, {tot/1048576:.2f} MB")

    if missing:
        print(f"\n!! 缺失 {len(missing)} 项（说明对应步骤没跑完或不产出该文件）:")
        for spec, tier, note in missing:
            print(f"    [tier{tier}] {spec}   —— {note}")
    else:
        print("\n所有列出的产物都在。")

    if args.list:
        return

    dest.mkdir(parents=True, exist_ok=True)
    lines = ["# CQ500 结果包清单", f"# 来源: {root}", f"# 层级: 1..{max_tier}",
             f"# 文件数: {len(found) + len(weights)}", ""]
    lines.append("## 带校验和的文件（小文件）")
    for p, tier, note in sorted(found, key=lambda x: str(x[0])):
        rel = p.relative_to(root)
        dst = dest / "results" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        if p.stat().st_size < 20 * 1024 * 1024:
            lines.append(f"{sha256(p)}  {p.stat().st_size:>12}  {tier}  {rel.as_posix()}  # {note}")
        else:
            lines.append(f"{'-':>16}  {p.stat().st_size:>12}  {tier}  {rel.as_posix()}  # {note}")

    if weights:
        lines += ["", "## 模型权重"]
        for p in weights:
            dst = dest / "models" / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            lines.append(f"{sha256(p)}  {p.stat().st_size:>12}  4  models/{p.name}")

    readme = f"""# CQ500 结果包

从 `{root}` 打包，层级 1..{max_tier}
{'（含模型权重）' if weights else '（不含权重）'}。
只含 {DS} 数据集，physionet / rsna 的结果未纳入。

## 文件 -> 结论 的对应关系

**Q1（增强是否超过加噪图）/ Q2（该 filter 是否最优）**
    results/optimality_cq500/<variant>/oracle_summary_cq500.csv
        关键列：noisy_accuracy / adaptive_accuracy / q1_delta_adaptive_minus_noisy /
                q1_mcnemar_p / oracle_global_filter / oracle_global_accuracy /
                oracle_restricted_accuracy / oracle_per_patient_accuracy /
                q2_gap_to_global / q2_agreement_with_global
    results/optimality_cq500/<variant>/filter_sweep_unknown_mixed_cq500.csv
        10 个固定 filter 的患者级 ACC，oracle 就取自这里
    results/optimality_cq500/<variant>/per_patient_oracle_cq500.csv
        逐患者对照，用来算"选对率"和做配对检验

**两个判别器的对比（6class vs 7class）**
    results/detector_comparison_cq500/detector_comparison_cq500.csv
    results/detector_comparison_cq500/detector_variant_delta_cq500.csv
    results/noise_detector_cq500/<variant>/all_condition_behavior.csv
        重点看 unknown_mixed 行：6class 的 in_vocab=False、top_pred 是某个单噪声、
        mean_confidence 偏低；7class 应高准确率命中 unknown_mixed

**噪声破坏程度（未增强基线）**
    results/noisy_evaluation_cq500/all_noisy_results_cq500.csv

**类别->filter 映射（val 拟合，避免 test 挑选）**
    results/enhanced_known_noise_cq500/best_filter_map_cq500.csv
    results/enhanced_known_noise_cq500/val_grid_cq500.csv
    results/enhanced_known_noise_cq500/all_enhanced_known_noise_cq500.csv

**论文表格**
    results/final_tables_cq500/table*.csv 与同名 .tex

**图像质量**
    results/quality_metrics_cq500/quality_summary_cq500.csv

**逐切片复核（tier 3）**
    results/adaptive_unknown_cq500/<variant>/unknown_*_log_cq500.csv
    results/noisy_evaluation_cq500/*/*_slice_preds.csv

## 自检

在打包机上跑过 `python 17_verify_cq500_pipeline.py` 且 FAIL=0 时，
包内数据才与代码口径一致。若结果是在别处生成的，请把那边的
17 号自检输出一并附上。
"""
    (dest / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")
    (dest / "RESULTS_README.md").write_text(readme, encoding="utf-8")
    print(f"\n[打包完成] {dest}")
    print(f"  results/...        结果文件")
    if weights:
        print(f"  models/...         权重")
    print(f"  MANIFEST.txt       清单（小文件含 sha256 前 16 位）")
    print(f"  RESULTS_README.md  文件->结论 对应说明")
    print(f"\n直接回传整个 {dest} 即可。")


if __name__ == "__main__":
    main()
