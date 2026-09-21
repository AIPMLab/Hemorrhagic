# -*- coding: utf-8 -*-
""" 
CQ500 论文表格汇总

表1  患者级三条件对比（raw / unknown_noisy / adaptive）
表2  已知噪声增强全网格（test）
表3  每个噪声的最优 filter（**val 拟合**，附 test 表现）
表4  PSNR/SSIM 质量指标（含相对加噪图的增益）
表5  自适应结果明细（两个判别器变体）
表6  **Q1 恢复性 + Q2 最优性汇总**（逐变体）
表7  复合未知噪声下各固定 filter 的患者级指标（oracle 的依据）
表8  判别器在各噪声条件下的行为（逐变体）
表9  判别器变体对比（6class vs 7class）
表10 7class 相对 6class 的增益

输出: results/final_tables_cq500/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from cq500_commons import DETECTOR_VARIANTS, RESULT_ROOT

OUT = RESULT_ROOT / "final_tables_cq500"
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = sorted(DETECTOR_VARIANTS)


def load(p):
    p = Path(p)
    if not p.exists():
        print(f"[skip] 缺少 {p}")
        return None
    return pd.read_csv(p)


def load_variants(*parts):
    """把各变体子目录下的同名产物纵向合并（带 variant 列）。

    load_variants("optimality_cq500", "oracle_summary_cq500.csv")
      -> RESULT_ROOT/optimality_cq500/{variant}/oracle_summary_cq500.csv
    """
    rel = Path(*parts)
    base = RESULT_ROOT / rel.parent
    frames = []
    for v in VARIANTS:
        d = load(base / v / rel.name)
        if d is None:
            continue
        # 有些产物（如 16 的 oracle_summary）自己就带 variant 列，避免重复插入
        if "variant" in d.columns:
            d["variant"] = d["variant"].fillna(v)
        else:
            d.insert(0, "variant", v)
        frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def save(df, name_csv, name_tex, caption, label):
    if df is None:
        return
    df.to_csv(OUT / name_csv, index=False)
    with open(OUT / name_tex, "w", encoding="utf-8") as fh:
        fh.write(df.to_latex(index=False, float_format="%.4f", escape=False,
                             caption=caption, label=label))
    print(f"[ok] {name_csv}  shape={df.shape}")


def main():
    R = RESULT_ROOT

    # ---- 表1: 三条件对比（逐变体）----
    comp = load_variants("final_unknown_comparison_cq500",
                         "final_unknown_comparison_summary_cq500.csv")
    if comp is not None:
        comp = comp[["variant", "model", "condition", "num_patients", "accuracy",
                     "precision", "recall", "f1", "auc"]]
    save(comp, "table1_patient_condition_comparison.csv", "latex_table1.tex",
         "Patient-level comparison of raw, noisy and adaptively enhanced CQ500 test sets.",
         "tab:cq500_conditions")

    # ---- 表2: 已知噪声增强全网格（test）----
    enh = load(R / "enhanced_known_noise_cq500" / "all_enhanced_known_noise_cq500.csv")
    if enh is not None:
        enh = enh[["model", "noise_type", "enhancement_method", "num_patients",
                   "accuracy", "precision", "recall", "f1", "auc"]]
    save(enh, "table2_enhanced_known_noise.csv", "latex_table2.tex",
         "Patient-level performance of every enhancement filter under each noise condition.",
         "tab:cq500_enhanced")

    # ---- 表3: 每噪声最优 filter（val 拟合）----
    best = load(R / "enhanced_known_noise_cq500" / "best_filter_map_cq500.csv")
    if best is not None:
        best = best[["model", "noise_type", "enhancement_method", "fitted_on",
                     "val_accuracy", "test_accuracy", "test_f1", "test_auc"]]
    save(best, "table3_best_filter_per_noise.csv", "latex_table3.tex",
         "Optimal enhancement filter per noise condition, selected on the validation split "
         "and evaluated on the test split.",
         "tab:cq500_best_filter")

    # ---- 表4: 质量指标 ----
    q = load(R / "quality_metrics_cq500" / "quality_summary_cq500.csv")
    save(q, "table4_psnr_ssim.csv", "latex_table4.tex",
         "PSNR/SSIM of each enhancement filter against the original clean slices, "
         "with the gain relative to the noisy input.",
         "tab:cq500_quality")

    # ---- 表5: 自适应明细（逐变体）----
    ad = load_variants("adaptive_unknown_cq500", "adaptive_unknown_results_cq500.csv")
    if ad is not None:
        ad = ad[["variant", "model", "condition", "num_patients", "accuracy",
                 "precision", "recall", "f1", "auc"]]
    save(ad, "table5_adaptive_unknown.csv", "latex_table5.tex",
         "Adaptive enhancement results on compound unknown noise, per detector variant.",
         "tab:cq500_adaptive")

    # ---- 表6: Q1 + Q2 核心汇总（逐变体）----
    oracle = load_variants("optimality_cq500", "oracle_summary_cq500.csv")
    save(oracle, "table6_q1_q2_summary.csv", "latex_table6_q1_q2.tex",
         "Q1: does adaptive enhancement beat the noisy input? "
         "Q2: is the detector-selected filter optimal? "
         "Compared against global, restricted and per-patient oracles, per detector variant.",
         "tab:cq500_q1q2")

    # ---- 表7: 复合噪声下的 filter 穷举 ----
    sweep = load_variants("optimality_cq500", "filter_sweep_unknown_mixed_cq500.csv")
    save(sweep, "table7_filter_sweep_unknown.csv", "latex_table7_sweep.tex",
         "Patient-level performance of each fixed filter under compound unknown noise.",
         "tab:cq500_sweep")

    # ---- 表8: 判别器行为（逐变体）----
    det = load_variants("optimality_cq500", "detector_accuracy_cq500.csv")
    save(det, "table8_detector_behavior.csv", "latex_table8_detector.tex",
         "Behaviour of each detector variant on every noise condition; for the 6-class "
         "detector compound noise is out-of-vocabulary (accuracy undefined, predictions shown).",
         "tab:cq500_detector")

    # ---- 表9/表10: 判别器变体对比 ----
    comp = load(R / "detector_comparison_cq500" / "detector_comparison_cq500.csv")
    save(comp, "table9_detector_comparison.csv", "latex_table9_detector_comparison.tex",
         "Side-by-side comparison of the 6-class and 7-class noise detectors on compound "
         "unknown noise (patient-level, CQ500).",
         "tab:cq500_detector_comparison")

    delta = load(R / "detector_comparison_cq500" / "detector_variant_delta_cq500.csv")
    save(delta, "table10_detector_variant_delta.csv", "latex_table10_detector_delta.tex",
         "Gain obtained by adding the compound-noise class to the detector.",
         "tab:cq500_detector_delta")

    print(f"\n[done] 全部表格输出: {OUT}")


if __name__ == "__main__":
    main()
