# -*- coding: utf-8 -*-
"""
PhysioNet 论文表格汇总（与 CQ500 的 15 同构）

表1  患者级三条件对比（raw / unknown_noisy / adaptive，逐变体）
表2  已知噪声增强全网格（test）
表3  每个噪声的最优 filter（**val 拟合**，附 test 表现）
表4  PSNR/SSIM 质量指标（含相对加噪图的增益）
表5  自适应结果明细（两个判别器变体）
表6  Q1 恢复性 + Q2 最优性汇总（逐变体）
表7  复合未知噪声下各固定 filter 的患者级指标
表8  判别器在各噪声条件下的行为（逐变体）
表9  判别器变体对比
表10 7class 相对 6class 的增益

输出: results/final_tables_physionet/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from physionet_commons import DETECTOR_VARIANTS, RESULT_ROOT

OUT = RESULT_ROOT / "final_tables_physionet"
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = sorted(DETECTOR_VARIANTS)


def load(p):
    p = Path(p)
    if not p.exists():
        print(f"[skip] 缺少 {p}")
        return None
    return pd.read_csv(p)


def load_variants(*parts):
    """把各变体子目录下的同名产物纵向合并（带 variant 列）。"""
    rel = Path(*parts)
    base = RESULT_ROOT / rel.parent
    frames = []
    for v in VARIANTS:
        d = load(base / v / rel.name)
        if d is None:
            continue
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

    comp = load_variants("final_unknown_comparison_physionet",
                         "final_unknown_comparison_summary_physionet.csv")
    if comp is not None:
        comp = comp[["variant", "model", "condition", "num_patients", "accuracy",
                     "precision", "recall", "f1", "auc"]]
    save(comp, "table1_patient_condition_comparison.csv", "latex_table1.tex",
         "PhysioNet ICH subset: patient-level comparison of raw, noisy and adaptively "
         "enhanced test sets.", "tab:physionet_conditions")

    enh = load(R / "enhanced_known_noise_physionet" / "all_enhanced_known_noise_physionet.csv")
    if enh is not None:
        enh = enh[["model", "noise_type", "enhancement_method", "num_patients",
                   "accuracy", "precision", "recall", "f1", "auc"]]
    save(enh, "table2_enhanced_known_noise.csv", "latex_table2.tex",
         "PhysioNet ICH subset: patient-level performance of every enhancement filter "
         "under each noise condition.", "tab:physionet_enhanced")

    best = load(R / "enhanced_known_noise_physionet" / "best_filter_map_physionet.csv")
    if best is not None:
        best = best[["model", "noise_type", "enhancement_method", "fitted_on",
                     "val_accuracy", "test_accuracy", "test_f1", "test_auc"]]
    save(best, "table3_best_filter_per_noise.csv", "latex_table3.tex",
         "PhysioNet ICH subset: optimal enhancement filter per noise condition, selected on the "
         "validation split and evaluated on the test split.", "tab:physionet_best_filter")

    q = load(R / "quality_metrics_physionet" / "quality_summary_physionet.csv")
    save(q, "table4_psnr_ssim.csv", "latex_table4.tex",
         "PhysioNet ICH subset: PSNR/SSIM of each filter against the clean slices, with the "
         "gain relative to the noisy input.", "tab:physionet_quality")

    ad = load_variants("adaptive_unknown_physionet", "adaptive_unknown_results_physionet.csv")
    if ad is not None:
        ad = ad[["variant", "model", "condition", "num_patients", "accuracy",
                 "precision", "recall", "f1", "auc"]]
    save(ad, "table5_adaptive_unknown.csv", "latex_table5.tex",
         "PhysioNet ICH subset: adaptive enhancement results on compound unknown noise, "
         "per detector variant.", "tab:physionet_adaptive")

    oracle = load_variants("optimality_physionet", "oracle_summary_physionet.csv")
    save(oracle, "table6_q1_q2_summary.csv", "latex_table6_q1_q2.tex",
         "PhysioNet ICH subset: Q1 recovery and Q2 optimality of the adaptive pipeline on "
         "compound unknown noise, against global/restricted/per-patient oracles.",
         "tab:physionet_q1q2")

    sweep = load_variants("optimality_physionet", "filter_sweep_unknown_mixed_physionet.csv")
    save(sweep, "table7_filter_sweep_unknown.csv", "latex_table7_sweep.tex",
         "PhysioNet ICH subset: patient-level performance of each fixed filter under compound "
         "unknown noise.", "tab:physionet_sweep")

    det = load_variants("optimality_physionet", "detector_accuracy_physionet.csv")
    save(det, "table8_detector_behavior.csv", "latex_table8_detector.tex",
         "PhysioNet ICH subset: behaviour of each detector variant on every noise condition; "
         "compound noise is out-of-vocabulary for the 6-class detector.",
         "tab:physionet_detector")

    c = load(R / "detector_comparison_physionet" / "detector_comparison_physionet.csv")
    save(c, "table9_detector_comparison.csv", "latex_table9_detector_comparison.tex",
         "PhysioNet ICH subset: side-by-side comparison of the 6-class and 7-class noise "
         "detectors on compound unknown noise.", "tab:physionet_detector_comparison")

    d = load(R / "detector_comparison_physionet" / "detector_variant_delta_physionet.csv")
    save(d, "table10_detector_variant_delta.csv", "latex_table10_detector_delta.tex",
         "PhysioNet ICH subset: gain obtained by adding the compound-noise class to the detector.",
         "tab:physionet_detector_delta")

    print(f"\n[done] 全部表格输出: {OUT}")


if __name__ == "__main__":
    main()
