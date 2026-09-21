# -*- coding: utf-8 -*-
"""
PhysioNet 两个噪声判别器变体对比（与 CQ500 的 18 同构）

对比对象：
  6class —— 只在 6 类单一退化上训练。复合噪声对其完全分布外，只能硬分到某个已知
            单噪声类，机制上无法选中"复合噪声的最优 filter"。
  7class —— 额外把复合未知噪声作为第 7 类显式训练，可直接识别并路由。

回答：加入复合噪声类后，自适应增强的患者级指标提升多少？与 oracle 的差距缩小多少？
      封闭集（6class）在复合噪声上到底预测成了什么、置信度多高？

输入: results/optimality_physionet/{variant}/oracle_summary_physionet.csv
      results/noise_detector_physionet/{variant}/all_condition_behavior.csv
输出: results/detector_comparison_physionet/

用法: python 18_compare_detectors_physionet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physionet_commons import DETECTOR_VARIANTS, RESULT_ROOT

OUT = RESULT_ROOT / "detector_comparison_physionet"
VARIANTS = sorted(DETECTOR_VARIANTS)


def load(p):
    return pd.read_csv(p) if Path(p).exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    oracle, behavior = {}, {}
    for v in VARIANTS:
        o = load(RESULT_ROOT / "optimality_physionet" / v / "oracle_summary_physionet.csv")
        if o is not None:
            oracle[v] = o
        b = load(RESULT_ROOT / "noise_detector_physionet" / v / "all_condition_behavior.csv")
        if b is not None:
            behavior[v] = b

    if not oracle:
        raise SystemExit("没有找到任何变体的 oracle_summary_physionet.csv；"
                         "请先运行 11 --variant <v> 与 16 --variant <v>")

    rows = []
    for v, df in oracle.items():
        b = behavior.get(v)
        for _, r in df.iterrows():
            rec = {
                "variant": v, "model": r["model"],
                "adaptive_accuracy": r["adaptive_accuracy"],
                "noisy_accuracy": r["noisy_accuracy"],
                "q1_delta_vs_noisy": r["q1_delta_adaptive_minus_noisy"],
                "q1_mcnemar_p": r["q1_mcnemar_p"],
                "oracle_global_accuracy": r["oracle_global_accuracy"],
                "oracle_per_patient_accuracy": r["oracle_per_patient_accuracy"],
                "q2_gap_to_global": r["q2_gap_to_global"],
                "q2_agreement_with_global": r["q2_agreement_with_global"],
                "filters_used": r["filters_used"],
            }
            if b is not None:
                hm = b[b["condition"] == "unknown_mixed"]
                if len(hm):
                    rec["detector_in_vocab_for_compound"] = bool(hm["in_vocab"].iloc[0])
                    rec["detector_top_pred_on_compound"] = hm["top_pred"].iloc[0]
                    rec["detector_top_share_on_compound"] = hm["top_pred_share"].iloc[0]
                    rec["detector_confidence_on_compound"] = hm["mean_confidence"].iloc[0]
            rows.append(rec)
    comp = pd.DataFrame(rows).sort_values(["model", "variant"]).reset_index(drop=True)
    comp.to_csv(OUT / "detector_comparison_physionet.csv", index=False)

    deltas = []
    if len(VARIANTS) == 2 and all(v in oracle for v in VARIANTS):
        a, bb = VARIANTS
        for model in sorted(set(comp["model"])):
            ra = comp[(comp["variant"] == a) & (comp["model"] == model)]
            rb = comp[(comp["variant"] == bb) & (comp["model"] == model)]
            if ra.empty or rb.empty:
                continue
            ra, rb = ra.iloc[0], rb.iloc[0]
            deltas.append({
                "model": model,
                "adaptive_6class": ra["adaptive_accuracy"],
                "adaptive_7class": rb["adaptive_accuracy"],
                "delta_adaptive_7_minus_6": rb["adaptive_accuracy"] - ra["adaptive_accuracy"],
                "gap_to_global_6class": ra["q2_gap_to_global"],
                "gap_to_global_7class": rb["q2_gap_to_global"],
                "agreement_6class": ra["q2_agreement_with_global"],
                "agreement_7class": rb["q2_agreement_with_global"],
                "filters_used_6class": ra["filters_used"],
                "filters_used_7class": rb["filters_used"],
            })
    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df.to_csv(OUT / "detector_variant_delta_physionet.csv", index=False)

    models = sorted(comp["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 4.8),
                             squeeze=False, constrained_layout=True)
    for ax, model in zip(axes[0], models):
        sub = comp[comp["model"] == model].set_index("variant")
        x = np.arange(len(VARIANTS))
        w = 0.2
        ax.bar(x - 1.5 * w, sub.loc[VARIANTS, "noisy_accuracy"], w,
               label="Noisy (no filter)", color="tab:red")
        ax.bar(x - 0.5 * w, sub.loc[VARIANTS, "adaptive_accuracy"], w,
               label="Adaptive", color="tab:blue")
        ax.bar(x + 0.5 * w, sub.loc[VARIANTS, "oracle_global_accuracy"], w,
               label="Oracle (best filter)", color="tab:green")
        ax.bar(x + 1.5 * w, sub.loc[VARIANTS, "oracle_per_patient_accuracy"], w,
               label="Oracle (per-patient)", color="tab:olive")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n(compound class: "
                            f"{'yes' if v == '7class' else 'no'})" for v in VARIANTS],
                           fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Patient-level accuracy")
        ax.set_title(model, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Detector variant comparison on compound unknown noise (PhysioNet ICH subset)",
                 fontweight="bold")
    fig.savefig(OUT / "fig_detector_comparison_physionet.png", dpi=200, bbox_inches="tight")
    plt.close()

    with open(OUT / "latex_detector_comparison_physionet.tex", "w", encoding="utf-8") as fh:
        fh.write(comp.to_latex(index=False, float_format="%.4f", escape=False,
                               caption="PhysioNet ICH subset: comparison of the 6-class and "
                                       "7-class noise detectors on compound unknown noise.",
                               label="tab:physionet_detector_variants"))

    print("[对比] 逐 (model, variant):")
    print(comp[["model", "variant", "noisy_accuracy", "adaptive_accuracy",
                "q2_gap_to_global", "q2_agreement_with_global"]].to_string(index=False))
    if not delta_df.empty:
        print("\n[增益] 7class 相对 6class:")
        print(delta_df[["model", "adaptive_6class", "adaptive_7class",
                        "delta_adaptive_7_minus_6"]].to_string(index=False))
    for v, b in behavior.items():
        hm = b[b["condition"] == "unknown_mixed"]
        if len(hm):
            r = hm.iloc[0]
            print(f"\n[判别器 {v} 在复合噪声上] 在词表内={r['in_vocab']} "
                  f"最常预测={r['top_pred']} ({r['top_pred_share']:.1%}) "
                  f"平均置信度={r['mean_confidence']:.3f}")
    print(f"\n[done] 输出目录: {OUT}")


if __name__ == "__main__":
    main()
