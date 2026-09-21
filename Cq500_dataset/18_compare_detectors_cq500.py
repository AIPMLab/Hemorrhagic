# -*- coding: utf-8 -*-
"""
CQ500 两个噪声判别器变体对比（论文核心对比之一）

对比对象：
  6class —— 只在 6 类单一退化（clean + 5 种单噪声）上训练。复合噪声对它完全分布外，
            它必须把复合噪声硬分到某个已知单噪声类，因而只能路由到"某个单噪声的最优 filter"，
            机制上无法选中"复合噪声的最优 filter"。
  7class —— 额外把复合未知噪声 unknown_mixed 作为第 7 类显式训练，可直接识别复合噪声
            并路由到复合噪声在 val 上拟合出的最优 filter。

本脚本把两者的结果并列，回答：
  * 加入复合噪声类之后，自适应增强的患者级指标提升多少？（Δ adaptive）
  * 与 oracle（test 上穷举最优）的差距缩小了多少？（Δ gap to oracle）
  * 选择一致性提升了多少？
  * 封闭集（6class）在复合噪声上到底预测成了什么、置信度多高？

输入: results/optimality_cq500/{variant}/oracle_summary_cq500.csv
      results/optimality_cq500/{variant}/detector_accuracy_cq500.csv
      results/noise_detector_cq500/{variant}/all_condition_behavior.csv
输出: results/detector_comparison_cq500/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cq500_commons import DETECTOR_VARIANTS, RESULT_ROOT

OUT = RESULT_ROOT / "detector_comparison_cq500"
VARIANTS = sorted(DETECTOR_VARIANTS)


def load(p):
    return pd.read_csv(p) if Path(p).exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    oracle, det_behavior = {}, {}
    for v in VARIANTS:
        o = load(RESULT_ROOT / "optimality_cq500" / v / "oracle_summary_cq500.csv")
        if o is not None:
            oracle[v] = o
        b = load(RESULT_ROOT / "noise_detector_cq500" / v / "all_condition_behavior.csv")
        if b is not None:
            det_behavior[v] = b

    if not oracle:
        raise SystemExit(
            "没有找到任何变体的 oracle_summary_cq500.csv；"
            "请先运行 11 --variant <v> 与 16 --variant <v>")

    # ---- 逐 (model, variant) 汇总 ----
    rows = []
    for v, df in oracle.items():
        b = det_behavior.get(v)
        for _, r in df.iterrows():
            rec = {
                "variant": v,
                "model": r["model"],
                "adaptive_accuracy": r["adaptive_accuracy"],
                "noisy_accuracy": r["noisy_accuracy"],
                "q1_delta_vs_noisy": r["q1_delta_adaptive_minus_noisy"],
                "q1_mcnemar_p": r["q1_mcnemar_p"],
                "oracle_global_accuracy": r["oracle_global_accuracy"],
                "oracle_per_patient_accuracy": r["oracle_per_patient_accuracy"],
                "q2_gap_to_global": r["q2_gap_to_global"],
                "q2_gap_to_per_patient": r["q2_gap_to_per_patient"],
                "q2_agreement_with_global": r["q2_agreement_with_global"],
                "filters_used": r["filters_used"],
            }
            if b is not None:
                hm = b[(b["condition"] == "unknown_mixed")]
                if len(hm):
                    rec["detector_in_vocab_for_compound"] = bool(hm["in_vocab"].iloc[0])
                    rec["detector_top_pred_on_compound"] = hm["top_pred"].iloc[0]
                    rec["detector_top_share_on_compound"] = hm["top_pred_share"].iloc[0]
                    rec["detector_confidence_on_compound"] = hm["mean_confidence"].iloc[0]
            rows.append(rec)
    comp = pd.DataFrame(rows).sort_values(["model", "variant"]).reset_index(drop=True)
    comp.to_csv(OUT / "detector_comparison_cq500.csv", index=False)

    # ---- 关键差值：7class 相对 6class 的增益 ----
    deltas = []
    if len(VARIANTS) == 2 and all(v in oracle for v in VARIANTS):
        a, b = VARIANTS   # 6class, 7class
        for model in sorted(set(comp["model"])):
            ra = comp[(comp["variant"] == a) & (comp["model"] == model)]
            rb = comp[(comp["variant"] == b) & (comp["model"] == model)]
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
                "delta_gap_7_minus_6": rb["q2_gap_to_global"] - ra["q2_gap_to_global"],
                "agreement_6class": ra["q2_agreement_with_global"],
                "agreement_7class": rb["q2_agreement_with_global"],
                "filters_used_6class": ra["filters_used"],
                "filters_used_7class": rb["filters_used"],
            })
    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df.to_csv(OUT / "detector_variant_delta_cq500.csv", index=False)

    # ---- 图 ----
    models = sorted(comp["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 4.8),
                             squeeze=False, constrained_layout=True)
    for ax, model in zip(axes[0], models):
        sub = comp[comp["model"] == model]
        x = np.arange(len(VARIANTS))
        w = 0.2
        ax.bar(x - 1.5 * w, sub.set_index("variant").loc[VARIANTS, "noisy_accuracy"],
               w, label="Noisy (no filter)", color="tab:red")
        ax.bar(x - 0.5 * w, sub.set_index("variant").loc[VARIANTS, "adaptive_accuracy"],
               w, label="Adaptive", color="tab:blue")
        ax.bar(x + 0.5 * w, sub.set_index("variant").loc[VARIANTS, "oracle_global_accuracy"],
               w, label="Oracle (best filter)", color="tab:green")
        ax.bar(x + 1.5 * w, sub.set_index("variant").loc[VARIANTS, "oracle_per_patient_accuracy"],
               w, label="Oracle (per-patient)", color="tab:olive")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n(compound class: {'yes' if v == '7class' else 'no'})"
                            for v in VARIANTS], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Patient-level accuracy")
        ax.set_title(f"{model}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Detector variant comparison on compound unknown noise (CQ500)",
                 fontweight="bold")
    fig.savefig(OUT / "fig_detector_comparison_cq500.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- LaTeX ----
    with open(OUT / "latex_detector_comparison.tex", "w", encoding="utf-8") as fh:
        fh.write(comp.to_latex(index=False, float_format="%.4f", escape=False,
                               caption="Comparison of the 6-class (single-noise only) and "
                                       "7-class (with compound noise) noise detectors on "
                                       "compound unknown noise, patient-level, CQ500.",
                               label="tab:cq500_detector_variants"))
    if not delta_df.empty:
        with open(OUT / "latex_detector_variant_delta.tex", "w", encoding="utf-8") as fh:
            fh.write(delta_df.to_latex(index=False, float_format="%.4f", escape=False,
                                       caption="Gain from adding the compound-noise class to "
                                               "the detector.",
                                       label="tab:cq500_detector_delta"))

    print("[对比] 逐 (model, variant):")
    cols = ["model", "variant", "noisy_accuracy", "adaptive_accuracy",
            "q2_gap_to_global", "q2_agreement_with_global"]
    print(comp[cols].to_string(index=False))
    if not delta_df.empty:
        print("\n[增益] 7class 相对 6class:")
        print(delta_df[["model", "adaptive_6class", "adaptive_7class",
                        "delta_adaptive_7_minus_6", "gap_to_global_6class",
                        "gap_to_global_7class"]].to_string(index=False))
    # 判别器在复合噪声上的行为对比
    b6 = det_behavior.get("6class")
    b7 = det_behavior.get("7class")
    for v, b in (("6class", b6), ("7class", b7)):
        if b is None:
            continue
        hm = b[b["condition"] == "unknown_mixed"]
        if len(hm):
            r = hm.iloc[0]
            print(f"\n[判别器 {v} 在复合噪声上] "
                  f"在词表内={r['in_vocab']} 最常预测={r['top_pred']} "
                  f"({r['top_pred_share']:.1%}) 平均置信度={r['mean_confidence']:.3f}")
    print(f"\n[done] 输出目录: {OUT}")


if __name__ == "__main__":
    main()
