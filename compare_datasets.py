# -*- coding: utf-8 -*-
"""
CQ500 与 RSNA 的**并排对比**汇总（论文用）

两套数据集各自跑完 16（最优性）后，本脚本把它们的结果并排成一张表 + 一张图，
直接回答"同一个自适应框架在两个队列上表现如何"。

因为要同时读两边结果，本脚本放在 code 根目录（不属于任何单一数据集）。
缺哪一边就只报哪一边，不会报错 —— 单跑一个数据集时也能用。

输入:
    results/optimality_cq500/{variant}/oracle_summary_cq500.csv
    results/optimality_rsna/{variant}/oracle_summary_rsna.csv
    results/noise_detector_cq500/{variant}/all_condition_behavior.csv
    results/noise_detector_rsna/{variant}/all_condition_behavior.csv
    Cq500_dataset/cq500_slices_index.csv、Rsna_dataset/rsna_slices_index.csv（队列规模）

输出: results/cross_dataset_comparison/
    cross_dataset_side_by_side.csv / .tex
    fig_cross_dataset.png

用法:
    python compare_datasets.py
"""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ich_common import DETECTOR_VARIANTS

RESULTS = CODE / "results"
CV_PHYSIONET = CODE / "results_cv_physionet"
OUT = RESULTS / "cross_dataset_comparison"

SPECS = [
    ("CQ500", "optimality_cq500", "oracle_summary_cq500.csv",
     "noise_detector_cq500", CODE / "Cq500_dataset" / "cq500_slices_index.csv"),
    ("RSNA", "optimality_rsna", "oracle_summary_rsna.csv",
     "noise_detector_rsna", CODE / "Rsna_dataset" / "rsna_slices_index.csv"),
    ("PhysioNet", "optimality_physionet", "oracle_summary_physionet.csv",
     "noise_detector_physionet", CODE / "PhysioNet" / "physionet_slices_index.csv"),
]
VARIANTS = sorted(DETECTOR_VARIANTS)

# 逐列搬运：源列可能不存在（CV 合并结果就没有 q1_mcnemar_p / filters_used），给 NaN。
ORACLE_COLS = [
    ("noisy_accuracy", "noisy_accuracy"),
    ("adaptive_accuracy", "adaptive_accuracy"),
    ("q1_delta_adaptive_minus_noisy", "q1_delta_vs_noisy"),
    ("q1_mcnemar_p", "q1_mcnemar_p"),
    ("oracle_global_accuracy", "oracle_global_accuracy"),
    ("oracle_per_patient_accuracy", "oracle_per_patient_accuracy"),
    ("q2_gap_to_global", "q2_gap_to_global"),
    ("q2_agreement_with_global", "q2_agreement_with_global"),
    ("filters_used", "filters_used"),
]


def load(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def cv_source(name):
    """PhysioNet 的正式口径是 5 折患者级交叉验证，不是 results/ 下的单次划分。

    若 results_cv_physionet/cv_pooled.csv 存在就用它；否则退回单次划分结果并在
    eval_mode 列标明 —— 避免把 14 例 test 的 holdout 数字当成 CV 数字写进论文。
    """
    if name != "PhysioNet":
        return None
    pooled = load(CV_PHYSIONET / "cv_pooled.csv")
    return None if pooled is None or pooled.empty else pooled


def cohort_stats(index_csv):
    if not Path(index_csv).exists():
        return {}
    # patient_id 显式读成字符串：PhysioNet 用零填充（"049"），pandas 会当成整数 49
    d = pd.read_csv(index_csv, dtype={"patient_id": str})
    pat = d.groupby("patient_id").size()
    pz = d.groupby("patient_id")["label"].first()
    return {"patients": int(d["patient_id"].nunique()),
            "slices": int(len(d)),
            "slices_per_patient_mean": round(float(pat.mean()), 1),
            "patient_pos_rate": round(float((pz == "Hemorrhagic").mean()), 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cv_det = load(CV_PHYSIONET / "cv_detector_behavior.csv")
    for name, res_dir, fname, det_dir, index_csv in SPECS:
        stats = cohort_stats(index_csv)
        cv = cv_source(name)
        for v in VARIANTS:
            if cv is not None:
                o = cv[cv["variant"] == v]
                eval_mode = "5-fold CV"
            else:
                o = load(RESULTS / res_dir / v / fname)
                eval_mode = "holdout"
            if o is None or len(o) == 0:
                continue
            b = load(RESULTS / det_dir / v / "all_condition_behavior.csv")
            for _, r in o.iterrows():
                rec = {"dataset": name, "variant": v, "model": r["model"],
                       "eval_mode": eval_mode, **stats}
                for src, dst in ORACLE_COLS:
                    rec[dst] = r[src] if src in o.columns else np.nan
                if cv is not None and cv_det is not None:
                    d = cv_det[cv_det["variant"] == v]
                    if len(d):
                        dr = d.iloc[0]
                        rec["detector_in_vocab_for_compound"] = bool(dr["in_vocab"])
                        rec["detector_top_pred_on_compound"] = dr["top_pred_mode"]
                        rec["detector_conf_on_compound"] = dr["confidence_mean"]
                elif b is not None:
                    hm = b[b["condition"] == "unknown_mixed"]
                    if len(hm):
                        rec["detector_in_vocab_for_compound"] = bool(hm["in_vocab"].iloc[0])
                        rec["detector_top_pred_on_compound"] = hm["top_pred"].iloc[0]
                        rec["detector_conf_on_compound"] = hm["mean_confidence"].iloc[0]
                rows.append(rec)

    if not rows:
        raise SystemExit(
            "两边都没有结果可对比。请先各自跑完 11/16：\n"
            "  Cq500_dataset: python run_cq500_pipeline.py\n"
            "  Rsna_dataset : python run_rsna_pipeline.py\n"
            "  PhysioNet    : python physionet_run_cv.py && python physionet_aggregate_cv.py")

    comp = pd.DataFrame(rows).sort_values(["dataset", "model", "variant"]).reset_index(drop=True)
    comp.to_csv(OUT / "cross_dataset_side_by_side.csv", index=False)
    with open(OUT / "cross_dataset_side_by_side.tex", "w", encoding="utf-8") as fh:
        fh.write(comp.to_latex(index=False, float_format="%.4f", escape=False,
                               caption="Side-by-side comparison of the adaptive enhancement "
                                       "framework on CQ500, the RSNA ICH subset and PhysioNet "
                                       "CT-ICH (patient-level, compound unknown noise). "
                                       "The eval\\_mode column distinguishes the single "
                                       "train/val/test split from 5-fold patient-level "
                                       "cross-validation (mean over folds).",
                               label="tab:cross_dataset"))

    datasets = sorted(comp["dataset"].unique())
    models = sorted(comp["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.0 * len(models), 5.0),
                             squeeze=False, constrained_layout=True)
    for ax, model in zip(axes[0], models):
        sub = comp[comp["model"] == model]
        labels, x = [], []
        i = 0
        colors = {"Noisy (no filter)": "tab:red", "Adaptive": "tab:blue",
                  "Oracle (best fixed filter)": "tab:green",
                  "Oracle (per-patient)": "tab:olive"}
        for ds in datasets:
            for v in VARIANTS:
                row = sub[(sub["dataset"] == ds) & (sub["variant"] == v)]
                if row.empty:
                    continue
                r = row.iloc[0]
                for k, (lab, val) in enumerate([
                        ("Noisy (no filter)", r["noisy_accuracy"]),
                        ("Adaptive", r["adaptive_accuracy"]),
                        ("Oracle (best fixed filter)", r["oracle_global_accuracy"]),
                        ("Oracle (per-patient)", r["oracle_per_patient_accuracy"])]):
                    ax.bar(i + (k - 1.5) * 0.2, val, 0.2, color=colors[lab],
                           label=lab if i == 0 else None)
                labels.append(f"{ds}\n{v}")
                x.append(i)
                i += 1
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Patient-level accuracy")
        ax.set_title(model, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Adaptive enhancement on compound unknown noise — "
                 "CQ500 vs RSNA vs PhysioNet (patient-level)", fontweight="bold")
    fig.savefig(OUT / "fig_cross_dataset.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("[并排对比]")
    show = ["dataset", "variant", "model", "eval_mode", "patients",
            "slices_per_patient_mean", "noisy_accuracy", "adaptive_accuracy",
            "q1_delta_vs_noisy", "oracle_global_accuracy", "q2_gap_to_global",
            "q2_agreement_with_global"]
    print(comp[[c for c in show if c in comp.columns]].to_string(index=False))
    small = (comp[(comp["patients"] < 30) & (comp["eval_mode"] == "holdout")]["dataset"]
             .unique().tolist() if {"patients", "eval_mode"} <= set(comp.columns) else [])
    if small:
        print(f"\n[注意] {small} 的患者数很少（单次划分模式下 test 集只有十几例）："
              f"患者级准确率步长很大、配对检验功效很低。\n"
              f"       建议改用患者级交叉验证，并在论文中说明。")
    if "detector_conf_on_compound" in comp.columns:
        print("\n[判别器在复合噪声上的行为]")
        print(comp[["dataset", "variant", "detector_in_vocab_for_compound",
                    "detector_top_pred_on_compound",
                    "detector_conf_on_compound"]].drop_duplicates().to_string(index=False))
    print(f"\n[done] 输出目录: {OUT}")


if __name__ == "__main__":
    main()
