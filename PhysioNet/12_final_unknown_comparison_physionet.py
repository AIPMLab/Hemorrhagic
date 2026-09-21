# -*- coding: utf-8 -*-
"""
PhysioNet 最终对比：Raw / Unknown Noisy / Adaptive Enhanced 患者级指标汇总表与图
（数据来自 11 --variant X 的输出）

输出: results/final_unknown_comparison_physionet/{variant}/

用法:
    python 12_final_unknown_comparison_physionet.py --variant 7class
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from physionet_commons import RESULT_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="7class")
args = ap.parse_args()

in_file = (RESULT_ROOT / "adaptive_unknown_physionet" / args.variant
           / "adaptive_unknown_results_physionet.csv")
out = RESULT_ROOT / "final_unknown_comparison_physionet" / args.variant
out.mkdir(parents=True, exist_ok=True)
if not in_file.exists():
    raise FileNotFoundError(f"缺少 {in_file}，请先运行 11 --variant {args.variant}")

df = pd.read_csv(in_file)
summary = df[["model", "condition", "num_patients", "accuracy",
              "precision", "recall", "f1", "auc"]].sort_values(["model", "condition"])
summary.to_csv(out / "final_unknown_comparison_summary_physionet.csv", index=False)
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.5))
cond_order = ["raw", "unknown_noisy", "adaptive"]
colors = {"raw": "tab:green", "unknown_noisy": "tab:red", "adaptive": "tab:blue"}
models = list(df["model"].unique())
for i, mname in enumerate(models):
    sub = df[df["model"] == mname]
    for j, cond in enumerate(cond_order):
        row = sub[sub["condition"] == cond]
        if len(row) == 0:
            continue
        ax.bar(i * 3 + j, row.iloc[0]["accuracy"], color=colors[cond],
               label=cond if i == 0 else None)
ax.set_xticks([i * 3 + 1 for i in range(len(models))])
ax.set_xticklabels(models)
ax.set_ylabel("Patient-level accuracy")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_title(f"PhysioNet ICH [{args.variant}]", fontweight="bold")
plt.tight_layout()
plt.savefig(out / "fig_patient_accuracy_physionet.png", dpi=150)
plt.close()
print(f"\n[done] 图已保存: {out / 'fig_patient_accuracy_physionet.png'}")
