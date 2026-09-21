# -*- coding: utf-8 -*-
"""
CQ500 最终对比：Raw / Unknown Noisy / Adaptive Enhanced 的患者级指标汇总表与图
（数据来自 11 --variant X 的输出）
输出: results/final_unknown_comparison_cq500/{variant}/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cq500_commons import RESULT_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="7class", help="用哪个判别器变体的自适应结果")
args = ap.parse_args()

in_file = (RESULT_ROOT / "adaptive_unknown_cq500" / args.variant
           / "adaptive_unknown_results_cq500.csv")
out = RESULT_ROOT / "final_unknown_comparison_cq500" / args.variant
out.mkdir(parents=True, exist_ok=True)
if not in_file.exists():
    raise FileNotFoundError(f"缺少 {in_file}，请先运行 11 --variant {args.variant}")

df = pd.read_csv(in_file)
summary = df[["model", "condition", "num_patients", "accuracy", "precision", "recall", "f1", "auc"]]
summary = summary.sort_values(["model", "condition"])
summary.to_csv(out / "final_unknown_comparison_summary_cq500.csv", index=False)
print(summary.to_string(index=False))

# 患者级 accuracy 对比图
fig, ax = plt.subplots(figsize=(8, 4.5))
cond_order = ["raw", "unknown_noisy", "adaptive"]
colors = {"raw": "tab:green", "unknown_noisy": "tab:red", "adaptive": "tab:blue"}
for i, mname in enumerate(df["model"].unique()):
    sub = df[df["model"] == mname]
    for cond in cond_order:
        row = sub[sub["condition"] == cond].iloc[0]
        ax.bar(i * 3 + cond_order.index(cond), row["accuracy"], color=colors[cond], label=cond if i == 0 else None)
ax.set_xticks([i * 3 + 1 for i in range(len(df["model"].unique()))])
ax.set_xticklabels(df["model"].unique())
ax.set_ylabel("Patient-level Accuracy"); ax.set_ylim(0, 1)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(out / "fig_patient_accuracy_cq500.png", dpi=150); plt.close()
print("[done] 图已保存:", out / "fig_patient_accuracy_cq500.png")