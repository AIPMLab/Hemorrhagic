# -*- coding: utf-8 -*-
"""
CQ500 可视化：已知噪声增强性能 热力图 + 分面柱状图（患者级指标）
输出: results/enhanced_known_noise_cq500/fig_heatmap_cq500.png, fig_bar_cq500.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from cq500_commons import RESULT_ROOT

DATA = RESULT_ROOT / "enhanced_known_noise_cq500" / "all_enhanced_known_noise_cq500.csv"
NOISE_ORDER = ["gaussian", "salt_pepper", "speckle", "motion_blur", "low_light"]
ENH_ORDER = ["median", "gaussian_filter", "bilateral", "clahe", "gamma",
             "clahe_gamma", "sharpen", "unsharp", "hist_equalization"]
METRICS = {"accuracy": "Accuracy", "f1": "F1", "auc": "AUC"}
df = pd.read_csv(DATA)
MODELS = sorted(df["model"].unique())

for metric in METRICS:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5 * len(MODELS), 4.5), constrained_layout=True)
    axes = [axes] if len(MODELS) == 1 else list(axes)
    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        mat = pd.DataFrame(index=ENH_ORDER, columns=NOISE_ORDER, dtype=float)
        for e in ENH_ORDER:
            for n in NOISE_ORDER:
                m = sub[(sub["enhancement_method"] == e) & (sub["noise_type"] == n)]
                vals = m[metric]
                mat.loc[e, n] = vals.values[0] if len(vals) else np.nan
        sns.heatmap(mat.astype(float), ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                    annot_kws={"size": 8}, linewidths=0.4)
        ax.set_title(f"{model} - {METRICS[metric]}")
    fig.suptitle("CQ500 Known-Noise Enhancement (Patient-level)", y=1.02, fontweight="bold")
    fig.savefig(RESULT_ROOT / "enhanced_known_noise_cq500" / f"fig_heatmap_cq500_{metric}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[done] heatmap {metric}")

# 柱状图：accuracy，model x noise
COLORS = {n: c for n, c in zip(NOISE_ORDER, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])}
fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5 * len(MODELS), 4.5), constrained_layout=True)
axes = [axes] if len(MODELS) == 1 else list(axes)
for ax, model in zip(axes, MODELS):
    sub = df[df["model"] == model]
    x = np.arange(len(ENH_ORDER)); w = 0.14
    for k, noise in enumerate(NOISE_ORDER):
        vals = [sub[(sub["noise_type"] == noise) & (sub["enhancement_method"] == e)]["accuracy"]
                .values[0] if len(sub[(sub["noise_type"] == noise) & (sub["enhancement_method"] == e)]) else 0
                for e in ENH_ORDER]
        ax.bar(x + (k - 2) * w, vals, w, color=COLORS[noise], label=noise)
    ax.set_xticks(x); ax.set_xticklabels(ENH_ORDER, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_title(model); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("CQ500 Enhancement Accuracy by Noise (Patient-level)", y=1.02, fontweight="bold")
fig.savefig(RESULT_ROOT / "enhanced_known_noise_cq500" / "fig_bar_cq500.png", dpi=150, bbox_inches="tight")
plt.close()
print("[done] bar")