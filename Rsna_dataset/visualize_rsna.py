# -*- coding: utf-8 -*-
"""
RSNA 可视化：已知噪声增强性能 热力图 + 分面柱状图（患者级指标）

读 07 产出的 all_enhanced_known_noise_rsna.csv（该表已按 val 拟合 / test 评估口径生成）。
输出: results/enhanced_known_noise_rsna/fig_heatmap_rsna_{metric}.png, fig_bar_rsna.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rsna_commons import RESULT_ROOT

DATA = RESULT_ROOT / "enhanced_known_noise_rsna" / "all_enhanced_known_noise_rsna.csv"
OUTDIR = RESULT_ROOT / "enhanced_known_noise_rsna"
NOISE_ORDER = ["gaussian", "salt_pepper", "speckle", "motion_blur", "low_light"]
ENH_ORDER = ["median", "gaussian_filter", "bilateral", "clahe", "gamma",
             "clahe_gamma", "sharpen", "unsharp", "hist_equalization"]
METRICS = {"accuracy": "Accuracy", "f1": "F1", "auc": "AUC"}


def main():
    if not DATA.exists():
        raise SystemExit(f"缺少 {DATA}，请先运行 07_evaluate_enhanced_known_noise_rsna.py")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    models = sorted(df["model"].unique())
    print(f"[data] {len(df)} 行，模型 {models}")

    for metric in METRICS:
        fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.5),
                                 constrained_layout=True)
        axes = [axes] if len(models) == 1 else list(axes)
        for ax, model in zip(axes, models):
            sub = df[df["model"] == model]
            mat = pd.DataFrame(index=ENH_ORDER, columns=NOISE_ORDER, dtype=float)
            for e in ENH_ORDER:
                for n in NOISE_ORDER:
                    m = sub[(sub["enhancement_method"] == e) & (sub["noise_type"] == n)]
                    mat.loc[e, n] = m[metric].values[0] if len(m) else np.nan
            sns.heatmap(mat.astype(float), ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                        annot_kws={"size": 8}, linewidths=0.4)
            ax.set_title(f"{model} - {METRICS[metric]}")
        fig.suptitle("RSNA Known-Noise Enhancement (Patient-level)", y=1.02,
                     fontweight="bold")
        fig.savefig(OUTDIR / f"fig_heatmap_rsna_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[done] heatmap {metric}")

    colors = dict(zip(NOISE_ORDER, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]))
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.5),
                             constrained_layout=True)
    axes = [axes] if len(models) == 1 else list(axes)
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        x = np.arange(len(ENH_ORDER))
        w = 0.14
        for k, noise in enumerate(NOISE_ORDER):
            vals = []
            for e in ENH_ORDER:
                m = sub[(sub["noise_type"] == noise) & (sub["enhancement_method"] == e)]
                vals.append(m["accuracy"].values[0] if len(m) else 0)
            ax.bar(x + (k - 2) * w, vals, w, color=colors[noise], label=noise)
        ax.set_xticks(x)
        ax.set_xticklabels(ENH_ORDER, rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("RSNA Enhancement Accuracy by Noise (Patient-level)", y=1.02,
                 fontweight="bold")
    fig.savefig(OUTDIR / "fig_bar_rsna.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[done] bar")


if __name__ == "__main__":
    main()
