"""
可视化脚本：已知噪声下不同增强方式的模型性能对比
方案A：热力图（主图，4个指标 × 2个模型）
方案B：分面柱状图（以增强方式为分组，噪声类型为颜色）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "results" / "enhanced_known_noise_dataset1" / "all_enhanced_known_noise_results.csv"
OUTPUT_HEATMAP = BASE / "results" / "enhanced_known_noise_dataset1" / "fig_heatmap_enhancement.png"
OUTPUT_BAR = BASE / "results" / "enhanced_known_noise_dataset1" / "fig_bar_enhancement.png"

# 显示名称映射
NOISE_LABELS = {
    "gaussian": "Gaussian",
    "salt_pepper": "Salt & Pepper",
    "speckle": "Speckle",
    "motion_blur": "Motion Blur",
    "low_light": "Low Light",
}
ENH_LABELS = {
    "median": "Median",
    "gaussian_filter": "Gaussian Filter",
    "bilateral": "Bilateral",
    "clahe": "CLAHE",
    "gamma": "Gamma",
    "clahe_gamma": "CLAHE+Gamma",
    "sharpen": "Sharpen",
    "unsharp": "Unsharp Mask",
    "hist_equalization": "Hist. Equal.",
}
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "swin_tiny": "Swin Transformer",
}
METRICS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision",
    "recall_macro": "Recall",
    "f1_macro": "F1 Score",
}

NOISE_ORDER = list(NOISE_LABELS.keys())
ENH_ORDER = list(ENH_LABELS.keys())
METRIC_ORDER = list(METRICS.keys())
MODELS = ["resnet50", "swin_tiny"]

# 噪声类型配色（方案B使用）
NOISE_COLORS = {
    "gaussian": "#4C72B0",
    "salt_pepper": "#DD8452",
    "speckle": "#55A868",
    "motion_blur": "#C44E52",
    "low_light": "#8172B2",
}


# ══════════════════════════════════════════════════════════════════════
# 方案A：热力图
# ══════════════════════════════════════════════════════════════════════
def plot_heatmap(df):
    """
    布局：2行（模型）× 4列（指标）
    每格：热力图，行=增强方式，列=噪声类型
    """
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(22, 11),
        constrained_layout=True,
    )
    fig.suptitle(
        "Enhancement Method Performance under Known Noise\n(Heatmap: row = Enhancement, col = Noise Type)",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for row_i, model in enumerate(MODELS):
        df_m = df[df["model"] == model]
        for col_i, metric in enumerate(METRIC_ORDER):
            ax = axes[row_i][col_i]

            # 构造矩阵：行=增强方式，列=噪声类型
            matrix = pd.DataFrame(index=ENH_ORDER, columns=NOISE_ORDER, dtype=float)
            for enh in ENH_ORDER:
                for noise in NOISE_ORDER:
                    mask = (df_m["enhancement_method"] == enh) & (df_m["noise_type"] == noise)
                    vals = df_m.loc[mask, metric]
                    matrix.loc[enh, noise] = vals.values[0] if len(vals) > 0 else np.nan

            matrix = matrix.astype(float)

            # 根据指标范围动态设定 vmin/vmax，使颜色差异更显著
            vmin = max(0.5, matrix.min().min() - 0.02)
            vmax = min(1.0, matrix.max().max() + 0.01)

            sns.heatmap(
                matrix,
                ax=ax,
                annot=True,
                fmt=".3f",
                annot_kws={"size": 7.5},
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax,
                linewidths=0.4,
                linecolor="#e0e0e0",
                cbar_kws={"shrink": 0.8, "pad": 0.02},
                xticklabels=[NOISE_LABELS[n] for n in NOISE_ORDER],
                yticklabels=[ENH_LABELS[e] for e in ENH_ORDER],
            )

            ax.set_title(
                f"{METRICS[metric]}",
                fontsize=11, fontweight="bold", pad=6,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=8, rotation=30)
            ax.tick_params(axis="y", labelsize=8, rotation=0)

            # 每行左侧标注模型名
            if col_i == 0:
                ax.set_ylabel(
                    MODEL_LABELS[model],
                    fontsize=12, fontweight="bold", labelpad=8,
                )

    plt.savefig(OUTPUT_HEATMAP, dpi=180, bbox_inches="tight")
    print(f"[✓] 热力图已保存：{OUTPUT_HEATMAP}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 方案B：分面柱状图
# 布局：2行（模型）× 4列（指标）
# 每格：x轴=增强方式，颜色=噪声类型（5色，比9色友好）
# ══════════════════════════════════════════════════════════════════════
def plot_grouped_bar(df):
    n_rows, n_cols = 2, 4
    n_enh = len(ENH_ORDER)
    n_noise = len(NOISE_ORDER)
    bar_w = 0.12
    x = np.arange(n_enh)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(26, 10),
        constrained_layout=True,
        sharey=False,
    )
    fig.suptitle(
        "Enhancement Method Performance under Known Noise"
        "(Grouped Bar: x = Enhancement Method, color = Noise Type)",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for row_i, model in enumerate(MODELS):
        df_m = df[df["model"] == model]
        for col_i, metric in enumerate(METRIC_ORDER):
            ax = axes[row_i][col_i]

            all_vals = []
            for ni, noise in enumerate(NOISE_ORDER):
                df_n = df_m[df_m["noise_type"] == noise]
                vals = []
                for enh in ENH_ORDER:
                    mask = df_n["enhancement_method"] == enh
                    v = df_n.loc[mask, metric]
                    vals.append(v.values[0] if len(v) > 0 else np.nan)

                offset = (ni - (n_noise - 1) / 2) * bar_w
                bars = ax.bar(
                    x + offset, vals,
                    width=bar_w,
                    color=NOISE_COLORS[noise],
                    label=NOISE_LABELS[noise],
                    alpha=0.88,
                    edgecolor="white",
                    linewidth=0.4,
                )
                all_vals.extend([v for v in vals if not np.isnan(v)])

            # y轴范围：稍留白
            lo = max(0.0, min(all_vals) - 0.05)
            hi = min(1.02, max(all_vals) + 0.04)
            ax.set_ylim(lo, hi)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [ENH_LABELS[e] for e in ENH_ORDER],
                rotation=38, ha="right", fontsize=7.5,
            )
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(METRICS[metric], fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)

            if col_i == 0:
                ax.set_ylabel(
                    MODEL_LABELS[model],
                    fontsize=12, fontweight="bold", labelpad=8,
                )

    # 公共图例（底部居中）
    legend_patches = [
        mpatches.Patch(color=NOISE_COLORS[n], label=NOISE_LABELS[n])
        for n in NOISE_ORDER
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=10,
        frameon=True,
        title="Noise Type",
        title_fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.savefig(OUTPUT_BAR, dpi=180, bbox_inches="tight")
    print(f"[✓] 分面柱状图已保存：{OUTPUT_BAR}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成：{df.shape[0]} 行")

    print("正在生成热力图 ...")
    plot_heatmap(df)

    print("正在生成分面柱状图 ...")
    plot_grouped_bar(df)

    print("全部完成。")
