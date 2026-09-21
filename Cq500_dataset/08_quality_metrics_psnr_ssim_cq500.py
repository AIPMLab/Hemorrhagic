# -*- coding: utf-8 -*-
"""
CQ500 图像质量指标：原始 test vs 各增强方法输出的 PSNR/SSIM

路径修正：增强图在 enhanced_tests/cq500/{noise}/{method}/...，旧版漏了噪声层级，
导致所有噪声读到同一批图。

额外加入 "none" 作为基线（none 的增强图就是加噪图本身），
于是能得到 psnr_gain = psnr(raw, enhanced) - psnr(raw, noisy)，
即"增强把图拉回原始图多少"，这是论文里真正想看的恢复量。

输出: results/quality_metrics_cq500/quality_summary_cq500.csv, all_quality_metrics_cq500.csv
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from cq500_commons import (DATA_ROOT, FILTER_NAMES, NOISE_NAMES, NOISY_ROOT,
                           RESULT_ROOT, enhanced_dir, load_index)


def imread_gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (224, 224)) if img is not None else None


def process(task):
    noise, rel = task
    raw = imread_gray(DATA_ROOT / rel)
    noisy = imread_gray(NOISY_ROOT / noise / rel)
    if raw is None or noisy is None:
        return []
    psnr_noisy = peak_signal_noise_ratio(raw, noisy, data_range=255)
    ssim_noisy = structural_similarity(raw, noisy, data_range=255)

    rows = []
    for method in FILTER_NAMES:
        enh = noisy if method == "none" else imread_gray(enhanced_dir(noise, method) / rel)
        if enh is None:
            continue
        psnr = peak_signal_noise_ratio(raw, enh, data_range=255)
        ssim = structural_similarity(raw, enh, data_range=255)
        rows.append({
            "noise_type": noise, "enhancement_method": method,
            "psnr": psnr, "ssim": ssim,
            "psnr_noisy": psnr_noisy, "ssim_noisy": ssim_noisy,
            "psnr_gain": psnr - psnr_noisy, "ssim_gain": ssim - ssim_noisy,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 张切片（冒烟测试用）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = load_index()
    sub = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        sub = sub.head(args.limit).reset_index(drop=True)

    tasks = [(n, rel) for n in NOISE_NAMES for rel in sub["image_path"]]
    print(f"[start] {len(tasks)} 任务（{len(sub)} 切片 x {len(NOISE_NAMES)} 噪声 x {len(FILTER_NAMES)} 方法）",
          flush=True)

    rows, i = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(process, tasks, chunksize=32):
            rows.extend(r)
            i += 1
            if i % 5000 == 0:
                print(f"[progress] {i}/{len(tasks)}", flush=True)

    all_df = pd.DataFrame(rows)
    out = RESULT_ROOT / "quality_metrics_cq500"
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "all_quality_metrics_cq500.csv", index=False)

    summary = all_df.groupby(["noise_type", "enhancement_method"]).agg(
        psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
        psnr_gain_mean=("psnr_gain", "mean"), ssim_gain_mean=("ssim_gain", "mean"),
        n=("psnr", "count")).reset_index()
    summary.to_csv(out / "quality_summary_cq500.csv", index=False)

    print("\n[summary] 各噪声下质量最好的方法（按 psnr_gain）:")
    best = (summary.sort_values("psnr_gain_mean", ascending=False)
            .drop_duplicates("noise_type")[["noise_type", "enhancement_method",
                                            "psnr_mean", "psnr_gain_mean", "ssim_gain_mean"]])
    print(best.to_string(index=False))
    print(f"\n[done] 输出目录: {out}")


if __name__ == "__main__":
    main()
