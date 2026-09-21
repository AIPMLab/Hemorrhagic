# -*- coding: utf-8 -*-
"""
PhysioNet 增强图像生成：对加噪切片施加 9 种增强（filter）

路径为 enhanced_tests/physionet/{noise}/{method}/...
—— **含噪声层级**。旧版写成 ENHANCED_ROOT/{method}/...，
   5 类噪声会互相覆盖同一文件，07 评估时每个噪声拿到的是同一批图。

本步骤为**可选**：仅用于出图与 08 的 PSNR/SSIM。
07（最优 filter 表）与 16（最优性分析）都在内存态穷举，不依赖这里落盘。

用法:
    python 06_generate_enhanced_images_physionet.py
    python 06_generate_enhanced_images_physionet.py --noises unknown_mixed --limit 100
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2

from physionet_commons import (FILTER_NAMES, NOISE_NAMES, NOISY_ROOT, apply_filter,
                          enhanced_dir, load_index)

# 可落盘的增强方法（不含 none —— 未增强图已在 noisy_tests 里）
METHODS = [f for f in FILTER_NAMES if f != "none"]


def process(task):
    rel, src, noise, method = task
    dst = enhanced_dir(noise, method) / rel
    if dst.exists():
        return 0
    img = cv2.imread(str(src))
    if img is None:
        return 0
    out = apply_filter(img, method)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--noises", default=",".join(NOISE_NAMES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    noises = [n.strip() for n in args.noises.split(",") if n.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    df = load_index()
    sub = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        sub = sub.head(args.limit).reset_index(drop=True)

    tasks = []
    for noise in noises:
        for _, row in sub.iterrows():
            src = NOISY_ROOT / noise / row["image_path"]
            for method in methods:
                tasks.append((row["image_path"], src, noise, method))
    print(f"[start] 增强任务 {len(tasks)}"
          f"（{len(sub)} 切片 x {len(noises)} 噪声 x {len(methods)} 方法）", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, n in enumerate(ex.map(process, tasks, chunksize=256), 1):
            done += n
            if i % 5000 == 0:
                print(f"[progress] {i}/{len(tasks)}，新生成 {done}", flush=True)
    print(f"[done] 新生成 {done} 张增强图")


if __name__ == "__main__":
    main()
