# -*- coding: utf-8 -*-
"""
CQ500 增强图像生成：对加噪切片施加 9 种增强（filter）
输出: enhanced_tests/cq500/{noise}/{method}/{label}/{patient}/seqX_frameY.png

注意路径含**噪声层级** —— 旧版写成 ENHANCED_ROOT/{method}/...，
导致 5 类噪声互相覆盖同一文件，07 评估时每个噪声拿到的是同一批图。

本步骤为**可选**：仅用于出图与 08 的 PSNR/SSIM。
07（已知噪声最优 filter 表）与 16（最优性分析）都在内存态穷举，不依赖这里落盘。

用法:
    python 06_generate_enhanced_images_cq500.py                    # 6 类噪声 x 9 方法
    python 06_generate_enhanced_images_cq500.py --noises unknown_mixed
    python 06_generate_enhanced_images_cq500.py --limit 100        # 冒烟测试
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2

from cq500_commons import (NOISE_NAMES, NOISY_ROOT, apply_filter, enhanced_dir,
                           load_index)

# 可落盘的增强方法（不含 none —— 未增强图已在 noisy_tests 里）
METHODS = ["median", "gaussian_filter", "bilateral", "clahe", "gamma",
           "clahe_gamma", "sharpen", "unsharp", "hist_equalization"]


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
    ap.add_argument("--noises", default=",".join(NOISE_NAMES),
                    help="逗号分隔的噪声列表，默认全部 6 类")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="逗号分隔的增强方法列表，默认全部 9 种")
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 张切片（冒烟测试用）")
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
        for i, n in enumerate(ex.map(process, tasks, chunksize=32), 1):
            done += n
            if i % 10000 == 0:
                print(f"[progress] {i}/{len(tasks)}，新生成 {done}", flush=True)
    print(f"[done] 新生成 {done} 张增强图")
    for noise in noises:
        for method in methods:
            print(f"  {enhanced_dir(noise, method)}")


if __name__ == "__main__":
    main()
