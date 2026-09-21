# -*- coding: utf-8 -*-
"""
CQ500 噪声测试集生成：对 test 患者切片施加 6 类噪声
输出: noisy_tests/cq500/{noise}/{label}/{patient}/seqX_frameY.png（结构同 png，方便下游切换 base_dir）

噪声实现全部来自 cq500_commons.NOISE_FUNCS —— 与判别器数据集(09)同源，
避免"判别器训练用一种噪声参数、测试用另一种"的分布不匹配。

用法:
    python 04_generate_noisy_test_cq500.py                 # 默认 test 划分
    python 04_generate_noisy_test_cq500.py --split val
    python 04_generate_noisy_test_cq500.py --limit 200     # 仅前 200 张，冒烟测试用
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd

from cq500_commons import (DATA_ROOT, NOISY_ROOT, NOISE_NAMES, apply_noise,
                           deterministic_seed, load_index)


def process_slice(task):
    rel, src, noise = task
    # 按 (切片, 噪声) 播种：与 worker 数量/调度顺序无关，可复现
    np.random.seed(deterministic_seed(noise, rel))
    img = cv2.imread(str(src))
    if img is None:
        return 0
    out = apply_noise(img, noise)
    dst = NOISY_ROOT / noise / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="划分: train/val/test（默认 test）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 张切片（冒烟测试用，0=全部）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = load_index()
    sub = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        sub = sub.head(args.limit).reset_index(drop=True)
    print(f"[start] {args.split} 切片 {len(sub)} 张 x {len(NOISE_NAMES)} 噪声", flush=True)

    tasks = [(row["image_path"], DATA_ROOT / row["image_path"], nf)
             for _, row in sub.iterrows() for nf in NOISE_NAMES]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, n in enumerate(ex.map(process_slice, tasks, chunksize=64), 1):
            done += n
            if i % 5000 == 0:
                print(f"[progress] {i}/{len(tasks)} 帧，成功 {done}", flush=True)
    print(f"[done] 共处理 {done} 帧 -> {NOISY_ROOT}")

    pd.DataFrame({"noise": NOISE_NAMES}).to_csv(
        NOISY_ROOT.parent / f"{NOISY_ROOT.name}_noise_list.csv", index=False)


if __name__ == "__main__":
    main()
