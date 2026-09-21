# -*- coding: utf-8 -*-
"""
PhysioNet 噪声测试集生成：对 test 患者切片施加 6 类噪声

噪声实现全部来自 physionet_commons（即 cq500_commons 那一份）——
与 CQ500 完全同一批函数，避免两套数据集"噪声参数不同"导致对比失效，
也避免判别器训练数据与测试数据参数不一致。

输出: noisy_tests/physionet/{noise}/{label}/{patient_id}/sXX_fXXXX.png

用法:
    python 04_generate_noisy_test_physionet.py
    python 04_generate_noisy_test_physionet.py --limit 200      # 冒烟测试
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd

from physionet_commons import (DATA_ROOT, NOISE_NAMES, NOISY_ROOT, apply_noise,
                          deterministic_seed, load_index)


def process_slice(task):
    rel, src, noise = task
    # 按 (噪声, 切片) 播种：与 worker 数量/调度顺序无关，结果可复现
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
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 张切片（冒烟测试）")
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
        for i, n in enumerate(ex.map(process_slice, tasks, chunksize=256), 1):
            done += n
            if i % 3000 == 0:
                print(f"[progress] {i}/{len(tasks)} 帧，成功 {done}", flush=True)
    print(f"[done] 共处理 {done} 帧 -> {NOISY_ROOT}")

    pd.DataFrame({"noise": NOISE_NAMES}).to_csv(
        NOISY_ROOT.parent / "physionet_noise_list.csv", index=False)


if __name__ == "__main__":
    main()
