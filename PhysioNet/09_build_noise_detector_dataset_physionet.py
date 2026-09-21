# -*- coding: utf-8 -*-
"""
PhysioNet 噪声判别器数据集构建（7 类）

类别: clean / gaussian / salt_pepper / speckle / motion_blur / low_light / unknown_mixed

与 CQ500 的 09 同构：
  * 源切片**只取 PhysioNet 的 train + val 患者，绝不使用 test**（否则判别器见过测试图）。
  * 每患者等间距抽 SLICES_PER_PATIENT 张。
  * 噪声一律来自 physionet_commons（= cq500_commons 同一份函数），
    与 04（分类器测试集加噪）同源，消除训练/推理噪声不匹配。
  * 7 个类别由**同一批源切片**变换而来，类别天然均衡。

manifest.csv 记录 file/class/patient_id/source_split，
使 10 能按患者划分判别器自身的 train/val（沿用 PhysioNet 的患者边界，两侧都不含 test）。

输出: noise_detector_dataset_physionet/{class}/{patient}_{idx}_{stem}.png + manifest.csv

用法:
    python 09_build_noise_detector_dataset_physionet.py
    python 09_build_noise_detector_dataset_physionet.py --slices-per-patient 40
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd

from physionet_commons import (DATA_ROOT, DETECTOR_CLASSES, DETECTOR_DATASET_DIR,
                          apply_noise, deterministic_seed, load_index)


def process(task):
    cls_name, rel, src, out_name = task
    np.random.seed(deterministic_seed(cls_name, rel))
    img = cv2.imread(str(src))
    if img is None:
        return 0
    out = apply_noise(img, cls_name)
    dst = DETECTOR_DATASET_DIR / cls_name / out_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 必须与 04 生成的噪声测试图用**同一种编码**（PNG 无损）。
    # 曾用 JPEG q95，实测会把高斯噪声衰减 25.5%、斑点 23.4%，
    # 而椒盐/运动模糊几乎不受影响 —— 判别器学到的是"被压过的噪声"，
    # 一到推理就只在高斯/斑点上崩。训练与推理口径必须一致。
    cv2.imwrite(str(dst), out)
    return 1


def even_sample(rows, n):
    """等间距抽取 n 张，覆盖整个扫描范围。"""
    if n <= 0 or not rows:
        return []
    if len(rows) <= n:
        return rows
    if n == 1:
        return [rows[len(rows) // 2]]
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [rows[i] for i in sorted(set(idx))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices-per-patient", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = load_index()
    src = df[df["split"].isin(["train", "val"])].reset_index(drop=True)

    test_patients = set(df[df["split"] == "test"]["patient_id"])
    used = set(src["patient_id"])
    assert not (used & test_patients), "泄漏: 判别器数据源包含 test 患者"
    print(f"[check] 源患者({len(used)}) ∩ test 患者 = ∅  ✓")

    tasks, manifest = [], []
    for pid, grp in src.groupby("patient_id"):
        grp = grp.sort_values("image_path")
        picked = even_sample(grp.to_dict("records"), args.slices_per_patient)
        for i, row in enumerate(picked):
            out_name = f"{pid}_{i:03d}_{Path(row['image_path']).stem}.png"
            for cls_name in DETECTOR_CLASSES:
                tasks.append((cls_name, row["image_path"],
                              DATA_ROOT / row["image_path"], out_name))
                manifest.append({"file": f"{cls_name}/{out_name}", "class": cls_name,
                                 "patient_id": pid, "source_split": row["split"]})

    print(f"[start] {len(tasks)} 张"
          f"（{len(used)} 患者 x ≤{args.slices_per_patient} 切片 x "
          f"{len(DETECTOR_CLASSES)} 类）", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, n in enumerate(ex.map(process, tasks, chunksize=64), 1):
            done += n
            if i % 5000 == 0:
                print(f"[progress] {i}/{len(tasks)}，成功 {done}", flush=True)

    DETECTOR_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    mdf = pd.DataFrame(manifest)
    mdf.to_csv(DETECTOR_DATASET_DIR / "manifest.csv", index=False)

    print(f"\n[done] 生成 {done} 张 -> {DETECTOR_DATASET_DIR}")
    print("[dist] 类别分布:")
    print(mdf["class"].value_counts().to_string())
    print("[dist] 来源划分 (患者数):")
    print(mdf.drop_duplicates(["patient_id", "source_split"])["source_split"]
          .value_counts().to_string())


if __name__ == "__main__":
    main()
