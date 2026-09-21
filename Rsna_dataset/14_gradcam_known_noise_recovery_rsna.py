# -*- coding: utf-8 -*-
"""
RSNA 已知噪声最优增强恢复 Grad-CAM（与 CQ500 的 14 同构）

对 5 类已知噪声，用 07 输出的 best_filter_map_rsna.csv 选出的最优方法，
对比 Raw / Noisy / Best-Enhanced 三种输入的 Grad-CAM。

注意增强图路径含**噪声层级**：enhanced_tests/rsna/{noise}/{method}/...
（enhanced_dir 是唯一入口；旧写法漏掉噪声层会取不到图）

输出: results/gradcam_known_noise_recovery_rsna/{noise}/{patient}/*.png

用法:
    python 14_gradcam_known_noise_recovery_rsna.py
    python 14_gradcam_known_noise_recovery_rsna.py --model swin_tiny
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import pandas as pd
import torch

from rsna_commons import (DATA_ROOT, MODEL_DIR, NOISY_ROOT, RESULT_ROOT, IMG_SIZE,
                          enhanced_dir, grad_cam, initialize_gradcam_tf, load_index,
                          load_model, overlay)

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="resnet50", choices=["resnet50", "swin_tiny"])
_ap.add_argument("--max-patients", type=int, default=0, help="每个噪声只画前 N 例")
args = _ap.parse_args()

OUT_DIR = RESULT_ROOT / "gradcam_known_noise_recovery_rsna"
ARCH = {"resnet50": "resnet50",
        "swin_tiny": "swin_tiny_patch4_window7_224"}[args.model]
WEIGHT = MODEL_DIR / f"{ARCH}_rsna_best.pth"
BEST_CSV = RESULT_ROOT / "enhanced_known_noise_rsna" / "best_filter_map_rsna.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tf = initialize_gradcam_tf()

NOISES = ["gaussian", "salt_pepper", "speckle", "motion_blur", "low_light"]


def main():
    if not WEIGHT.exists():
        raise SystemExit(f"缺少分类器权重 {WEIGHT}，请先运行 02_train_rsna.py")
    if not BEST_CSV.exists():
        raise SystemExit(f"缺少 {BEST_CSV}，请先运行 "
                         f"07_evaluate_enhanced_known_noise_rsna.py")

    bf = pd.read_csv(BEST_CSV)
    bf = bf[bf["model"] == args.model]
    best = {r["noise_type"]: r["enhancement_method"] for _, r in bf.iterrows()}
    print("最优增强映射:", best)

    model = load_model(ARCH, WEIGHT, DEVICE)
    idx = load_index()
    test = idx[idx["split"] == "test"]
    chosen = test.groupby("patient_id").apply(
        lambda g: g.iloc[len(g) // 2]["image_path"]).reset_index(name="image_path")
    if args.max_patients:
        chosen = chosen.head(args.max_patients)

    for noise in NOISES:
        method = best.get(noise)
        if not method:
            print(f"[skip] {noise} 无最优方法（07 是否跑过？）")
            continue
        done = 0
        for _, row in chosen.iterrows():
            pid, rel = row["patient_id"], row["image_path"]
            variants = {"raw": DATA_ROOT / rel,
                        "noisy": NOISY_ROOT / noise / rel,
                        "best_enhanced": enhanced_dir(noise, method) / rel}
            tiles = []
            for _n, p in variants.items():
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                x = tf(torch.from_numpy(rgb).permute(2, 0, 1) / 255.0).unsqueeze(0).to(DEVICE)
                cam = grad_cam(model, x)
                tiles.append(overlay(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cam))
            if len(tiles) < 3:
                continue
            d = OUT_DIR / noise / pid
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"cmp_{Path(rel).stem}.png"), cv2.hconcat(tiles))
            done += 1
        print(f"[done] {noise} ({method}) 输出 {done} 张")

    print(f"[done] 输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
