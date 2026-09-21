# -*- coding: utf-8 -*-
"""
CQ500 未知噪声恢复 Grad-CAM（resnet50）：
对 test 每个患者取一张中间切片，对比 Raw / Unknown Noisy / Adaptive Enhanced 三种输入的 Grad-CAM。
输出: results/gradcam_recovery_cq500/{patient}/cmp_xxx.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch

from cq500_commons import (MODEL_DIR, RESULT_ROOT, DATA_ROOT, NOISY_ROOT, ADAPTIVE_ROOT,
                           load_index, load_model, IMG_SIZE, grad_cam, overlay,
                           initialize_gradcam_tf)

# 自适应增强图按 (变体, 分类器) 分目录保存在 11 --save-images 的产物里
_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", default="7class", help="用哪个判别器变体的自适应增强图")
_ap.add_argument("--model", default="resnet50",
                 choices=["resnet50", "swin_tiny"], help="用哪个分类器出 Grad-CAM")
args = _ap.parse_args()

OUT_DIR = RESULT_ROOT / f"gradcam_recovery_cq500_{args.variant}_{args.model}"
ARCH = {"resnet50": "resnet50",
        "swin_tiny": "swin_tiny_patch4_window7_224"}[args.model]
WEIGHT = MODEL_DIR / f"{ARCH}_cq500_best.pth"
ADAPTIVE_DIR = ADAPTIVE_ROOT / args.variant / args.model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tf = initialize_gradcam_tf()


def main():
    model = load_model(ARCH, WEIGHT, DEVICE)
    idx = load_index()
    test = idx[idx["split"] == "test"]
    chosen = test.groupby("patient_id").apply(
        lambda g: g.iloc[len(g) // 2]["image_path"]).reset_index(name="image_path")

    for i, (_, row) in enumerate(chosen.iterrows()):
        pid, rel = row["patient_id"], row["image_path"]
        variants = {"raw": DATA_ROOT / rel,
                    "unknown_noisy": NOISY_ROOT / "unknown_mixed" / rel,
                    "adaptive": ADAPTIVE_DIR / rel}
        tiles = []
        for name, p in variants.items():
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = tf(torch.from_numpy(rgb).permute(2, 0, 1) / 255.0).unsqueeze(0).to(DEVICE)
            cam = grad_cam(model, x)
            tiles.append(overlay(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cam))
        if not tiles:
            continue
        canvas = cv2.hconcat(tiles)
        d = OUT_DIR / pid
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"cmp_{Path(rel).name.replace('.png','')}.png"), canvas)
        if (i + 1) % 50 == 0:
            print(f"[progress] {i+1}/{len(chosen)} 患者")
    print(f"[done] Grad-CAM 输出:", OUT_DIR)


if __name__ == "__main__":
    main()