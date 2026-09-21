# -*- coding: utf-8 -*-
"""
CQ500 已知噪声最优增强恢复 Grad-CAM（resnet50）：
对 5 类已知噪声，用 best_filter_per_known_noise_cq500.csv 选出的最优方法，
对比 Raw / Noisy / Best-Enhanced 三种输入的 Grad-CAM。
输出: results/gradcam_known_noise_recovery_cq500/{noise}/{patient}/cmp_xxx.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch

from cq500_commons import (MODEL_DIR, RESULT_ROOT, DATA_ROOT, NOISY_ROOT,
                           enhanced_dir, load_index, load_model, IMG_SIZE, grad_cam,
                           overlay, initialize_gradcam_tf)

OUT_DIR = RESULT_ROOT / "gradcam_known_noise_recovery_cq500"
ARCH, WEIGHT = "resnet50", MODEL_DIR / "resnet50_cq500_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tf = initialize_gradcam_tf()

NOISES = ["gaussian", "salt_pepper", "speckle", "motion_blur", "low_light"]


def main():
    best = {}
    best_csv = RESULT_ROOT / "enhanced_known_noise_cq500" / "best_filter_per_known_noise_cq500.csv"
    if best_csv.exists():
        import pandas as pd
        bf = pd.read_csv(best_csv)
        bf = bf[bf["model"] == "resnet50"]
        best = {r["noise_type"]: r["enhancement_method"] for _, r in bf.iterrows()}
    print("最优增强映射:", best)

    model = load_model(ARCH, WEIGHT, DEVICE)
    idx = load_index()
    test = idx[idx["split"] == "test"]
    chosen = test.groupby("patient_id").apply(
        lambda g: g.iloc[len(g) // 2]["image_path"]).reset_index(name="image_path")

    for noise in NOISES:
        method = best.get(noise)
        if not method:
            print(f"[skip] {noise} 无最优方法，跳过")
            continue
        for _, row in chosen.iterrows():
            pid, rel = row["patient_id"], row["image_path"]
            variants = {"raw": DATA_ROOT / rel,
                        "noisy": NOISY_ROOT / noise / rel,
                        # 增强图路径含噪声层级（enhanced_dir 是唯一入口；
                        # 旧写法 ENHANCED_ROOT/method 会漏掉噪声层，取不到图）
                        "best_enhanced": enhanced_dir(noise, method) / rel}
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
            d = OUT_DIR / noise / pid
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"cmp_{Path(rel).name.replace('.png','')}.png"), canvas)
        print(f"[done] {noise} ({method}) 恢复图已输出")
    print("[done] 输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()