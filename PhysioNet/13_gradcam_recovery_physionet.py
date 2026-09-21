# -*- coding: utf-8 -*-
"""
PhysioNet 未知噪声恢复 Grad-CAM（与 CQ500 的 13 同构）

对 test 每位患者取一张中间切片，对比 Raw / Unknown Noisy / Adaptive Enhanced
三种输入的 Grad-CAM，用来展示"增强后注意力是否回到出血区域"。

需要 11 --save-images 产生的自适应增强图
    adaptive_unknown_tests/physionet/{variant}/{model}/...
输出: results/gradcam_recovery_physionet_{variant}_{model}/{patient}/*.png

用法:
    python 13_gradcam_recovery_physionet.py --variant 7class --model resnet50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import torch

from physionet_commons import (ADAPTIVE_ROOT, DATA_ROOT, MODEL_DIR, NOISY_ROOT, RESULT_ROOT,
                          IMG_SIZE, grad_cam, initialize_gradcam_tf, load_index,
                          load_model, overlay)

_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", default="7class", help="用哪个判别器变体的自适应增强图")
_ap.add_argument("--model", default="resnet50",
                 choices=["resnet50", "swin_tiny"], help="用哪个分类器出 Grad-CAM")
_ap.add_argument("--max-patients", type=int, default=0, help="只画前 N 例（0=全部）")
args = _ap.parse_args()

OUT_DIR = RESULT_ROOT / f"gradcam_recovery_physionet_{args.variant}_{args.model}"
ARCH = {"resnet50": "resnet50",
        "swin_tiny": "swin_tiny_patch4_window7_224"}[args.model]
WEIGHT = MODEL_DIR / f"{ARCH}_physionet_best.pth"
ADAPTIVE_DIR = ADAPTIVE_ROOT / args.variant / args.model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tf = initialize_gradcam_tf()


def main():
    if not WEIGHT.exists():
        raise SystemExit(f"缺少分类器权重 {WEIGHT}，请先运行 02_train_physionet.py")
    if not ADAPTIVE_DIR.exists():
        raise SystemExit(
            f"缺少自适应增强图 {ADAPTIVE_DIR}\n"
            f"请先运行 11_adaptive_unknown_pipeline_physionet.py --variant {args.variant} "
            f"--save-images")

    model = load_model(ARCH, WEIGHT, DEVICE)
    idx = load_index()
    test = idx[idx["split"] == "test"]
    chosen = test.groupby("patient_id").apply(
        lambda g: g.iloc[len(g) // 2]["image_path"]).reset_index(name="image_path")
    if args.max_patients:
        chosen = chosen.head(args.max_patients)
    print(f"[start] {len(chosen)} 例  variant={args.variant} model={args.model}")

    done = 0
    for _, row in chosen.iterrows():
        pid, rel = row["patient_id"], row["image_path"]
        variants = {"raw": DATA_ROOT / rel,
                    "unknown_noisy": NOISY_ROOT / "unknown_mixed" / rel,
                    "adaptive": ADAPTIVE_DIR / rel}
        tiles = []
        for _name, p in variants.items():
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = tf(torch.from_numpy(rgb).permute(2, 0, 1) / 255.0).unsqueeze(0).to(DEVICE)
            cam = grad_cam(model, x)
            tiles.append(overlay(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cam))
        if len(tiles) < 3:
            print(f"[skip] {pid} 缺图（raw/noisy/adaptive 需齐三张）")
            continue
        d = OUT_DIR / pid
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"cmp_{Path(rel).stem}.png"), cv2.hconcat(tiles))
        done += 1
        if done % 25 == 0:
            print(f"[progress] {done}/{len(chosen)}", flush=True)

    print(f"[done] 输出 {done} 张 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
