# -*- coding: utf-8 -*-
"""
RSNA 自适应未知噪声管线（per-slice），支持两个判别器变体

与 CQ500 的 11 同构：
  1) 指定变体的**RSNA 自训练**判别器对 unknown_mixed 每张测试切片预测噪声类别
       - 6class：只认识 6 类单一退化，对复合噪声是分布外
       - 7class：额外认识复合噪声，可直接判为 unknown_mixed
  2) 用 07 在 **val** 上拟合的 (类别 -> 最优 filter) 表换成 filter
  3) 对同一张切片施加该 filter，再送分类器
  4) 患者级投票，输出 raw / unknown_noisy / adaptive 三条件指标

旧版的问题：硬编码 BEST_FILTERS、复用 CQ500 的判别器权重、按 patient 取第一张切片的
预测代表整例、filter 实现与评估不一致、缺失类别静默落到 median。均已修正。

输出（按变体分目录）:
  results/adaptive_unknown_rsna/{variant}/adaptive_unknown_results_rsna.csv
  results/adaptive_unknown_rsna/{variant}/unknown_noise_detection_log_rsna.csv
  results/adaptive_unknown_rsna/{variant}/unknown_filter_selection_log_rsna.csv

用法:
    python 11_adaptive_unknown_pipeline_rsna.py --variant 6class
    python 11_adaptive_unknown_pipeline_rsna.py --variant 7class
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

from rsna_commons import (ADAPTIVE_ROOT, CLASSES, DATA_ROOT, DETECTOR_VARIANTS,
                          MODEL_DIR, NOISY_ROOT, RESULT_ROOT, apply_filter,
                          build_eval_transform, detector_weight_path, load_index,
                          load_model, patient_vote_metrics)

NOISE_CONDITION = "unknown_mixed"

MODELS = {
    "resnet50": ("resnet50", MODEL_DIR / "resnet50_rsna_best.pth"),
    "swin_tiny": ("swin_tiny_patch4_window7_224",
                  MODEL_DIR / "swin_tiny_patch4_window7_224_rsna_best.pth"),
}

MAP_CSV = RESULT_ROOT / "enhanced_known_noise_rsna" / "best_filter_map_rsna.csv"


def variant_outdir(variant):
    return RESULT_ROOT / "adaptive_unknown_rsna" / variant


def _pil(arr_bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))


def load_detector(device, variant):
    weight_path = detector_weight_path(variant)
    if not weight_path.exists():
        raise FileNotFoundError(
            f"缺少判别器权重 {weight_path}，请先运行 09 与 10 --variant {variant}")
    ckpt = torch.load(weight_path, map_location=device)
    class_names = ckpt["class_names"]
    expected = DETECTOR_VARIANTS[variant]
    assert class_names == expected, (
        f"判别器类别与变体不符: checkpoint={class_names} vs {variant}={expected}")
    model = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[detector] variant={variant} classes({len(class_names)})={class_names}")
    return model, class_names


def load_filter_map():
    if not MAP_CSV.exists():
        raise FileNotFoundError(f"缺少 {MAP_CSV}，请先运行 "
                                f"07_evaluate_enhanced_known_noise_rsna.py")
    df = pd.read_csv(MAP_CSV)
    return {(r["model"], r["noise_type"]): r["enhancement_method"]
            for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(DETECTOR_VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--save-images", action="store_true",
                    help="另存自适应增强图（13 出 Grad-CAM 需要）")
    args = ap.parse_args()

    variant = args.variant
    OUTDIR = variant_outdir(variant)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {k: v for k, v in MODELS.items() if k in wanted}
    if not models:
        raise SystemExit(f"--models 未匹配到任何模型，可选: {list(MODELS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tf = build_eval_transform()
    detector, det_classes = load_detector(device, variant)
    filter_map = load_filter_map()

    df = load_index()
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit).reset_index(drop=True)
    print(f"[data] test 切片 {len(test)} 张 / 患者 {test['patient_id'].nunique()} 例")

    for mname in models:
        missing = [c for c in det_classes if (mname, c) not in filter_map]
        if missing:
            raise KeyError(f"{mname} 缺少类别->filter 映射: {missing}；请检查 {MAP_CSV}")
    print(f"[check] {variant} 判别器的 {len(det_classes)} 个类别 filter 映射齐全 ✓")
    if "unknown_mixed" not in det_classes:
        print("[note] 该变体不含 unknown_mixed 类，永远不会预测复合噪声 -> "
              "只能路由到某个已知单噪声的最优 filter")

    classifiers = {}
    for mname, (arch, wpath) in models.items():
        if not wpath.exists():
            raise FileNotFoundError(f"缺少分类器权重 {wpath}，请先运行 02_train_rsna.py")
        classifiers[mname] = load_model(arch, wpath, device)

    rels = test["image_path"].tolist()
    pids = test["patient_id"].tolist()
    y_true = [CLASSES.index(l) for l in test["label"].tolist()]

    probs = {m: {c: [] for c in ("raw", "unknown_noisy", "adaptive")} for m in models}
    det_rows, sel_rows = [], []

    def forward(model, tensors):
        return F.softmax(model(tensors.to(device)), dim=1)[:, 1].cpu().numpy()

    bs = args.batch_size
    for s in range(0, len(rels), bs):
        chunk = rels[s:s + bs]
        arr_noisy, arr_raw = [], []
        for rel in chunk:
            n = cv2.imread(str(NOISY_ROOT / NOISE_CONDITION / rel))
            r = cv2.imread(str(DATA_ROOT / rel))
            if n is None or r is None:
                raise FileNotFoundError(f"读图失败: {rel}")
            arr_noisy.append(n)
            arr_raw.append(r)

        x_noisy = torch.stack([tf(_pil(a)) for a in arr_noisy])
        x_raw = torch.stack([tf(_pil(a)) for a in arr_raw])

        with torch.no_grad():
            det_p = F.softmax(detector(x_noisy.to(device)), dim=1).cpu().numpy()
        classes = [det_classes[i] for i in det_p.argmax(1)]

        for rel, pid, cls_name, conf in zip(chunk, pids[s:s + bs], classes, det_p.max(1)):
            det_rows.append({"patient_id": pid, "image_path": rel,
                             "predicted_noise": cls_name, "confidence": float(conf)})

        with torch.no_grad():
            for mname, model in classifiers.items():
                probs[mname]["raw"].append(forward(model, x_raw))
                probs[mname]["unknown_noisy"].append(forward(model, x_noisy))

                selected = [filter_map[(mname, c)] for c in classes]
                for rel, pid, cls_name, f in zip(chunk, pids[s:s + bs], classes, selected):
                    sel_rows.append({"model": mname, "patient_id": pid, "image_path": rel,
                                     "predicted_noise": cls_name, "selected_filter": f})

                x_ad = torch.stack([tf(_pil(apply_filter(arr, f)))
                                    for arr, f in zip(arr_noisy, selected)])
                probs[mname]["adaptive"].append(forward(model, x_ad))

                if args.save_images:
                    for rel, arr, f in zip(chunk, arr_noisy, selected):
                        dst = ADAPTIVE_ROOT / variant / mname / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(dst), apply_filter(arr, f))

        print(f"[progress] {min(s + bs, len(rels))}/{len(rels)}", end="\r", flush=True)
    print()

    rows = []
    for mname in models:
        for cond in ("raw", "unknown_noisy", "adaptive"):
            p = np.concatenate(probs[mname][cond])
            sdf = pd.DataFrame({"patient_id": pids, "label": y_true, "prob": p})
            sdf["pred"] = (sdf["prob"] >= 0.5).astype(int)
            m = patient_vote_metrics(sdf, tag=f"{mname}-{cond}")
            m.update({"model": mname, "condition": cond})
            rows.append(m)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    det_log = pd.DataFrame(det_rows)
    det_log["label"] = test["label"].tolist()
    det_log.to_csv(OUTDIR / "unknown_noise_detection_log_rsna.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(
        OUTDIR / "unknown_filter_selection_log_rsna.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(["model", "condition"])
    summary.to_csv(OUTDIR / "adaptive_unknown_results_rsna.csv", index=False)

    print("\n[distribution] 判别器在复合噪声上的逐切片预测分布:")
    print(det_log["predicted_noise"].value_counts().to_string())
    print("\n[summary]")
    print(summary[["model", "condition", "num_patients", "accuracy",
                   "f1", "auc"]].to_string(index=False))
    print(f"\n[done] 输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
