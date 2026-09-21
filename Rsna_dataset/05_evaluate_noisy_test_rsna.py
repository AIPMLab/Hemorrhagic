# -*- coding: utf-8 -*-
"""
RSNA 加噪图患者级评测：每个噪声类型 -> 患者级指标

输出: results/noisy_evaluation_rsna/{model}/... 与 all_noisy_results_rsna.csv

用法:
    python 05_evaluate_noisy_test_rsna.py
    python 05_evaluate_noisy_test_rsna.py --models resnet50 --limit 300
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_commons import (MODEL_DIR, NOISY_ROOT, NOISE_NAMES, RESULT_ROOT,
                          PatientSliceDataset, build_eval_transform, load_model,
                          patient_vote_metrics, predict_slices)

MODELS = {
    "resnet50": ("resnet50", MODEL_DIR / "resnet50_rsna_best.pth"),
    "swin_tiny": ("swin_tiny_patch4_window7_224",
                  MODEL_DIR / "swin_tiny_patch4_window7_224_rsna_best.pth"),
}
BATCH_SIZE = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--noises", default=",".join(NOISE_NAMES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {k: v for k, v in MODELS.items() if k in wanted}
    noises = [n.strip() for n in args.noises.split(",") if n.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tf = build_eval_transform()
    outdir = RESULT_ROOT / "noisy_evaluation_rsna"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for mname, (arch, wpath) in models.items():
        if not wpath.exists():
            raise FileNotFoundError(f"缺少权重 {wpath}，请先运行 02_train_rsna.py")
        model = load_model(arch, wpath, device)
        for noise in noises:
            base = NOISY_ROOT / noise
            if not base.exists():
                print(f"[skip] 缺少 {base}，请先运行 04_generate_noisy_test_rsna.py")
                continue
            ds = PatientSliceDataset("test", base_dir=base, transform=tf,
                                     limit=args.limit)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
            sdf = predict_slices(model, dl, device)
            m = patient_vote_metrics(
                sdf, out_path=outdir / f"{mname}_{noise}_patient_metrics.csv",
                tag=f"{mname}-{noise}")
            m["model"] = mname
            m["noise_type"] = noise
            rows.append(m)
            sdf.to_csv(outdir / f"{mname}_{noise}_slice_preds.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(["model", "noise_type"])
    summary.to_csv(outdir / "all_noisy_results_rsna.csv", index=False)
    print("\n[summary]")
    print(summary[["model", "noise_type", "num_patients", "accuracy",
                   "f1", "auc"]].to_string(index=False))
    print(f"\n[done] {outdir}")


if __name__ == "__main__":
    main()
