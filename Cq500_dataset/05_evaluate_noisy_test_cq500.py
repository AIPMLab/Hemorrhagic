# -*- coding: utf-8 -*-
"""
CQ500 噪声图患者级评测：每个噪声类型 -> 患者级指标
输出: results/noisy_evaluation_cq500/{model}/... 与 all_noisy_results_cq500.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from cq500_commons import (MODEL_DIR, RESULT_ROOT, NOISY_ROOT, CLASSES, NUM_CLASSES,
                           PatientSliceDataset, predict_slices, patient_vote_metrics)

MODELS = {
    "resnet50": MODEL_DIR / "resnet50_cq500_best.pth",
    "swin_tiny": MODEL_DIR / "swin_tiny_patch4_window7_224_cq500_best.pth",
}
NOISES = ["gaussian", "salt_pepper", "speckle", "motion_blur", "low_light", "unknown_mixed"]
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def main():
    rows = []
    for mname, wpath in MODELS.items():
        from cq500_commons import load_model
        arch = "resnet50" if mname == "resnet50" else "swin_tiny_patch4_window7_224"
        model = load_model(arch, wpath, DEVICE)
        for noise in NOISES:
            ds = PatientSliceDataset("test", base_dir=NOISY_ROOT / noise, transform=tf)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            sdf = predict_slices(model, dl, DEVICE)
            out = RESULT_ROOT / "noisy_evaluation_cq500" / str(pd.Series(mname).iloc[0])
            out.mkdir(parents=True, exist_ok=True)
            m = patient_vote_metrics(sdf, out_path=out / f"{noise}_patient_metrics.csv", tag=f"{mname}-{noise}")
            m["model"] = mname; m["noise_type"] = noise
            rows.append(m)
            sdf.to_csv(out / f"{noise}_slice_preds.csv", index=False)

    summary = pd.DataFrame(rows).rename(columns={"accuracy": "accuracy"}).sort_values(["model", "noise_type"])
    summary.to_csv(RESULT_ROOT / "noisy_evaluation_cq500" / "all_noisy_results_cq500.csv", index=False)
    print(summary[["model", "noise_type", "num_patients", "accuracy", "f1", "auc"]].to_string())


if __name__ == "__main__":
    main()