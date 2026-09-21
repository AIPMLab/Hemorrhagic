# -*- coding: utf-8 -*-
"""
PhysioNet 子集二分类训练（Normal / Hemorrhagic）

与 CQ500 的 02_train_cq500.py 同构，只在数据集相关处不同：
  - 用 physionet_commons（索引 physionet_slices_index.csv，患者键为 patient_id）
  - 权重存为 models/*_physionet_best.pth
  - 预训练权重同样走 weights/ 本地文件（换机器/离线都能跑）

训练循环已按 CQ500 那边的经验修正：
  pin_memory + non_blocking、persistent_workers、GPU 侧累加器（每轮只同步一次）、
  逐 batch 进度条、worker_init_fn（增强随机性与 worker 数解耦）。

用法:
    python 02_train_physionet.py
    python 02_train_physionet.py --models resnet50
    python 02_train_physionet.py --epochs 1 --limit 600      # 冒烟测试
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from physionet_commons import (CLASSES, MODEL_DIR, NUM_CLASSES, RESULT_ROOT,
                          PatientSliceDataset, create_model_with_pretrained,
                          patient_vote_metrics, predict_slices)

ALL_MODELS = ["resnet50", "swin_tiny_patch4_window7_224"]
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15


def worker_init_fn(worker_id):
    """增强随机流由主进程种子派生，保证与 num_workers 无关的可复现性。"""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="train/val 各取前 N 张（冒烟测试）")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="改动 batch size 会改变训练结果，需全程一致")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="每 N 轮做一次验证集患者级评估（最后一轮必做）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-column", choices=["label", "slice_label"], default="label",
                    help="label=患者级标签（与 CQ500 同口径，默认）；"
                         "slice_label=逐帧标注（PhysioNet 独有，可作消融）")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    models_to_train = [m.strip() for m in args.models.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | batch={args.batch_size} workers={args.num_workers} "
          f"seed={args.seed} label={args.label_column}")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainset = PatientSliceDataset("train", transform=train_tf,
                                   label_column=args.label_column, limit=args.limit)
    valset = PatientSliceDataset("val", transform=eval_tf,
                                 label_column=args.label_column, limit=args.limit)
    print(f"[data] train {len(trainset)} 切片 / val {len(valset)} 切片")

    if args.label_column == "label":
        labels = trainset.df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    else:
        labels = trainset.df["slice_label"].astype(int).values
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    print(f"[data] 类别计数 {dict(zip(CLASSES, counts.astype(int)))}")
    weights = 1.0 / np.maximum(counts, 1)[labels]

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=args.num_workers > 0,
                     prefetch_factor=4 if args.num_workers > 0 else None,
                     worker_init_fn=worker_init_fn)
    loader_kw = {k: v for k, v in loader_kw.items() if v is not None}

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, drop_last=True,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
        **loader_kw)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            **loader_kw)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name in models_to_train:
        print(f"\n========== 训练 {model_name} (PhysioNet 二分类) ==========")
        model = create_model_with_pretrained(model_name, NUM_CLASSES,
                                             pretrained=not args.no_pretrained)
        model.to(device)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best_f1, best_path = -1.0, MODEL_DIR / f"{model_name}_physionet_best.pth"
        for ep in range(args.epochs):
            model.train()
            # 累加器放 GPU：原来每 batch 两次 .item() 会强制同步、打断流水线
            n_seen = 0
            corr_t = torch.zeros((), device=device)
            loss_t = torch.zeros((), device=device)

            pbar = tqdm(train_loader, desc=f"[{model_name}] epoch {ep+1}/{args.epochs}",
                        leave=False)
            for x, y, _ in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                loss = crit(out, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                n_seen += y.size(0)
                corr_t += (out.argmax(1) == y).sum()
                loss_t += loss.detach() * y.size(0)
                if pbar.n % 200 == 0:
                    pbar.set_postfix(loss=f"{(loss_t / max(n_seen, 1)).item():.4f}")

            sched.step()
            tot = n_seen
            corr = int(corr_t.item())
            loss_sum = float(loss_t.item())
            pbar.close()
            print(f"[{model_name}] epoch {ep+1}/{args.epochs} | "
                  f"loss {loss_sum/max(tot,1):.4f} | train_acc {corr/max(tot,1):.4f}")

            if (ep + 1) % args.eval_every != 0 and (ep + 1) != args.epochs:
                continue
            vdf = predict_slices(model, val_loader, device)
            m = patient_vote_metrics(vdf, tag=f"{model_name}-val-ep{ep+1}")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                torch.save(model.state_dict(), best_path)
                print(f"    -> 新最优 f1={m['f1']:.4f}，已保存 {best_path}")

        print(f"[{model_name}] 最优患者级 F1={best_f1:.4f}")
        rows.append({"dataset": "physionet", "model": model_name, "best_patient_f1": best_f1,
                     "epochs": args.epochs, "batch_size": args.batch_size,
                     "num_workers": args.num_workers, "seed": args.seed,
                     "label_column": args.label_column, "weight": str(best_path)})

    out = RESULT_ROOT / "physionet_train_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[done] 训练汇总 -> {out}")


if __name__ == "__main__":
    main()
