# -*- coding: utf-8 -*-
"""
CQ500 二分类训练（无出血 Normal / 出血 Hemorrhagic）
- ResNet-50 与 Swin-Tiny，切片级训练，患者级验证投票
- 权重保存为 models/*_cq500_best.pth（不与废弃的 dataset1 混淆）

用法:
    python 02_train_cq500.py
    python 02_train_cq500.py --epochs 1 --limit 1200     # 冒烟测试
    python 02_train_cq500.py --models resnet50           # 只训一个模型
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

from cq500_commons import (CLASSES, MODEL_DIR, NUM_CLASSES, RESULT_ROOT,
                           PatientSliceDataset, create_model_with_pretrained,
                           patient_vote_metrics, predict_slices)

ALL_MODELS = ["resnet50", "swin_tiny_patch4_window7_224"]
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15


def worker_init_fn(worker_id):
    """让每个 worker 的增强随机流由主进程种子派生。

    这样 train 增强的可复现性与 num_workers 无关 —— 否则调 worker 数就会悄悄改变
    训练轨迹，论文结果无法对齐。
    """
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=0,
                    help="train/val 各取前 N 张切片（冒烟测试用；0=全部）")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="注意：改动 batch size 会改变训练结果，需全程一致")
    ap.add_argument("--num-workers", type=int, default=6,
                    help="DataLoader 进程数（本机 16 逻辑核，原为 2 导致 GPU 饥饿）")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="每 N 个 epoch 做一次验证集患者级评估（默认每轮）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不加载 ImageNet 预训练权重（离线/冒烟测试用；结果无科学意义）")
    args = ap.parse_args()

    # 固定种子，保证与 worker 数无关的可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    models_to_train = [m.strip() for m in args.models.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | batch={args.batch_size} workers={args.num_workers} seed={args.seed}")

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

    trainset = PatientSliceDataset("train", transform=train_tf, limit=args.limit)
    valset = PatientSliceDataset("val", transform=eval_tf, limit=args.limit)
    print(f"[data] train {len(trainset)} 切片 / val {len(valset)} 切片")

    # 切片级逆频率加权采样（类别不均）
    labels = trainset.df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    print(f"[data] 类别计数 {dict(zip(CLASSES, counts.astype(int)))}")
    weights = 1.0 / np.maximum(counts, 1)[labels]
    # pin_memory + non_blocking 让 H2D 拷贝与 GPU 计算重叠；
    # persistent_workers 避免每轮重新 spawn（Windows spawn 每次起解释器很贵）；
    # prefetch_factor 让 worker 提前备好若干 batch，填住 GPU 空隙。
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
    summary_path = RESULT_ROOT / "cq500_train_summary.csv"
    rows = []

    for model_name in models_to_train:
        print(f"\n========== 训练 {model_name} (CQ500 二分类) ==========")
        model = create_model_with_pretrained(model_name, NUM_CLASSES,
                                             pretrained=not args.no_pretrained)
        model.to(device)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best_f1, best_path = -1.0, MODEL_DIR / f"{model_name}_cq500_best.pth"
        for ep in range(args.epochs):
            model.train()
            # 累加器放在 GPU 上：原来每个 batch 调 2 次 .item()，每次都会强制同步，
            # 把 GPU 流水线打断，是 GPU 利用率长期只有 40% 的元凶之一。
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
            # 每轮只同步一次
            tot = n_seen
            corr = int(corr_t.item())
            loss_sum = float(loss_t.item())
            pbar.close()
            print(f"[{model_name}] epoch {ep+1}/{args.epochs} | "
                  f"loss {loss_sum/max(tot,1):.4f} | train_acc {corr/max(tot,1):.4f}")

            # 验证集患者级投票：跑到整个 val（23,499 切片）是一次完整前向，
            # 默认每轮做；--eval-every 可降低频率，但最后一轮一定做。
            if (ep + 1) % args.eval_every != 0 and (ep + 1) != args.epochs:
                continue

            vdf = predict_slices(model, val_loader, device)
            m = patient_vote_metrics(vdf, tag=f"{model_name}-val-ep{ep+1}")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                torch.save(model.state_dict(), best_path)
                print(f"    -> 新最优 f1={m['f1']:.4f}，已保存 {best_path}")

        print(f"[{model_name}] 最优患者级 F1={best_f1:.4f}")
        rows.append({"model": model_name, "best_patient_f1": best_f1,
                     "epochs": args.epochs, "batch_size": args.batch_size,
                     "num_workers": args.num_workers, "seed": args.seed,
                     "limit": args.limit, "weight": str(best_path)})

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n[done] 训练汇总 -> {summary_path}")
    print("\nCQ500 训练完成")


if __name__ == "__main__":
    main()
