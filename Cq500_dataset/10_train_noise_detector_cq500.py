# -*- coding: utf-8 -*-
"""
CQ500 噪声判别器训练（EfficientNet-B0，两个变体对比）

变体（论文中互相对比，共用 09 生成的同一份 7 类数据集，只是取的类别子集不同）：
  6class —— clean + 5 种单噪声，共 6 类。复合噪声对它完全是**分布外**：
            它只能把复合噪声硬分到某个已知单噪声类，因此永远无法路由到"复合噪声最优 filter"。
  7class —— 额外把复合未知噪声 unknown_mixed 作为第 7 类显式学习，
            因而能直接识别复合噪声并路由到对应 filter。

训练后除自身验证集外，**一律再喂全部 7 种条件**做逐类评估：
  对 7class 是有意义的准确率；对 6class 故意暴露它在复合噪声上的行为
  （准确率≈0 且只能预测成某个已知类），这正是两者对比的核心证据。

按患者划分判别器自身 train/val：沿用 09 manifest 的 source_split
（CQ500 train 患者 -> 训练；CQ500 val 患者 -> 验证），两侧都不含 test 患者。

权重: models/noise_detector_efficientnetb0_cq500_{variant}_best.pth
输出: results/noise_detector_cq500/{variant}/

用法:
    python 10_train_noise_detector_cq500.py --variant 6class
    python 10_train_noise_detector_cq500.py --variant 7class
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cq500_commons import (DETECTOR_CLASSES, DETECTOR_DATASET_DIR, DETECTOR_VARIANTS,
                           create_model_with_pretrained, detector_result_dir,
                           detector_weight_path)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
LR = 2e-4
WEIGHT_DECAY = 1e-4


class DetectorDataset(Dataset):
    def __init__(self, df, transform, class_to_idx):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(DETECTOR_DATASET_DIR / row["file"]).convert("RGB")
        # 用 get(..., -1)：评估"分布外"条件（如 6class 变体遇到 unknown_mixed）时
        # 该类别不在词表内，标签无意义，返回 -1 而不是 KeyError。
        return self.transform(img), self.class_to_idx.get(row["class"], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(DETECTOR_VARIANTS),
                    help="6class=只学单一噪声；7class=额外学复合未知噪声")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="每类仅取前 N 张（冒烟测试用）")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不加载 ImageNet 预训练权重（冒烟测试用；结果无科学意义）")
    args = ap.parse_args()

    variant = args.variant
    classes = DETECTOR_VARIANTS[variant]
    weight_path = detector_weight_path(variant)
    outdir = detector_result_dir(variant)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = DETECTOR_DATASET_DIR / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"缺少 {manifest}，请先运行 09_build_noise_detector_dataset_cq500.py")
    full = pd.read_csv(manifest)

    df = full[full["class"].isin(classes)].reset_index(drop=True)
    if args.limit:
        df = df.groupby("class", group_keys=False).head(args.limit).reset_index(drop=True)

    train_df = df[df["source_split"] == "train"].reset_index(drop=True)
    val_df = df[df["source_split"] == "val"].reset_index(drop=True)
    tr_pat, va_pat = set(train_df["patient_id"]), set(val_df["patient_id"])
    assert not (tr_pat & va_pat), f"判别器 train/val 患者重叠: {tr_pat & va_pat}"
    print(f"[variant] {variant}  类别({len(classes)})={classes}")
    print(f"[check] train 患者({len(tr_pat)}) ∩ val 患者({len(va_pat)}) = ∅  ✓")
    print(f"[data] train {len(train_df)} 张 / val {len(val_df)} 张")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    class_to_idx = {c: i for i, c in enumerate(classes)}
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

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=args.num_workers > 0,
                     prefetch_factor=4 if args.num_workers > 0 else None)
    loader_kw = {k: v for k, v in loader_kw.items() if v is not None}

    train_loader = DataLoader(DetectorDataset(train_df, train_tf, class_to_idx),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(DetectorDataset(val_df, eval_tf, class_to_idx),
                            batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = create_model_with_pretrained(
        "efficientnet_b0", len(classes),
        pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def evaluate_loader(loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in loader:
                preds = model(x.to(device)).argmax(1)
                y_true.extend(y.numpy())
                y_pred.extend(preds.cpu().numpy())
        return y_true, y_pred

    best_acc, best_cm, history = -1.0, None, []
    for epoch in range(args.epochs):
        model.train()
        # 累加器留在 GPU 上，避免每个 batch 一次 .item() 同步打断流水线
        n = 0
        loss_t = torch.zeros((), device=device)
        for x, y in tqdm(train_loader, desc=f"[{variant}] epoch {epoch+1}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            n += y.size(0)
            loss_t += loss.detach() * y.size(0)
        total_loss = float(loss_t.item())

        y_true, y_pred = evaluate_loader(val_loader)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"[{variant}] epoch {epoch+1}: loss={total_loss/max(n,1):.4f} "
              f"acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        history.append({"variant": variant, "epoch": epoch + 1,
                        "loss": total_loss / max(n, 1), "accuracy": acc,
                        "precision_macro": prec, "recall_macro": rec, "f1_macro": f1})

        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "class_names": classes,
                        "variant": variant, "arch": "efficientnet_b0"}, weight_path)
            print(f"    -> 新最优 acc={acc:.4f}，已保存 {weight_path.name}")

    pd.DataFrame(history).to_csv(outdir / "training_log.csv", index=False)

    cm_df = pd.DataFrame(best_cm, index=classes, columns=classes)
    cm_df.to_csv(outdir / "confusion_matrix.csv")
    per_class = pd.DataFrame({
        "class": classes,
        "accuracy": best_cm.diagonal() / np.maximum(best_cm.sum(axis=1), 1),
        "support": best_cm.sum(axis=1),
    })
    per_class.to_csv(outdir / "per_class_accuracy.csv", index=False)

    plt.figure(figsize=(8.5, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(f"Noise Detector CQ500 [{variant}] — val acc={best_acc:.4f}",
              fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12, fontweight="bold")
    plt.ylabel("Actual", fontsize=12, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 关键对比证据：喂全部 7 种条件（含复合噪声）----
    print(f"\n[{variant}] 在全 7 种条件上的表现（含分布外的复合噪声）:")
    rows = []
    for cond in DETECTOR_CLASSES:
        cdf = full[(full["class"] == cond) & (full["source_split"] == "val")]
        if args.limit:
            cdf = cdf.head(args.limit)
        if cdf.empty:
            continue
        loader = DataLoader(DetectorDataset(cdf, eval_tf, class_to_idx),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        model.eval()
        preds, confs = [], []
        with torch.no_grad():
            for x, _ in loader:
                p = torch.softmax(model(x.to(device)), dim=1)
                preds.extend(p.argmax(1).cpu().numpy())
                confs.extend(p.max(1).values.cpu().numpy())
        pred_names = pd.Series([classes[i] for i in preds])
        in_vocab = cond in classes
        # 分布外的条件没有正确的类可命中，准确率无定义 -> NaN
        acc = float((pred_names == cond).mean()) if in_vocab else float("nan")
        vc = pred_names.value_counts(normalize=True)
        rows.append({
            "variant": variant, "condition": cond, "in_vocab": in_vocab,
            "n": len(pred_names), "accuracy": acc,
            "top_pred": vc.index[0], "top_pred_share": float(vc.iloc[0]),
            "mean_confidence": float(np.mean(confs)),
        })
        tag = "在类内" if in_vocab else "分布外(无该类)"
        shown = f"{acc:.4f}" if in_vocab else " n/a "
        print(f"  {cond:15s} {tag:14s} acc={shown} "
              f"-> 最常预测为 {rows[-1]['top_pred']} "
              f"({rows[-1]['top_pred_share']:.1%}, 平均置信度 {rows[-1]['mean_confidence']:.3f})")
    pd.DataFrame(rows).to_csv(outdir / "all_condition_behavior.csv", index=False)

    print(f"\n[done] {variant} 最优 val acc={best_acc:.4f}")
    print("[per-class]\n" + per_class.to_string(index=False))
    print(f"[weights] {weight_path}")


if __name__ == "__main__":
    main()
