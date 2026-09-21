# -*- coding: utf-8 -*-
"""
PhysioNet 已知噪声增强评测（与 CQ500 的 07 同构）

旧版两个问题（与 CQ500 早期完全相同）：
  1) base_dir 恒为 ENHANCED_ROOT/{method}，循环里 5 个噪声读到同一批图
     -> 5 行数字完全相同（路径缺噪声层级的连锁反应）。
  2) 在 **test** 上挑最优 filter 又在 **test** 上报告同一数字 —— 选择性偏差，
     用它当"最优 filter"基线会让 16 的最优性比较变成循环论证。

现在：
  - 全程内存态穷举（复用 physionet_commons.sweep_filters，即 CQ500 那份实现）。
  - **拟合 phase**：在 val 上对每个噪声条件穷举 10 个 filter，选出最优。
  - **评估 phase**：用 val 拟合出的 filter 在 test 上评估，并给出全网格表。

输出: results/enhanced_known_noise_physionet/
     best_filter_map_physionet.csv            <- 11 消费的 类别->filter 表
     best_filter_per_known_noise_physionet.csv
     all_enhanced_known_noise_physionet.csv   <- test 全网格
     val_grid_physionet.csv                   <- val 全网格（拟合依据）

用法:
    python 07_evaluate_enhanced_known_noise_physionet.py
    python 07_evaluate_enhanced_known_noise_physionet.py --val-limit 400 --noises gaussian
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import torch

from physionet_commons import (DETECTOR_CLASSES, FILTER_NAMES, MODEL_DIR, NOISY_ROOT,
                          RESULT_ROOT, load_index, load_model, patient_vote_metrics,
                          sweep_filters)

MODELS = {
    "resnet50": ("resnet50", MODEL_DIR / "resnet50_physionet_best.pth"),
    "swin_tiny": ("swin_tiny_patch4_window7_224",
                  MODEL_DIR / "swin_tiny_patch4_window7_224_physionet_best.pth"),
}

# 需要拟合 filter 的噪声条件 = 判别器的 7 个类别（clean 也要）
CONDITIONS = list(DETECTOR_CLASSES)
OUTDIR = RESULT_ROOT / "enhanced_known_noise_physionet"


def assert_split_disjoint(a, b):
    df = load_index()
    sa = set(df[df["split"] == a]["patient_id"])
    sb = set(df[df["split"] == b]["patient_id"])
    inter = sa & sb
    assert not inter, f"患者级泄漏: {a} ∩ {b} = {sorted(inter)[:5]}"
    print(f"[check] {a}({len(sa)}) ∩ {b}({len(sb)}) = ∅  ✓")


def run_sweep(model, condition, split, limit, device, num_workers):
    """val 一律内存加噪（避免为 val 落盘）；test 读 04 已生成的加噪图。"""
    from_disk = (split == "test") and (condition != "clean")
    return sweep_filters(
        split=split, noise_name=condition, filter_names=FILTER_NAMES,
        model=model, device=device, num_workers=num_workers,
        from_disk=from_disk, seed=2024, limit=limit)


def grid_rows(sweeps, model_name, condition, split):
    rows = []
    for method, sdf in sweeps.items():
        m = patient_vote_metrics(sdf, tag=f"{model_name}-{condition}-{method}")
        rows.append({
            "model": model_name, "noise_type": condition, "enhancement_method": method,
            "split": split, "num_patients": m["num_patients"],
            "n_classes": int(sdf["label"].nunique()),
            "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "auc": m["auc"],
            "slice_accuracy": m["slice_accuracy"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noises", default=",".join(CONDITIONS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    conditions = [c.strip() for c in args.noises.split(",") if c.strip()]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {k: v for k, v in MODELS.items() if k in wanted}
    if not models:
        raise SystemExit(f"--models 未匹配到任何模型，可选: {list(MODELS)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    assert_split_disjoint("val", "test")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    val_rows, test_rows, map_rows = [], [], []
    for model_name, (arch, wpath) in models.items():
        if not wpath.exists():
            raise FileNotFoundError(f"缺少权重 {wpath}，请先运行 02_train_physionet.py")
        model = load_model(arch, wpath, device)
        print(f"\n===== {model_name} =====")

        for condition in conditions:
            print(f"[fit] {model_name} / {condition} / val")
            val_sweeps = run_sweep(model, condition, "val", args.val_limit,
                                   device, args.num_workers)
            rows = grid_rows(val_sweeps, model_name, condition, "val")
            val_rows.extend(rows)
            val_best = max(rows, key=lambda r: (r["accuracy"], r["f1"]))
            print(f"  -> val 最优 filter: {val_best['enhancement_method']} "
                  f"(acc={val_best['accuracy']:.4f})")

            print(f"[eval] {model_name} / {condition} / test")
            test_sweeps = run_sweep(model, condition, "test", args.test_limit,
                                    device, args.num_workers)
            trows = grid_rows(test_sweeps, model_name, condition, "test")
            test_rows.extend(trows)
            tmap = {r["enhancement_method"]: r for r in trows}
            chosen = tmap[val_best["enhancement_method"]]
            print(f"  -> 该 filter 在 test 上 acc={chosen['accuracy']:.4f} "
                  f"(test 全网格最优: "
                  f"{max(trows, key=lambda r: r['accuracy'])['enhancement_method']})")

            map_rows.append({
                "model": model_name, "noise_type": condition,
                "enhancement_method": val_best["enhancement_method"],
                "fitted_on": "val", "val_accuracy": val_best["accuracy"],
                "val_f1": val_best["f1"],
                "test_accuracy": chosen["accuracy"],
                "test_f1": chosen["f1"], "test_auc": chosen["auc"],
            })

    val_df, test_df, map_df = (pd.DataFrame(val_rows), pd.DataFrame(test_rows),
                               pd.DataFrame(map_rows))

    # 回归保护：旧版路径缺噪声层级的签名是**所有**噪声拿到同一批图
    # -> 全部条件的 (filter->accuracy) 向量完全一致。
    # 只要求"两两不同"会在小样本上误报（15 例患者时准确率只有 ~16 个取值，
    # 10 个 filter 撞车很正常），所以只在**全部条件都相同**时才判定为 bug。
    n_pat = int(test_df["num_patients"].max())
    n_cls = int(test_df["n_classes"].max())
    non_degenerate = n_pat >= 5 and n_cls >= 2
    for model_name in map_df["model"].unique():
        vectors = {n: tuple(sorted(
            test_df[(test_df["model"] == model_name) & (test_df["noise_type"] == n)]
            ["accuracy"].round(10).tolist()))
            for n in map_df[map_df["model"] == model_name]["noise_type"]}
        distinct = set(vectors.values())
        if len(distinct) == 1 and len(vectors) > 1 and non_degenerate:
            raise AssertionError(
                f"{model_name}: 全部 {len(vectors)} 个噪声的 (filter->accuracy) 向量"
                f"完全相同 —— 疑似增强图路径缺噪声层级")
        if len(distinct) < len(vectors):
            dup = [n for n, v in vectors.items() if list(vectors.values()).count(v) > 1]
            print(f"[note] {model_name}: 噪声 {dup} 指标相同；当前测试集仅 {n_pat} 例/"
                  f"{n_cls} 类，准确率可取的值很少，多为小样本巧合")
        else:
            print(f"[check] {model_name}: 各噪声的 filter 指标向量互不相同 ✓")

    val_df.to_csv(OUTDIR / "val_grid_physionet.csv", index=False)
    test_df.to_csv(OUTDIR / "all_enhanced_known_noise_physionet.csv", index=False)
    map_df.to_csv(OUTDIR / "best_filter_map_physionet.csv", index=False)
    map_df.to_csv(OUTDIR / "best_filter_per_known_noise_physionet.csv", index=False)

    print("\n[map] val 拟合的 类别->filter 表:")
    print(map_df[["model", "noise_type", "enhancement_method",
                  "val_accuracy", "test_accuracy"]].to_string(index=False))
    print(f"\n[done] 输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
