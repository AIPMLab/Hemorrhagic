# -*- coding: utf-8 -*-
"""
CQ500 已知噪声增强评测（重写）

旧版两个致命问题：
  1) base_dir 恒为 ENHANCED_ROOT/{method}，循环里 5 个噪声读到同一批图
     -> 5 行数字完全相同（路径缺噪声层级的连锁反应）。
  2) 在 **test** 上挑最优 filter 又在 **test** 上报告同一数字 —— 选择性偏差，
     用它当"最优 filter"基线会让 16 的最优性比较变成循环论证。

现在：
  - 全程内存态穷举（单次解码 -> 全 filter 前向），不依赖 06 落盘，省下约 120 GB。
  - **拟合 phase**：在 val 上、对每个噪声条件穷举 10 个 filter，选出最优 ->
    best_filter_per_known_noise_cq500.csv（表内标注 fitted_on=val）。
  - **评估 phase**：用 val 拟合出的 filter，在 test 上评估并给出全网格表。

输出:
  results/enhanced_known_noise_cq500/
    best_filter_map_cq500.csv           <- 11 消费的 类别->filter 表
    best_filter_per_known_noise_cq500.csv
    all_enhanced_known_noise_cq500.csv  <- test 全网格（visualize_cq500.py 消费）
    val_grid_cq500.csv                  <- val 全网格（拟合依据，供审阅）

用法:
    python 07_evaluate_enhanced_known_noise_cq500.py
    python 07_evaluate_enhanced_known_noise_cq500.py --val-limit 400 --noises gaussian
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import torch

from cq500_commons import (DETECTOR_CLASSES, FILTER_NAMES, MODEL_DIR,
                           RESULT_ROOT, load_index, load_model, patient_vote_metrics,
                           sweep_filters)

MODELS = {
    "resnet50": ("resnet50", MODEL_DIR / "resnet50_cq500_best.pth"),
    "swin_tiny": ("swin_tiny_patch4_window7_224",
                  MODEL_DIR / "swin_tiny_patch4_window7_224_cq500_best.pth"),
}

# 需要拟合 filter 的噪声条件 = 判别器的 7 个类别（clean 也要，别假设它直接选 none）
CONDITIONS = [c for c in DETECTOR_CLASSES]

OUTDIR = RESULT_ROOT / "enhanced_known_noise_cq500"


def assert_split_disjoint(a, b):
    df = load_index()
    sa = set(df[df["split"] == a]["patient_id"])
    sb = set(df[df["split"] == b]["patient_id"])
    inter = sa & sb
    assert not inter, f"患者级泄漏: {a} ∩ {b} = {inter}"
    print(f"[check] {a}({len(sa)}) ∩ {b}({len(sb)}) = ∅  ✓")


def run_sweep(model, condition, split, limit, device, num_workers):
    """val 一律在内存加噪（否则要为 val 另落盘 25,480 x 6 张图）；
    test 读 04 已生成的加噪图，保证与 05/11/16 用的是同一批噪声实现。
    condition == 'clean' 时读原图，等价于不加噪。"""
    from_disk = (split == "test") and (condition != "clean")
    return sweep_filters(
        split=split, noise_name=condition, filter_names=FILTER_NAMES,
        model=model, device=device, num_workers=num_workers,
        from_disk=from_disk, seed=2024, limit=limit,
    )


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
    ap.add_argument("--models", default=",".join(MODELS),
                    help="逗号分隔的模型列表，默认全部")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="val 拟合阶段仅用前 N 张切片（0=全部）；快速验证用")
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader 进程数：全部 CPU 预处理在 worker 内并行，越大越能喂饱 GPU")
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
    if args.val_limit or args.test_limit:
        print(f"[limit] val_limit={args.val_limit} test_limit={args.test_limit}")

    val_rows, test_rows, map_rows = [], [], []

    for model_name, (arch, wpath) in models.items():
        if not wpath.exists():
            raise FileNotFoundError(f"缺少权重 {wpath}，请先运行 02_train_cq500.py")
        model = load_model(arch, wpath, device)
        print(f"\n===== {model_name} =====")

        for condition in conditions:
            # ---------- 拟合 phase: val ----------
            print(f"[fit] {model_name} / {condition} / val")
            val_sweeps = run_sweep(model, condition, "val", args.val_limit,
                                   device, args.num_workers)
            rows = grid_rows(val_sweeps, model_name, condition, "val")
            val_rows.extend(rows)
            val_best = max(rows, key=lambda r: (r["accuracy"], r["f1"]))
            print(f"  -> val 最优 filter: {val_best['enhancement_method']} "
                  f"(acc={val_best['accuracy']:.4f})")

            # ---------- 评估 phase: test ----------
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

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    map_df = pd.DataFrame(map_rows)

    # 回归保护：旧版路径缺噪声层级的签名是**所有**噪声拿到同一批图
    # -> 全部条件的 (filter->accuracy) 向量完全一致。
    # 只要求"两两不同"会在小样本上误报（患者数少时准确率可取的值很少），
    # 所以只在**全部条件都相同**时才判定为 bug。
    n_pat = int(test_df["num_patients"].max())
    n_cls = int(test_df["n_classes"].max())
    non_degenerate = n_pat >= 5 and n_cls >= 2

    for model_name in map_df["model"].unique():
        sub = map_df[map_df["model"] == model_name]
        vectors = {n: tuple(sorted(
            test_df[(test_df["model"] == model_name) & (test_df["noise_type"] == n)]
            ["accuracy"].round(10).tolist()))
            for n in sub["noise_type"]}
        distinct = set(vectors.values())
        if len(distinct) == 1 and len(vectors) > 1 and non_degenerate:
            raise AssertionError(
                f"{model_name}: 全部 {len(vectors)} 个噪声的 (filter->accuracy) 向量"
                f"完全相同 —— 疑似增强图路径缺噪声层级（回归 B1）")
        if len(distinct) < len(vectors):
            dup = [n for n, v in vectors.items() if list(vectors.values()).count(v) > 1]
            print(f"[note] {model_name}: 噪声 {dup} 指标相同；当前测试集仅 {n_pat} 例/"
                  f"{n_cls} 类，准确率可取的值很少，多为小样本巧合")
        else:
            print(f"[check] {model_name}: 各噪声的 filter 指标向量互不相同 ✓")

    val_df.to_csv(OUTDIR / "val_grid_cq500.csv", index=False)
    test_df.to_csv(OUTDIR / "all_enhanced_known_noise_cq500.csv", index=False)
    map_df.to_csv(OUTDIR / "best_filter_map_cq500.csv", index=False)
    map_df.to_csv(OUTDIR / "best_filter_per_known_noise_cq500.csv", index=False)

    print("\n[map] val 拟合的 类别->filter 表:")
    print(map_df[["model", "noise_type", "enhancement_method",
                  "val_accuracy", "test_accuracy"]].to_string(index=False))
    print(f"\n[done] 输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
