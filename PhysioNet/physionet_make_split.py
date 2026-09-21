# -*- coding: utf-8 -*-
"""
PhysioNet 划分构建（**按患者**分层抽样 + 70/15/15，可选 K 折）

数据来源: physionet_slice_labels.csv（逐层标注）

抽样策略:
  患者级标签 = 该患者**任一层**阳性则阳性（与 CQ500/RSNA 同口径）。
  按患者数分层划分；本数据集患者级阳性率 43.9%，与 CQ500(41.6%)、RSNA(40.4%) 接近，
  因此三者类别比例天然可比。

**重要：本数据集只有 82 例患者。**
  70/15/15 划分后 test 仅约 12 例 —— 患者级准确率只能取 1/12 的整数倍（步长 8.3%），
  配对检验（McNemar）几乎没有统计功效。这是数据集本身的规模限制，不是代码问题。
  两个选择：
    (a) 沿用 70/15/15（与另两套数据集口径一致），论文中如实报告 test N=12；
    (b) 用 --n-folds K 生成 K 折患者级交叉验证分配（原论文即用 5 折），
        每个患者都被评估一次，可报 mean±std，统计上更稳。
  本脚本默认 (a)，同时把 --n-folds >1 时的折分配写到 physionet_folds.csv 备用。

输出（均在 PhysioNet/）：
  physionet_split.csv              patient_id -> split
  {train,val,test}_patients.csv
  physionet_split_summary.csv      各集合患者数/层数
  physionet_sampling_frame.csv     全部患者及其是否入选
  physionet_dataset_provenance.csv 数据集来源与规模记账
  physionet_folds.csv              仅当 --n-folds > 1

用法:
    python physionet_make_split.py
    python physionet_make_split.py --n-folds 5
    python physionet_make_split.py --dry-run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS_CSV = ROOT / "physionet_slice_labels.csv"
SPLIT_CSV = ROOT / "physionet_split.csv"
SUMMARY_CSV = ROOT / "physionet_split_summary.csv"
FRAME_CSV = ROOT / "physionet_sampling_frame.csv"
PROV_CSV = ROOT / "physionet_dataset_provenance.csv"
FOLDS_CSV = ROOT / "physionet_folds.csv"

SEED = 42
RATIO_TRAIN, RATIO_VAL = 0.70, 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=1,
                    help=">1 时额外生成 K 折患者级交叉验证分配（原论文用 5）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not LABELS_CSV.exists():
        raise SystemExit("缺少 physionet_slice_labels.csv；请先运行 "
                         "physionet_prepare_labels.py")
    lab = pd.read_csv(LABELS_CSV)

    pat = lab.groupby("patient_id").agg(
        any=("any", "max"), n_slices=("any", "count")).reset_index()
    print(f"[患者] {len(pat)} 例，患者级阳性率 {pat['any'].mean():.4f}"
          f"（阳性 {int(pat['any'].sum())}）")
    q = pat["n_slices"]
    print(f"[每患者层数] min={q.min()} P25={q.quantile(.25):.0f} 中位={q.median():.0f} "
          f"P75={q.quantile(.75):.0f} max={q.max()} 平均={q.mean():.1f}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for cls in (1, 0):
        # to_numpy() 有时返回只读视图，shuffle 会报 "array is read-only"，故显式复制
        ids = pat[pat["any"] == cls]["patient_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        k = len(ids)
        k_tr, k_va = int(k * RATIO_TRAIN), int(k * RATIO_VAL)
        rows.append(pd.DataFrame({
            "patient_id": ids,
            "split": ["train"] * k_tr + ["val"] * k_va + ["test"] * (k - k_tr - k_va),
        }))
    split = pd.concat(rows, ignore_index=True)

    s = {sp: set(split[split["split"] == sp]["patient_id"]) for sp in
         ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if s[a] & s[b]:
            raise SystemExit(f"患者级泄漏！{a} ∩ {b} = {sorted(s[a] & s[b])[:5]}")
    print("[check] 患者两两互斥 ✓")

    merged = pat.merge(split, on="patient_id")
    summary = (merged.groupby(["split", "any"]).agg(
        patients=("patient_id", "nunique"), slices=("n_slices", "sum"))
        .reset_index().sort_values(["split", "any"]))
    print("\n[划分]")
    print(summary.to_string(index=False))
    tot_pat, tot_sli = int(summary["patients"].sum()), int(summary["slices"].sum())
    n_test = len(s["test"])
    print(f"       合计 {tot_pat} 例 / {tot_sli} 层")
    if n_test < 20:
        print(f"       [注意] test 仅 {n_test} 例：患者级准确率的步长为 "
              f"{1/n_test*100:.1f}%，配对检验功效很低。可考虑 --n-folds 做交叉验证。")

    if args.dry_run:
        print("\n[dry-run] 未写任何文件")
        return

    split.to_csv(SPLIT_CSV, index=False)
    for sp in ("train", "val", "test"):
        split[split["split"] == sp][["patient_id"]].to_csv(
            ROOT / f"{sp}_patients.csv", index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    frame = pat.copy()
    frame["sampled"] = True
    frame["excluded_reason"] = "kept"
    frame.to_csv(FRAME_CSV, index=False)

    pd.DataFrame([
        {"item": "原始 hemorrhage_diagnosis.csv 切片数", "value": len(lab)},
        {"item": "全库患者数", "value": len(pat)},
        {"item": "患者级阳性数", "value": int(pat["any"].sum())},
        {"item": "患者级阳性率", "value": round(float(pat["any"].mean()), 4)},
        {"item": "切片级阳性率", "value": round(float(lab["any"].mean()), 4)},
        {"item": "每患者层数 (mean)", "value": round(float(pat["n_slices"].mean()), 1)},
        {"item": "划分随机种子", "value": args.seed},
        {"item": "划分比例 train/val/test (按患者分层)", "value": "0.70/0.15/0.15"},
        {"item": "划分粒度", "value": "patient_id"},
        {"item": "test 患者数", "value": n_test},
    ]).to_csv(PROV_CSV, index=False)

    if args.n_folds > 1:
        k = args.n_folds
        fold_rows = []
        for cls in (1, 0):
            ids = pat[pat["any"] == cls]["patient_id"].to_numpy(copy=True)
            rng.shuffle(ids)
            # 轮流分配，使各折的类别比例尽量接近
            for i, pid in enumerate(ids):
                fold_rows.append({"patient_id": pid, "any": int(cls),
                                  "fold": i % k})
        folds = pd.DataFrame(fold_rows).sort_values(["fold", "patient_id"])
        folds.to_csv(FOLDS_CSV, index=False)
        print(f"\n[折] {k} 折患者级交叉验证分配 -> {FOLDS_CSV.name}")
        print(folds.groupby(["fold", "any"]).size().unstack(fill_value=0).to_string())

        # ---- 为每一折写出独立的 train/val/test 划分 ----
        # 折 k 作 test；折 (k+1)%K 作 val（用于拟合 类别->filter 映射与选模型）；
        # 其余 K-2 折作 train。三者按患者严格互斥。
        for f in range(k):
            val_fold = (f + 1) % k
            rows = []
            for pid, fd in zip(folds["patient_id"], folds["fold"]):
                sp = ("test" if fd == f else
                      "val" if fd == val_fold else "train")
                rows.append({"patient_id": pid, "split": sp})
            fdf = pd.DataFrame(rows)
            s = {sp: set(fdf[fdf["split"] == sp]["patient_id"])
                 for sp in ("train", "val", "test")}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                if s[a] & s[b]:
                    raise SystemExit(f"fold {f}: 患者级泄漏 {a}∩{b}")
            fdf.to_csv(ROOT / f"physionet_split_fold{f}.csv", index=False)
            print(f"      fold {f}: test={len(s['test'])} val={len(s['val'])} "
                  f"train={len(s['train'])} 例  -> physionet_split_fold{f}.csv")
        print(f"\n[用法] 用 PHYSIONET_FOLD=k 选择第 k 折（见 physionet_commons.py）：")
        print(f"       PHYSIONET_FOLD=0 python physionet_images_to_png.py")
        print(f"       PHYSIONET_FOLD=0 python run_physionet_pipeline.py")
        print(f"       或用驱动脚本一次跑完：python physionet_run_cv.py")

    print(f"\n[out] {SPLIT_CSV.name} / {{train,val,test}}_patients.csv / "
          f"{SUMMARY_CSV.name} / {FRAME_CSV.name} / {PROV_CSV.name}")
    print("\n[提示] 下一步: python physionet_images_to_png.py")


if __name__ == "__main__":
    main()
