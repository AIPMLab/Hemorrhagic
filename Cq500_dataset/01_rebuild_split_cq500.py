# -*- coding: utf-8 -*-
"""
CQ500 划分重建（按"原数据集中真实存在的患者"重新分层划分）

问题：reads.csv 有 491 例患者标签，但本地原始数据集
      D:\\Li-kai\\project\\Data\\medical\\cq500 只有 473 例患者目录，
      有 18 例（如 CQ500CT120/123/160/173/189/221/233/244/269/296/313/336/386/388/389/454/458/80）
      本地根本没有 DICOM。于是 cq500_split_summary.csv 里写的 491 例（test 76 例）
      与实际能评估的 473 例（test 74 例）对不上，论文里的 N 会不一致。

做法：
  1. 以**原始数据集实际存在的患者目录**为准，与 reads.csv 标签取交集
  2. 按患者分层 70/15/15 重新划分（seed 42，与原脚本一致）
  3. 只移动**患者目录**（473 个目录级 move，不是 16 万个文件），
     把 png/{旧split}/{label}/{pid} 挪到 png/{新split}/{label}/{pid}
  4. 重写 cq500_split.csv / {train,val,test}_patients.csv / cq500_split_summary.csv
     并重建 cq500_slices_index.csv

用法:
    python 01_rebuild_split_cq500.py --dry-run     # 先看会怎么变
    python 01_rebuild_split_cq500.py               # 实际执行
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from cq500_commons import DATA_ROOT as PNG_ROOT
from cq500_commons import ROOT

LABELS_CSV = ROOT / "cq500_patient_labels.csv"
SPLIT_CSV = ROOT / "cq500_split.csv"
INDEX_CSV = ROOT / "cq500_slices_index.csv"
SUMMARY_CSV = ROOT / "cq500_split_summary.csv"
EXCLUDED_CSV = ROOT / "cq500_excluded_patients.csv"
PROVENANCE_CSV = ROOT / "cq500_dataset_provenance.csv"

CLASSES = ["Normal", "Hemorrhagic"]
SEED = 42
RATIO_TRAIN, RATIO_VAL = 0.70, 0.15


def resolve_raw(explicit=None):
    """决定原始数据集根目录：--raw > 环境变量 CQ500_RAW_DIR > 本机历史路径。"""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"--raw 指定的路径不存在: {p}")
        return p
    try:
        from cq500_commons import raw_dataset_dir
        return raw_dataset_dir()
    except FileNotFoundError as e:
        raise SystemExit(str(e))


def raw_patients(raw_root):
    """原始数据集里真实存在的患者 ID（目录名形如 'CQ500CT0 CQ500CT0'）。"""
    ids = set()
    for d in raw_root.rglob("*"):
        if d.is_dir() and d.name.startswith("CQ500CT"):
            ids.add(d.name.split()[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告，不移动文件")
    ap.add_argument("--raw", default=None,
                    help="原始数据集根目录（默认自动选 cq500_2，回退 cq500）")
    args = ap.parse_args()

    raw_root = resolve_raw(args.raw)
    print(f"[raw] 原始数据集: {raw_root}")

    labels = pd.read_csv(LABELS_CSV)
    have = raw_patients(raw_root)
    print(f"[源] reads.csv 患者      : {len(labels)}")
    print(f"[源] 原始数据集患者目录  : {len(have)}")

    missing = sorted(set(labels["patient_id"]) - have)
    print(f"[缺] 本地无 DICOM 的患者 : {len(missing)}")
    if missing:
        print("     " + ", ".join(missing))

    avail = labels[labels["patient_id"].isin(have)].copy()
    print(f"[用] 参与划分的患者      : {len(avail)}")
    print(avail["label"].value_counts().to_string())

    # ---- 分层 70/15/15（与原脚本同一套逻辑，仅样本集合变了）----
    rng = __import__("numpy").random.default_rng(SEED)
    rows = []
    for cls in CLASSES:
        ids = avail.loc[avail["label"] == cls, "patient_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(n * RATIO_TRAIN)
        n_va = int(n * RATIO_VAL)
        rows.append(pd.DataFrame({
            "patient_id": ids,
            "split": ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va),
        }))
    new_split = pd.concat(rows, ignore_index=True)
    assert new_split["patient_id"].nunique() == len(avail), "划分患者数与可用患者数不一致"

    summary = (new_split.merge(avail[["patient_id", "label"]], on="patient_id")
               .groupby(["split", "label"]).size().reset_index(name="count")
               .sort_values(["split", "label"]))
    print("\n[新] 划分统计:")
    print(summary.to_string(index=False))

    old_split = pd.read_csv(SPLIT_CSV) if SPLIT_CSV.exists() else None
    moved, stayed, absent = [], 0, []
    for _, r in new_split.iterrows():
        pid, new_sp = r["patient_id"], r["split"]
        # 该患者当前在 png 里的位置（按现有目录找，不依赖旧 csv）
        found = None
        for sp in ("train", "val", "test"):
            for lbl in CLASSES:
                d = PNG_ROOT / sp / lbl / pid
                if d.exists():
                    found = (sp, lbl, d)
                    break
            if found:
                break
        if found is None:
            absent.append(pid)
            continue
        old_sp, lbl, src = found
        if old_sp == new_sp:
            stayed += 1
            continue
        dst = PNG_ROOT / new_sp / lbl / pid
        moved.append((pid, old_sp, new_sp, src, dst))

    print(f"\n[移动] 需要换 split 的患者: {len(moved)}  （原地不动 {stayed}，缺失目录 {len(absent)}）")
    if absent:
        print(f"       png 下找不到目录的患者: {absent[:10]}")

    if args.dry_run:
        print("\n[dry-run] 未做任何改动。示例移动:")
        for m in moved[:8]:
            print(f"    {m[0]}: {m[1]} -> {m[2]}")
    else:
        for i, (pid, old_sp, new_sp, src, dst) in enumerate(moved, 1):
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            if i % 50 == 0:
                print(f"[move] {i}/{len(moved)}", flush=True)
        print(f"[move] 完成 {len(moved)} 例")
        # 清理空的 split/label 目录
        for sp in ("train", "val", "test"):
            for lbl in CLASSES:
                d = PNG_ROOT / sp / lbl
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()

    # ---- 写回划分相关文件 ----
    if not args.dry_run:
        new_split.to_csv(SPLIT_CSV, index=False)
        for sp in ("train", "val", "test"):
            sub = new_split[new_split["split"] == sp]
            sub[["patient_id"]].to_csv(ROOT / f"{sp}_patients.csv", index=False)
        summary.to_csv(SUMMARY_CSV, index=False)

        # 重建索引（以目录为准，此时目录已按新划分就位）
        img_rows = []
        for p in PNG_ROOT.rglob("*.png"):
            parts = p.parent.parts
            img_rows.append({"patient_id": parts[-1], "split": parts[-3],
                             "label": parts[-2],
                             "image_path": p.relative_to(PNG_ROOT).as_posix()})
        idx = pd.DataFrame(img_rows)
        idx.to_csv(INDEX_CSV, index=False)

        print(f"\n[写回] {SPLIT_CSV.name} / {{train,val,test}}_patients.csv / {SUMMARY_CSV.name}")
        print(f"[重建] {INDEX_CSV.name}: {len(idx)} 行, {idx['patient_id'].nunique()} 患者")

        # ---- 记录被排除的患者：让"N = 473 / 491"这件事可审计 ----
        exc = labels[labels["patient_id"].isin(missing)][["patient_id", "label"]].copy()
        exc["reason"] = "本地原始数据集无该患者目录（无 DICOM）"
        exc.sort_values("patient_id").to_csv(EXCLUDED_CSV, index=False)

        prov = pd.DataFrame([
            {"item": "reads.csv 标注患者总数（官方 CQ500 队列）", "value": len(labels)},
            {"item": "本地原始数据集实际存在的患者目录", "value": len(have)},
            {"item": "本地缺失、被排除的患者", "value": len(missing)},
            {"item": "参与训练/验证/测试的患者", "value": len(avail)},
            {"item": "收集到的切片总数", "value": len(idx)},
            {"item": "原始数据集路径", "value": str(raw_root)},
            {"item": "划分随机种子", "value": SEED},
            {"item": "划分比例 train/val/test (按类别分层)", "value": "0.70/0.15/0.15"},
        ])
        prov.to_csv(PROVENANCE_CSV, index=False)

        assert len(avail) + len(missing) == len(labels), "患者数对不上：可用 + 排除 != 标注总数"
        print(f"[记账] {len(avail)} (用) + {len(missing)} (排除) = {len(labels)} (reads.csv)  ✓")
        print(f"[输出] {EXCLUDED_CSV.name} 与 {PROVENANCE_CSV.name}")

        print("\n[最终] 索引中的划分统计:")
        print(idx.groupby(["split", "label"]).agg(
            patients=("patient_id", "nunique"), slices=("image_path", "count")
        ).to_string())
    else:
        print("\n[dry-run] 未写回任何文件")


if __name__ == "__main__":
    main()
