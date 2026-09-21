# -*- coding: utf-8 -*-
"""
PhysioNet 图像整理：brain-window JPEG -> png/{split}/{label}/{patient_id}/sXXX.png

与 CQ500/RSNA 的关键差别：
  本数据集**没有 DICOM**，作者提供的就是已经用 Siemens syngo 窗好的
  650x650 JPEG（brain / bone 两套）。所以我们不复做 HU 换算与脑窗，
  只取 brain-window 图并缩放到 224x224。

  => 三套数据的预处理"同类但非同一实现"：
     CQ500 / RSNA 是我们自己按 WL40/WW80 从 DICOM 生成的；
     PhysioNet 的脑窗由上游软件决定、数据集未给出具体数值。
     并排比较时这一点必须声明（physionet_verify.py 的 F1 会明确标出）。

命名: s{层号:03d}.png —— 保留原始层号，便于与 hemorrhage_diagnosis.csv 对照。

输出:
  PhysioNet/png/{split}/{label}/{patient_id}/sXXX.png
  PhysioNet/physionet_slices_index.csv
    列: patient_id, split, label, slice_label, image_path

用法:
    python physionet_images_to_png.py
    python physionet_images_to_png.py --clean            # 先清空 png/
    python physionet_images_to_png.py --limit 5          # 冒烟测试
"""
import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent
# 路径全部取自 commons —— 它会根据 PHYSIONET_FOLD 自动切到 png_fold{k} 等
PNG_ROOT = None          # 在 main() 里从 commons 取（需先设好 FOLD）
INDEX_CSV = None


def _convert_one(task):
    """(src, dst, size) -> 成功返回 dst 的相对路径片段，失败返回 None。"""
    import cv2
    src, dst, size = task
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img3)
    return str(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="转换前删除整个 png/")
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 例患者（冒烟测试）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # 延迟导入：只有真正转换时才需要读原始数据/标签
    from physionet_commons import (IMG_SIZE, DATA_ROOT as PNG_ROOT_IN,
                                   INDEX_CSV as INDEX_CSV_IN, SPLIT_CSV,
                                   fold_tag, raw_dataset_dir)
    global PNG_ROOT, INDEX_CSV
    PNG_ROOT, INDEX_CSV = PNG_ROOT_IN, INDEX_CSV_IN
    print(f"[fold] {fold_tag()}   输出 {PNG_ROOT.name}/ ，索引 {INDEX_CSV.name}")

    labels_csv = ROOT / "physionet_slice_labels.csv"
    for p in (SPLIT_CSV, labels_csv):
        if not p.exists():
            raise SystemExit(f"缺少 {p.name}；请先运行 "
                             f"physionet_make_split.py / physionet_prepare_labels.py")

    import pandas as pd
    split = pd.read_csv(SPLIT_CSV)
    lab = pd.read_csv(labels_csv)
    RAW = raw_dataset_dir()
    print(f"[raw] {RAW}")

    d = lab.merge(split, on="patient_id", how="inner")
    # **患者级**标签：任一层出血即 Hemorrhagic。这是目录与 label 列的依据，
    # 必须与 CQ500/RSNA 同口径；若误用逐层标签，同一患者会被拆进
    # Normal 与 Hemorrhagic 两个目录，评估用的 label 也就错了。
    pat_any = lab.groupby("patient_id")["any"].max()
    d["label"] = d["patient_id"].map(pat_any).map({1: "Hemorrhagic", 0: "Normal"})
    d["slice_label"] = d["any"].astype(int)      # 逐层原始标注，另存一列供消融
    patients = sorted(d["patient_id"].unique())
    if args.limit:
        patients = patients[:args.limit]
        d = d[d["patient_id"].isin(patients)]

    # 自检：同一患者只能有一个 label，否则说明又混进了逐层标签
    nl = d.groupby("patient_id")["label"].nunique()
    if int(nl.max()) != 1:
        raise SystemExit(f"有 {int((nl > 1).sum())} 例患者的 label 不唯一 —— "
                         f"标签口径错误")

    if args.clean and PNG_ROOT.exists():
        n = sum(1 for _ in PNG_ROOT.rglob("*") if _.is_file())
        print(f"[clean] 删除旧 png/（{n} 个文件）")
        shutil.rmtree(PNG_ROOT)

    tasks, rows = [], []
    for _, r in d.iterrows():
        pid = int(r["patient_id"])
        src = RAW / "Patients_CT" / f"{pid:03d}" / "brain" / f"{int(r['slice_no'])}.jpg"
        if not src.exists():
            continue
        rel = f"{r['split']}/{r['label']}/{pid:03d}/s{int(r['slice_no']):03d}.png"
        tasks.append((src, PNG_ROOT / rel, IMG_SIZE))
        rows.append({"patient_id": f"{pid:03d}", "split": r["split"],
                     "label": r["label"], "slice_label": int(r["slice_label"]),
                     "image_path": rel})

    print(f"[start] {len(tasks)} 层（{len(patients)} 例）", flush=True)
    t0 = time.time()
    done = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, ok in enumerate(ex.map(_convert_one, tasks, chunksize=64), 1):
                done += bool(ok)
                if i % 500 == 0:
                    print(f"[progress] {i}/{len(tasks)}", flush=True)

    if done != len(tasks):
        print(f"[warn] 期望 {len(tasks)} 张，实际写出 {done} 张")

    idx = pd.DataFrame(rows).sort_values(["split", "patient_id", "image_path"])
    idx = idx.reset_index(drop=True)
    idx.to_csv(INDEX_CSV, index=False)

    print(f"\n[done] {done} 张 PNG，{len(patients)} 例，{time.time()-t0:.0f}s")
    print("[索引统计]")
    print(idx.groupby(["split", "label"]).agg(
        patients=("patient_id", "nunique"), slices=("image_path", "count")).to_string())
    print(f"切片级阳性率: {idx['slice_label'].mean():.4f}"
          f"（对比 CQ500 ~0.42-0.46，语义不同，见 physionet_prepare_labels.py）")
    print(f"\n[out] {INDEX_CSV}")


if __name__ == "__main__":
    main()
