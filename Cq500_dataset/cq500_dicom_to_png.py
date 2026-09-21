# -*- coding: utf-8 -*-
"""
CQ500 数据预处理（步骤4）：DICOM -> PNG
- 读取原生 DICOM，Rescale 到 HU，脑窗窗位 (WL 40 / WW 80) 截断，转 8-bit 三通道 224x224 PNG
- 切片按 z 坐标(ImagePositionPatient[2])排序，缺失时用 InstanceNumber，再缺失用文件名序
- 输出结构:
    Cq500_dataset/png/{split}/{label}/{patient_id}/frame_0001.png ...
  （ImageFolder 可直接加载 train/val/test -> Normal/Hemorrhagic -> images）
- 另输出 Cq500_dataset/cq500_slices_index.csv（patient_id, split, label, image_path）

用法:
    python cq500_dicom_to_png.py                # 全量转换
    python cq500_dicom_to_png.py --max-patients 2 --workers 2   # 小规模测试
"""
import argparse
import csv
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

from cq500_commons import raw_dataset_dir
from ich_common import HU_HI, HU_LO, IMG_SIZE, WL, WW   # 脑窗常量与 RSNA 共用

OUT = Path(__file__).resolve().parent          # 本脚本在 Cq500_dataset 内，输出直接写到当前目录
PNG_ROOT = OUT / "png"


def read_series_key(dcm_path):
    """返回用于切片排序的 (z, instance) 键。读取失败返回 (nan, nan, 文件名)。"""
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    pos = getattr(ds, "ImagePositionPatient", None)
    z = float(pos[2]) if pos else float("nan")
    inst = getattr(ds, "InstanceNumber", None)
    return z, float(inst) if inst is not None else float("nan"), dcm_path


def dicom_to_png(dcm_path, out_png):
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + inter
    hu = np.clip(hu, HU_LO, HU_HI)
    img8 = ((hu - HU_LO) / (HU_HI - HU_LO + 1e-6) * 255.0).astype(np.uint8)
    img3 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    if img3.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_png), img3)


def process_patient(job):
    patient_id, split, label, dcm_paths = job
    sdir = PNG_ROOT / split / label / patient_id
    import shutil
    if sdir.exists():                      # 同名患者可能在多个 qct 分组重复，写前清空避免旧帧残留/并发覆盖
        shutil.rmtree(sdir)
    try:
        keys = []
        for p in dcm_paths:
            try:
                keys.append(read_series_key(p))
            except Exception:
                keys.append((float("nan"), float("nan"), p))
        # 按序列目录分组，组内按 z / InstanceNumber / 文件名排序
        groups = {}
        for z, inst, dp in keys:
            groups.setdefault(dp.parent.name, []).append((z, inst, dp))
        sdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for seq_i, series_name in enumerate(sorted(groups), start=1):
            items = sorted(groups[series_name],
                           key=lambda k: (float("inf") if k[0] != k[0] else k[0],
                                          float("inf") if k[1] != k[1] else k[1],
                                          k[2].name))
            for idx, (_, _, dp) in enumerate(items, start=1):
                out_png = sdir / f"seq{seq_i:02d}_frame{idx:04d}.png"
                dicom_to_png(dp, out_png)
                rows.append({"patient_id": patient_id, "split": split, "label": label,
                             "series": dp.parent.name, "image_path": out_png.name})
        return len(rows), []
    except Exception as e:  # noqa: BLE001
        return 0, [(patient_id, repr(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-patients", type=int, default=0, help="0=全部")
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 1))
    ap.add_argument("--clean", action="store_true", help="转换前删除已有 png 目录")
    args = ap.parse_args()

    if args.clean and PNG_ROOT.exists():
        import shutil
        shutil.rmtree(PNG_ROOT)

    # 原始 DICOM 根目录由 CQ500_RAW_DIR 或默认候选决定（换机器不必改代码）
    CQ500_RAW = raw_dataset_dir()
    print(f"[info] 原始数据集: {CQ500_RAW}")

    split_df = pd.read_csv(OUT / "cq500_split.csv")
    label_df = pd.read_csv(OUT / "cq500_patient_labels.csv")
    info = split_df.merge(label_df[["patient_id", "label"]], on="patient_id")

    # 收集患者 -> dcm 列表（不移动任何原始文件）
    # 患者目录位于 qct01~qct19 分组下，命名为 "CQ500CT{id} CQ500CT{id}"
    jobs = []
    raw_dirs = sorted([d for d in CQ500_RAW.rglob("*")
                       if d.is_dir() and d.name.startswith("CQ500CT")])
    print(f"[info] 找到患者目录: {len(raw_dirs)} 个")
    for d in raw_dirs:
        patient_id = d.name.split()[0]                    # "CQ500CT0 CQ500CT0" -> "CQ500CT0"
        row = info[info["patient_id"] == patient_id]
        if row.empty:
            print(f"[skip] 患者目录 {d.name} 不在 reads.csv 中，跳过")
            continue
        dcm_paths = sorted(d.rglob("*.dcm"))
        if not dcm_paths:
            print(f"[skip] 患者 {patient_id} 无 DICOM 文件")
            continue
        jobs.append((patient_id, row.iloc[0]["split"], row.iloc[0]["label"], dcm_paths))

    if args.max_patients:                                 # 测试模式：取前 N 个患者
        jobs = jobs[:args.max_patients]
        print(f"[test] 仅转换 {len(jobs)} 例患者")

    print(f"[start] 待转换患者: {len(jobs)}，workers={args.workers}")
    t0 = time.time()
    index_rows, failed, total_imgs = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_patient, j): j[0] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            pid = futs[fut]
            n, errs = fut.result()
            done += 1
            total_imgs += n
            index_rows.extend([None] * 0)  # 占位
            if errs:
                failed.extend(errs)
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[progress] {done}/{len(jobs)} 患者 | 已生成 {total_imgs} 张 | 耗时 {el:.0f}s")

    # 重建索引（直接扫输出目录，避免跨进程传递大列表）
    img_rows = []
    for p in PNG_ROOT.rglob("*.png"):
        parts = p.parent.parts                  # (..., split, label, patient_id)
        split = parts[-3]
        label = parts[-2]
        patient = parts[-1]
        img_rows.append({"patient_id": patient, "split": split, "label": label,
                         "image_path": str(p.relative_to(PNG_ROOT))})
    (pd.DataFrame(img_rows).to_csv(OUT / "cq500_slices_index.csv", index=False))

    print(f"[done] 总计生成 {total_imgs} 张 PNG（索引 {len(img_rows)} 行）")
    print(f"[done] 失败患者: {len(failed)}")
    for f in failed[:20]:
        print("   ", f)
    if failed:
        pd.DataFrame(failed, columns=["patient_id", "error"]).to_csv(OUT / "cq500_convert_log.csv", index=False)


if __name__ == "__main__":
    main()