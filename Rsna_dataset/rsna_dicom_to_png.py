# -*- coding: utf-8 -*-
"""
RSNA DICOM -> PNG（只转换 rsna_split.csv 选中的患者）

两个关键点：

1. **窗宽窗位与 CQ500 保持一致（默认固定脑窗 WL40/WW80）**
   旧脚本用的是"每张图自带的 WindowCenter/WindowWidth"（RSNA 多为 WC30/WW80），
   而 CQ500 用的是固定脑窗 WL40/WW80。两套数据集如果预处理口径不同，
   横向对比就不成立，评审也会追问。故默认改为与 CQ500 相同的固定脑窗；
   想复现旧行为可用 --window auto。

2. **目录以 patient_id 为单位**
   旧版以 study_id 为单位，一个患者多次检查就会有多个目录，划分时容易跨集合。
   现在同一患者的所有检查都落在 png/{split}/{label}/{patient_id}/ 下，
   文件名用 s{检查序号}_f{帧序号} 保留检查分组与层序。

输出:
  Rsna_dataset/png/{split}/{label}/{patient_id}/s01_f0001.png ...
  Rsna_dataset/rsna_slices_index.csv
    列: patient_id, study_id, split, label, slice_label, image_path
    label       患者级标签（该患者任一检查任一帧出血 -> Hemorrhagic），与 CQ500 口径一致
    slice_label 该帧自带的原始标注（0/1），供需要逐帧监督的消融实验使用

用法:
    python rsna_dicom_to_png.py
    python rsna_dicom_to_png.py --window auto     # 用 DICOM 自带窗（旧行为）
    python rsna_dicom_to_png.py --limit 20        # 冒烟测试
"""
import argparse
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd
import pydicom

from rsna_commons import raw_dataset_dir
from ich_common import HU_HI, HU_LO, IMG_SIZE, WL, WW   # 脑窗常量与 CQ500 共用

warnings.filterwarnings("ignore", message="Invalid value for VR UI")

ROOT = Path(__file__).resolve().parent            # Rsna_dataset
PNG_ROOT = ROOT / "png"
# DICOM 目录在 main() 里惰性解析 —— 目标机器上没打包原始 DICOM 时，
# 本模块仍能被 import（只有真的重转才会报错）。
DCM_SUBDIR = "stage_2_train"


def read_key(dcm_path):
    """(z, instance, 文件名) 用于层序排序。"""
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True,
                         specific_tags=["ImagePositionPatient", "InstanceNumber"])
    pos = getattr(ds, "ImagePositionPatient", None)
    try:
        z = float(pos[2]) if pos else float("nan")
    except Exception:  # noqa: BLE001
        z = float("nan")
    inst = getattr(ds, "InstanceNumber", None)
    try:
        inst = float(inst) if inst is not None else float("nan")
    except Exception:  # noqa: BLE001
        inst = float("nan")
    return z, inst, dcm_path


def window_to_uint8(ds, mode):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    hu = arr * slope + inter
    if mode == "auto":
        wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
        try:
            lo = float(np.atleast_1d(np.asarray(wc))[0]) - float(
                np.atleast_1d(np.asarray(ww))[0]) / 2
            hi = float(np.atleast_1d(np.asarray(wc))[0]) + float(
                np.atleast_1d(np.asarray(ww))[0]) / 2
        except Exception:  # noqa: BLE001
            lo, hi = float(hu.min()), float(hu.max())
    else:
        lo, hi = HU_LO, HU_HI
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)


def convert_patient(job):
    patient_id, split, plabel, items, mode = job
    pdir = PNG_ROOT / split / plabel / patient_id
    if pdir.exists():
        shutil.rmtree(pdir)                        # 避免旧帧残留
    try:
        keys = []
        for img_id, dp, study_id, slabel in items:
            try:
                z, inst, _ = read_key(dp)
            except Exception:  # noqa: BLE001
                z, inst = float("nan"), float("nan")
            keys.append((study_id, z, inst, dp, img_id, slabel))

        groups = {}
        for study_id, z, inst, dp, img_id, slabel in keys:
            groups.setdefault(study_id, []).append((z, inst, dp, img_id, slabel))

        pdir.mkdir(parents=True, exist_ok=True)
        out_rows = []
        for si, study_id in enumerate(sorted(groups), start=1):
            grp = sorted(groups[study_id], key=lambda k: (
                float("inf") if k[0] != k[0] else k[0],
                float("inf") if k[1] != k[1] else k[1],
                k[2].name))
            for fi, (_, _, dp, img_id, slabel) in enumerate(grp, start=1):
                ds = pydicom.dcmread(str(dp))
                img8 = window_to_uint8(ds, mode)
                img3 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
                if img3.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_AREA)
                name = f"s{si:02d}_f{fi:04d}.png"
                cv2.imwrite(str(pdir / name), img3)
                out_rows.append({"patient_id": patient_id, "study_id": study_id,
                                 "split": split, "label": plabel,
                                 "slice_label": int(slabel),
                                 "image_path": f"{split}/{plabel}/{patient_id}/{name}"})
        return out_rows, []
    except Exception as e:  # noqa: BLE001
        return [], [(patient_id, repr(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["brain", "auto"], default="brain",
                    help="brain=固定脑窗 WL40/WW80（与 CQ500 一致，默认）；auto=DICOM 自带窗")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 例（冒烟测试）")
    ap.add_argument("--clean", action="store_true",
                    help="转换前删除整个 png/（旧版按 study 划分的产物与本次按 patient "
                         "划分的样本不同，混在一起会污染索引）")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if args.clean and PNG_ROOT.exists():
        n = sum(1 for _ in PNG_ROOT.rglob("*") if _.is_file())
        print(f"[clean] 删除旧 png/（{n} 个文件）")
        shutil.rmtree(PNG_ROOT)

    DCM_DIR = raw_dataset_dir() / DCM_SUBDIR
    if not DCM_DIR.is_dir():
        raise SystemExit(f"找不到 DICOM 目录 {DCM_DIR}；"
                         f"如需重转请设置 RSNA_RAW_DIR")

    split_p = ROOT / "rsna_split.csv"
    if not split_p.exists():
        raise SystemExit("缺少 rsna_split.csv；请先运行 rsna_make_split.py")
    split = pd.read_csv(split_p)
    pm = pd.read_csv(ROOT / "rsna_patient_map.csv")
    sl = pd.read_csv(ROOT / "rsna_slice_labels.csv")

    pm = pm[(pm["patient_id"].fillna("") != "")
            & (~pm["study_id"].astype(str).str.startswith("ERR:"))]
    d = sl.merge(pm, on="img_id", how="inner")

    pat = d.groupby("patient_id")["any"].max()
    sel = split.merge(pat.rename("any").reset_index(), on="patient_id", how="left")
    sel["label"] = np.where(sel["any"] == 1, "Hemorrhagic", "Normal")
    print(f"[info] 抽样患者 {len(sel)}，阳性 {int((sel['label']=='Hemorrhagic').sum())}，"
          f"窗={'脑窗 WL%.0f/WW%.0f' % (WL, WW) if args.window == 'brain' else 'DICOM 自带'}")

    d = d.merge(split, on="patient_id")
    by_pat = d.groupby("patient_id")[
        ["img_id", "study_id", "any"]].apply(
        lambda g: list(zip(g["img_id"], g["study_id"]))).to_dict()
    slice_label_of = dict(zip(sl["img_id"], sl["any"]))

    jobs = []
    for _, r in sel.iterrows():
        pid = r["patient_id"]
        items = []
        for img_id, study_id in by_pat.get(pid, []):
            fp = DCM_DIR / f"{img_id}.dcm"
            if fp.exists():
                items.append((img_id, fp, study_id, slice_label_of.get(img_id, 0)))
        if items:
            jobs.append((pid, r["split"], r["label"], items, args.window))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[start] 待转换 {len(jobs)} 例，{sum(len(j[3]) for j in jobs)} 张切片")

    t0 = time.time()
    rows, errs, done = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_patient, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            r, e = fut.result()
            rows.extend(r)
            errs.extend(e)
            done += 1
            if done % 50 == 0:
                print(f"[progress] {done}/{len(jobs)} 例  共 {len(rows)} 张"
                      f"  {time.time() - t0:.0f}s", flush=True)

    # worker 返回的行已含 slice_label，直接整理成索引
    idx = pd.DataFrame(rows)
    if idx.empty:
        raise SystemExit("没有转换出任何图像，请检查 split 与 DICOM 是否匹配")
    idx = idx[["patient_id", "study_id", "split", "label",
               "slice_label", "image_path"]]
    idx = idx.sort_values(["split", "patient_id", "image_path"]).reset_index(drop=True)
    idx.to_csv(ROOT / "rsna_slices_index.csv", index=False)

    print(f"\n[done] {len(idx)} 张 PNG，{len(sel)} 例，{time.time() - t0:.0f}s")
    if errs:
        print(f"[warn] {len(errs)} 例失败，前 5 条：")
        for e in errs[:5]:
            print("   ", e)
    print("\n[索引统计]")
    print(idx.groupby(["split", "label"]).agg(
        patients=("patient_id", "nunique"), slices=("image_path", "count")).to_string())
    print(f"切片级阳性率: {idx['slice_label'].mean():.4f}"
          f"（对比 CQ500 ~0.42-0.46，语义不同，见 rsna_prepare_labels.py 说明）")
    print(f"\n[out] {ROOT / 'rsna_slices_index.csv'}")


if __name__ == "__main__":
    main()
