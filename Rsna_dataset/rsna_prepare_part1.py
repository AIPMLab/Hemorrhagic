# -*- coding: utf-8 -*-
"""
RSNA ICH 数据预处理（子集 500 例患者）Part 1：
1) 解析 stage_2_train.csv -> 图像级 any 标签（5 亚型取 max）
2) 并行扫描 DICOM 读 StudyInstanceUID -> 图像->患者(study) 映射
3) 患者级 any 聚合 -> 分层采样 500 例（保持 any 阳性比例）-> 7/15/15 患者级切分

输出 Rsna_dataset/rsna_patient_labels.csv / rsna_study_map.csv / rsna_split.csv / rsna_split_summary.csv
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pydicom

RSNA_ROOT = Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection")
TRAIN_CSV = RSNA_ROOT / "stage_2_train.csv"
DCM_DIR = RSNA_ROOT / "stage_2_train"
OUT = Path(__file__).resolve().parent / "Rsna_dataset"
OUT.mkdir(parents=True, exist_ok=True)

SUB_TYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
SEED = 42
N_PATIENTS = 500
RATIO_TRAIN, RATIO_VAL = 0.70, 0.15


def read_study(dcm_path):
    """返回 (file_stem, StudyInstanceUID)。失败返回 (stem, '')。"""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return dcm_path.stem, str(getattr(ds, "StudyInstanceUID", ""))
    except Exception:  # noqa: BLE001
        return dcm_path.stem, ""


def main():
    # ---------- 1) 图像级 any 标签 ----------
    t0 = time.time()   
    # stage_2 的 csv 较大（~450 万行），按需只读 ID/Label
    df = pd.read_csv(TRAIN_CSV, usecols=["ID", "Label"])
    df["img_id"] = df["ID"].str.rsplit("_", n=1, expand=True)[0]   # ID_xxx_subtype -> ID_xxx
    df["subtype"] = df["ID"].str.rsplit("_", n=1, expand=True)[1]
    df = df[df["subtype"].isin(SUB_TYPES)]
    img_label = (df.assign(Label=df["Label"].astype(int))
                 .groupby("img_id")["Label"].max().reset_index().rename(columns={"Label": "any"}))
    print(f"[1] 图像级标签: {len(img_label)} 张，any 阳性率 {img_label['any'].mean():.4f}（耗时 {time.time()-t0:.0f}s）", flush=True)

    # ---------- 2) 图 -> study(患者) 映射（仅扫描 csv 中存在的图像） ----------
    dcm_files = [f for f in DCM_DIR.glob("*.dcm") if f.stem in set(img_label["img_id"])]
    print(f"[2] 匹配到 DICOM 文件 {len(dcm_files)} 个，开始读 StudyInstanceUID ...", flush=True)
    # chunksize>1 避免每文件一次 IPC（75 万次 IPC 是之前的性能瓶颈）
    pairs = []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, study) in enumerate(ex.map(read_study, dcm_files, chunksize=512), 1):
            pairs.append((stem, study))
            if i % 100000 == 0:
                print(f"    ... {i}/{len(dcm_files)}（{time.time()-t1:.0f}s）", flush=True)
    study_map = pd.DataFrame(pairs, columns=["img_id", "study_id"])
    study_map = study_map[study_map["study_id"] != ""]
    study_map.to_csv(OUT / "rsna_study_map.csv", index=False)
    print(f"[2] 映射完成: {len(study_map)} 张，患者(study)数 {study_map['study_id'].nunique()}（耗时 {time.time()-t0:.0f}s）")

    # ---------- 3) 患者级聚合 + 分层采样 + 切分 ----------
    merged = img_label.merge(study_map, on="img_id")
    agg = (merged.groupby("study_id")
           .agg(n_images=("img_id", "count"), any=("any", "max")).reset_index())
    print(f"[3] 患者级: {len(agg)} 例，any 阳性率 {agg['any'].mean():.4f}，每例平均切片 {agg['n_images'].mean():.0f}")

    rng = np.random.default_rng(SEED)
    pos = agg[agg["any"] == 1]["study_id"].tolist()
    neg = agg[agg["any"] == 0]["study_id"].tolist()
    rng.shuffle(pos); rng.shuffle(neg)
    # 分层保持原始阳性比例
    pos_n = int(round(N_PATIENTS * agg["any"].mean()))
    neg_n = N_PATIENTS - pos_n
    sample_ids = set(pos[:pos_n]) | set(neg[:neg_n])
    sampled = agg[agg["study_id"].isin(sample_ids)].copy()
    print(f"   采样 {len(sampled)} 例（阳性 {int(sampled['any'].sum())} / 阴性 {int((sampled['any']==0).sum())}），"
          f"阳性率 {sampled['any'].mean():.4f}")

    rng.shuffle(list(sample_ids))
    rows = []
    for cls in [1, 0]:
        ids = sampled[sampled["any"] == cls]["study_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(n * RATIO_TRAIN), int(n * RATIO_VAL)
        rows.append(pd.DataFrame({"study_id": ids,
                                  "split": ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)}))
    split_df = pd.concat(rows, ignore_index=True)
    split_df.merge(sampled[["study_id", "any", "n_images"]], on="study_id").to_csv(
        OUT / "rsna_patient_labels.csv", index=False)
    split_df.to_csv(OUT / "rsna_split.csv", index=False)
    summary = (split_df.merge(sampled[["study_id", "any"]], on="study_id")
               .groupby(["split", "any"]).size().reset_index(name="count"))
    summary.to_csv(OUT / "rsna_split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[done] 全部输出至 {OUT}")


if __name__ == "__main__":
    main()