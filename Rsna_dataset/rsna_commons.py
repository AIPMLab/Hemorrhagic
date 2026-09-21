# -*- coding: utf-8 -*-
"""
RSNA ICH 流水线公共模块（**只放数据集相关部分**）

数据集无关的计算（脑窗常量、噪声函数、增强 filter、判别器类别、预训练权重加载、
患者级投票、filter 穷举、Grad-CAM）全部在 **code 根目录的 ich_common.py**。

**本模块不依赖 CQ500**（早期版本 import 了 cq500_commons，导致 RSNA 无法单独
拷到别的机器上跑）。现在 RSNA 只需要 ich_common.py + Rsna_dataset/ 两样东西，
既独立部署，又与 CQ500 共用同一份计算定义，两边结果可以并排比较。

索引列（见 rsna_dicom_to_png.py）：
  patient_id, study_id, split, label, slice_label, image_path

  label       患者级标签（与 CQ500 同口径）—— 分类训练/评估默认用它
  slice_label 该帧自带的原始标注 —— 需要逐帧监督的消融实验可用

数据根: Rsna_dataset/png/{split}/{label}/{patient_id}/sXX_fXXXX.png
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # code 根，供 import ich_common

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ich_common import (  # noqa: F401  转出，供各脚本使用
    CLASSES, CODE_ROOT, DETECTOR_CLASSES, DETECTOR_VARIANTS, FILTER_FUNCS,
    FILTER_NAMES, HU_HI, HU_LO, IMG_SIZE, NOISE_FUNCS, NOISE_NAMES, NOISE_PARAMS,
    NUM_CLASSES, PRETRAINED_FILE, TTA_VARIANTS, WEIGHTS_DIR, WL, WW, apply_filter,
    apply_noise, boot_ci, build_eval_transform, create_model_with_pretrained,
    deterministic_seed, grad_cam, initialize_gradcam_tf, load_model,
    load_pretrained_into, match_prevalence, match_prevalence_rate, mcnemar,
    model_adaptation, overlay, patient_vote_metrics, predict_slices,
    prior_correction_mode, uses_prior_correction, _SweepDataset, _to_tensor,
)
from ich_common import sweep_filters as _sweep_filters  # noqa: E402

ROOT = Path(__file__).resolve().parent            # Rsna_dataset
_CODE = ROOT.parent

# 冒烟测试隔离：设 RSNA_SMOKE=1 时所有产物写入 *_smoke 目录，绝不污染正式结果。
# 注意 DATA_ROOT(png) 不隔离 —— 那是真实输入数据，测试也要读它。
SMOKE = os.environ.get("RSNA_SMOKE") == "1"
_SFX = "_smoke" if SMOKE else ""

DATA_ROOT = ROOT / "png"                                  # {split}/{label}/{patient}/...
NOISY_ROOT = _CODE / "noisy_tests" / f"rsna{_SFX}"        # {noise}/{label}/{patient}/...
ENHANCED_ROOT = _CODE / "enhanced_tests" / f"rsna{_SFX}"  # {noise}/{method}/{label}/{patient}/...
ADAPTIVE_ROOT = _CODE / "adaptive_unknown_tests" / f"rsna{_SFX}"
RESULT_ROOT = _CODE / f"results{_SFX}"
MODEL_DIR = _CODE / f"models{_SFX}"

INDEX_CSV = ROOT / "rsna_slices_index.csv"
SPLIT_CSV = ROOT / "rsna_split.csv"
DETECTOR_DATASET_DIR = _CODE / f"noise_detector_dataset_rsna{_SFX}"


def detector_weight_path(variant):
    return MODEL_DIR / f"noise_detector_efficientnetb0_rsna_{variant}_best.pth"


def detector_result_dir(variant):
    return RESULT_ROOT / "noise_detector_rsna" / variant


DETECTOR_WEIGHT = detector_weight_path("7class")   # 兼容旧引用


def load_index():
    """读 RSNA 切片索引，并把 image_path 统一成 POSIX 分隔符（跨平台）。"""
    df = pd.read_csv(INDEX_CSV)
    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
    return df


def raw_dataset_dir():
    """RSNA 原始 DICOM 目录（仅重新转换/重建划分时才需要）。

    优先取环境变量 RSNA_RAW_DIR；否则用本机历史路径。换机器设一下即可。
    """
    env = os.environ.get("RSNA_RAW_DIR")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"RSNA_RAW_DIR 指向的路径不存在: {p}")
        return p
    for cand in (Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset"
                      r"\rsna-intracranial-hemorrhage-detection"),
                 Path("/data/rsna"), Path("/mnt/rsna")):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "找不到 RSNA 原始数据集。请设置环境变量 RSNA_RAW_DIR 指向它，"
        "例如 export RSNA_RAW_DIR=/data/rsna")


def enhanced_dir(noise, method):
    """增强图目录：{noise}/{method}/... —— 唯一入口，防止漏掉噪声层级。"""
    return ENHANCED_ROOT / noise / method


class PatientSliceDataset(Dataset):
    """读 RSNA 索引中某 split 的切片；base_dir 可切到噪声/增强产物（结构相同）。

    label_column 默认 "label"（患者级，与 CQ500 同口径）；
    传 "slice_label" 可做逐帧监督的消融。
    返回 (img, label, patient_id)。
    """

    def __init__(self, split, base_dir=None, transform=None,
                 label_column="label", limit=0):
        self.df = load_index()
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if limit:
            self.df = self.df.head(limit).reset_index(drop=True)
        self.base = Path(base_dir) if base_dir is not None else DATA_ROOT
        self.transform = transform
        self.label_column = label_column
        self.label_map = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.base / row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.label_column == "slice_label":
            label = int(row["slice_label"])
        else:
            label = self.label_map[row["label"]]
        return img, label, row["patient_id"]


def sweep_filters(split, noise_name, filter_names, model, device="cuda",
                  batch_size=64, num_workers=8, from_disk=False, seed=None, limit=0):
    """复用 ich_common 的 sweep 实现，但绑定 RSNA 的索引与目录。

    这样两套数据集跑的是**同一份穷举代码**，不会各自漂移。
    """
    return _sweep_filters(
        split=split, noise_name=noise_name, filter_names=filter_names,
        model=model, device=device, batch_size=batch_size, num_workers=num_workers,
        from_disk=from_disk, seed=seed, limit=limit,
        index_df=load_index(), data_root=DATA_ROOT, noisy_root=NOISY_ROOT,
        classes=CLASSES,
    )
