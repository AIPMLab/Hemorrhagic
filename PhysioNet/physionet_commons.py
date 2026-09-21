# -*- coding: utf-8 -*-
"""
PhysioNet CT-ICH 流水线公共模块（**只放数据集相关部分**）

数据集无关的计算（脑窗常量、噪声函数、增强 filter、判别器类别、预训练权重加载、
患者级投票、filter 穷举、Grad-CAM）全部在 **code 根目录的 ich_common.py**。
本模块只负责 PhysioNet 自己的路径、索引与数据集类 —— 因此
**不依赖 CQ500 / RSNA 中的任何文件**，可以单独部署。

数据集来源
    Hssayeni et al., "Computed Tomography Images for Intracranial Hemorrhage
    Detection and Segmentation" (PhysioNet, 2020)。
    82 例患者的非增强头颅 CT，约 30 层/例，层厚 5 mm；
    每层由两位放射科医生标注出血亚型与骨折。

图像来源与其它两套数据集的区别（论文里要写清楚）
    CQ500 / RSNA：我们自己从 DICOM 做 HU 换算 + 固定脑窗 WL40/WW80 得到 PNG。
    PhysioNet    ：原始数据只提供**已经用 Siemens syngo 窗好的 650x650 JPEG**
                   （brain / bone 两套），没有 DICOM。
                   因此这里直接使用作者提供的 brain-window 图，只做 224x224 缩放。
                   脑窗参数由上游软件决定、数据集未给出具体数值，与另两套属"同类
                   但非同一实现"的预处理 —— 并排比较时这是需要声明的差异。

索引列（见 physionet_images_to_png.py）：
  patient_id, split, label, slice_label, image_path

  label       患者级标签（任一层出血则 Hemorrhagic）—— 与 CQ500/RSNA 同口径
  slice_label 该层自带的原始标注（0/1）—— PhysioNet 与 RSNA 都是逐层标注

数据根: PhysioNet/png/{split}/{label}/{patient_id}/sXXX.png
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

ROOT = Path(__file__).resolve().parent            # PhysioNet
_CODE = ROOT.parent

# 冒烟测试隔离：设 PHYSIONET_SMOKE=1 时所有产物写入 *_smoke 目录，绝不污染正式结果。
# 注意 DATA_ROOT(png) 不隔离 —— 那是真实输入数据，测试也要读它。
SMOKE = os.environ.get("PHYSIONET_SMOKE") == "1"
_SFX = "_smoke" if SMOKE else ""

# 交叉验证折：设 PHYSIONET_FOLD=k 时，数据/索引/结果/权重全部带 _fold{k} 后缀。
# 本数据集只有 82 例，单次划分的 test 仅 14 例（准确率步长 7.1%，配对检验几乎没功效），
# 因此正式做法是 5 折患者级交叉验证：每折 test 一折、val 下一折、train 其余三折
# （见 physionet_make_split.py --n-folds 5 生成的 physionet_split_fold{k}.csv）。
# 不设该变量时退回固定的 70/15/15 单次划分。
FOLD = os.environ.get("PHYSIONET_FOLD")
if FOLD is not None:
    try:
        int(FOLD)
    except ValueError:
        raise SystemExit(f"PHYSIONET_FOLD 必须是整数，收到 {FOLD!r}")
_FOLD_SFX = f"_fold{FOLD}" if FOLD is not None else ""

DATA_ROOT = ROOT / f"png{_FOLD_SFX}"                                    # {split}/{label}/{patient}/...
NOISY_ROOT = _CODE / "noisy_tests" / f"physionet{_FOLD_SFX}{_SFX}"
ENHANCED_ROOT = _CODE / "enhanced_tests" / f"physionet{_FOLD_SFX}{_SFX}"
ADAPTIVE_ROOT = _CODE / "adaptive_unknown_tests" / f"physionet{_FOLD_SFX}{_SFX}"
RESULT_ROOT = _CODE / f"results{_FOLD_SFX}{_SFX}"
MODEL_DIR = _CODE / f"models{_FOLD_SFX}{_SFX}"

# 划分文件（随折变化）与切片索引（随折变化，因为索引里带 split 列与目录层级）
SPLIT_CSV = ROOT / f"physionet_split{_FOLD_SFX}.csv"
INDEX_CSV = ROOT / f"physionet_slices_index{_FOLD_SFX}.csv"
DETECTOR_DATASET_DIR = _CODE / f"noise_detector_dataset_physionet{_FOLD_SFX}{_SFX}"


def fold_tag():
    """当前折的短标签，用于结果表里区分。"""
    return f"fold{FOLD}" if FOLD is not None else "holdout"


def detector_weight_path(variant):
    return MODEL_DIR / f"noise_detector_efficientnetb0_physionet_{variant}_best.pth"


def detector_result_dir(variant):
    return RESULT_ROOT / "noise_detector_physionet" / variant


DETECTOR_WEIGHT = detector_weight_path("7class")   # 兼容旧引用


def load_index():
    """读 PhysioNet 切片索引，并把 image_path 统一成 POSIX 分隔符（跨平台）。

    patient_id 显式读成字符串：索引里是零填充的 "049"，pandas 默认会解析成整数 49，
    于是与标签 CSV（未填充）对不上，groupby/merge 也会静默错配。
    """
    df = pd.read_csv(INDEX_CSV, dtype={"patient_id": str})
    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
    return df


def raw_dataset_dir():
    """PhysioNet 原始数据集根目录（仅重新生成图片/重建划分时才需要）。

    优先取环境变量 PHYSIONET_RAW_DIR；否则用本机历史路径。换机器设一下即可。
    """
    env = os.environ.get("PHYSIONET_RAW_DIR")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"PHYSIONET_RAW_DIR 指向的路径不存在: {p}")
        return p
    for cand in (Path(r"D:\Li-kai\project\Data\medical\PhysioNet CT-ICH"
                      r"\computed-tomography-images-for-intracranial-hemorrhage-"
                      r"detection-and-segmentation-1.0.0"),
                 Path("/data/physionet_ct_ich"), Path("/mnt/physionet_ct_ich")):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "找不到 PhysioNet 原始数据集。请设置环境变量 PHYSIONET_RAW_DIR 指向它，"
        "例如 export PHYSIONET_RAW_DIR=/data/physionet_ct_ich")


def enhanced_dir(noise, method):
    """增强图目录：{noise}/{method}/... —— 唯一入口，防止漏掉噪声层级。"""
    return ENHANCED_ROOT / noise / method


class PatientSliceDataset(Dataset):
    """读 PhysioNet 索引中某 split 的切片；base_dir 可切到噪声/增强产物（结构相同）。

    label_column 默认 "label"（患者级，与 CQ500/RSNA 同口径）；
    传 "slice_label" 可做逐层监督（PhysioNet 本身就是逐层标注）。
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
    """复用 ich_common 的 sweep 实现，但绑定 PhysioNet 的索引与目录。"""
    return _sweep_filters(
        split=split, noise_name=noise_name, filter_names=filter_names,
        model=model, device=device, batch_size=batch_size, num_workers=num_workers,
        from_disk=from_disk, seed=seed, limit=limit,
        index_df=load_index(), data_root=DATA_ROOT, noisy_root=NOISY_ROOT,
        classes=CLASSES,
    )
