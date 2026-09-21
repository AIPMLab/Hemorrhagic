# -*- coding: utf-8 -*-
"""
CQ500 流水线公共模块（**只放数据集相关部分**）

数据集无关的计算（脑窗常量、噪声函数、增强 filter、判别器类别、预训练权重加载、
患者级投票、filter 穷举、Grad-CAM）全部在 **code 根目录的 ich_common.py**。
本模块只负责：

    * CQ500 的目录约定（*_cq500 后缀，与其它数据集完全隔离）
    * CQ500 的索引读取与数据集类
    * 绑定 CQ500 路径的 sweep_filters 包装
    * 把 ich_common 的名字**转出**，使既有脚本 `from cq500_commons import ...`
      一行都不用改

与 RSNA 的关系：**互不依赖**。两者都只依赖 ich_common.py，
所以任一套都能单独拷到别的机器上跑，而计算定义又不会分叉。

分辨率约定
    噪声与滤波在**原生分辨率**施加，Resize(224) 只在送入网络的 transform 里；
    判别器数据集与判别器推理遵循同一约定。
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

ROOT = Path(__file__).resolve().parent            # Cq500_dataset
_CODE = ROOT.parent

# 冒烟测试隔离：设 CQ500_SMOKE=1 时所有产物写入 *_smoke 目录，绝不污染正式结果
# （正式产物一旦被小规模测试覆盖，run_cq500_pipeline.py 会误判为"已完成"而跳过）
SMOKE = os.environ.get("CQ500_SMOKE") == "1"
_SFX = "_smoke" if SMOKE else ""

DATA_ROOT = ROOT / "png"                                   # {split}/{label}/{patient}/...
NOISY_ROOT = _CODE / "noisy_tests" / f"cq500{_SFX}"        # {noise}/{label}/{patient}/...
ENHANCED_ROOT = _CODE / "enhanced_tests" / f"cq500{_SFX}"  # {noise}/{method}/{label}/{patient}/...
ADAPTIVE_ROOT = _CODE / "adaptive_unknown_tests" / f"cq500{_SFX}"
RESULT_ROOT = _CODE / f"results{_SFX}"
MODEL_DIR = _CODE / f"models{_SFX}"

INDEX_CSV = ROOT / "cq500_slices_index.csv"
DETECTOR_DATASET_DIR = _CODE / f"noise_detector_dataset_cq500{_SFX}"


def detector_weight_path(variant):
    return MODEL_DIR / f"noise_detector_efficientnetb0_cq500_{variant}_best.pth"


def detector_result_dir(variant):
    return RESULT_ROOT / "noise_detector_cq500" / variant


DETECTOR_WEIGHT = detector_weight_path("7class")   # 兼容旧引用


def load_index():
    """读 CQ500 切片索引，并把 image_path 统一成 POSIX 分隔符（跨平台）。

    索引 CSV 里存的是 Windows 反斜杠（test\\Hemorrhagic\\CQ500CT114\\frame_0001.png）。
    在 Linux 上 Path("a\\b\\c.png") 会被当成**单个文件名**，全部取图都会失败。
    统一转成正斜杠后 Windows/Linux 都能解析 —— 换服务器无需重写索引。
    """
    df = pd.read_csv(INDEX_CSV)
    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
    return df


def raw_dataset_dir():
    """CQ500 原始 DICOM 数据集根目录（仅重转换/重建划分时才需要）。

    优先取环境变量 CQ500_RAW_DIR；否则尝试本机历史路径。换机器设一下即可。
    """
    env = os.environ.get("CQ500_RAW_DIR")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"CQ500_RAW_DIR 指向的路径不存在: {p}")
        return p
    for cand in (Path(r"D:\Li-kai\project\Data\medical\cq500_2"),
                 Path(r"D:\Li-kai\project\Data\medical\cq500"),
                 Path("/data/cq500"), Path("/mnt/cq500")):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "找不到原始 DICOM 数据集。请设置环境变量 CQ500_RAW_DIR 指向它，"
        "例如 export CQ500_RAW_DIR=/data/cq500")


def enhanced_dir(noise, method):
    """增强图目录：{noise}/{method}/... —— 唯一入口，防止漏掉噪声层级。"""
    return ENHANCED_ROOT / noise / method


class PatientSliceDataset(Dataset):
    """读 CQ500 索引中某 split 的切片；base_dir 可切到噪声/增强产物（结构相同）。

    noise / filter_name 给定则在**内存**加噪/滤波（用于 val 拟合，避免落盘）。
    返回 (img, label, patient_id)。
    """

    def __init__(self, split, base_dir=None, transform=None, label_map=None,
                 noise=None, filter_name=None, limit=0):
        self.df = load_index()
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if limit:
            self.df = self.df.head(limit).reset_index(drop=True)
        self.base = Path(base_dir) if base_dir is not None else DATA_ROOT
        self.transform = transform
        self.label_map = label_map or {c: i for i, c in enumerate(CLASSES)}
        self.noise = noise
        self.filter_name = filter_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import cv2
        row = self.df.iloc[i]
        img = Image.open(self.base / row["image_path"]).convert("RGB")
        if self.noise is not None or self.filter_name is not None:
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if self.noise is not None:
                arr = apply_noise(arr, self.noise)
            if self.filter_name is not None:
                arr = apply_filter(arr, self.filter_name)
            img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[row["label"]], row["patient_id"]


def sweep_filters(split, noise_name, filter_names, model, device="cuda",
                  batch_size=64, num_workers=8, from_disk=False, seed=None, limit=0):
    """复用 ich_common 的 sweep 实现，但绑定 CQ500 的索引与目录。

    这样两套数据集跑的是**同一份穷举代码**，不会各自漂移。
    """
    return _sweep_filters(
        split=split, noise_name=noise_name, filter_names=filter_names,
        model=model, device=device, batch_size=batch_size, num_workers=num_workers,
        from_disk=from_disk, seed=seed, limit=limit,
        index_df=load_index(), data_root=DATA_ROOT, noisy_root=NOISY_ROOT,
        classes=CLASSES,
    )
