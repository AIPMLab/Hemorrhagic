# -*- coding: utf-8 -*-
"""
ICH 噪声鲁棒性流水线的**共用计算模块**（数据集无关）

为什么单独拆出这一层：
  CQ500 与 RSNA 两套数据集要做并排对比，就必须保证"同样的脑窗预处理、同样的噪声、
  同样的滤波器、同样的患者级投票、同样的 filter 穷举"。早先把这些放在 cq500_commons
  里导致 RSNA 必须 import 另一个数据集的代码 —— 换机器时 RSNA 就无法独立部署。
  现在两边都只依赖本模块（位于 code 根目录），既独立又不会分叉。

本模块只包含**与数据集无关**的东西：
    - 脑窗常量 WL/WW/IMG_SIZE（两个 DICOM->PNG 转换脚本共用，预处理口径因此结构一致）
    - 噪声函数（唯一事实来源）、增强 filter（唯一事实来源）
    - 判别器类别与两个变体（6class / 7class）
    - 预训练权重加载（本地文件优先，绕开 TLS/镜像问题）
    - 患者级投票评估、filter 穷举（sweep）
    - Grad-CAM 工具

数据集相关的路径、索引读取、数据集类留在各自的 <dataset>_commons.py：
    Cq500_dataset/cq500_commons.py
    Rsna_dataset/rsna_commons.py

分辨率约定
    噪声与滤波一律在**原生分辨率**施加，Resize(224) 只发生在送入网络的 transform 里。
    判别器数据集与判别器推理都遵循同一约定，故二者的噪声分布一致。
"""
import os
import sys
from pathlib import Path

# Windows 控制台默认 cp1252/GBK，脚本里的中文与 ✓ 等符号会抛 UnicodeEncodeError。
# 所有脚本都 import 本模块，这里统一把标准流改成 UTF-8。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001  某些重定向场景不支持 reconfigure
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy.stats import binomtest

CODE_ROOT = Path(__file__).resolve().parent       # code 根（weights/ 所在）

IMG_SIZE = 224

# 脑窗（两个 DICOM->PNG 转换脚本共用；改这里即两边同时改，预处理口径不会不一致）
WL, WW = 40.0, 80.0
HU_LO, HU_HI = WL - WW / 2, WL + WW / 2           # -> [0, 80]

CLASSES = ["Normal", "Hemorrhagic"]
NUM_CLASSES = 2

# 判别器类别（7 类）：以目录名组织，ImageFolder 按字母序固定
DETECTOR_CLASSES = ["clean", "gaussian", "low_light", "motion_blur",
                    "salt_pepper", "speckle", "unknown_mixed"]

# 两个判别器变体（论文中互相对比）：
#   6class —— 只在 6 类"单一"退化上学习（clean + 5 种单噪声），复合噪声对它完全是分布外
#   7class —— 额外把复合未知噪声 unknown_mixed 作为第 7 类显式学习
# 二者共用 09 生成的同一份 7 类数据集，只是训练时取的类别子集不同。
DETECTOR_VARIANTS = {
    "6class": ["clean", "gaussian", "low_light", "motion_blur",
               "salt_pepper", "speckle"],
    "7class": list(DETECTOR_CLASSES),
}

# 预训练权重本地缓存：本机 huggingface_hub/requests 的 TLS 会被重置，
# 由 00_fetch_pretrained_*.py 用 curl 下载到此处，训练时直接本地加载。
WEIGHTS_DIR = CODE_ROOT / "weights"
PRETRAINED_FILE = {
    "resnet50": "resnet50.a1_in1k.bin",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.ms_in1k.bin",
    "efficientnet_b0": "efficientnet_b0.ra_in1k.bin",
}


def load_pretrained_into(model, sd):
    """把预训练 state_dict 灌进模型，容忍分类头形状不同。

    load_state_dict(strict=False) 只容忍"多键/少键"，遇到**形状不同**仍会抛错，
    而分类头（resnet 的 fc、swin 的 head、efficientnet 的 classifier）恰恰会因
    类别数不同而形状不同。因此先按目标形状过滤掉这些张量。
    返回 (missing_backbone, skipped_head, unexpected)。
    """
    target = model.state_dict()
    filtered, skipped_head = {}, []
    for k, v in sd.items():
        if k in target:
            if target[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped_head.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    backbone_missing = [k for k in missing
                        if not k.startswith("head.") and not k.startswith("fc.")
                        and "classifier" not in k]
    return backbone_missing, skipped_head, unexpected


def create_model_with_pretrained(arch, num_classes, pretrained=True):
    """建模型并加载预训练权重。

    优先用 weights/ 下的本地文件（离线、可复现、绕开 TLS/镜像问题）；
    本地没有才回退到 timm 的 pretrained=True（本机可能因 TLS 重置而失败）。
    """
    import timm
    import torch

    local = WEIGHTS_DIR / PRETRAINED_FILE.get(arch, "")
    if pretrained and local.exists():
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        try:
            sd = torch.load(local, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"本地预训练权重无法读取 {local}: {e}") from e
        if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
            sd = sd["model"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        backbone_missing, skipped, _ = load_pretrained_into(model, sd)
        if backbone_missing:
            raise RuntimeError(
                f"{arch}: 本地权重与结构不匹配，主干缺少 {len(backbone_missing)} 个张量，"
                f"例如 {backbone_missing[:5]}")
        print(f"[pretrained] {arch} <- {local.name}（本地文件，"
              f"跳过 {len(skipped)} 个分类头张量）")
        return model

    if pretrained:
        print(f"[pretrained] {arch} <- timm 在线下载（本地无 {local.name}，"
              f"本机可能因 TLS 重置失败）")
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def load_index():
    """占位：请用各数据集的 <dataset>_commons.load_index（索引路径因数据集而异）。

    保留这个名字是为了让 `from cq500_commons import *` 之类的旧引用立刻报错，
    而不是静默读到错误的索引。
    """
    raise NotImplementedError(
        "load_index 是数据集相关的，请从 cq500_commons / rsna_commons 导入")


def raw_dataset_dir():
    raise NotImplementedError(
        "raw_dataset_dir 是数据集相关的，请从 cq500_commons 导入")


def enhanced_dir(noise, method):
    raise NotImplementedError(
        "enhanced_dir 是数据集相关的，请从 cq500_commons / rsna_commons 导入")


# =====================================================================
# 噪声函数：唯一事实来源
# 参数集中于此，04（测试集落盘）、09（判别器数据集）、07（val 内存态）共用
# =====================================================================
NOISE_PARAMS = {
    "gaussian":     {"sigma": 18},
    "salt_pepper":  {"amount": 0.03},
    "speckle":      {"sigma": 0.20},
    "motion_blur":  {"kernel_size": 9},
    "low_light":    {"gamma": 2.2},                                  # pow(img/255, gamma) -> 变暗
    "unknown_mixed": {"sigma": 18, "kernel_size": 9, "gamma": 1.6},  # 复合：高斯 + 运动模糊 + 低光照
}


def add_gaussian(img, sigma=None):
    sigma = NOISE_PARAMS["gaussian"]["sigma"] if sigma is None else sigma
    out = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=None):
    amount = NOISE_PARAMS["salt_pepper"]["amount"] if amount is None else amount
    out = img.copy()
    h, w = out.shape[:2]
    n = int(amount * h * w)
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    vals = np.where(np.random.rand(n) < 0.5, 255, 0).astype(np.uint8)
    if out.ndim == 3:
        # 两下标索引结果是 (n, 通道)，故取值需补一维广播；直接赋 (n,) 会 shape mismatch
        out[ys, xs] = vals[:, None]
    else:
        out[ys, xs] = vals
    return out


def add_speckle(img, sigma=None):
    sigma = NOISE_PARAMS["speckle"]["sigma"] if sigma is None else sigma
    f = img.astype(np.float32)
    out = f + f * np.random.randn(*img.shape) * sigma
    return np.clip(out, 0, 255).astype(np.uint8)


def add_motion_blur(img, kernel_size=None):
    import cv2
    kernel_size = NOISE_PARAMS["motion_blur"]["kernel_size"] if kernel_size is None else kernel_size
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0
    k /= kernel_size
    return cv2.filter2D(img, -1, k)


def add_low_light(img, gamma=None):
    """低光照：pow(img/255, gamma)，gamma>1 使图像变暗（与其名称语义一致）。"""
    gamma = NOISE_PARAMS["low_light"]["gamma"] if gamma is None else gamma
    out = (img.astype(np.float32) / 255.0) ** gamma * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def add_unknown_mixed(img):
    """复合未知噪声：高斯 -> 运动模糊 -> 低光照，全部复用上方同一批函数。"""
    p = NOISE_PARAMS["unknown_mixed"]
    out = add_gaussian(img, p["sigma"])
    out = add_motion_blur(out, p["kernel_size"])
    out = add_low_light(out, p["gamma"])
    return out


NOISE_FUNCS = {
    "clean": None,
    "gaussian": add_gaussian,
    "salt_pepper": add_salt_pepper,
    "speckle": add_speckle,
    "motion_blur": add_motion_blur,
    "low_light": add_low_light,
    "unknown_mixed": add_unknown_mixed,
}

# 落盘需要生成噪声的类别（不含 clean）
NOISE_NAMES = ["gaussian", "salt_pepper", "speckle", "motion_blur", "low_light", "unknown_mixed"]


def apply_noise(img_bgr, noise_name):
    """对 BGR uint8 施加指定噪声；noise_name=='clean' 时原样返回。"""
    fn = NOISE_FUNCS.get(noise_name)
    if fn is None:
        return img_bgr
    out = fn(img_bgr)
    if out.ndim == 2:
        import cv2
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return np.clip(out, 0, 255).astype(np.uint8)


# =====================================================================
# 增强(filter)函数：唯一事实来源
# 07 评估、11 施加、16 穷举 共用同一份实现，杜绝"评估的 filter 与实际施加的不一致"
# =====================================================================
def median_filter(img):
    import cv2
    return cv2.medianBlur(img, 5)


def gaussian_filter(img):
    import cv2
    return cv2.GaussianBlur(img, (5, 5), 0)


def bilateral_filter(img):
    import cv2
    return cv2.bilateralFilter(img, 9, 75, 75)


def clahe_enhance(img):
    """CLAHE。注意必须把 LAB 转回 BGR——旧版 CQ500 的 06 漏了这一步。"""
    import cv2
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab2 = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def gamma_correction(img, gamma=0.6):
    out = (img.astype(np.float32) / 255.0) ** gamma * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_gamma(img):
    """CLAHE + gamma：先 CLAHE 再 gamma(0.7)，两步都要有（旧版 11 只有 gamma，丢了 CLAHE）。"""
    return gamma_correction(clahe_enhance(img), gamma=0.7)


def sharpen_filter(img):
    import cv2
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, k)


def unsharp_mask(img):
    import cv2
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def hist_equalization(img):
    import cv2
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def identity_filter(img):
    return img


FILTER_FUNCS = {
    "none": identity_filter,
    "median": median_filter,
    "gaussian_filter": gaussian_filter,
    "bilateral": bilateral_filter,
    "clahe": clahe_enhance,
    "gamma": gamma_correction,
    "clahe_gamma": clahe_gamma,
    "sharpen": sharpen_filter,
    "unsharp": unsharp_mask,
    "hist_equalization": hist_equalization,
}

# 穷举/评估顺序（none 在最前，作为"不增强"基线）
FILTER_NAMES = ["none", "median", "gaussian_filter", "bilateral", "clahe", "gamma",
                "clahe_gamma", "sharpen", "unsharp", "hist_equalization"]


def apply_filter(img_bgr, method_name):
    fn = FILTER_FUNCS.get(method_name)
    if fn is None:
        raise KeyError(f"未知增强方法: {method_name}（可选: {sorted(FILTER_FUNCS)}）")
    out = fn(img_bgr)
    if out.ndim == 2:
        import cv2
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return np.clip(out, 0, 255).astype(np.uint8)


def deterministic_seed(*keys):
    """由任务标识派生稳定种子。

    ProcessPoolExecutor 的 initializer 收不到 worker_id，无法靠 worker 编号播种；
    而 spawn 子进程会各自重新播种同一状态，导致噪声实现跨 worker 重复。
    改为按"任务内容"播种后，每个 (切片, 噪声) 的噪声实现固定，
    与 worker 数量、调度顺序、chunk 划分全都无关。
    """
    import zlib
    return zlib.crc32("|".join(str(k) for k in keys).encode("utf-8")) % (2 ** 31)


# =====================================================================
# 评估工具（数据集无关）
# =====================================================================
def build_eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_slices(model, loader, device="cuda"):
    """遍历 loader，返回 DataFrame: patient_id, label, pred, prob_positive(出血概率 1)。"""
    model.eval()
    rows = []
    for x, y, pid in loader:
        x = x.to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        pred = logits.argmax(1)
        for p, l, pr in zip(pid, y.tolist(), prob[:, 1].cpu().numpy()):
            rows.append({"patient_id": p, "label": l, "prob": pr})
    df = pd.DataFrame(rows)
    df["pred"] = (df["prob"] >= 0.5).astype(int)
    return df


def patient_vote_metrics(slice_df, out_path=None, tag=""):
    """按患者多数投票聚合切片预测，输出患者级指标与混淆矩阵。

    slice_df: 列 patient_id, label, pred, prob
    患者级预测: 该患者所有切片 prob 均值 >= 0.5 -> 出血；同时给出切片级准确率作参考。
    """
    g = slice_df.groupby("patient_id").agg(
        label=("label", "first"),
        mean_prob=("prob", "mean"),
        n_slices=("label", "count"))
    g["pred"] = (g["mean_prob"] >= 0.5).astype(int)
    y_true, y_pred = g["label"].values, g["pred"].values
    probs = g["mean_prob"].values
    res = {
        "num_patients": len(g),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
        "slice_accuracy": accuracy_score(slice_df["label"], slice_df["pred"]),
    }
    res["confusion"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()  # [[TN, FP], [FN, TP]]
    s = pd.Series({k: v for k, v in res.items() if k != "confusion"}, name="value")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(out_path)
        np.save(out_path.with_suffix(".cm.npy"), res["confusion"])
    print(f"[{tag}] 患者级(数={res['num_patients']}) acc={res['accuracy']:.4f} "
          f"prec={res['precision']:.4f} rec={res['recall']:.4f} f1={res['f1']:.4f} auc={res['auc']:.4f}")
    print(f"        混淆矩阵(TN,FP / FN,TP): {res['confusion']}")
    return res


def load_model(arch, weight_path, device="cuda"):
    import timm
    try:
        model = timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES)
    except RuntimeError as e:
        # 各脚本的 MODELS 字典里 key 是**显示短名**、值是 (完整架构名, 权重路径)。
        # 误把短名当架构名传进来时，timm 只会丢一句 "Unknown model"，很难查。
        raise RuntimeError(
            f"timm 不认识架构名 {arch!r}。必须传**完整**架构名"
            f"（如 'swin_tiny_patch4_window7_224'，而不是 'swin_tiny'）；"
            f"MODELS 字典的 key 只是结果表里的显示短名。"
        ) from e
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# =====================================================================
# TTA（测试时自适应）：不改输入图像，只让**分类器**在评测时适配目标域
#
#   接在"载入分类器 → 预测"之间：
#       载入 checkpoint → TTA 自适应（过一遍当前测试 loader）→ 预测概率 → 患者级投票
#
# 四个变体：
#   none      不 TTA，纯推理（对比基线）
#   bn        重估 BatchNorm 的 running mean/var 为测试分布统计量（不动权重）
#   tent      熵最小化：冻结主干，只用 -Σ p·log p 微调归一化层的 weight/bias
#   bn_tent   先 bn 对齐统计量，再 tent 锐化
#
# 两个必须知道的架构事实（已实测）：
#   resnet50 : BatchNorm=53, LayerNorm=0, 无 dropout    → bn 有效，tent 可调 0.053M 参数
#   swin     : BatchNorm=0 , LayerNorm=29, 49 Dropout + 22 DropPath
#              => bn 对 swin 是**数学上的空操作**（没有 BN 可更新），bn_tent ≡ tent
#
# 因此这里**不用 model.train()**：只把 BatchNorm 子模块置为 train（这样才有批统计），
# 其余模块保持 eval。swin 的 71 个随机化模块（Dropout/DropPath）一旦被打开，
# 熵损失的梯度会被随机深度污染，TENT 基本不可用；只切 BN 则 swin 的前向保持确定性。
# =====================================================================
TTA_VARIANTS = ("none", "bn", "tent", "bn_tent",
                "prior", "bn_prior", "priorq", "bn_priorq")

# 变体 -> 需要的**模型自适应**（None = 完全不动模型）
_MODEL_ADAPT = {
    "none": None, "bn": "bn", "tent": "tent", "bn_tent": "bn_tent",
    "prior": None, "bn_prior": "bn",
    "priorq": None, "bn_priorq": "bn",
}

# 变体 -> 先验校正口径：None / "mean"（匹配均值概率）/ "rate"（匹配阳性率）
_PRIOR_MODE = {"prior": "mean", "bn_prior": "mean",
               "priorq": "rate", "bn_priorq": "rate"}


def prior_correction_mode(variant):
    """该变体用的先验校正口径；None 表示不做校正。"""
    return _PRIOR_MODE.get(variant)


def uses_prior_correction(variant):
    """该变体是否需要在患者级决策前做先验校正。"""
    return prior_correction_mode(variant) is not None


def model_adaptation(variant):
    """该变体对应的模型自适应类型（None 表示不改模型）。"""
    if variant not in _MODEL_ADAPT:
        raise ValueError(f"未知 TTA 变体: {variant}（可选 {TTA_VARIANTS}）")
    return _MODEL_ADAPT[variant]


def match_prevalence(prob, target_prev, tol=1e-10, max_iter=200):
    """决策层先验校正：找一个 logit 偏移 b，使 sigmoid(logit(p)+b) 的均值 == 目标患病率。

    只挪决策边界，**不改模型、不改特征、不重排** —— 因此按构造不可能让 AUC 变差，
    最多是不起作用。这与熵最小化（改参数、会动排序）有本质区别。

    返回 (校正后的概率数组, b)。对 b 单调，二分即可，无需求解器。
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    logit = np.log(p / (1.0 - p))
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cur = float((1.0 / (1.0 + np.exp(-(logit + mid)))).mean())
        if abs(cur - target_prev) < tol:
            lo = hi = mid
            break
        if cur < target_prev:
            lo = mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)
    return 1.0 / (1.0 + np.exp(-(logit + b))), float(b)


def match_prevalence_rate(prob, target_prev):
    """决策层先验校正（按**阳性率**匹配）：找 logit 偏移 b，使被判阳性的患者比例 == 目标患病率。

    与 match_prevalence 的区别（实测得出）：均值概率匹配在模型过度自信时会**系统性过冲**——
    要让"平均概率"降到目标，阳性率会被压到目标之下（实测 0.7397→0.3562，目标 0.4143）。
    这里直接匹配决策率，这才是决策层校正的正确定义：
      * 阳性率已经正确时，校正量恒为 0（不会像均值匹配那样把对的推坏）；
      * 其余情况下精确落在目标率上，不再过冲。

    做法是取 logit 的第 k 大值作阈值（k = round(target*n)），因此仍严格单调 ⇒ AUC 不变。
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logit = np.log(p / (1.0 - p))
    n = len(logit)
    if n == 0:
        return np.asarray(prob, dtype=np.float64), 0.0
    k = int(round(target_prev * n))
    if k <= 0:
        b = -30.0
    elif k >= n:
        b = 30.0
    else:
        b = -float(np.sort(logit)[::-1][k - 1])      # 第 k 大的 logit 作为阈值
    return 1.0 / (1.0 + np.exp(-(logit + b))), float(b)


def mcnemar(a_correct, b_correct):
    """配对 McNemar 精确检验，返回 (b, c, p)。

    b = a 对且 b 错（变差），c = a 错且 b 对（变好）。n=0（无不一致对）时 p=1。
    """
    b = int(np.sum((a_correct == 1) & (b_correct == 0)))
    c = int(np.sum((a_correct == 0) & (b_correct == 1)))
    n = b + c
    p = 1.0 if n == 0 else binomtest(min(b, c), n, 0.5).pvalue
    return b, c, p


def boot_ci(diff, n_boot=5000, seed=0):
    """患者级配对差值的 bootstrap 95% CI。"""
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def norm_layer_summary(model):
    """归一化层构成，用来判断某个 TTA 变体在这个网络上是否可能有效。"""
    bn = ln = 0
    norm_p = tot_p = 0
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            bn += 1
        elif isinstance(mod, torch.nn.LayerNorm):
            ln += 1
    for p in model.parameters():
        tot_p += p.numel()
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.LayerNorm)):
            for p in mod.parameters(recurse=False):
                norm_p += p.numel()
    return {"batchnorm": bn, "layernorm": ln,
            "norm_params": norm_p, "total_params": tot_p,
            "norm_frac": (norm_p / tot_p) if tot_p else 0.0}


def collect_norm_params(model):
    """TENT 可学习的参数：归一化层的 weight/bias（BN 与 LayerNorm）。

    只动这一小撮（实测 resnet50 0.23%、swin 0.09%），主干权重全程冻结 ——
    这是 TENT 能在无标签测试集上不发散的前提。
    """
    params = []
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d, torch.nn.LayerNorm)):
            if mod.weight is not None:
                params.append(mod.weight)
            if mod.bias is not None:
                params.append(mod.bias)
    return params


def _set_bn_train(model, on):
    """只切换 BatchNorm 子模块的 train 状态，其余模块不受影响（见文件头说明）。"""
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
            mod.train(on)


def update_bn_stats(model, loader, device="cuda", max_batches=0):
    """bn 变体：把 BN 的 running mean/var 重估为**当前测试分布**的统计量。

    做法是清空旧统计量 + momentum=None（对全部测试样本做累积平均），
    在 no_grad 下前向；不改任何权重。
    """
    bns = [m for m in model.modules()
           if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                             torch.nn.BatchNorm3d))]
    if not bns:
        return {"bn_layers": 0, "batches": 0,
                "note": "该网络无 BatchNorm，bn 为空操作"}

    saved = [(m, m.momentum) for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                      # 累积平均 => 用全部测试样本
    was = model.training
    _set_bn_train(model, True)
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            model(x)
            n += 1
            if max_batches and n >= max_batches:
                break
    for m, mom in saved:
        m.momentum = mom
    _set_bn_train(model, False)
    if was:
        model.train()
    else:
        model.eval()
    return {"bn_layers": len(bns), "batches": n}


def tent_adapt(model, loader, device="cuda", lr=1e-4, max_batches=0):
    """tent 变体：冻结主干，用熵损失 -Σ p·log p 微调归一化层参数。

    熵越小 => softmax 分布越尖锐 => 模型对目标域更"有主见"。

    **归一化统计量全程冻结**（前向走 eval，BN 用 running stats 且不更新）：
      - 这样被优化的目标（eval 口径下的熵）与实际评测口径完全一致。
        若改在 train 模式下优化，梯度是按批统计算的、评测却按 running 统计，
        实测会出现"训练目标在降、评测熵反而升"的错配
        （clean 0.146->0.171、gaussian 0.112->0.170）。
      - 也让 tent 与 bn 不重叠：bn 负责对齐统计量，tent 只负责锐化参数的锐化，
        bn_tent 才是"先对齐再锐化"。
    """
    params = collect_norm_params(model)
    if not params:
        return {"tunable": 0, "steps": 0, "entropy_before": None,
                "entropy_after": None, "note": "无可学习的归一化参数"}

    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    model.eval()                               # 统计量冻结 + 关闭 Dropout/DropPath
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)

    def _mean_entropy():
        tot, nb = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                p = F.softmax(model(x), dim=1)
                tot += float(-(p * torch.log(p.clamp_min(1e-12))).sum(1).mean())
                nb += 1
                if max_batches and nb >= max_batches:
                    break
        return tot / max(nb, 1)

    ent_before = _mean_entropy()
    steps = 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        p = F.softmax(model(x), dim=1)
        loss = -(p * torch.log(p.clamp_min(1e-12))).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        if max_batches and steps >= max_batches:
            break
    ent_after = _mean_entropy()

    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return {"tunable": sum(p.numel() for p in params), "steps": steps,
            "entropy_before": ent_before, "entropy_after": ent_after}


def apply_tta(model, loader, device="cuda", variant="none", lr=1e-4, max_batches=0):
    """按变体跑 TTA，返回 (model, info)。model 每次都应从 checkpoint 重新载入。

    loader 契约：每个 batch 是 (x, ...)，且 x 必须是**已堆叠好的** batch 张量
    （即用 DataLoader 默认 collate；自定义 collate 若返回 list-of-tuple 会在
    batch[0].to(device) 处报 'tuple' object has no attribute 'to'）。
    """
    if variant not in TTA_VARIANTS:
        raise ValueError(f"未知 TTA 变体: {variant}（可选 {TTA_VARIANTS}）")
    info = {"tta": variant}
    adapt = model_adaptation(variant)
    if adapt is None:
        # none / prior：不动模型。prior 的校正发生在患者级决策前（见 uses_prior_correction）
        model.eval()
        return model, info

    if adapt in ("bn", "bn_tent"):
        info["bn"] = update_bn_stats(model, loader, device, max_batches=max_batches)
    if adapt in ("tent", "bn_tent"):
        info["tent"] = tent_adapt(model, loader, device, lr=lr,
                                  max_batches=max_batches)
    model.eval()
    return model, info


# =====================================================================
# filter 穷举：CPU 预处理全部放进 DataLoader worker 并行执行
# 16（最优性分析）与 07（val 拟合 / test 评估）共用
# =====================================================================
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(arr_bgr):
    """与 build_eval_transform 等价，但走纯 numpy/cv2，省掉 PIL 往返。

    两者的一致性由 17_verify 的 K1 检查逐元素把关（atol=1e-6）。
    数据本来就是 224x224，所以只在尺寸不符时才真的做重采样：
    PIL 的同尺寸 resize 也会走一次重采样，实测占了这里绝大部分开销。
    """
    rgb = arr_bgr[..., ::-1]                       # BGR->RGB，视图，等价 cvtColor
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = np.asarray(Image.fromarray(np.ascontiguousarray(rgb))
                         .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(((x - _MEAN) / _STD).transpose(2, 0, 1))


class _SweepDataset(Dataset):
    """按 (切片, filter) 展开的数据集：长度 = n_slices * n_filters。

    把"解码 + 加噪 + 滤波 + resize + normalize"全放进 worker 并行做。
    旧实现把这些串行放在主进程里、每张切片只喂 batch_size 张进 GPU，
    导致 GPU 长期饥饿（实测利用率约 25%，worker 反而闲置在 5%）。

    k = i * n_filters + j 的排布让同一张切片的 n_filters 项相邻，
    worker 一次取到的就是同患者同序列的图，磁盘局部性也好。

    加噪种子只由 (noise_name, image_path) 或 (seed, slice_index) 决定，
    **与 filter 无关** => 同一张切片的 n_filters 项共享同一噪声实现，
    filter 之间仍然可比（与原实现语义一致）。

    cache 非空时从内存缓存取图（见 sweep_filters 的 preload）：
    每张切片会被 n_filters 个样本各读一次文件，10 倍读放大。
    数据集小（如 RSNA 子集）而磁盘是机械盘时，这部分 I/O 会彻底拖死 GPU
    （实测 GPU 6%、worker 空闲、主进程在等盘）。把它放进内存即可。
    """

    def __init__(self, df, base, noise_name, filter_names, from_disk, seed, cache=None):
        self.df = df.reset_index(drop=True)
        self.base = Path(base)
        self.noise_name = noise_name
        self.filter_names = list(filter_names)
        self.from_disk = from_disk
        self.seed = seed
        self.cache = cache

    def __len__(self):
        return len(self.df) * len(self.filter_names)

    def __getitem__(self, k):
        import cv2
        nf = len(self.filter_names)
        i, j = divmod(k, nf)
        if self.cache is not None:
            arr = self.cache[i]
        else:
            rel = self.df.iloc[i]["image_path"]
            img = Image.open(self.base / rel).convert("RGB")
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if not self.from_disk:
                if self.seed is None:
                    np.random.seed(deterministic_seed(self.noise_name, rel))
                else:
                    np.random.seed(self.seed + i)
                arr = apply_noise(arr, self.noise_name)
        arr = apply_filter(arr, self.filter_names[j])
        return _to_tensor(arr), i, j


def sweep_filters(split, noise_name, filter_names, model, device="cuda",
                  batch_size=64, num_workers=8, from_disk=False, seed=None, limit=0,
                  index_df=None, data_root=None, noisy_root=None, classes=None,
                  preload=None, preload_max_slices=8000):
    """对同一批切片穷举 filter_name，各跑一次分类器。

    路径与索引可显式覆盖（index_df / data_root / noisy_root / classes），
    使 RSNA 能复用**同一实现**而不必再写一份 —— 两份实现迟早会漂移，
    那正是 CQ500 早期踩过的坑。默认（全为 None）保持 CQ500 原行为。

    split       : "val" / "test"
    noise_name  : 噪声类别（"unknown_mixed" / "gaussian" / ... / "clean"）
    from_disk   : True  -> 读 {noisy_root}/{noise_name} 下已生成的加噪图（与 04 同一实现）
                  False -> 从 {data_root} 读原图，在内存加噪（用于 val 拟合，不落盘）
    limit       : 仅取该 split 的前 N 张切片（0=全部），快速验证用
    preload     : True/False 强制开/关内存预载；None=按规模自动决定。
                  每张切片会被 n_filters 个样本各读一次文件（10 倍读放大），
                  在机械盘上会拖死 GPU。切片数 <= preload_max_slices 时自动预载。
    返回        : {filter_name: DataFrame[patient_id, label, prob, slice_index]}
    """
    from torch.utils.data import DataLoader

    # 数据集相关的入参必须由调用方（各 <dataset>_commons 的包装）显式给出，
    # 否则容易静默读到别的数据集的索引/目录。
    if index_df is None or data_root is None or noisy_root is None:
        raise ValueError(
            "sweep_filters 需要显式传入 index_df / data_root / noisy_root；"
            "请使用 cq500_commons.sweep_filters 或 rsna_commons.sweep_filters"
            "（它们已绑定各自数据集的路径）")
    classes = list(classes) if classes is not None else CLASSES
    df = index_df
    df = df[df["split"] == split].reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    data_root = Path(data_root)
    noisy_root = Path(noisy_root)

    base = (noisy_root / noise_name) if from_disk else data_root

    # 内存预载：只在数据量小的时候做（大 split 会把内存吃光）
    if preload is None:
        preload = len(df) <= preload_max_slices
    cache = None
    if preload:
        import cv2
        cache = []
        for i, rel in enumerate(df["image_path"]):
            img = Image.open(base / rel).convert("RGB")
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if not from_disk:
                if seed is None:
                    np.random.seed(deterministic_seed(noise_name, rel))
                else:
                    np.random.seed(seed + i)
                arr = apply_noise(arr, noise_name)
            cache.append(arr)
        mb = sum(a.nbytes for a in cache) / 1048576
        print(f"[preload] {split}/{noise_name}: {len(cache)} 张切片读入内存 "
              f"({mb:.0f} MB)，避免 {len(filter_names)}x 读放大")
        # 内存里已有数据，多进程反而要把缓存 pickle 给每个 worker，故不并行
        num_workers = 0

    nf = len(filter_names)
    ds = _SweepDataset(df, base, noise_name, filter_names, from_disk, seed, cache=cache)
    loader_kw = dict(num_workers=num_workers, pin_memory=True,
                     persistent_workers=num_workers > 0,
                     prefetch_factor=4 if num_workers > 0 else None)
    loader_kw = {k: v for k, v in loader_kw.items() if v is not None}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kw)

    labels = [classes.index(l) for l in df["label"]]
    pids = df["patient_id"].tolist()
    probs = np.zeros((len(df), nf), dtype=np.float32)

    model.eval()
    total, done = len(df) * nf, 0
    with torch.no_grad():
        for xb, idx_b, jb in dl:
            p = F.softmax(model(xb.to(device, non_blocking=True)), dim=1)[:, 1]
            p = p.cpu().numpy()
            for n in range(len(p)):
                probs[int(idx_b[n]), int(jb[n])] = p[n]
            done += len(p)
            print(f"[sweep] {split}/{noise_name} {done}/{total} 张", end="\r", flush=True)
    print()

    out = {}
    for j, f in enumerate(filter_names):
        d = pd.DataFrame({"patient_id": pids, "label": labels,
                          "prob": probs[:, j],
                          "slice_index": np.arange(len(df), dtype=int)})
        d["pred"] = (d["prob"] >= 0.5).astype(int)
        out[f] = d
    return out


# ---- Grad-CAM 工具（resnet 系，基于 layer4 钩子） ----
GRADCAM_TF = None  # 在 initialize_gradcam_tf() 中构建


def initialize_gradcam_tf():
    global GRADCAM_TF
    GRADCAM_TF = build_eval_transform()
    return GRADCAM_TF


def grad_cam(model, x, target_layer_name="layer4"):
    import cv2
    grads, acts = {}, {}

    def fwd_hook(m, i, o): acts["v"] = o
    def bwd_hook(m, i, o): grads["v"] = o[0]

    layer = dict(model.named_modules())[target_layer_name]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    model.zero_grad()
    out = model(x)
    out[0, 1].backward()                     # 出血类
    h1.remove(); h2.remove()
    a = acts["v"][0]; g = grads["v"][0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = (w * a).sum(0).clamp(min=0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cv2.resize(cam.detach().cpu().numpy(), (x.shape[-2], x.shape[-1]))
    return cam


def overlay(img_bgr, cam):
    import cv2
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.55, heat, 0.45, 0)
