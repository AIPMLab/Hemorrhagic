# -*- coding: utf-8 -*-
"""
19 TTA 消融（PhysioNet）：在复合未知噪声上，用测试时自适应把分类器的实际效果再抬一截

回答的问题：TTA 能否提升"检测器 → 匹配 filter"链路的最终效果，尤其是 6class 那条路径？

流程（**每个组合都从 checkpoint 重新载入模型**，避免变体之间互相污染）：
    载入分类器 → 在该条件的测试分布上自适应 → 预测 → 患者级投票 →（可选）决策层先验校正

条件：
    raw             干净测试图（对照：TTA 不应损害干净性能）
    unknown_noisy   复合噪声、未增强（等价 filter=none）
    adaptive        复合噪声 + 11 逐切片选出的 filter（6class / 7class 各一套）

TTA 变体（定义见 ich_common）：
    none / bn / tent / bn_tent        模型侧自适应
    prior / bn_prior                  决策层先验校正（均值概率口径）
    priorq / bn_priorq                决策层先验校正（阳性率口径）

5 折交叉验证下本脚本必须按折运行 —— 路径全部来自 physionet_commons，
设了 PHYSIONET_FOLD=k 就会自动落到 results_fold{k}/tta_unknown_physionet/。
整轮跑完由 physionet_run_cv.py 逐折驱动。

三点必须留意的口径：
  1. 自适应**只使用测试图像，不使用任何标签**（transductive TTA）——这是 TTA 的正当用法，
     但论文里必须写明"自适应与评测使用同一批无标签测试数据"。
     先验校正的目标患病率取自 **val** 划分，不使用 test 标签。
  2. `none` 行会与 11 的 adaptive_accuracy 交叉核对；对不上说明两次口径不同，结果不可比。
  3. `bn` 对 swin 是空操作（swin 无 BatchNorm），故 swin 的 bn 行应与 none 几乎相同、
     bn_tent 行应与 tent 几乎相同——这是预期，不是 bug。

PhysioNet 侧的背景（解读时要留神）：
  - 5 折下每折 test 只有 16~18 名患者：**1 名患者 ≈ 6 个百分点**。
    单折内的 Δ 基本不可能显著，请以 physionet_aggregate_cv.py 合并 82 例后的结果为准；
    本脚本的 vs_none_p 只用于逐折诊断。
  - 目标患病率来自该折的 val 划分，因此**逐折不同**（各折 val 的阳性率不一样），
    这是交叉验证下的正确做法，但并排比较时不要误当成同一个阈值。

用法:
    python 19_tta_unknown_physionet.py                      # 全部组合
    python 19_tta_unknown_physionet.py --tta none,bn,priorq # 只跑部分变体
    python 19_tta_unknown_physionet.py --limit-patients 4   # 冒烟

    # 交叉验证下逐折跑（或直接 python physionet_run_cv.py 一次跑完 5 折）
    PHYSIONET_FOLD=0 python 19_tta_unknown_physionet.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import physionet_commons as C  # noqa: E402  先导入它：它会确保 code 根在 sys.path 上
# 只从共享层取**数据集无关**的东西；路径与索引一律走 physionet_commons
# （ich_common 的 load_index/raw_dataset_dir/enhanced_dir 是有意留的占位，调用即报错）
from ich_common import (CLASSES, TTA_VARIANTS, _to_tensor,  # noqa: E402
                        apply_filter, apply_tta, boot_ci, load_model,
                        match_prevalence, match_prevalence_rate, mcnemar,
                        norm_layer_summary, patient_vote_metrics,
                        prior_correction_mode, uses_prior_correction)

RESULT_DIR = C.RESULT_ROOT / "tta_unknown_physionet"
NOISE = "unknown_mixed"

# key = 结果表里显示的短名（与 07/11/16 保持一致，便于 15/18 按 model 列对齐）
# value = (timm 架构名, 权重路径) —— 架构名必须完整，timm 不认 "swin_tiny" 这种简写
MODELS = {
    "resnet50": ("resnet50", C.MODEL_DIR / "resnet50_physionet_best.pth"),
    "swin_tiny": ("swin_tiny_patch4_window7_224",
                  C.MODEL_DIR / "swin_tiny_patch4_window7_224_physionet_best.pth"),
}


class ConditionDataset(Dataset):
    """按条件产出分类器输入；adaptive 的 filter 来自 11 的逐切片决策日志。"""

    def __init__(self, df, condition, filter_by_path=None):
        self.df = df.reset_index(drop=True)
        self.condition = condition
        self.filter_by_path = filter_by_path or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        rel = row["image_path"]
        if self.condition == "raw":
            arr = cv2.imread(str(C.DATA_ROOT / rel))
            src = C.DATA_ROOT / rel
        else:
            src = C.NOISY_ROOT / NOISE / rel
            arr = cv2.imread(str(src))
        if arr is None:
            raise FileNotFoundError(f"读不到图: {src}")
        if self.condition == "adaptive":
            f = self.filter_by_path.get(rel)
            if f is None:
                raise KeyError(f"决策日志里没有该切片: {rel}")
            arr = apply_filter(arr, f)
        # _to_tensor 的契约是 BGR numpy（不是 PIL），整条链路保持 numpy
        return _to_tensor(arr), CLASSES.index(row["label"]), row["patient_id"]


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    pids = [b[2] for b in batch]
    return xs, ys, pids


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    rows = []
    for x, y, pids in loader:
        p = torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1]
        p = p.float().cpu().numpy()
        for j, pid in enumerate(pids):
            rows.append({"patient_id": pid, "label": int(y[j]),
                         "prob": float(p[j])})
    d = pd.DataFrame(rows)
    d["pred"] = (d["prob"] >= 0.5).astype(int)
    return d


def load_filter_log(variant, model_name):
    """11 的逐切片决策日志 -> {image_path: selected_filter}。"""
    p = C.RESULT_ROOT / "adaptive_unknown_physionet" / variant / \
        "unknown_filter_selection_log_physionet.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["model"] == model_name]
    if d.empty:
        return None
    return dict(zip(d["image_path"], d["selected_filter"]))


def run_combo(display, arch, weight, condition, tta, df, filter_by_path,
              device, args):
    """fresh 载入 -> 自适应 -> 评估，返回 (指标 dict, 自适应 info, 患者级表)。"""
    ds = ConditionDataset(df, condition, filter_by_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=_collate,
                        persistent_workers=args.num_workers > 0)

    model = load_model(arch, weight, device)
    info = {}
    if tta != "none":
        # 自适应用的 loader 与评估 loader 是同一批无标签测试数据（transductive TTA）
        model, info = apply_tta(model, loader, device, tta,
                                lr=args.tent_lr, max_batches=args.tta_batches)

    slice_df = collect_probs(model, loader, device)

    # 决策层先验校正：只挪患者级决策边界，不改模型
    #   mean 口径 = prior / bn_prior      （匹配均值概率；实测会系统性过冲）
    #   rate 口径 = priorq / bn_priorq    （匹配阳性率；率已正确时校正量为 0）
    extra = {}
    mode = prior_correction_mode(tta)
    if mode is not None:
        pp = slice_df.groupby("patient_id")["prob"].mean()
        fn = match_prevalence if mode == "mean" else match_prevalence_rate
        corr, bias = fn(pp.values, args.target_prev)
        cmap = dict(zip(pp.index, corr))
        # 用校正后的患者概率覆盖 prob 列；pred 列保持不动，
        # 这样 slice_accuracy 仍是未校正的切片级准确率（语义不变）
        slice_df = slice_df.copy()
        slice_df["prob"] = slice_df["patient_id"].map(cmap)
        extra = {"prior_mode": mode,
                 "prior_target": round(args.target_prev, 4),
                 "prior_bias": round(bias, 4),
                 "pos_rate_before": round(float((pp.values >= 0.5).mean()), 4),
                 "pos_rate_after": round(float((corr >= 0.5).mean()), 4)}

    m = patient_vote_metrics(slice_df, tag=f"{display}-{condition}-{tta}")
    m.update(extra)
    # 患者级 (label, pred)，供配对 McNemar / bootstrap 使用
    per_patient = (slice_df.groupby("patient_id")
                   .agg(label=("label", "first"), prob=("prob", "mean")))
    per_patient["pred"] = (per_patient["prob"] >= 0.5).astype(int)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return m, info, per_patient


def _flat(prefix, d, out):
    """把 apply_tta 返回的嵌套 info 摊平成一行，便于写 CSV。"""
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            _flat(key, v, out)
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resnet50,swin_tiny")
    ap.add_argument("--variants", default="6class,7class")
    ap.add_argument("--conditions", default="raw,unknown_noisy,adaptive")
    ap.add_argument("--tta", default=",".join(TTA_VARIANTS),
                    help=f"逗号分隔，可选 {TTA_VARIANTS}；prior/bn_prior 为决策层先验校正")
    ap.add_argument("--target-prev", type=float, default=None,
                    help="先验校正的目标患病率；默认取该折 val 划分的患者级阳性率")
    ap.add_argument("--tent-lr", type=float, default=1e-4)
    ap.add_argument("--tta-batches", type=int, default=0,
                    help="自适应最多用多少个 batch（0=全部测试集）")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="TENT 需要反向传播，显存小时别开太大")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 张切片（冒烟用）")
    ap.add_argument("--limit-patients", type=int, default=0,
                    help="每类各取前 N 个患者（冒烟用；索引按类别排序，"
                         "单用 --limit 只会取到一个类，患者级指标会退化）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [m for m in args.models.split(",") if m]
    variants = [v for v in args.variants.split(",") if v]
    conditions = [c for c in args.conditions.split(",") if c]
    tta_list = [t for t in args.tta.split(",") if t]
    for t in tta_list:
        if t not in TTA_VARIANTS:
            raise SystemExit(f"未知 TTA 变体 {t}，可选 {TTA_VARIANTS}")

    df = C.load_index()
    df = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit_patients:
        keep = []
        for lab, g in df.groupby("label", sort=True):
            keep.extend(g["patient_id"].drop_duplicates().head(args.limit_patients))
        df = df[df["patient_id"].isin(keep)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"[19] 折 {C.fold_tag()}  test 切片 {len(df)}，"
          f"患者 {df['patient_id'].nunique()}，设备 {device}")

    # 先验校正的目标患病率：默认取该折 val 的患者级阳性率（标注参考集，与 test 无关）
    if args.target_prev is None:
        val = C.load_index()
        val = val[val["split"] == "val"]
        vp = val.groupby("patient_id")["label"].first()
        args.target_prev = float((vp == CLASSES[1]).mean())
    base_df = df
    if any(uses_prior_correction(t) for t in tta_list):
        if not 0.0 < args.target_prev < 1.0:
            raise SystemExit(f"目标患病率非法: {args.target_prev}")
        obs = base_df.groupby("patient_id")["label"].first()
        obs = float((obs == CLASSES[1]).mean())
        print(f"[19] 目标患病率(来自 val) = {args.target_prev:.4f}；"
              f"test 实际 = {obs:.4f}（仅供核对，校正不使用它）")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 归一化层构成（解释 bn 对 swin 为何是空操作）
    norm_rows = []
    for mn in models:
        arch, weight = MODELS[mn]
        if not weight.exists():
            raise SystemExit(f"缺权重: {weight}（先跑 02）")
        m0 = load_model(arch, weight, device)
        s = norm_layer_summary(m0)
        s["model"] = mn
        norm_rows.append(s)
        del m0
        print(f"[19] {mn}: BatchNorm={s['batchnorm']} LayerNorm={s['layernorm']} "
              f"可调归一化参数={s['norm_params']} ({100 * s['norm_frac']:.2f}%)")
    pd.DataFrame(norm_rows).to_csv(RESULT_DIR / "norm_layers_physionet.csv",
                                   index=False, encoding="utf-8-sig")

    rows, infos = [], []
    for model_name in models:
        arch, weight = MODELS[model_name]
        for condition in conditions:
            variant_list = variants if condition == "adaptive" else ["-"]
            for variant in variant_list:
                fmap = None
                if condition == "adaptive":
                    fmap = load_filter_log(variant, model_name)
                    if fmap is None:
                        print(f"[19] 跳过 {model_name}/{condition}/{variant}: "
                              f"缺 11 的决策日志")
                        continue
                group = []                      # 本组（同 model/variant/condition）的行
                per_tta = {}                    # tta -> 患者级 (label, pred)
                for tta in tta_list:
                    m, info, pp = run_combo(model_name, arch, weight, condition, tta,
                                            df, fmap, device, args)
                    per_tta[tta] = pp
                    rec = {"model": model_name, "detector_variant": variant,
                           "condition": condition, "tta": tta,
                           "num_patients": m["num_patients"],
                           "accuracy": m["accuracy"], "precision": m["precision"],
                           "recall": m["recall"], "f1": m["f1"], "auc": m["auc"],
                           "slice_accuracy": m["slice_accuracy"],
                           # 仅先验校正变体有值，用于审计校正幅度
                           "prior_mode": m.get("prior_mode", ""),
                           "prior_target": m.get("prior_target", ""),
                           "prior_bias": m.get("prior_bias", ""),
                           "pos_rate_before": m.get("pos_rate_before", ""),
                           "pos_rate_after": m.get("pos_rate_after", "")}
                    group.append(rec)
                    rows.append(rec)
                    if info:
                        rec_i = {"model": model_name, "condition": condition,
                                 "detector_variant": variant, "tta": tta}
                        _flat("", info, rec_i)
                        infos.append(rec_i)
                    tail = ""
                    if m.get("pos_rate_before", "") != "":
                        tail = (f" 阳性率 {m['pos_rate_before']:.3f}"
                                f"->{m['pos_rate_after']:.3f}"
                                f"(目标{m['prior_target']:.3f})")
                    print(f"  {model_name:9s} {condition:14s} "
                          f"{variant:8s} tta={tta:10s} acc={m['accuracy']:.4f} "
                          f"f1={m['f1']:.4f}{tail}")

                # 配对检验：每个变体 vs 同组内的 none（同一模型、同一条件的基线）
                base = per_tta.get("none")
                if base is not None:
                    for rec in group:
                        if rec["tta"] == "none":
                            continue
                        other = per_tta.get(rec["tta"])
                        if other is None:
                            continue
                        j = base.join(other, lsuffix="_a", rsuffix="_b", how="inner")
                        a = (j["pred_a"] == j["label_a"]).to_numpy(dtype=int)
                        bb = (j["pred_b"] == j["label_b"]).to_numpy(dtype=int)
                        nb, nc, p = mcnemar(a, bb)
                        lo, hi = boot_ci(bb - a)
                        rec.update({"vs_none_delta": float(bb.mean() - a.mean()),
                                    "vs_none_b": nb, "vs_none_c": nc,
                                    "vs_none_p": p,
                                    "vs_none_ci_low": lo, "vs_none_ci_high": hi})

    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "tta_unknown_results_physionet.csv",
               index=False, encoding="utf-8-sig")
    if infos:
        pd.DataFrame(infos).to_csv(RESULT_DIR / "tta_adaptation_info_physionet.csv",
                                   index=False, encoding="utf-8-sig")

    # ---- 交叉核对：tta=none 的 adaptive 行必须与 11 报告的数字一致 ----
    print("\n[交叉核对] adaptive + tta=none  vs  11 的 adaptive_accuracy")
    base = out[(out["condition"] == "adaptive") & (out["tta"] == "none")]
    for _, r in base.iterrows():
        p = C.RESULT_ROOT / "adaptive_unknown_physionet" / r["detector_variant"] / \
            "adaptive_unknown_results_physionet.csv"
        if not p.exists():
            print(f"  {r['model']}/{r['detector_variant']}: 缺 11 的结果，跳过")
            continue
        d = pd.read_csv(p)
        d = d[(d["model"] == r["model"]) & (d["condition"] == "adaptive")]
        if d.empty:
            continue
        ref = float(d["accuracy"].iloc[0])
        ok = "一致" if abs(ref - r["accuracy"]) < 1e-9 else "**不一致**"
        print(f"  {r['model']:9s} {r['detector_variant']:8s} "
              f"{r['accuracy']:.6f} vs 11: {ref:.6f}  {ok}")

    # ---- 关键对比：TTA 带来的增量（含配对检验） ----
    print("\n[TTA 增益] adaptive 条件下，相对 tta=none 的 Δaccuracy / Δrecall")
    ad = out[out["condition"] == "adaptive"]
    for (mn, v), g in ad.groupby(["model", "detector_variant"]):
        b = g[g["tta"] == "none"]
        if b.empty:
            continue
        b0 = float(b["accuracy"].iloc[0])
        r0 = float(b["recall"].iloc[0])
        for _, r in g.sort_values("tta").iterrows():
            p = r.get("vs_none_p", "")
            ps = f" p={p:.4f}" if isinstance(p, float) else ""
            print(f"  {mn:9s} {v:8s} {r['tta']:10s} acc={r['accuracy']:.4f} "
                  f"({r['accuracy'] - b0:+.4f})  rec={r['recall']:.4f} "
                  f"({r['recall'] - r0:+.4f}){ps}")

    print(f"\n[19] 写出 {RESULT_DIR / 'tta_unknown_results_physionet.csv'}")
    if len(out):
        print(f"[19] 提示：单折 test 仅 {int(out['num_patients'].max())} 名患者，"
              f"单折 Δ 基本不可能显著；请以 physionet_aggregate_cv.py "
              f"合并 82 例后的结果为准。")


if __name__ == "__main__":
    main()
