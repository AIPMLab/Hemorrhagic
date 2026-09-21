# -*- coding: utf-8 -*-
"""
20 边界位移校正（PhysioNet）：把先验校正铺到**全部噪声条件**（clean + 5 类已知噪声 + 复合）

动机（来自 CQ500 与 RSNA 的已有结果）
    噪声造成的端到端损失主要不是"排序坏了"，而是"决策边界位移"：
    CQ500 上 resnet50 在 motion_blur 的 AUC 仍有 0.8233，准确率却只有 0.5479 —— 27.5 点落差。
    RSNA 上同一个模型在复合噪声下更极端：准确率 0.6234（看着"还行"），
    细看混淆矩阵 [[43,3],[26,5]] —— 召回只有 0.161，26/31 的出血患者被判成 Normal。
    即模型并没有丢失排序能力，而是整体把分数压到了阈值以下。
    本脚本就是去实测这个可恢复空间在**每个**条件下到底有多少。

设计（关键：不重复推理）
    none / prior / priorq 只在**患者级分数**上后处理，不需要重新前向。
    所以每个 (模型, 条件) 最多 3 次前向：
        plain 评估(1) + BN 自适应(1) + BN 后评估(1)
    6 个变体全部由这两组分数后处理得到。
    若模型没有 BatchNorm（如 swin），bn 系列与无 bn **逐位等价**（已在 19 中验证
    为 0 个不一致对），直接复用 plain 分数，省掉 2/3 的前向。

产出
    results_fold{k}/boundary_correction_physionet/boundary_all_noise_physionet.csv
    每行 = (模型, 噪声条件, 变体)，含校正前后阳性率、相对同条件 none 的配对检验，
    以及由 AUC 推出的可恢复上限（oracle_thresh_acc），用于判断"还剩多少没拿到"。

PhysioNet 解读提醒（比另两套更要注意）
    5 折下每折 test 只有 16~18 名患者：**1 名患者 ≈ 6 个百分点**。
    单折内 Δ 小于 ~12% 时配对检验基本不可能显著，请以 vs_none_p / vs_none_ci_*
    为准，不要只看 Δ 的正负；跨折结论请把各折结果交给 physionet_aggregate_cv.py 汇总。
    另外目标患病率取自**该折的 val 划分**，因此逐折不同 —— 这是交叉验证下的正确做法。

用法:
    python 20_boundary_correction_physionet.py
    python 20_boundary_correction_physionet.py --conditions unknown_mixed
    python 20_boundary_correction_physionet.py --limit-patients 4   # 冒烟

    # 交叉验证下逐折跑（或直接 python physionet_run_cv.py 一次跑完 5 折）
    PHYSIONET_FOLD=0 python 20_boundary_correction_physionet.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import physionet_commons as C  # noqa: E402
from ich_common import (CLASSES, _to_tensor, boot_ci, load_model,  # noqa: E402
                        match_prevalence, match_prevalence_rate, mcnemar,
                        norm_layer_summary, patient_vote_metrics,
                        prior_correction_mode, update_bn_stats)

OUT_DIR = C.RESULT_ROOT / "boundary_correction_physionet"

# clean 是原始图目录，其余在 NOISY_ROOT/<noise>/
CONDITIONS = ["clean", "gaussian", "salt_pepper", "speckle",
              "motion_blur", "low_light", "unknown_mixed"]
# {无自适应, BN} x {不校正, 均值口径, 阳性率口径}
VARIANTS = ["none", "prior", "priorq", "bn", "bn_prior", "bn_priorq"]
BN_VARIANTS = {"bn", "bn_prior", "bn_priorq"}

POS = CLASSES[1]                      # 正类（Hemorrhagic）


# ---------------------------------------------------------------- 数据
def src_dir(cond):
    return C.DATA_ROOT if cond == "clean" else C.NOISY_ROOT / cond


class _CondDataset(Dataset):
    """按噪声条件产出分类器输入（干净图直接读，其余读 04 生成的加噪图）。"""

    def __init__(self, df, cond):
        self.df = df.reset_index(drop=True)
        self.base = src_dir(cond)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        rel = row["image_path"]
        arr = cv2.imread(str(self.base / rel))
        if arr is None:
            raise FileNotFoundError(f"读不到图像: {self.base / rel}（先跑 04）")
        return _to_tensor(arr), CLASSES.index(row["label"]), row["patient_id"]


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys, [b[2] for b in batch]


def make_loader(df, cond, bs, workers):
    return DataLoader(_CondDataset(df, cond), batch_size=bs, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      collate_fn=_collate,
                      persistent_workers=workers > 0)


# ---------------------------------------------------------------- 推理
def patient_scores(model, loader, device):
    """返回切片级 DataFrame（patient_id, label, prob, pred）。"""
    model.eval()
    rows = []
    with torch.no_grad():
        for xb, yb, pids in loader:
            p = torch.softmax(model(xb.to(device, non_blocking=True)), 1)[:, 1]
            p = p.detach().cpu().numpy()
            for k, pid in enumerate(pids):
                rows.append((pid, int(yb[k]), float(p[k])))
    df = pd.DataFrame(rows, columns=["patient_id", "label", "prob"])
    df["pred"] = (df["prob"] >= 0.5).astype(int)
    return df


# ---------------------------------------------------------------- 校正
def oracle_thresh_acc(slice_df):
    """遍历阈值所能达到的最高患者级准确率 —— 这才是"只动边界"这类方法的真上限。

    做法：按分数降序排列，枚举"预测前 k 名为正"的所有 k，取正确数最大者。
    注意它**用了标签**，所以只是个诊断用的上界，不能当作可实现的成绩。
    priorq 与它的差距 = 患病率匹配离最优工作点还差多远。
    """
    g = per_patient(slice_df)
    y = g["label"].to_numpy(int)[np.argsort(-g["prob"].to_numpy())]
    n = len(y)
    if n == 0:
        return float("nan")
    tp = np.cumsum(y)                       # 前 k 名中真阳数
    fp = np.arange(1, n + 1) - tp           # 前 k 名中假阳数
    n_neg = int((1 - y).sum())
    correct = np.concatenate([[n_neg], tp + (n_neg - fp)])   # k=0 也要算
    return float(correct.max() / n)


def apply_correction(slice_df, mode, target):
    """在**患者级**概率上做边界校正；返回 (新切片 df, 审计字典)。

    只改 prob 列（按患者广播），pred 列保持原样 —— 于是 patient_vote_metrics 里
    患者级指标用校正后的值，而 slice_accuracy 仍是未校正的切片准确率（语义正确）。
    """
    if mode is None:
        return slice_df, {}
    pp = slice_df.groupby("patient_id")["prob"].mean()
    before = float((pp >= 0.5).mean())
    fn = match_prevalence if mode == "mean" else match_prevalence_rate
    corr, bias = fn(pp.to_numpy(), target)
    cmap = dict(zip(pp.index, corr))
    out = slice_df.copy()
    out["prob"] = out["patient_id"].map(cmap)
    return out, {"prior_mode": mode, "prior_target": target,
                 "prior_bias": round(float(bias), 4),
                 "pos_rate_before": round(before, 4),
                 "pos_rate_after": round(float((np.asarray(corr) >= 0.5).mean()), 4)}


def per_patient(slice_df):
    g = slice_df.groupby("patient_id").agg(
        label=("label", "first"), prob=("prob", "mean"))
    g["pred"] = (g["prob"] >= 0.5).astype(int)
    return g


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resnet50,swin_tiny")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--batch-size", type=int, default=32,
                    help="BN 自适应要反向，显存小就别开大")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张切片（冒烟）")
    ap.add_argument("--limit-patients", type=int, default=0,
                    help="每类各取 N 个患者（冒烟；索引按类别排序，别用 --limit）")
    ap.add_argument("--target-prev", type=float, default=None,
                    help="覆盖目标患病率；默认取该折 val 划分的患者级阳性率")
    ap.add_argument("--dump-scores", action="store_true",
                    help="额外落盘逐患者分数，供 21_prior_sensitivity.py 离线做先验敏感性扫描"
                         "（不增加任何前向，只多写一个小 CSV）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = C.load_index()
    vp = df[df["split"] == "val"].groupby("patient_id")["label"].first()
    target = float((vp == POS).mean()) if args.target_prev is None else args.target_prev
    print(f"[20] 折 {C.fold_tag()}  设备={device} 目标患病率={target:.4f}"
          f"（来自 val，{len(vp)} 名患者；正类={POS}）")

    df = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit_patients:
        keep = (df.groupby("label")["patient_id"].unique()
                .apply(lambda ps: list(ps)[:args.limit_patients]))
        keep = {p for ps in keep for p in ps}
        df = df[df["patient_id"].isin(keep)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    n_pat = df["patient_id"].nunique()
    print(f"[20] 测试切片 {len(df)} 张 / {n_pat} 名患者"
          f"（正类 {int((df.groupby('patient_id')['label'].first() == POS).sum())} 名）"
          f"  —— 1 名患者约 {100 / max(n_pat, 1):.1f} 个百分点")

    defs = {"resnet50": ("resnet50", C.MODEL_DIR / "resnet50_physionet_best.pth"),
            "swin_tiny": ("swin_tiny_patch4_window7_224",
                          C.MODEL_DIR / "swin_tiny_patch4_window7_224_physionet_best.pth")}
    conds = [c for c in args.conditions.split(",") if c]
    variants = [v for v in args.variants.split(",") if v]

    rows = []
    for mname in [m for m in args.models.split(",") if m]:
        if mname not in defs:
            raise SystemExit(f"未知模型 {mname}，可选 {list(defs)}")
        arch, weight = defs[mname]
        if not weight.exists():
            raise SystemExit(f"缺权重 {weight}（先跑 02）")
        print(f"\n===== {mname} =====")

        for cond in conds:
            base_dir = src_dir(cond)
            if not base_dir.is_dir():
                print(f"  [跳过] {cond}: 缺 {base_dir}（先跑 04）")
                continue

            loader = make_loader(df, cond, args.batch_size, args.num_workers)
            n_norm = norm_layer_summary(load_model(arch, weight, device))
            has_bn = n_norm.get("batchnorm", 0) > 0

            # --- 第 1 组：无自适应 ---
            m0 = load_model(arch, weight, device)
            print(f"  {cond:14s} 前向(plain) ...", end="", flush=True)
            s_plain = patient_scores(m0, loader, device)
            print(" 完成")
            del m0
            if device == "cuda":
                torch.cuda.empty_cache()

            # --- 第 2 组：BN 自适应 ---
            if has_bn:
                m1 = load_model(arch, weight, device)
                print(f"  {cond:14s} BN 自适应 ...", end="", flush=True)
                update_bn_stats(m1, loader, device)
                s_bn = patient_scores(m1, loader, device)
                print(" 完成")
                del m1
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                # swin 无 BatchNorm：bn 系列与无 bn 逐位等价（19 已验证 0 个不一致对）
                print(f"  {cond:14s} 无 BatchNorm -> 跳过 bn 组前向")
                s_bn = s_plain

            # --- 后处理出全部变体 ---
            pre = {"plain": s_plain, "bn": s_bn}

            # 落盘逐患者分数（供 21 先验敏感性离线扫描；不增加任何前向）
            if args.dump_scores:
                sdir = OUT_DIR / "scores"
                sdir.mkdir(parents=True, exist_ok=True)
                for stag, sdf in (("plain", s_plain), ("bn", s_bn)):
                    pp = sdf.groupby("patient_id").agg(
                        label=("label", "first"), prob=("prob", "mean")).reset_index()
                    pp.insert(0, "model", mname)
                    pp.insert(1, "condition", cond)
                    pp["bn"] = (stag == "bn")
                    pp["dataset"] = "PhysioNet"
                    pp["fold"] = C.fold_tag()
                    pp.to_csv(sdir / f"per_patient_scores_physionet_{mname}_{cond}"
                                     f"_{stag}.csv", index=False, encoding="utf-8-sig")

            group, per_variant = [], {}
            for v in variants:
                if v not in VARIANTS:
                    continue
                sdf = pre["bn"] if v in BN_VARIANTS else pre["plain"]
                mode = prior_correction_mode(v)
                cdf, audit = apply_correction(sdf, mode, target)
                met = patient_vote_metrics(cdf, tag=f"{mname}-{cond}-{v}")
                rec = {"model": mname, "condition": cond, "tta": v,
                       "num_patients": met["num_patients"],
                       "accuracy": met["accuracy"], "precision": met["precision"],
                       "recall": met["recall"], "f1": met["f1"], "auc": met["auc"],
                       "slice_accuracy": met["slice_accuracy"],
                       "oracle_thresh_acc": oracle_thresh_acc(cdf)}
                rec.update(audit)
                group.append(rec)
                rows.append(rec)
                per_variant[v] = per_patient(cdf)
                extra = ""
                if audit:
                    extra = (f" 阳性率 {audit['pos_rate_before']:.4f}"
                             f"->{audit['pos_rate_after']:.4f}"
                             f"(目标{target:.4f})")
                print(f"    {v:12s} acc={met['accuracy']:.4f} "
                      f"auc={met['auc']:.4f} 阈值上限={rec['oracle_thresh_acc']:.4f}{extra}")

            # 同条件内、相对 none 的患者级配对检验
            base = per_variant.get("none")
            if base is not None:
                for rec in group:
                    if rec["tta"] == "none":
                        continue
                    other = per_variant.get(rec["tta"])
                    if other is None:
                        continue
                    j = base.join(other, lsuffix="_a", rsuffix="_b", how="inner")
                    a = (j["pred_a"] == j["label_a"]).to_numpy(int)
                    b = (j["pred_b"] == j["label_b"]).to_numpy(int)
                    nb, nc, p = mcnemar(a, b)
                    lo, hi = boot_ci(b - a)
                    rec.update({"vs_none_delta": float(b.mean() - a.mean()),
                                "vs_none_b": nb, "vs_none_c": nc, "vs_none_p": p,
                                "vs_none_ci_low": lo, "vs_none_ci_high": hi})

    out = pd.DataFrame(rows)
    path = OUT_DIR / "boundary_all_noise_physionet.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[20] 写出 {path}（{len(out)} 行）")

    # ---- 核心摘要：每个条件下校正带来的最大增益 ----
    print("\n[20] 各噪声条件下边界校正的增益（同条件内最好变体 vs none）：")
    print(f"{'模型':10s} {'条件':14s} {'none':>7s} {'最好变体':>11s} "
          f"{'acc':>7s} {'Δ患者':>6s} {'p':>8s} {'阈值上限':>8s}")
    for (mname, cond), g in out.groupby(["model", "condition"], sort=False):
        g_none = g[g["tta"] == "none"]
        if g_none.empty:
            continue
        n0 = g_none.iloc[0]
        best = g.loc[g["accuracy"].idxmax()]
        d_pat = int(round((best["accuracy"] - n0["accuracy"]) * n0["num_patients"]))
        p = best.get("vs_none_p", np.nan)
        print(f"{mname:10s} {cond:14s} {n0['accuracy']:7.4f} {best['tta']:>11s} "
              f"{best['accuracy']:7.4f} {d_pat:+6d} "
              f"{p:8.4f} {n0['oracle_thresh_acc']:8.4f}")

    if len(out):
        n_pat = int(out["num_patients"].max())
        print(f"\n[20] 提示：本折 test 仅 {n_pat} 名患者（1 名 ≈ {100 / max(n_pat, 1):.1f} 点），"
              f"单折 Δ 多半不显著；跨折结论请用 physionet_aggregate_cv.py 汇总。")


if __name__ == "__main__":
    main()
