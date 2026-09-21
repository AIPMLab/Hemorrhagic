# -*- coding: utf-8 -*-
"""
PhysioNet 5 折交叉验证结果汇总（论文用）

把各折的结果合并成 mean ± std：
    Q1  是否恢复：adaptive − noisy（逐折计算差值再汇总，配对更稳）
    Q2  是否最优：与 oracle_global / oracle_restricted / oracle_per_patient 的差距
    判别器在复合噪声上的行为（6class 分布外 vs 7class 在类内）

同时给出"把 5 折的 test 患者拼起来"的**合并评估**：因为交叉验证下每个患者
恰好被评估一次，把所有折的 test 预测并起来就等价于对全部 82 例做了一次评估，
这个数字比逐折均值更稳，也可以直接和 CQ500/RSNA 的单次划分结果并排。

输入: results_fold{k}/optimality_physionet/{variant}/oracle_summary_physionet.csv
      results_fold{k}/noise_detector_physionet/{variant}/all_condition_behavior.csv
      results_fold{k}/tta_unknown_physionet/tta_unknown_results_physionet.csv   (19，可缺)
      results_fold{k}/boundary_correction_physionet/boundary_all_noise_physionet.csv (20，可缺)
输出: results_cv_physionet/
      cv_per_fold.csv         逐折明细
      cv_summary.csv          mean ± std（论文主表）
      cv_pooled.csv           5 折 test 合并后的评估
      latex_cv_summary.tex
      cv_tta_{per_fold,summary,pooled}.csv        19 的 TTA 消融
      cv_boundary_{per_fold,summary,pooled}.csv   20 的边界校正

用法:
    python physionet_aggregate_cv.py
    python physionet_aggregate_cv.py --folds 0 1 2      # 只汇总部分折
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent
OUT = CODE / "results_cv_physionet"
N_FOLDS = 5


def mcnemar_counts(b, c):
    """McNemar 精确检验（**不一致对计数**形式），返回 (b, c, p)。

    与共享层 ich_common.mcnemar 的 p 值口径完全一致（单边 binomtest）——
    这里单独写一份是为了让本脚本保持**不依赖 torch**，能在任何机器上直接跑起来。

    合并各折时 b、c 可以直接相加：各折 test 患者互不重叠，不一致对不会跨折重复计数。
    """
    n = int(b) + int(c)
    p = 1.0 if n == 0 else float(binomtest(min(int(b), int(c)), n, 0.5).pvalue)
    return int(b), int(c), p

# 逐折的指标（oracle_summary_physionet.csv 的列）
Q1 = ["noisy_accuracy", "adaptive_accuracy", "q1_delta_adaptive_minus_noisy",
      "q1_mcnemar_p"]
Q2 = ["oracle_global_accuracy", "oracle_restricted_accuracy",
      "oracle_per_patient_accuracy", "q2_gap_to_global",
      "q2_gap_to_restricted", "q2_gap_to_per_patient",
      "q2_agreement_with_global", "q2_agreement_with_per_patient"]


def load(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    args = ap.parse_args()

    per_fold, det_rows = [], []
    for k in args.folds:
        base = CODE / f"results_fold{k}" / "optimality_physionet"
        found = False
        for vdir in sorted(base.glob("*")) if base.exists() else []:
            o = load(vdir / "oracle_summary_physionet.csv")
            if o is None:
                continue
            found = True
            o = o.copy()
            o.insert(0, "fold", k)
            o["variant"] = vdir.name
            per_fold.append(o)
        dbase = CODE / f"results_fold{k}" / "noise_detector_physionet"
        for vdir in sorted(dbase.glob("*")) if dbase.exists() else []:
            b = load(vdir / "all_condition_behavior.csv")
            if b is None:
                continue
            hm = b[b["condition"] == "unknown_mixed"]
            if len(hm):
                r = hm.iloc[0]
                det_rows.append({"fold": k, "variant": vdir.name,
                                 "in_vocab": bool(r["in_vocab"]),
                                 "top_pred": r["top_pred"],
                                 "top_share": r["top_pred_share"],
                                 "mean_confidence": r["mean_confidence"]})
        if not found:
            print(f"[skip] fold {k}: 没有找到 optimality 结果")

    if not per_fold:
        raise SystemExit(
            "没有找到任何折的结果。请先跑：\n"
            "  python physionet_run_cv.py\n"
            f"（每折结果在 results_fold{{k}}/optimality_physionet/）")
    OUT.mkdir(parents=True, exist_ok=True)

    pf = pd.concat(per_fold, ignore_index=True)
    pf.to_csv(OUT / "cv_per_fold.csv", index=False)
    n_used = pf["fold"].nunique()
    print(f"[折] 汇总 {n_used} 折：{sorted(pf['fold'].unique())}")

    # ---- 逐 (variant, model) 汇总 mean ± std ----
    keys = ["variant", "model"]
    metrics = [m for m in Q1 + Q2 if m in pf.columns]
    rows = []
    for (v, m), g in pf.groupby(keys):
        rec = {"variant": v, "model": m, "n_folds": len(g)}
        for c in metrics:
            vals = g[c].astype(float)
            rec[f"{c}_mean"] = round(float(vals.mean()), 4)
            rec[f"{c}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)
        rows.append(rec)
    summ = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    summ.to_csv(OUT / "cv_summary.csv", index=False)

    # ---- 合并评估：把各折 test 患者并起来（每人恰好被评估一次）----
    # 注意 oracle_summary 里没有患者数，所以按折从 split 文件里取 test 例数。
    # 交叉验证下各折 test 互斥且并集 = 全部患者，故加权合并等价于对全体做一次评估。
    def fold_test_n(k):
        p = ROOT / f"physionet_split_fold{k}.csv"
        if not p.exists():
            return 0
        d = pd.read_csv(p)
        return int((d["split"] == "test").sum())

    n_by_fold = {int(k): fold_test_n(int(k)) for k in pf["fold"].unique()}
    tot_n = sum(n_by_fold.values())
    print(f"[合并] 各折 test 例数 {n_by_fold} -> 合计 {tot_n} 例")

    pooled = []
    for (v, m), g in pf.groupby(keys):
        g = g.copy()
        g["_n"] = g["fold"].map(n_by_fold).fillna(0).astype(float)
        tot = float(g["_n"].sum())
        w = (g["_n"] / tot) if tot > 0 else None
        rec = {"variant": v, "model": m, "n_patients_pooled": int(tot)}
        for c in ["noisy_accuracy", "adaptive_accuracy", "q1_delta_adaptive_minus_noisy",
                  "oracle_global_accuracy", "oracle_restricted_accuracy",
                  "oracle_per_patient_accuracy", "q2_gap_to_global",
                  "q2_agreement_with_global"]:
            if c not in g:
                continue
            rec[c] = round(float((g[c] * w).sum() if w is not None else g[c].mean()), 4)
        pooled.append(rec)
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(OUT / "cv_pooled.csv", index=False)

    # ---- 判别器在复合噪声上 ----
    det_df = pd.DataFrame(det_rows)
    if not det_df.empty:
        agg = (det_df.groupby(["variant", "top_pred"]).size()
               .rename("n_folds").reset_index())
        det_sum = det_df.groupby("variant").agg(
            folds=("fold", "nunique"),
            in_vocab=("in_vocab", "first"),
            top_pred_mode=("top_pred", lambda s: s.mode().iloc[0]),
            top_share_mean=("top_share", "mean"),
            confidence_mean=("mean_confidence", "mean")).reset_index()
        det_df.to_csv(OUT / "cv_detector_behavior_per_fold.csv", index=False)
        det_sum.to_csv(OUT / "cv_detector_behavior.csv", index=False)
        print("\n[判别器在复合噪声上（逐折汇总）]")
        print(det_sum.to_string(index=False))
        print("（按预测类别分布的折数）")
        print(agg.to_string(index=False))

    # ---- 19 TTA 消融 与 20 边界校正：逐折收集 → mean±std 与合并 ----
    # 这两个脚本都不落盘逐患者预测，但它们的 vs_none_b / vs_none_c 是**不一致对计数**。
    # 各折 test 患者互不重叠，于是：
    #   合并准确率  = Σ(acc_i × n_i) / Σ n_i          （精确，不是近似）
    #   合并 McNemar = 对 (Σb, Σc) 再做精确检验        （不相交集合上的计数可直接相加）
    #   合并 Δ       = (c − b) / n，其中 b=变差、c=变好（与 ich_common.mcnemar 同号）
    # 所以不需要改 19/20 去额外落盘逐患者表。
    secondary = []
    for tag, subdir, fname, keys in (
            ("tta", "tta_unknown_physionet", "tta_unknown_results_physionet.csv",
             ["model", "detector_variant", "condition", "tta"]),
            ("boundary", "boundary_correction_physionet",
             "boundary_all_noise_physionet.csv", ["model", "condition", "tta"])):
        frames = []
        for k in args.folds:
            d = load(CODE / f"results_fold{k}" / subdir / fname)
            if d is None or not len(d):
                continue
            d = d.copy()
            d.insert(0, "fold", k)
            frames.append(d)
        if not frames:
            print(f"[skip] {tag}: 没有找到结果（该步骤可能还没跑）")
            continue
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(OUT / f"cv_{tag}_per_fold.csv", index=False)
        gkeys = [c for c in keys if c in df.columns]
        mets = [c for c in ("accuracy", "precision", "recall", "f1", "auc")
                if c in df.columns]

        srows = []
        for kv, g in df.groupby(gkeys):
            rec = dict(zip(gkeys, kv if isinstance(kv, tuple) else (kv,)))
            rec["n_folds"] = int(g["fold"].nunique())
            for c in mets:
                v = g[c].astype(float)
                rec[f"{c}_mean"] = round(float(v.mean()), 4)
                rec[f"{c}_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
            if "vs_none_delta" in g.columns and g["vs_none_delta"].notna().any():
                v = g["vs_none_delta"].astype(float)
                rec["vs_none_delta_mean"] = round(float(v.mean()), 4)
                rec["vs_none_delta_std"] = round(
                    float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
            srows.append(rec)
        sdf = pd.DataFrame(srows).sort_values(gkeys).reset_index(drop=True)
        sdf.to_csv(OUT / f"cv_{tag}_summary.csv", index=False)

        prows = []
        for kv, g in df.groupby(gkeys):
            rec = dict(zip(gkeys, kv if isinstance(kv, tuple) else (kv,)))
            n = (g["num_patients"].astype(float).fillna(0).to_numpy()
                 if "num_patients" in g.columns
                 else np.ones(len(g), dtype=float))
            tot = float(n.sum())
            rec["n_patients_pooled"] = int(tot)
            if tot > 0:
                w = n / tot
                for c in mets:
                    rec[c] = round(float((g[c].astype(float).to_numpy() * w).sum()), 4)
                if "vs_none_b" in g.columns and g["vs_none_b"].notna().any():
                    # 口径与 ich_common.mcnemar 一致：b = none 对而变体错（变差），
                    #                       c = none 错而变体对（变好）
                    # 故 Δ = (c − b) / n。19/20 落盘的 vs_none_delta 也是这个方向，
                    # 这里必须同号，否则合并表会把"有效"读成"有害"。
                    bb = int(g["vs_none_b"].fillna(0).sum())
                    cc = int(g["vs_none_c"].fillna(0).sum())
                    _, _, p = mcnemar_counts(bb, cc)
                    rec["vs_none_b"] = bb
                    rec["vs_none_c"] = cc
                    rec["vs_none_delta"] = round((cc - bb) / tot, 4)
                    rec["vs_none_p"] = round(p, 4)
            prows.append(rec)
        pdf = pd.DataFrame(prows).sort_values(gkeys).reset_index(drop=True)
        pdf.to_csv(OUT / f"cv_{tag}_pooled.csv", index=False)
        secondary.append((tag, df, sdf, pdf))
        print(f"[{tag}] 汇总 {int(df['fold'].nunique())} 折，"
              f"{len(sdf)} 个组合 -> cv_{tag}_summary.csv / cv_{tag}_pooled.csv")

    # ---- LaTeX ----
    with open(OUT / "latex_cv_summary.tex", "w", encoding="utf-8") as fh:
        fh.write(summ.to_latex(index=False, float_format="%.4f", escape=False,
                               caption=f"PhysioNet CT-ICH: {n_used}-fold patient-level "
                                       f"cross-validation of the adaptive enhancement "
                                       f"framework on compound unknown noise "
                                       f"(mean $\\pm$ std over folds).",
                               label="tab:physionet_cv"))

    # ---- 图 ----
    variants = sorted(summ["variant"].unique())
    models = sorted(summ["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 4.8),
                             squeeze=False, constrained_layout=True)
    series = [("noisy_accuracy", "Noisy (no filter)", "tab:red"),
              ("adaptive_accuracy", "Adaptive", "tab:blue"),
              ("oracle_global_accuracy", "Oracle (best fixed filter)", "tab:green"),
              ("oracle_per_patient_accuracy", "Oracle (per-patient)", "tab:olive")]
    for ax, model in zip(axes[0], models):
        sub = summ[summ["model"] == model].set_index("variant")
        x = np.arange(len(variants))
        w = 0.2
        for i, (col, lab, col_c) in enumerate(series):
            vals = [sub.loc[v, f"{col}_mean"] if v in sub.index else np.nan
                    for v in variants]
            errs = [sub.loc[v, f"{col}_std"] if v in sub.index else 0 for v in variants]
            ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=3,
                   color=col_c, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n(compound class: "
                            f"{'yes' if v == '7class' else 'no'})" for v in variants],
                           fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Patient-level accuracy ({n_used}-fold mean)")
        ax.set_title(model, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"PhysioNet CT-ICH — {n_used}-fold patient-level cross-validation "
                 f"(error bars = std over folds)", fontweight="bold")
    fig.savefig(OUT / "fig_cv_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[逐折均值 ± 标准差]  ({n_used} 折)")
    show = ["variant", "model", "adaptive_accuracy_mean", "adaptive_accuracy_std",
            "noisy_accuracy_mean", "oracle_global_accuracy_mean",
            "q2_gap_to_global_mean", "q2_agreement_with_global_mean"]
    print(summ[[c for c in show if c in summ.columns]].to_string(index=False))
    print("\n[合并评估] 各折 test 患者拼接（每人恰好评估一次）")
    print(pooled_df.to_string(index=False))

    for tag, df, sdf, pdf in secondary:
        if tag == "tta":
            ad = pdf[pdf["condition"] == "adaptive"] if "condition" in pdf.columns else pdf
            if len(ad):
                print("\n[19 TTA 消融 · 合并 82 例] adaptive 条件下各变体相对 none")
                cols = [c for c in ("model", "detector_variant", "tta", "accuracy",
                                    "recall", "vs_none_delta", "vs_none_p")
                        if c in ad.columns]
                print(ad[cols].to_string(index=False))
        elif len(pdf) and "accuracy" in pdf.columns:
            print("\n[20 边界校正 · 合并 82 例] 各噪声条件下最好的变体")
            rows = []
            for (m, c), g in pdf.groupby(["model", "condition"], sort=False):
                best = g.loc[g["accuracy"].idxmax()]
                r = {"model": m, "condition": c, "best_tta": best["tta"],
                     "accuracy": best["accuracy"]}
                for c2 in ("auc", "vs_none_delta", "vs_none_p"):
                    if c2 in best.index:
                        r[c2] = best[c2]
                rows.append(r)
            print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n[done] 输出目录: {OUT}")
    print("       论文主表: cv_summary.csv（mean±std）与 cv_pooled.csv（合并）")


if __name__ == "__main__":
    main()
