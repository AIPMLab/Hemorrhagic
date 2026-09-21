# -*- coding: utf-8 -*-
"""
21 先验敏感性：目标患病率估偏了会损失多少？

回答的问题（论文最容易被攻击的一点）：
    边界校正的目标患病率 π 取自 val，而 val 与 test 同源且分层，两者患病率几乎相同。
    因此校正实质上是**使用真实类别比例**。若换到一个真正分布偏移的目标域，
    π 估偏 ε 会损失多少？

两种模式
    --scores DIR    从 20 落盘的逐患者分数做**完整扫描**（推荐；需先在服务器上跑一次
                    20 --dump-scores，之后本脚本可离线反复跑，不再需要模型与图像）
    --aggregate     只用现有的 boundary_all_noise_*.csv 做**一阶估计**（无需重跑任何东西，
                    但只能给割线斜率，不能给完整曲线）

用法:
    # 完整模式（离线，推荐）
    python 21_prior_sensitivity.py --scores D:\\...\\results_bundle --dataset CQ500

    # 一阶模式（立刻可跑）
    python 21_prior_sensitivity.py --aggregate --dataset CQ500
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 完整扫描
def load_scores(root, dataset):
    """读 20 --dump-scores 写出的逐患者分数。"""
    root = Path(root)
    pats = [root / f"results_fold*/**/per_patient_scores_*.csv",
            root / f"**/per_patient_scores_*.csv"]
    files = []
    for p in pats:
        files.extend(glob.glob(str(p), recursive=True))
    files = sorted(set(files))
    if not files:
        raise SystemExit(
            f"在 {root} 下找不到 per_patient_scores_*.csv。\n"
            f"请先在服务器上对每个数据集跑一次：\n"
            f"    python 20_boundary_correction_<ds>.py --dump-scores\n"
            f"（20 本来就有全部逐患者分数，这一步只多写一个很小的 CSV，不增加前向次数）")
    d = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in files],
                  ignore_index=True)
    if "dataset" in d.columns and dataset:
        d = d[d["dataset"] == dataset]
    return d


def sweep_from_scores(d, targets, bn=False):
    """对每个 (model, condition) 在给定目标患病率网格上重算准确率。

    校正按 match_prevalence_rate 的定义：取 logit 第 k 大值为阈值，k = round(π·n)。
    因此只需要分数与标签，不需要模型。
    """
    rows = []
    sub = d[d["bn"] == bn] if "bn" in d.columns else d
    for (model, cond), g in sub.groupby(["model", "condition"], sort=False):
        y = g["label"].to_numpy(int)
        p = np.clip(g["prob"].to_numpy(float), 1e-7, 1 - 1e-7)
        z = np.log(p / (1 - p))
        n = len(y)
        native = float(((p >= 0.5).astype(int) == y).mean())
        for t in targets:
            k = int(round(t * n))
            if k <= 0:
                pred = np.zeros(n, int)
            elif k >= n:
                pred = np.ones(n, int)
            else:
                thr = np.sort(z)[::-1][k - 1]
                pred = (z >= thr).astype(int)
            rows.append({"model": model, "condition": cond,
                         "bn": bool(bn), "target": round(float(t), 4),
                         "achieved_pos_rate": round(float(pred.mean()), 4),
                         "accuracy": round(float((pred == y).mean()), 4),
                         "n": n, "native_acc": round(native, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 一阶估计
def aggregate_boundary(ds_root):
    """读 boundary_all_noise_*.csv（兼容单队列与 5 折目录）。

    每个文件打上 `source` 标签（= 所在目录名），并在分组时使用 ——
    否则把多队列的 CSV 汇总到一个目录后会**跨队列配对**（拿 RSNA 的目标患病率
    配 CQ500 的准确率），算出来的"增益"完全没有意义。
    """
    ds_root = Path(ds_root)
    cands = sorted(ds_root.rglob("boundary_all_noise_*.csv"))
    if not cands:
        raise SystemExit(f"{ds_root} 下找不到 boundary_all_noise_*.csv")
    frames = []
    for p in cands:
        d = pd.read_csv(p)
        # 单队列 bundle：父目录名（如 boundary_correction_cq500）；多折：再带上折名
        d["source"] = f"{p.parent.parent.name}/{p.parent.name}"
        if "fold" in d.columns:
            d["source"] = d["source"] + ":" + d["fold"].astype(str)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="", help="含 per_patient_scores_*.csv 的目录")
    ap.add_argument("--aggregate", action="store_true",
                    help="只用 boundary_all_noise_*.csv 做一阶估计")
    ap.add_argument("--dataset", default="", help="数据集名（--scores 模式下筛选用）")
    ap.add_argument("--boundary-root", default="",
                    help="--aggregate 模式下的结果根目录")
    ap.add_argument("--grid", default="0.20,0.25,0.30,0.35,0.40,0.4390,0.45,0.50,0.55,0.60,0.65",
                    help="目标患病率网格（逗号分隔）")
    args = ap.parse_args()

    if args.scores:
        d = load_scores(args.scores, args.dataset)
        targets = [float(x) for x in args.grid.split(",") if x.strip()]
        print("=" * 100)
        print("先验敏感性扫描（完整模式：直接从逐患者分数重算）")
        print("=" * 100)
        for bn in (False, True):
            if "bn" in d.columns and bn not in set(d["bn"]):
                continue
            s = sweep_from_scores(d, targets, bn=bn)
            if not len(s):
                continue
            print(f"\n--- {'BN 自适应后' if bn else '无模型自适应'} ---")
            piv = s.pivot_table(index=["model", "condition"], columns="target",
                                values="accuracy")
            print(piv.round(4).to_string())
            # 相对峰值的位置
            print("\n  相对该行峰值的损失（越大越说明对 π 敏感）：")
            loss = piv.sub(piv.max(axis=1), axis=0).round(4)
            print(loss.to_string())
        out = HERE.parent / "results_merged"
        out.mkdir(exist_ok=True)
        for bn in (False, True):
            s = sweep_from_scores(d, targets, bn=bn)
            if len(s):
                s.to_csv(out / f"prior_sensitivity_{args.dataset or 'all'}"
                              f"{'_bn' if bn else ''}.csv", index=False)
        print(f"\n[out] 写出到 {out}/prior_sensitivity_*.csv")
        return

    if not args.aggregate:
        raise SystemExit("请指定 --scores DIR（完整模式）或 --aggregate（一阶模式）")

    # ---------------- 一阶模式 ----------------
    root = Path(args.boundary_root) if args.boundary_root else HERE
    bl = aggregate_boundary(root)
    print("=" * 100)
    print("先验敏感性（一阶模式：只用现有 boundary_all_noise_*.csv）")
    print("=" * 100)
    print("""
    做法：不去构造一个假的"曲线斜率"，而是用数据里**天然存在的**目标误差做自然实验——
          基线 none 工作在 r0，真实患病率是 π，于是它带着一个 (r0 − π) 的误差；
          把它校正回 π 得到的增益，就是"纠正大小为 |r0 − π| 的误差能拿回多少"。
          42 个 (队列 × 模型 × 条件) 组合构成一条经验曲线。

    能回答：目标患病率估偏 ε 时**大致**会损失多少（经验外推）。
    不能回答：逐条曲线的精确形状（需要逐患者分数，见 --scores 模式）。
    """)
    rows = []
    for (source, model, cond), g in bl.groupby(["source", "model", "condition"],
                                               sort=False):
        n0 = g[g["tta"] == "none"]
        if not len(n0):
            continue
        n0 = n0.iloc[0]
        cand = g[g["tta"].isin(["priorq", "bn_priorq"])]
        if not len(cand):
            continue
        best = cand.loc[cand["accuracy"].idxmax()]
        pb = best.get("pos_rate_before", np.nan)
        pi = best.get("prior_target", np.nan)
        if pd.isna(pb) or pd.isna(pi):
            continue
        a0, a1 = float(n0["accuracy"]), float(best["accuracy"])
        gap = abs(float(pb) - float(pi))
        # 割线斜率：仅当区间足够宽时才可报（分母趋 0 时它是舍入噪声，不是导数）
        slope = ((a1 - a0) / (float(pi) - float(pb))) if gap >= 0.05 else np.nan
        rows.append({"source": source, "model": model, "condition": cond,
                     "pos_rate_before": round(float(pb), 4),
                     "target_pi": round(float(pi), 4),
                     "error_at_baseline": round(gap, 4),
                     "base_acc": round(a0, 4), "corrected_acc": round(a1, 4),
                     "gain": round(a1 - a0, 4),
                     "chord_slope": (round(slope, 3) if slope == slope else np.nan),
                     "headroom_left": round(float(n0["oracle_thresh_acc"]) - a1, 4)})
    t = pd.DataFrame(rows)
    if len(t):
        print(t.to_string(index=False))
        out = HERE.parent / "results_merged"
        out.mkdir(exist_ok=True)
        t.to_csv(out / "prior_sensitivity_chord.csv", index=False)

        print("\n" + "-" * 100)
        print("经验曲线：把 |误差| 分箱后的平均增益（这是本模式真正的产出）")
        print("-" * 100)
        m = t[["error_at_baseline", "gain"]].dropna()
        m = m[m["error_at_baseline"] > 0]
        if len(m) >= 6:
            q = pd.qcut(m["error_at_baseline"], 4, duplicates="drop")
            b = m.groupby(q, observed=True).agg(
                err_mean=("error_at_baseline", "mean"),
                gain_mean=("gain", "mean"), n=("gain", "size")).round(4)
            print(b.to_string())
            lo = m[m["error_at_baseline"] <= 0.12]
            if len(lo) >= 4:
                ratio = float(lo["gain"].sum() / lo["error_at_baseline"].sum())
                print(f"\n  小误差区（|误差| ≤ 0.12，n={len(lo)}）的增益/误差比 ≈ {ratio:.2f}")
                print(f"  -> 目标患病率估偏 ε（ε ≤ 0.12）时，保住 ε×{ratio:.2f} 的准确率损失；")
                print(f"     即 ε = 0.05 约损失 {0.05 * ratio:.3f}，ε = 0.10 约损失 "
                      f"{0.10 * ratio:.3f}（**经验外推，非上界**）")
            t.to_csv(out / "prior_sensitivity_chord.csv", index=False)
            b.to_csv(out / "prior_sensitivity_empirical_bins.csv")
            print(f"\n[out] {out}/prior_sensitivity_chord.csv 与 "
                  f"prior_sensitivity_empirical_bins.csv")
        else:
            print("  （可用组合太少，不足以分箱）")
    else:
        print("  （没有可用的 rate-matching 行）")
    print("\n注意：本模式的 42 个点来自「模型天然带有的误差」，不是人为设定的误差。")
    print("      要做真正可控的 ε 扫描，请跑：20_*.py --dump-scores，再用 --scores 模式。")


if __name__ == "__main__":
    main()
