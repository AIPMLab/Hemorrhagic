# -*- coding: utf-8 -*-
"""
CQ500 复合未知噪声：Q1 恢复性 + Q2 最优性分析（本项目核心脚本）

回答两个问题：
  Q1  判别器选出的 filter 增强后，患者级指标是否**超过未增强的加噪图**？
      （adaptive vs unknown_noisy，含 McNemar 配对检验与 bootstrap 置信区间）
  Q2  这个 filter **是不是最优 filter**？
      与三层 oracle 比较：
        oracle_global      —— test 上按患者级 ACC 最优的**单个** filter（全 9+none 里挑）
        oracle_restricted  —— 只在该模型实际用到的 filter 集合里挑（分离"判别器分错类"）
        oracle_per_patient —— 逐患者取其自身最优 filter（绝对上界）
      并给出选择一致性：adaptive 所选 filter 与各 oracle 的一致比例。

实现要点：
  * filter 穷举与 07 共用 cq500_commons.sweep_filters：单次解码 -> 全 filter 前向。
  * **adaptive 行不是重跑**，而是用 11 落盘的逐切片决策去索引同一份 sweep 结果，
    因此 adaptive 与各 oracle 基于完全相同的噪声实现与 filter 实现；
    同时与 11 自报的 adaptive 指标交叉校验。
  * 判别器能力表：7 类设计使复合噪声首次有真值，可量化判别器在各噪声上的准确率。

输出: results/optimality_cq500/
  filter_sweep_unknown_mixed_cq500.csv   每个固定 filter 的患者级指标
  oracle_summary_cq500.csv               Q1/Q2 汇总结论
  per_patient_oracle_cq500.csv           逐患者 oracle 与 adaptive 对照
  detector_accuracy_cq500.csv            判别器在 7 种噪声条件下的准确率
  latex_*.tex

用法:
    python 16_optimality_analysis_cq500.py
    python 16_optimality_analysis_cq500.py --limit 600      # 冒烟测试
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

from cq500_commons import (CLASSES, DATA_ROOT, DETECTOR_CLASSES, DETECTOR_VARIANTS,
                           FILTER_NAMES, MODEL_DIR, NOISE_NAMES, NOISY_ROOT,
                           RESULT_ROOT, boot_ci, build_eval_transform,
                           detector_weight_path, load_index, load_model, mcnemar,
                           patient_vote_metrics, sweep_filters)

NOISE_CONDITION = "unknown_mixed"
MODELS = {
    "resnet50": ("resnet50", MODEL_DIR / "resnet50_cq500_best.pth"),
    "swin_tiny": ("swin_tiny_patch4_window7_224",
                  MODEL_DIR / "swin_tiny_patch4_window7_224_cq500_best.pth"),
}

MAP_CSV = RESULT_ROOT / "enhanced_known_noise_cq500" / "best_filter_map_cq500.csv"
ADAPTIVE_CSV = RESULT_ROOT / "adaptive_unknown_cq500" / "{variant}" / "adaptive_unknown_results_cq500.csv"
SEL_CSV = RESULT_ROOT / "adaptive_unknown_cq500" / "{variant}" / "unknown_filter_selection_log_cq500.csv"


def variant_outdir(variant):
    return RESULT_ROOT / "optimality_cq500" / variant


def _pil(arr_bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------- 工具
def patient_table(slice_df):
    """切片 -> 患者级 (label, mean_prob, correct)，与 patient_vote_metrics 的口径一致。"""
    g = slice_df.groupby("patient_id").agg(
        label=("label", "first"), mean_prob=("prob", "mean"), n_slices=("prob", "count"))
    g["pred"] = (g["mean_prob"] >= 0.5).astype(int)
    g["correct"] = (g["pred"] == g["label"]).astype(int)
    return g


def margin(g):
    """患者级"正确方向的余量"：出血患者越大越好，正常患者越小越好。"""
    return np.where(g["label"] == 1, g["mean_prob"] - 0.5, 0.5 - g["mean_prob"])


# ---------------------------------------------------------------- 配对检验
# mcnemar / boot_ci 已移入共享层 ich_common，19 也复用同一实现，避免两处口径漂移


# ---------------------------------------------------------------- 判别器能力表
def detector_accuracy(device, limit, variant):
    """在 clean / 5 类已知噪声 / unknown_mixed 上评估**指定变体**的判别器。

    7class：7 类都有真值，准确率有意义。
    6class：unknown_mixed 不在词表内，准确率无定义（NaN），只报告它的预测分布与置信度
            —— 这正是"封闭集判别器遇到复合噪声会硬分到某个已知类"的证据。
    """
    weight_path = detector_weight_path(variant)
    ckpt = torch.load(weight_path, map_location=device)
    det_classes = ckpt["class_names"]
    det = timm.create_model("efficientnet_b0", pretrained=False,
                            num_classes=len(det_classes))
    det.load_state_dict(ckpt["model_state"])
    det.to(device).eval()

    tf = build_eval_transform()
    df = load_index()
    df = df[df["split"] == "test"].reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)

    rows, cms = [], {}
    for cond in DETECTOR_CLASSES:
        base = DATA_ROOT if cond == "clean" else (NOISY_ROOT / cond)
        arrs = []
        for rel in df["image_path"]:
            img = cv2.imread(str(base / rel))
            if img is not None:
                arrs.append(img)
        preds, confs = [], []
        with torch.no_grad():
            for s in range(0, len(arrs), 32):
                xb = torch.stack([
                    tf(_pil(a)) for a in arrs[s:s + 32]]).to(device)
                p = F.softmax(det(xb), dim=1).cpu().numpy()
                preds.extend(det_classes[i] for i in p.argmax(1))
                confs.extend(p.max(1))
        in_vocab = cond in det_classes
        pred_series = pd.Series(preds)
        acc = float((pred_series == cond).mean()) if in_vocab else float("nan")
        vc = pred_series.value_counts(normalize=True) if len(pred_series) else pd.Series()
        cm = pd.crosstab(pd.Series([cond] * len(preds), name="true"),
                         pred_series.rename("pred")).reindex(
            index=DETECTOR_CLASSES, columns=det_classes, fill_value=0)
        cms[cond] = cm
        rows.append({
            "variant": variant, "condition": cond, "in_vocab": in_vocab,
            "n_slices": len(preds), "accuracy": acc,
            "top_pred": vc.index[0] if len(vc) else "",
            "top_pred_share": float(vc.iloc[0]) if len(vc) else float("nan"),
            "mean_confidence": float(np.mean(confs)) if confs else float("nan"),
        })
        shown = f"{acc:.4f}" if in_vocab else " n/a "
        print(f"[detector:{variant}] {cond:15s} acc={shown} "
              f"conf={rows[-1]['mean_confidence']:.3f} "
              f"top={rows[-1]['top_pred']}({rows[-1]['top_pred_share']:.1%})")

    return pd.DataFrame(rows), cms


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(DETECTOR_VARIANTS),
                    help="分析哪个判别器变体的自适应决策：6class 或 7class")
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 张切片（冒烟测试用）")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader 进程数：全部 CPU 预处理在 worker 内并行，越大越能喂饱 GPU")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="逗号分隔的模型列表，默认全部")
    ap.add_argument("--skip-detector", action="store_true",
                    help="跳过判别器能力表（省一次全量前向）")
    args = ap.parse_args()

    variant = args.variant
    OUTDIR = variant_outdir(variant)
    sel_csv = Path(str(SEL_CSV).format(variant=variant))
    adaptive_csv = Path(str(ADAPTIVE_CSV).format(variant=variant))

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {k: v for k, v in MODELS.items() if k in wanted}
    if not models:
        raise SystemExit(f"--models 未匹配到任何模型，可选: {list(MODELS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"[variant] {variant}  classes={DETECTOR_VARIANTS[variant]}")

    if not MAP_CSV.exists():
        raise FileNotFoundError(f"缺少 {MAP_CSV}，请先运行 07")
    if not sel_csv.exists():
        raise FileNotFoundError(f"缺少 {sel_csv}，请先运行 11 --variant {variant}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sel_df = pd.read_csv(sel_csv)
    fmap = pd.read_csv(MAP_CSV)

    # 索引（与 sweep_filters 内部的裁剪口径保持一致，用于把 slice_index 映射回 image_path）
    df = load_index()
    df = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    rel_by_idx = dict(enumerate(df["image_path"]))

    sweep_rows, oracle_rows, pp_rows = [], [], []

    for mname, (arch, wpath) in models.items():
        if not wpath.exists():
            raise FileNotFoundError(f"缺少分类器权重 {wpath}，请先运行 02_train_cq500.py")
        print(f"\n===== {mname} =====")
        model = load_model(arch, wpath, device)

        # ---- 1) 固定 filter 穷举（同一批噪声实现）----
        print(f"[sweep] {mname} / test / {NOISE_CONDITION} / {len(FILTER_NAMES)} filters")
        sweeps = sweep_filters(split="test", noise_name=NOISE_CONDITION,
                               filter_names=FILTER_NAMES, model=model, device=device,
                               num_workers=args.num_workers, from_disk=True,
                               limit=args.limit)

        pat = {}
        for f in FILTER_NAMES:
            g = patient_table(sweeps[f])
            pat[f] = g
            m = patient_vote_metrics(sweeps[f], tag=f"{mname}-fixed-{f}")
            sweep_rows.append({
                "model": mname, "filter": f, "num_patients": m["num_patients"],
                "accuracy": m["accuracy"], "precision": m["precision"],
                "recall": m["recall"], "f1": m["f1"], "auc": m["auc"],
                "slice_accuracy": m["slice_accuracy"],
            })

        sweep_model_df = pd.DataFrame(sweep_rows)
        sweep_model_df = sweep_model_df[sweep_model_df["model"] == mname].reset_index(drop=True)
        acc_by_filter = dict(zip(sweep_model_df["filter"], sweep_model_df["accuracy"]))

        # ---- 2) oracle 三层 ----
        oracle_global = max(FILTER_NAMES, key=lambda f: acc_by_filter[f])
        oracle_global_acc = acc_by_filter[oracle_global]

        # 逐患者最优 filter：按正确方向余量取 argmax
        # 必须显式带上患者索引 —— margin() 返回裸 ndarray，若让 pandas 自动生成索引
        # 会退化成 0..n-1 的位置索引，后面按 patient_id 取值就会 KeyError/错位。
        pat_index = pat["none"].index
        for f in FILTER_NAMES:
            assert pat[f].index.equals(pat_index), f"{f} 的患者索引与 none 不一致"
        margin_mat = pd.DataFrame({f: margin(pat[f]) for f in FILTER_NAMES},
                                  index=pat_index)
        oracle_pp_filter = margin_mat.idxmax(axis=1)
        oracle_pp_best = margin_mat.max(axis=1)
        oracle_pp_acc = float((oracle_pp_best > 0).mean())

        # 受限 oracle：只在该模型实际用到的 filter 集合内挑
        m_sel = sel_df[sel_df["model"] == mname]
        used_filters = sorted(set(m_sel["selected_filter"].unique()))
        missing = [f for f in used_filters if f not in acc_by_filter]
        if missing:
            raise KeyError(f"{mname}: 选择日志里的 filter 不在评估集合中: {missing}")
        oracle_restricted = max(used_filters, key=lambda f: acc_by_filter[f])
        oracle_restricted_acc = acc_by_filter[oracle_restricted]

        # ---- 3) adaptive：用 11 的逐切片决策索引同一份 sweep（不重跑）----
        sel_by_path = dict(zip(m_sel["image_path"], m_sel["selected_filter"]))
        idx_to_pid = dict(zip(sweeps["none"]["slice_index"], sweeps["none"]["patient_id"]))
        slices_by_filter = {f: sweeps[f].set_index("slice_index") for f in FILTER_NAMES}

        adaptive_prob, adaptive_label, adaptive_pid = [], [], []
        slice_choice = {}                       # slice_index -> 实际选用的 filter
        for i in range(len(df)):
            f = sel_by_path.get(rel_by_idx[i])
            if f is None or i not in slices_by_filter[f].index:
                continue
            row = slices_by_filter[f].loc[i]
            adaptive_prob.append(float(row["prob"]))
            adaptive_label.append(int(row["label"]))
            adaptive_pid.append(row["patient_id"])
            slice_choice[i] = f
        adaptive_sdf = pd.DataFrame({"patient_id": adaptive_pid, "label": adaptive_label,
                                     "prob": adaptive_prob})
        adaptive_sdf["pred"] = (adaptive_sdf["prob"] >= 0.5).astype(int)
        m_ad = patient_vote_metrics(adaptive_sdf, tag=f"{mname}-adaptive")
        g_ad = patient_table(adaptive_sdf)

        # 与 11 自报结果交叉校验
        cross = "n/a"
        if adaptive_csv.exists():
            ad = pd.read_csv(adaptive_csv)
            hit = ad[(ad["model"] == mname) & (ad["condition"] == "adaptive")]
            if len(hit):
                ref = float(hit["accuracy"].iloc[0])
                cross = f"{m_ad['accuracy']:.6f} vs 11: {ref:.6f}"
                if abs(ref - m_ad["accuracy"]) > 1e-6:
                    print(f"[warn] {mname} adaptive 与 11 自报不一致: {cross}")

        # ---- 4) Q1：adaptive vs unknown_noisy（= filter none）----
        g_none = pat["none"]
        common = g_ad.index.intersection(g_none.index)
        a_c = g_ad.loc[common, "correct"].values
        n_c = g_none.loc[common, "correct"].values
        b, c, p_mcnemar = mcnemar(a_c, n_c)
        diff = a_c.astype(float) - n_c.astype(float)
        lo, hi = boot_ci(diff)

        # ---- 5) Q2：选择一致性 ----
        agree_global = float((m_sel["selected_filter"] == oracle_global).mean())
        # 逐切片：adaptive 选的 filter 是否等于"该患者自身的 oracle 最优 filter"
        pairs = [(f, oracle_pp_filter.get(idx_to_pid.get(i)))
                 for i, f in slice_choice.items()
                 if idx_to_pid.get(i) in oracle_pp_filter.index]
        agree_pp = float(np.mean([f == best for f, best in pairs])) if pairs else float("nan")

        print(f"[oracle] global={oracle_global} ({oracle_global_acc:.4f}) | "
              f"restricted={oracle_restricted} ({oracle_restricted_acc:.4f}) | "
              f"per-patient={oracle_pp_acc:.4f}")
        print(f"[adaptive] acc={m_ad['accuracy']:.4f} (noisy={acc_by_filter['none']:.4f}) "
              f"| 与 11 校验: {cross}")

        oracle_rows.append({
            "variant": variant,
            "model": mname,
            "adaptive_accuracy": m_ad["accuracy"],
            "noisy_accuracy": acc_by_filter["none"],
            "q1_delta_adaptive_minus_noisy": m_ad["accuracy"] - acc_by_filter["none"],
            "q1_ci_low": lo, "q1_ci_high": hi,
            "q1_mcnemar_b": b, "q1_mcnemar_c": c, "q1_mcnemar_p": p_mcnemar,
            "oracle_global_filter": oracle_global,
            "oracle_global_accuracy": oracle_global_acc,
            "oracle_restricted_filter": oracle_restricted,
            "oracle_restricted_accuracy": oracle_restricted_acc,
            "oracle_per_patient_accuracy": oracle_pp_acc,
            "q2_gap_to_global": oracle_global_acc - m_ad["accuracy"],
            "q2_gap_to_restricted": oracle_restricted_acc - m_ad["accuracy"],
            "q2_gap_to_per_patient": oracle_pp_acc - m_ad["accuracy"],
            "q2_agreement_with_global": agree_global,
            "q2_agreement_with_per_patient": agree_pp,
            "filters_used": ",".join(used_filters),
            "adaptive_cross_check": cross,
        })

        for pid in oracle_pp_filter.index:
            pp_rows.append({
                "model": mname, "patient_id": pid,
                "label": int(pat["none"].loc[pid, "label"]),
                "oracle_best_filter": oracle_pp_filter.loc[pid],
                "oracle_best_margin": float(oracle_pp_best.loc[pid]),
                "oracle_correct": int(margin_mat.loc[pid].max() > 0),
                "adaptive_correct": int(g_ad.loc[pid, "correct"]) if pid in g_ad.index else -1,
                "adaptive_used_filters": ",".join(sorted(
                    set(m_sel[m_sel["patient_id"] == pid]["selected_filter"]))),
                "noisy_correct": int(pat["none"].loc[pid, "correct"]),
            })

    sweep_df_all = pd.DataFrame(sweep_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    pp_df = pd.DataFrame(pp_rows)

    sweep_df_all.to_csv(OUTDIR / "filter_sweep_unknown_mixed_cq500.csv", index=False)
    oracle_df.to_csv(OUTDIR / "oracle_summary_cq500.csv", index=False)
    pp_df.to_csv(OUTDIR / "per_patient_oracle_cq500.csv", index=False)

    # ---- 6) 不变量自检 ----
    for _, r in oracle_df.iterrows():
        assert r["oracle_per_patient_accuracy"] >= r["oracle_global_accuracy"] - 1e-9, \
            f"{r['model']}: per-patient oracle 应 >= global oracle"
        assert r["oracle_global_accuracy"] >= r["adaptive_accuracy"] - 1e-9, \
            f"{r['model']}: global oracle 应 >= adaptive"
    print("[check] oracle 不变量通过 ✓")

    # ---- 7) 判别器能力表 ----
    if not args.skip_detector and detector_weight_path(variant).exists():
        det_df, cms = detector_accuracy(device, args.limit, variant)
        det_df.to_csv(OUTDIR / "detector_accuracy_cq500.csv", index=False)
        for cond, cm in cms.items():
            cm.to_csv(OUTDIR / f"detector_confusion_{cond}_cq500.csv")

    # ---- 8) LaTeX ----
    with open(OUTDIR / "latex_q1_q2_summary.tex", "w", encoding="utf-8") as fh:
        fh.write(oracle_df.to_latex(index=False, float_format="%.4f", escape=False,
                                    caption="Q1 recovery and Q2 optimality of the adaptive "
                                            "pipeline on compound unknown noise (CQ500, patient-level).",
                                    label="tab:q1q2"))
    with open(OUTDIR / "latex_filter_sweep.tex", "w", encoding="utf-8") as fh:
        fh.write(sweep_df_all.to_latex(index=False, float_format="%.4f", escape=False,
                                       caption="Patient-level performance of each fixed "
                                               "enhancement filter under compound unknown noise.",
                                       label="tab:sweep"))

    print("\n[Q1] adaptive vs noisy:")
    print(oracle_df[["model", "noisy_accuracy", "adaptive_accuracy",
                     "q1_delta_adaptive_minus_noisy", "q1_ci_low", "q1_ci_high",
                     "q1_mcnemar_p"]].to_string(index=False))
    print("\n[Q2] oracle 对比:")
    print(oracle_df[["model", "adaptive_accuracy", "oracle_global_filter",
                     "oracle_global_accuracy", "oracle_restricted_accuracy",
                     "oracle_per_patient_accuracy", "q2_gap_to_global",
                     "q2_agreement_with_global"]].to_string(index=False))
    print(f"\n[done] 输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
