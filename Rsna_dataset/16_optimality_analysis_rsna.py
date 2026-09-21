# -*- coding: utf-8 -*-
"""
RSNA 复合未知噪声：Q1 恢复性 + Q2 最优性分析（与 CQ500 的 16 同构）

回答：
  Q1  判别器选出的 filter 增强后，患者级指标是否**超过未增强的加噪图**？
      （adaptive vs unknown_noisy，含 McNemar 配对检验与 bootstrap 置信区间）
  Q2  这个 filter **是不是最优 filter**？
      与三层 oracle 比较：
        oracle_global      —— test 上按患者级 ACC 最优的**单个** filter
        oracle_restricted  —— 只在该模型实际用到的 filter 集合里挑（分离"判别器分错类"）
        oracle_per_patient —— 逐患者取其自身最优 filter（绝对上界）
      并给出选择一致性。

**adaptive 行不是重跑**：用 11 落盘的逐切片决策去索引同一份 sweep 结果，
因此 adaptive 与各 oracle 基于完全相同的噪声实现与 filter 实现；
同时与 11 自报的 adaptive 指标交叉校验。

输出: results/optimality_rsna/{variant}/

用法:
    python 16_optimality_analysis_rsna.py --variant 7class
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
from rsna_commons import (CLASSES, DATA_ROOT, DETECTOR_CLASSES, DETECTOR_VARIANTS,
                          FILTER_NAMES, MODEL_DIR, NOISE_NAMES, NOISY_ROOT,
                          RESULT_ROOT, boot_ci, build_eval_transform,
                          detector_weight_path, load_index, load_model, mcnemar,
                          patient_vote_metrics, sweep_filters)

NOISE_CONDITION = "unknown_mixed"
MODELS = {
    "resnet50": ("resnet50", MODEL_DIR / "resnet50_rsna_best.pth"),
    "swin_tiny": ("swin_tiny_patch4_window7_224",
                  MODEL_DIR / "swin_tiny_patch4_window7_224_rsna_best.pth"),
}

MAP_CSV = RESULT_ROOT / "enhanced_known_noise_rsna" / "best_filter_map_rsna.csv"
ADAPTIVE_CSV = (RESULT_ROOT / "adaptive_unknown_rsna" / "{variant}"
                / "adaptive_unknown_results_rsna.csv")
SEL_CSV = (RESULT_ROOT / "adaptive_unknown_rsna" / "{variant}"
           / "unknown_filter_selection_log_rsna.csv")


def variant_outdir(variant):
    return RESULT_ROOT / "optimality_rsna" / variant


def _pil(arr_bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))


def patient_table(slice_df):
    g = slice_df.groupby("patient_id").agg(
        label=("label", "first"), mean_prob=("prob", "mean"), n_slices=("prob", "count"))
    g["pred"] = (g["mean_prob"] >= 0.5).astype(int)
    g["correct"] = (g["pred"] == g["label"]).astype(int)
    return g


def margin(g):
    return np.where(g["label"] == 1, g["mean_prob"] - 0.5, 0.5 - g["mean_prob"])


# mcnemar / boot_ci 已移入共享层 ich_common，19 也复用同一实现，避免两处口径漂移
def detector_accuracy(device, limit, variant):
    """在全部 7 种条件上评估指定变体的判别器（6class 对复合噪声无真值 -> NaN）。"""
    ckpt = torch.load(detector_weight_path(variant), map_location=device)
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

    rows = []
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
                xb = torch.stack([tf(_pil(a)) for a in arrs[s:s + 32]]).to(device)
                p = F.softmax(det(xb), dim=1).cpu().numpy()
                preds.extend(det_classes[i] for i in p.argmax(1))
                confs.extend(p.max(1))
        in_vocab = cond in det_classes
        ps = pd.Series(preds)
        acc = float((ps == cond).mean()) if in_vocab else float("nan")
        vc = ps.value_counts(normalize=True) if len(ps) else pd.Series()
        rows.append({"variant": variant, "condition": cond, "in_vocab": in_vocab,
                     "n_slices": len(preds), "accuracy": acc,
                     "top_pred": vc.index[0] if len(vc) else "",
                     "top_pred_share": float(vc.iloc[0]) if len(vc) else float("nan"),
                     "mean_confidence": float(np.mean(confs)) if confs else float("nan")})
        shown = f"{acc:.4f}" if in_vocab else " n/a "
        print(f"[detector:{variant}] {cond:15s} acc={shown} "
              f"conf={rows[-1]['mean_confidence']:.3f} "
              f"top={rows[-1]['top_pred']}({rows[-1]['top_pred_share']:.1%})")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(DETECTOR_VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--skip-detector", action="store_true")
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

    df = load_index()
    df = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    rel_by_idx = dict(enumerate(df["image_path"]))

    sweep_rows, oracle_rows, pp_rows = [], [], []

    for mname, (arch, wpath) in models.items():
        if not wpath.exists():
            raise FileNotFoundError(f"缺少分类器权重 {wpath}，请先运行 02_train_rsna.py")
        print(f"\n===== {mname} =====")
        model = load_model(arch, wpath, device)

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
            sweep_rows.append({"model": mname, "filter": f,
                               "num_patients": m["num_patients"],
                               "accuracy": m["accuracy"], "precision": m["precision"],
                               "recall": m["recall"], "f1": m["f1"], "auc": m["auc"],
                               "slice_accuracy": m["slice_accuracy"]})

        sweep_model_df = pd.DataFrame(sweep_rows)
        sweep_model_df = sweep_model_df[sweep_model_df["model"] == mname].reset_index(drop=True)
        acc_by_filter = dict(zip(sweep_model_df["filter"], sweep_model_df["accuracy"]))

        oracle_global = max(FILTER_NAMES, key=lambda f: acc_by_filter[f])
        oracle_global_acc = acc_by_filter[oracle_global]

        pat_index = pat["none"].index
        for f in FILTER_NAMES:
            assert pat[f].index.equals(pat_index), f"{f} 的患者索引与 none 不一致"
        margin_mat = pd.DataFrame({f: margin(pat[f]) for f in FILTER_NAMES},
                                  index=pat_index)
        oracle_pp_filter = margin_mat.idxmax(axis=1)
        oracle_pp_best = margin_mat.max(axis=1)
        oracle_pp_acc = float((oracle_pp_best > 0).mean())

        m_sel = sel_df[sel_df["model"] == mname]
        used_filters = sorted(set(m_sel["selected_filter"].unique()))
        missing = [f for f in used_filters if f not in acc_by_filter]
        if missing:
            raise KeyError(f"{mname}: 选择日志里的 filter 不在评估集合中: {missing}")
        oracle_restricted = max(used_filters, key=lambda f: acc_by_filter[f])
        oracle_restricted_acc = acc_by_filter[oracle_restricted]

        sel_by_path = dict(zip(m_sel["image_path"], m_sel["selected_filter"]))
        idx_to_pid = dict(zip(sweeps["none"]["slice_index"], sweeps["none"]["patient_id"]))
        slices_by_filter = {f: sweeps[f].set_index("slice_index") for f in FILTER_NAMES}

        adaptive_prob, adaptive_label, adaptive_pid = [], [], []
        slice_choice = {}
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

        cross = "n/a"
        if adaptive_csv.exists():
            ad = pd.read_csv(adaptive_csv)
            hit = ad[(ad["model"] == mname) & (ad["condition"] == "adaptive")]
            if len(hit):
                ref = float(hit["accuracy"].iloc[0])
                cross = f"{m_ad['accuracy']:.6f} vs 11: {ref:.6f}"
                if abs(ref - m_ad["accuracy"]) > 1e-6:
                    print(f"[warn] {mname} adaptive 与 11 自报不一致: {cross}")

        g_none = pat["none"]
        common = g_ad.index.intersection(g_none.index)
        a_c = g_ad.loc[common, "correct"].values
        n_c = g_none.loc[common, "correct"].values
        b, c, p_mcnemar = mcnemar(a_c, n_c)
        diff = a_c.astype(float) - n_c.astype(float)
        lo, hi = boot_ci(diff)

        agree_global = float((m_sel["selected_filter"] == oracle_global).mean())
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
            "variant": variant, "model": mname,
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
                "n_slices": int(pat["none"].loc[pid, "n_slices"]),
                "oracle_best_filter": oracle_pp_filter.loc[pid],
                "oracle_correct": int(margin_mat.loc[pid].max() > 0),
                "adaptive_correct": int(g_ad.loc[pid, "correct"]) if pid in g_ad.index else -1,
                "noisy_correct": int(pat["none"].loc[pid, "correct"]),
            })

    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(OUTDIR / "oracle_summary_rsna.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(OUTDIR / "filter_sweep_unknown_mixed_rsna.csv", index=False)
    pp_df = pd.DataFrame(pp_rows)
    pp_df.insert(0, "variant", variant)
    pp_df.to_csv(OUTDIR / "per_patient_oracle_rsna.csv", index=False)

    for _, r in oracle_df.iterrows():
        assert r["oracle_per_patient_accuracy"] >= r["oracle_global_accuracy"] - 1e-9, \
            f"{r['model']}: per-patient oracle 应 >= global oracle"
        assert r["oracle_global_accuracy"] >= r["adaptive_accuracy"] - 1e-9, \
            f"{r['model']}: global oracle 应 >= adaptive"
    print("[check] oracle 不变量通过 ✓")

    if not args.skip_detector and detector_weight_path(variant).exists():
        det_df = detector_accuracy(device, args.limit, variant)
        det_df.to_csv(OUTDIR / "detector_accuracy_rsna.csv", index=False)

    with open(OUTDIR / "latex_q1_q2_summary.tex", "w", encoding="utf-8") as fh:
        fh.write(oracle_df.to_latex(index=False, float_format="%.4f", escape=False,
                                    caption="RSNA ICH subset: Q1 recovery and Q2 optimality "
                                            "of the adaptive pipeline on compound unknown noise "
                                            "(patient-level).",
                                    label="tab:rsna_q1q2"))

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
