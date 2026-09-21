# -*- coding: utf-8 -*-
"""
RSNA 划分构建（**按患者**分层抽样 + 70/15/15）

修复的核心问题：
  旧版按 StudyInstanceUID（检查）划分，并把 study_id 直接当成"患者"
  （见旧脚本 rsna_study_map_all.csv -> "患者" 的注释与用法）。
  同一患者多次检查时，两次检查会被分到不同集合 -> 患者级泄漏。
  本版一律按 PatientID 划分；实测全库确有患者有多次检查。

数据来源：
  rsna_slice_labels.csv    每张切片的 any 标签（逐切片标注）
  rsna_patient_map.csv     img_id -> (patient_id, study_id)，从 DICOM 头抽取

抽样策略：
  患者级标签 = 该患者**任意一张切片**阳性则阳性（与 CQ500 的患者级定义一致）。
  按患者数抽样（默认 500，与 CQ500 的 473 例同量级），保持自然阳性率，
  使两数据集的类别比例天然可比（实测均 ~41%）。

输出（均在 Rsna_dataset/）：
  rsna_split.csv               patient_id -> split
  {train,val,test}_patients.csv
  rsna_split_summary.csv       各集合患者数/切片数
  rsna_sampling_frame.csv      全部患者及其是否入选（可审计）
  rsna_dataset_provenance.csv  数据集来源与规模记账

用法:
    python rsna_make_split.py
    python rsna_make_split.py --n-patients 800 --min-slices 10
    python rsna_make_split.py --dry-run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS_CSV = ROOT / "rsna_slice_labels.csv"
PMAP_CSV = ROOT / "rsna_patient_map.csv"
SPLIT_CSV = ROOT / "rsna_split.csv"
SUMMARY_CSV = ROOT / "rsna_split_summary.csv"
FRAME_CSV = ROOT / "rsna_sampling_frame.csv"
PROV_CSV = ROOT / "rsna_dataset_provenance.csv"

SEED = 42
RATIO_TRAIN, RATIO_VAL = 0.70, 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=500,
                    help="抽样的患者数（默认 500，与 CQ500 的 473 例同量级）")
    ap.add_argument("--min-slices", type=int, default=0,
                    help="排除切片数少于该值的患者（0=不排除）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    args = ap.parse_args()

    if not LABELS_CSV.exists():
        raise SystemExit("缺少 rsna_slice_labels.csv；请先运行 rsna_prepare_labels.py")
    if not PMAP_CSV.exists():
        raise SystemExit("缺少 rsna_patient_map.csv；请先运行 rsna_extract_patient_map.py")

    lab = pd.read_csv(LABELS_CSV)
    pm = pd.read_csv(PMAP_CSV)
    bad = int(pm["study_id"].astype(str).str.startswith("ERR:").sum())
    if bad:
        print(f"[warn] 患者映射里有 {bad} 行读取失败，已剔除")
        pm = pm[~pm["study_id"].astype(str).str.startswith("ERR:")]
    nopat = int((pm["patient_id"].fillna("") == "").sum())
    if nopat:
        print(f"[warn] 患者映射里有 {nopat} 行缺 PatientID，已剔除")
        pm = pm[pm["patient_id"].fillna("") != ""]

    d = lab.merge(pm, on="img_id", how="inner")
    print(f"[join] 切片标签 {len(lab)} ∩ 患者映射 {len(pm)} -> {len(d)} 张切片")

    pat = d.groupby("patient_id").agg(
        any=("any", "max"),
        n_slices=("img_id", "count"),
        n_studies=("study_id", "nunique")).reset_index()
    print(f"[患者] 全库 {len(pat)} 例，患者级阳性率 {pat['any'].mean():.4f}"
          f"（阳性 {int(pat['any'].sum())}）")

    q = pat["n_slices"].describe(percentiles=[.05, .25, .5, .75, .95])
    print("[每患者切片数] "
          + "  ".join([f"P5={q['5%']:.0f}", f"P25={q['25%']:.0f}",
                       f"P50={q['50%']:.0f}", f"P75={q['75%']:.0f}",
                       f"P95={q['95%']:.0f}"])
          + f"  min={q['min']:.0f}  max={q['max']:.0f}  mean={q['mean']:.1f}")

    frame = pat.copy()
    # 用显式 sentinel 而不是空串：空串写进 CSV 后会被 pandas 读成 NaN，
    # 后续用 == "" 过滤会全部落空（这个坑踩过一次）。
    frame["excluded_reason"] = "kept"
    if args.min_slices > 0:
        drop = frame["n_slices"] < args.min_slices
        frame.loc[drop, "excluded_reason"] = f"slices<{args.min_slices}"
        print(f"[过滤] 因切片数 <{args.min_slices} 排除 {int(drop.sum())} 例")
    pool = frame[frame["excluded_reason"] == "kept"].copy()
    print(f"[池] 可抽样患者 {len(pool)} 例，阳性率 {pool['any'].mean():.4f}")

    n = min(args.n_patients, len(pool))
    if n < args.n_patients:
        print(f"[warn] 池中仅 {len(pool)} 例，少于请求的 {args.n_patients}")

    # 按患者分层抽样，保持自然阳性率
    rate = float(pool["any"].mean())
    n_pos = int(round(n * rate))
    n_neg = n - n_pos
    rng = np.random.default_rng(args.seed)
    pos = pool[pool["any"] == 1]["patient_id"].to_numpy()
    neg = pool[pool["any"] == 0]["patient_id"].to_numpy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    chosen = set(pos[:n_pos].tolist()) | set(neg[:n_neg].tolist())
    print(f"[抽样] {len(chosen)} 例（阳性 {n_pos} / 阴性 {n_neg}，"
          f"目标阳性率 {rate:.4f}）")
    frame["sampled"] = frame["patient_id"].isin(chosen)

    sel = pool[pool["patient_id"].isin(chosen)].copy()
    # 按患者分层的 70/15/15
    rows = []
    for cls in (1, 0):
        ids = sel[sel["any"] == cls]["patient_id"].to_numpy()
        rng.shuffle(ids)
        k = len(ids)
        k_tr, k_va = int(k * RATIO_TRAIN), int(k * RATIO_VAL)
        rows.append(pd.DataFrame({
            "patient_id": ids,
            "split": ["train"] * k_tr + ["val"] * k_va + ["test"] * (k - k_tr - k_va),
        }))
    split = pd.concat(rows, ignore_index=True)

    # ---- 校验：患者级严格互斥（本次重建的核心目的）----
    s = {sp: set(split[split["split"] == sp]["patient_id"]) for sp in
         ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if s[a] & s[b]:
            raise SystemExit(f"患者级泄漏！{a} ∩ {b} = {sorted(s[a] & s[b])[:5]}")
    # 按患者划分必然保证同一检查不跨集合，这里显式确认一次
    st = d.merge(split, on="patient_id")
    bad_st = int((st.groupby("study_id")["split"].nunique() > 1).sum())
    if bad_st:
        raise SystemExit(f"有 {bad_st} 个检查跨集合，异常")
    print("[check] 患者两两互斥 ✓   检查跨集合数 = 0 ✓")

    merged = sel.merge(split, on="patient_id")
    summary = (merged.groupby(["split", "any"]).agg(
        patients=("patient_id", "nunique"), slices=("n_slices", "sum"))
        .reset_index().sort_values(["split", "any"]))
    print("\n[划分]")
    print(summary.to_string(index=False))
    tot_pat = int(summary["patients"].sum())
    tot_sli = int(summary["slices"].sum())
    print(f"       合计 {tot_pat} 例 / {tot_sli} 张切片"
          f"（平均 {tot_sli / max(tot_pat, 1):.1f} 张/例）")

    if args.dry_run:
        print("\n[dry-run] 未写任何文件")
        return

    split.to_csv(SPLIT_CSV, index=False)
    for sp in ("train", "val", "test"):
        split[split["split"] == sp][["patient_id"]].to_csv(
            ROOT / f"{sp}_patients.csv", index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    frame.to_csv(FRAME_CSV, index=False)

    pd.DataFrame([
        {"item": "stage_2_train 切片总数（带标签）", "value": len(lab)},
        {"item": "可映射到患者的切片", "value": len(d)},
        {"item": "全库患者数", "value": len(pat)},
        {"item": "全库检查数", "value": int(d["study_id"].nunique())},
        {"item": "抽样患者数", "value": tot_pat},
        {"item": "抽样切片数", "value": tot_sli},
        {"item": "每患者切片数 (mean)", "value": round(tot_sli / max(tot_pat, 1), 1)},
        {"item": "抽样前阳性率 (患者级)", "value": round(rate, 4)},
        {"item": "抽样后阳性率 (患者级)", "value": round(float(merged["any"].mean()), 4)},
        {"item": "划分随机种子", "value": args.seed},
        {"item": "划分比例 train/val/test (按患者分层)", "value": "0.70/0.15/0.15"},
        {"item": "划分粒度", "value": "PatientID（非 StudyInstanceUID）"},
    ]).to_csv(PROV_CSV, index=False)

    print(f"\n[out] {SPLIT_CSV.name} / {{train,val,test}}_patients.csv / "
          f"{SUMMARY_CSV.name} / {FRAME_CSV.name} / {PROV_CSV.name}")
    print("\n[提示] 下一步: python rsna_dicom_to_png.py")


if __name__ == "__main__":
    main()
