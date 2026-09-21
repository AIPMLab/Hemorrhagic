# -*- coding: utf-8 -*-
"""
PhysioNet CT-ICH 标签整理

原始 hemorrhage_diagnosis.csv 是"每层一行"，列为 5 个出血亚型 + No_Hemorrhage
+ Fracture_Yes_No。本脚本压成：

    any = 1 - No_Hemorrhage            （等价于 5 个亚型的 max，已校验一致）
    patient-level = 该患者任一层 any=1

注意标签口径（论文里要写清楚）：
    PhysioNet 是**逐层**由两位放射科医生标注的（与 RSNA 同类）；
    CQ500 则是**逐患者**标注再传播到该患者所有切片。
    所以切片级阳性率差很多（PhysioNet ~12.7% vs CQ500 ~42-46%），
    跨数据集的**切片级**指标不可直接比较，患者级才可比。
    有趣的是三套数据的**患者级**阳性率很接近（41.6% / 40.4% / 43.9%）。

输出: PhysioNet/physionet_slice_labels.csv
     列: patient_id, slice_no, any, Intraventricular, Intraparenchymal,
         Subarachnoid, Epidural, Subdural, Fracture_Yes_No

用法:
    python physionet_prepare_labels.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from physionet_commons import raw_dataset_dir

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "physionet_slice_labels.csv"

SUBTYPES = ["Intraventricular", "Intraparenchymal", "Subarachnoid",
            "Epidural", "Subdural"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="原始数据集根目录（默认自动解析）")
    args = ap.parse_args()

    raw = Path(args.raw) if args.raw else raw_dataset_dir()
    csv = raw / "hemorrhage_diagnosis.csv"
    if not csv.exists():
        raise SystemExit(f"找不到 {csv}")
    print(f"[read] {csv}")

    d = pd.read_csv(csv)
    need = {"PatientNumber", "SliceNumber", "No_Hemorrhage"} | set(SUBTYPES)
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"CSV 缺列: {missing}")

    d["any"] = (1 - d["No_Hemorrhage"]).astype(int)
    # 与亚型取 max 交叉校验（防止 No_Hemorrhage 与亚型列不一致）
    chk = d[SUBTYPES].max(axis=1)
    bad = int((d["any"] != chk).sum())
    if bad:
        raise SystemExit(f"有 {bad} 行：No_Hemorrhage 与 5 个亚型不一致，需人工确认")
    print("[check] any == max(亚型) 全部一致 ✓")

    out = d.rename(columns={"PatientNumber": "patient_id",
                            "SliceNumber": "slice_no"})
    out = out[["patient_id", "slice_no", "any"] + SUBTYPES + ["Fracture_Yes_No"]]
    out = out.sort_values(["patient_id", "slice_no"]).reset_index(drop=True)

    pat = out.groupby("patient_id")["any"].max()
    print(f"[out] 切片 {len(out)}，患者 {len(pat)}")
    print(f"      切片级阳性率 {out['any'].mean():.4f}")
    print(f"      患者级阳性 {int(pat.sum())}/{len(pat)} = {pat.mean():.4f}")
    per = out.groupby("patient_id").size()
    print(f"      每患者层数 min={per.min()} 中位={per.median():.0f} "
          f"max={per.max()} 平均={per.mean():.1f}")
    print("[subtype] 阳性切片数:")
    print(out[SUBTYPES].sum().to_string())

    out.to_csv(OUT_CSV, index=False)
    print(f"[out] {OUT_CSV}")


if __name__ == "__main__":
    main()
