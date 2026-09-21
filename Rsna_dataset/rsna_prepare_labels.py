# -*- coding: utf-8 -*-
"""
RSNA 切片级标签：stage_2_train.csv -> 每张切片一个二值标签

stage_2_train.csv 是"每张切片 × 5 个出血亚型"的长表；本脚本把它压成
每张切片一个 any 标签 = 5 个亚型里**任一为 1**。

注意 label 语义（论文里要写清楚）：
  RSNA 的标签是**逐张切片**标注的；
  而 CQ500 的标签是**逐患者**的（3 位放射科医生投票），再传播到该患者所有切片。
  所以两者切片级阳性率差很多（RSNA ~14% vs CQ500 ~42-46%），
  跨数据集的**切片级**指标不可直接比较，患者级才可比。

输出: Rsna_dataset/rsna_slice_labels.csv   (img_id, any)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset"
           r"\rsna-intracranial-hemorrhage-detection")
TRAIN_CSV = RAW / "stage_2_train.csv"
OUT_CSV = ROOT / "rsna_slice_labels.csv"

SUBTYPES = ["epidural", "intraparenchymal", "intraventricular",
            "subarachnoid", "subdural"]


def main():
    if not TRAIN_CSV.exists():
        raise SystemExit(f"找不到 {TRAIN_CSV}")
    print(f"[read] {TRAIN_CSV.name}")
    d = pd.read_csv(TRAIN_CSV)
    d["subtype"] = d["ID"].str.rsplit("_", n=1).str[1]
    d["img_id"] = d["ID"].str.rsplit("_", n=1).str[0]
    print(f"       行数 {len(d)}  subtypes={sorted(d['subtype'].unique())}")

    sub = d[d["subtype"].isin(SUBTYPES)].copy()
    dropped = len(d) - len(sub)
    if dropped:
        print(f"       [note] 丢弃 {dropped} 行非 5 亚型（如 any 行）")

    lab = (sub.assign(Label=sub["Label"].astype(int))
           .groupby("img_id")["Label"].max().reset_index()
           .rename(columns={"Label": "any"}))
    print(f"[out] 切片 {len(lab)}，阳性 {int(lab['any'].sum())}，"
          f"切片级阳性率 {lab['any'].mean():.4f}")

    dup = int(lab["img_id"].duplicated().sum())
    if dup:
        raise SystemExit(f"img_id 有 {dup} 个重复，异常")
    lab.to_csv(OUT_CSV, index=False)
    print(f"      {OUT_CSV}")


if __name__ == "__main__":
    main()
