# -*- coding: utf-8 -*-
"""
CQ500 数据预处理（步骤2-3）
1) 从 reads.csv 生成患者级标签（3位放射科医生 ICH 多数组投票）
2) 按患者分层划分 train/val/test（70/15/15），防患者级数据泄漏

输出（全部写入本脚本同级目录 Cq500_dataset/）：
- cq500_patient_labels.csv   患者级标签
- cq500_split.csv            患者 -> split 映射
- train_patients.csv / val_patients.csv / test_patients.csv
- cq500_split_summary.csv    三个集合的类别统计
"""
from pathlib import Path
import numpy as np
import pandas as pd

from cq500_commons import raw_dataset_dir

CQ500_RAW = raw_dataset_dir()                  # 由 CQ500_RAW_DIR 或默认候选决定
OUT = Path(__file__).resolve().parent          # 本脚本在 Cq500_dataset 内，输出直接写到当前目录
OUT.mkdir(parents=True, exist_ok=True)

READS = CQ500_RAW / "reads.csv"
ICH_COLS = ["R1:ICH", "R2:ICH", "R3:ICH"]           # 三位医生出血判定
OTHER_COLS = (
    ["R1:Fracture", "R2:Fracture", "R3:Fracture",
     "R1:MassEffect", "R2:MassEffect", "R3:MassEffect",
     "R1:MidlineShift", "R2:MidlineShift", "R3:MidlineShift"]
)

SEED = 42
RATIO_TRAIN, RATIO_VAL = 0.70, 0.15

# ============ 步骤2：患者级标签 ============
df = pd.read_csv(READS, encoding="utf-8-sig")
df["patient_id"] = df["name"].str.replace("CQ500-CT-", "CQ500CT", regex=False).str.strip()

edges = df[ICH_COLS].astype(int)
df["ich_votes"] = edges.sum(axis=1)
df["label"] = np.where(df["ich_votes"] >= 2, "Hemorrhagic", "Normal")   # >=2/3 医生判出血

others = df[OTHER_COLS].astype(int).max(axis=1)                          # 任一组医生确认存在
df["has_other_findings"] = np.where(others > 0, 1, 0)                    # 骨折/占位/中线移位标记（供可选过滤）

labels = df[["patient_id", "label", "ich_votes", "has_other_findings"]]
labels.to_csv(OUT / "cq500_patient_labels.csv", index=False)
print(f"[labels] 共 {len(labels)} 例患者")
print(labels["label"].value_counts().to_string())
print(f"[labels] 其中混杂其他发现(骨折/占位/中线移位)的例数: {int(labels['has_other_findings'].sum())}")

# ============ 步骤3：按患者分层划分 ============
rng = np.random.default_rng(SEED)
rows = []
for cls in ["Normal", "Hemorrhagic"]:
    ids = labels.loc[labels["label"] == cls, "patient_id"].tolist()
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * RATIO_TRAIN)
    n_val = int(n * RATIO_VAL)
    rows.append(pd.DataFrame({
        "patient_id": ids,
        "split": ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val),
    }))

split_df = pd.concat(rows, ignore_index=True)
split_df.to_csv(OUT / "cq500_split.csv", index=False)
for sp in ["train", "val", "test"]:
    sub = split_df[split_df["split"] == sp]
    sub[["patient_id"]].to_csv(OUT / f"{sp}_patients.csv", index=False)
    print(f"[split:{sp}] 患者数={len(sub)}  类别分布: " +
          str(split_df.merge(labels, on="patient_id").loc[split_df["split"] == sp, "label"].value_counts().to_dict()))

summary = (split_df.merge(labels, on="patient_id")
           .groupby(["split", "label"]).size().reset_index(name="count"))
summary.to_csv(OUT / "cq500_split_summary.csv", index=False)
print("[split] summary 已保存:", summary.to_string(index=False))
print(f"[split] 合计 {len(split_df)} 例患者（应等于 {len(labels)}）")