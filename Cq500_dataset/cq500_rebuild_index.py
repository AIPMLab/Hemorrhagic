# -*- coding: utf-8 -*-
"""重建 cq500_slices_index.csv（扫描 png 目录实际文件）

路径全部相对本文件定位，不再硬编码盘符，换机器可直接跑。
image_path 统一写 POSIX 正斜杠，Windows/Linux 都能解析
（cq500_commons.load_index 也会再兜底转换一次）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from cq500_commons import DATA_ROOT as PNG_ROOT
from cq500_commons import ROOT as OUT

rows = []
for p in PNG_ROOT.rglob("*.png"):
    parts = p.parent.parts
    rows.append({"patient_id": parts[-1], "split": parts[-3], "label": parts[-2],
                 "image_path": p.relative_to(PNG_ROOT).as_posix()})
idx = pd.DataFrame(rows)
idx.to_csv(OUT / "cq500_slices_index.csv", index=False)
print(f"[rebuild] 索引 {len(idx)} 行, {idx['patient_id'].nunique()} 患者 -> {PNG_ROOT}")
print(idx["split"].value_counts().to_string())
print(idx["label"].value_counts().to_string())
