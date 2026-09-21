# -*- coding: utf-8 -*-
"""
RSNA 患者映射提取：img_id -> (patient_id, study_id)

为什么必须做这一步：
  stage_2_train.csv 只给出每张切片的 img_id 与标签，**没有任何患者/检查的分组信息**。
  DICOM 头里同时有 PatientID (0010,0020) 与 StudyInstanceUID (0020,000D)，两者并不
  一一对应 —— 同一个患者可以多次检查（不同 StudyInstanceUID）。实测 21,744 次检查
  里就有一批患者有多次检查。

  旧的 rsna_extract_studyuid.py 只取 StudyInstanceUID 并把注释写成"患者"，
  于是划分是按**检查**而不是按**患者**做的：同一患者的两次检查可能一个进 train、
  一个进 test，造成患者级泄漏。本脚本同时抽出 PatientID，后续划分一律按患者。

实现要点：用 pydicom 的 specific_tags + stop_before_pixels —— 拿到两个标签就停，
          只读几 KB 头，不读像素；实测 ~2,600 文件/s，全量约 5 分钟。
          （自己用固定窗口硬解析反而更慢：要么窗口不够漏标签，要么读太多字节。）

输出: Rsna_dataset/rsna_patient_map.csv   (img_id, patient_id, study_id)

用法:
    python rsna_extract_patient_map.py                # 全量（已存在则跳过）
    python rsna_extract_patient_map.py --force
    python rsna_extract_patient_map.py --limit 5000   # 小样本自测
"""
import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

warnings.filterwarnings("ignore", message="Invalid value for VR UI")

ROOT = Path(__file__).resolve().parent          # Rsna_dataset
RAW_DEFAULT = Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset"
                   r"\rsna-intracranial-hemorrhage-detection\stage_2_train")
OUT_CSV = ROOT / "rsna_patient_map.csv"
TAGS = ["PatientID", "StudyInstanceUID"]


def read_ids(path):
    import pydicom
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=TAGS)
        return (path.stem,
                str(getattr(ds, "PatientID", "") or "").strip(),
                str(getattr(ds, "StudyInstanceUID", "") or "").strip())
    except Exception as e:  # noqa: BLE001
        return path.stem, "", f"ERR:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重新提取")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（自测）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--raw", default=str(RAW_DEFAULT))
    args = ap.parse_args()

    if not args.force and OUT_CSV.exists():
        d = pd.read_csv(OUT_CSV)
        print(f"[skip] {OUT_CSV.name} 已存在：{len(d)} 张切片 / "
              f"{d.patient_id.nunique()} 患者 / {d.study_id.nunique()} 检查")
        print("       要重跑请加 --force")
        return

    raw = Path(args.raw)
    if not raw.exists():
        raise SystemExit(f"找不到 DICOM 目录: {raw}")
    files = sorted(raw.glob("*.dcm"))
    if args.limit:
        files = files[:args.limit]
    print(f"[start] 扫描 {len(files)} 个 DICOM（{args.workers} workers）", flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, tup in enumerate(ex.map(read_ids, files, chunksize=512), 1):
            rows.append(tup)
            if i % 100000 == 0:
                el = time.time() - t0
                print(f"[progress] {i}/{len(files)}  {el:.0f}s "
                      f"({i/el:.0f} 文件/s)", flush=True)

    df = pd.DataFrame(rows, columns=["img_id", "patient_id", "study_id"])
    el = time.time() - t0
    n_err = int(df.study_id.str.startswith("ERR:", na=False).sum())
    n_nopat = int((df.patient_id == "").sum())

    print(f"\n[done] {len(df)} 行，{el:.0f}s")
    print(f"       患者 {df.loc[df.patient_id != '', 'patient_id'].nunique()} / "
          f"检查 {df.loc[~df.study_id.str.startswith('ERR:', na=False), 'study_id'].nunique()}")
    if n_err:
        print(f"       [警告] {n_err} 个文件读取失败")
    if n_nopat:
        print(f"       [警告] {n_nopat} 个文件缺 PatientID")

    # 关键校验：按检查 vs 按患者，确实不是一回事
    d = df[(df.patient_id != "") & (~df.study_id.str.startswith("ERR:", na=False))]
    per_pat = d.groupby("patient_id")["study_id"].nunique()
    multi = int((per_pat > 1).sum())
    print(f"       有多次检查的患者: {multi} / {len(per_pat)}  "
          f"(这些患者按检查划分会同时出现在 train 与 test)")

    df.to_csv(OUT_CSV, index=False)
    print(f"[out] {OUT_CSV}")


if __name__ == "__main__":
    main()
