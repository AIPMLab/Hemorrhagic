# -*- coding: utf-8 -*-
"""v2 全量：只处理 known map 缺失的图，结果合并写入 rsna_study_map_all.csv"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsna_extract_v2_probe import extract_uid  # noqa: E402

OUT = Path(r"D:\Li-kai\paper\samiya\code\Rsna_dataset")
DCM = Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection\stage_2_train")


def main():
    done = pd.read_csv(OUT / "rsna_study_map.csv")
    done_ids = set(done["img_id"])
    files = [f for f in DCM.glob("*.dcm") if f.stem not in done_ids]
    print(f"[v2] 待处理 {len(files)}（{time.time():.0f}）", flush=True)
    t0 = time.time()
    pairs = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, uid) in enumerate(ex.map(extract_uid, files, chunksize=1024), 1):
            pairs.append((stem, uid))
            if i % 100000 == 0:
                print(f"    ... {i}/{len(files)}（{time.time()-t0:.0f}s）", flush=True)
    fb = pd.DataFrame(pairs, columns=["img_id", "study_id"])
    fb = fb[fb["study_id"] != ""]
    print(f"[v2] 命中 {len(fb)} / 缺失 {len(files)}（{time.time()-t0:.0f}s）", flush=True)
    all_map = pd.concat([done, fb], ignore_index=True).drop_duplicates("img_id")
    all_map.to_csv(OUT / "rsna_study_map_all.csv", index=False)
    print(f"[done] 总映射 {len(all_map)} 图 / {all_map['study_id'].nunique()} 患者")


if __name__ == "__main__":
    main()