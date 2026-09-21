# -*- coding: utf-8 -*-
"""轻量 DICOM StudyInstanceUID 提取：只读文件头 4KB，按显式 VR 解析元素，命中 (0020,000D) 即返回。
   I/O 由整文件(~6MB)降为 4KB，速度提升千倍。"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RSNA_ROOT = Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection")
DCM_DIR = RSNA_ROOT / "stage_2_train"
OUT = Path(__file__).resolve().parent

LONG_VRS = b"OBOWOFQUNUTODOL"
WK = 4096


def extract_study_uid(p):
    try:
        with open(p, "rb") as f:
            h = f.read(WK)
        if h[128:132] != b"DICM":
            return p.stem, ""
        off = 132
        n = len(h)
        while off + 8 <= n:
            g = int.from_bytes(h[off:off + 2], "little")
            e = int.from_bytes(h[off + 2:off + 4], "little")
            vr = h[off + 4:off + 6]
            if vr in LONG_VRS:
                if off + 12 > n:
                    break
                length = int.from_bytes(h[off + 8:off + 12], "little")
                voff = off + 12
            else:
                length = int.from_bytes(h[off + 6:off + 8], "little")
                voff = off + 8
            if g == 0x0020 and e == 0x000D:
                val = h[voff:voff + length].decode("ascii", "ignore").rstrip(" \x00")
                return p.stem, val
            off = voff + length
            if off > n - 8:
                break
        return p.stem, ""
    except Exception:  # noqa: BLE001
        return p.stem, ""


def main():
    files = sorted(DCM_DIR.glob("*.dcm"))
    print(f"[scan] 文件 {len(files)} 个，轻量提取 StudyUID ...", flush=True)
    t0 = time.time()
    pairs = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, uid) in enumerate(ex.map(extract_study_uid, files, chunksize=1024), 1):
            pairs.append((stem, uid))
            if i % 100000 == 0:
                print(f"    ... {i}/{len(files)}（{time.time()-t0:.0f}s）", flush=True)
    import pandas as pd
    df = pd.DataFrame(pairs, columns=["img_id", "study_id"])
    df = df[df["study_id"] != ""]
    df.to_csv(OUT / "rsna_study_map.csv", index=False)
    print(f"[done] {len(df)} 图 / {df['study_id'].nunique()} 患者（{time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()