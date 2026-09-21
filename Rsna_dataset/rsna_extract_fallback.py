# -*- coding: utf-8 -*-
"""fallback：对首次未命中 StudyUID 的 dcm 使用双解析器（explicit/implicit VR）重提取。
   结果合并写入 rsna_study_map_all.csv"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

RSNA_ROOT = Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection")
DCM_DIR = RSNA_ROOT / "stage_2_train"
OUT = Path(__file__).resolve().parent
WK = 65536                     # 读 64KB，覆盖 UID 不在开头的情况
LONG_VRS = b"OBOWOFQUNUTODOL"


def parse_explicit(h):
    off, n = 132, len(h)
    while off + 8 <= n:
        g = int.from_bytes(h[off:off + 2], "little")
        e = int.from_bytes(h[off + 2:off + 4], "little")
        vr = h[off + 4:off + 6]
        if not (vr.isalpha() and len(vr) == 2):
            return None
        if vr in LONG_VRS:
            if off + 12 > n:
                break
            length = int.from_bytes(h[off + 8:off + 12], "little")
            voff = off + 12
        else:
            length = int.from_bytes(h[off + 6:off + 8], "little")
            voff = off + 8
        if g == 0x0020 and e == 0x000D:
            return h[voff:voff + length].decode("ascii", "ignore").rstrip(" \x00")
        off = voff + length
    return None


def parse_implicit(h):
    off, n = 132, len(h)
    while off + 8 <= n:
        g = int.from_bytes(h[off:off + 2], "little")
        e = int.from_bytes(h[off + 2:off + 4], "little")
        length = int.from_bytes(h[off + 4:off + 8], "little")
        voff = off + 8
        if g == 0x0020 and e == 0x000D:
            return h[voff:voff + length].decode("ascii", "ignore").rstrip(" \x00")
        off = voff + length
    return None


def extract(p):
    try:
        with open(p, "rb") as f:
            h = f.read(WK)
        if h[128:132] != b"DICM":
            return p.stem, ""
        uid = parse_explicit(h) or parse_implicit(h)
        return p.stem, uid or ""
    except Exception:  # noqa: BLE001
        return p.stem, ""


def main():
    done = pd.read_csv(OUT / "rsna_study_map.csv")
    done_ids = set(done["img_id"])
    print(f"[已知映射] {len(done_ids)}，待处理剩余 ...")
    files = [f for f in DCM_DIR.glob("*.dcm") if f.stem not in done_ids]
    print(f"[fallback] 剩余 {len(files)} 个", flush=True)
    t0 = time.time()
    pairs = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, uid) in enumerate(ex.map(extract, files, chunksize=1024), 1):
            pairs.append((stem, uid))
            if i % 100000 == 0:
                print(f"    ... {i}/{len(files)}（{time.time()-t0:.0f}s）", flush=True)
    fb = pd.DataFrame(pairs, columns=["img_id", "study_id"])
    fb = fb[fb["study_id"] != ""]
    print(f"[fallback] 新命中 {len(fb)} / 仍缺失 {len(files)-len(fb)}（{time.time()-t0:.0f}s）", flush=True)
    all_map = pd.concat([done, fb], ignore_index=True).drop_duplicates("img_id")
    all_map.to_csv(OUT / "rsna_study_map_all.csv", index=False)
    print(f"[done] 总映射 {len(all_map)} 图 / {all_map['study_id'].nunique()} 患者")


if __name__ == "__main__":
    main()