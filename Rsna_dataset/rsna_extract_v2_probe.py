# -*- coding: utf-8 -*-
"""v2 提取器：先解 explicit meta(0002) 段，再按 implicit 解析主集。小样本验证命中率"""
import sys
from pathlib import Path
import pandas as pd

LONG_VRS = b"OBOWOFQUNUTODOL"


def extract_uid(p, wk=65536):
    try:
        with open(p, "rb") as f:
            h = f.read(wk)
        if h[128:132] != b"DICM":
            return p.stem, ""
        off, n = 132, len(h)
        # 1) explicit meta (group 0002)
        while off + 8 <= n:
            g = int.from_bytes(h[off:off + 2], "little")
            if g != 0x0002:
                break
            e = int.from_bytes(h[off + 2:off + 4], "little")
            vr = h[off + 4:off + 6]
            if vr in LONG_VRS:
                if off + 12 > n:
                    break
                ln = int.from_bytes(h[off + 8:off + 12], "little")
                voff = off + 12
            else:
                ln = int.from_bytes(h[off + 6:off + 8], "little")
                voff = off + 8
            off = voff + ln
        # 2) 主集 implicit
        while off + 8 <= n:
            g = int.from_bytes(h[off:off + 2], "little")
            e = int.from_bytes(h[off + 2:off + 4], "little")
            ln = int.from_bytes(h[off + 4:off + 8], "little")
            if g == 0x0020 and e == 0x000D:
                return p.stem, h[off + 8:off + 8 + ln].decode("ascii", "ignore").rstrip(" \x00")
            if ln <= 0 or ln > n:
                break
            off += 8 + ln
        return p.stem, ""
    except Exception:  # noqa: BLE001
        return p.stem, ""


if __name__ == "__main__":
    OUT = Path(r"D:\Li-kai\paper\samiya\code\Rsna_dataset")
    DCM = Path(r"D:\Li-kai\project\Data\medical\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection\stage_2_train")
    done = set(pd.read_csv(OUT / "rsna_study_map.csv")["img_id"])
    miss = [f for f in DCM.glob("*.dcm") if f.stem not in done][:500]
    hit = sum(1 for f in miss if extract_uid(f)[1])
    print(f"v2 命中率: {hit}/{len(miss)} = {hit/len(miss)*100:.1f}%")
    for f in miss[:3]:
        s, u = extract_uid(f)
        print(" ", s, "->", u)