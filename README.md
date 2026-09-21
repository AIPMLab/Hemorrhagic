# Python Source Package (Base Files + Three Datasets)

This package contains **only .py source code** — no data, no weights, no results.

## Directory Structure (do not change the hierarchy)

```
python_bundle/
    ich_common.py              # shared compute layer: noise, filters, detectors, voting, Grad-CAM
    compare_datasets.py        # side-by-side results table for the three cohorts
    <remaining root-level scripts>   # see the "status" column in MANIFEST.txt
    Cq500_dataset/*.py
    Rsna_dataset/*.py
    PhysioNet/*.py
```

## Why `ich_common.py` must stay in the root directory

All three `<ds>_commons.py` files do this:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ich_common import (...)
```

`parent.parent` is the directory one level above the dataset directory. **Move `ich_common.py` into any subdirectory and every script in all three datasets will fail to import.** The definitions shared across the datasets (brain window, noise functions, filter library, patient-level voting) all live in this one file, so the three code sets never drift apart.

## Deployment

```bash
# 1) Place it in any directory on the target machine, keeping the hierarchy above
# 2) Self-check (no data required — first confirm the code itself works)
cd Cq500_dataset  && python 17_verify_cq500_pipeline.py
cd ../Rsna_dataset && python rsna_verify.py
cd ../PhysioNet   && python physionet_verify.py
# 3) Run dataset by dataset
cd PhysioNet && python physionet_run_cv.py      # 5-fold cross-validation
```

## Data Dependencies (not included in this package)

| Dataset | Must be prepared separately |
|---|---|
| CQ500 | raw DICOM, `D:\Li-kai\project\Data\medical\cq500_2` |
| RSNA | raw DICOM, the full rsna-intracranial-hemorrhage-detection set |
| PhysioNet | `computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0` |
| Pretrained weights | run `Cq500_dataset/00_fetch_pretrained_cq500.py` (uses curl; a direct download from Python will fail) |

Packaged: 2026-09-18 15:12
Total files: 109
