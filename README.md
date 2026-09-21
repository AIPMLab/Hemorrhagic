# Python 源码包（基础文件 + 三个数据集）

本包**只含 .py 源码**，不含数据、权重与结果。

## 目录结构（不要改动层级）

```
python_bundle/
    ich_common.py              # 共享计算层：噪声、滤波器、检测器、投票、Grad-CAM
    compare_datasets.py        # 三队列并排结果表
    <其余根级脚本>              # 见 MANIFEST.txt 的「状态」列
    Cq500_dataset/*.py
    Rsna_dataset/*.py
    PhysioNet/*.py
```

## 为什么 `ich_common.py` 必须在根目录

三个 `<ds>_commons.py` 都这么做：

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ich_common import (...)
```

`parent.parent` 就是数据集目录的上一级。**把 `ich_common.py` 挪进任何子目录，
三个数据集的每个脚本都会 import 失败。** 数据集之间的公共定义（脑窗、噪声函数、
滤波器库、患者级投票）都在这一个文件里，三套代码因此不会各自漂移。

## 部署

```bash
# 1) 放到目标机器的任意目录，保持上面的层级
# 2) 自检（不需要数据，先确认代码本身可用）
cd Cq500_dataset  && python 17_verify_cq500_pipeline.py
cd ../Rsna_dataset && python rsna_verify.py
cd ../PhysioNet   && python physionet_verify.py
# 3) 逐数据集跑
cd PhysioNet && python physionet_run_cv.py      # 5 折交叉验证
```

## 数据依赖（本包不含）

| 数据集 | 需要另行准备 |
|---|---|
| CQ500 | 原始 DICOM，`D:\Li-kai\project\Data\medical\cq500_2` |
| RSNA | 原始 DICOM，完整 rsna-intracranial-hemorrhage-detection |
| PhysioNet | `computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0` |
| 预训练权重 | 跑 `Cq500_dataset/00_fetch_pretrained_cq500.py`（用 curl，Python 直连会失败） |

打包时间：2026-09-18 15:12
文件总数：109
