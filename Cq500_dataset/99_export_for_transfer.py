# -*- coding: utf-8 -*-
"""
迁移打包：把流水线拷到另一个服务器/机器所需的最小文件集。

回答"要拷哪些文件"：只拷真正需要的，并生成 MANIFEST 与接收端说明，
避免漏文件（尤其是跨目录依赖）或带上无用的大目录。

支持 CQ500 / RSNA / 两者，默认 both（论文里两套数据集要对比）。

**关键：RSNA 依赖 CQ500 的两个文件**（不是可选项）
    Rsna_dataset/rsna_commons.py 会把 ../Cq500_dataset 加进 sys.path，
    并 `from cq500_commons import ...` —— 噪声/滤波/投票/扫描全部复用那一份，
    以保证两个数据集的计算定义不会分叉。
    Rsna_dataset/rsna_verify.py 还会 import cq500_dicom_to_png（校验脑窗一致）
    并读 cq500_slices_index.csv（生成队列对比表）。
  所以只拷 RSNA 时，本脚本会自动带上这几个 CQ500 文件。
  也因此 **两个目录必须保持同级**（都在同一个 $CODE 下），否则 import 会失败。

各数据集内容:
    Cq500_dataset/  代码 + cq500_*.csv + png/（约 5.9 GB，16.5 万张 224x224 PNG）
    Rsna_dataset/   代码 + rsna_*.csv + png/（约 0.6 GB，19,646 张）+ 自动补的 CQ500 依赖
    weights/        3 个 timm 预训练权重（约 227 MB，两套共用）

**不需要**:
    原始 DICOM（cq500 / cq500_2 / rsna-ihd-dataset）—— 只有"重新转换/重建划分"
    才用得上。png 与划分已定稿，新机器直接复用，省下 400+ GB。
    已废弃的 dataset1 相关任何东西、__pycache__、日志。

用法:
    python 99_export_for_transfer.py --dest E:\\transfer                 # 两者都打
    python 99_export_for_transfer.py --dest /mnt/usb/x --datasets rsna   # 只要 RSNA
    python 99_export_for_transfer.py --dest out --no-png                 # 数据另行同步
    python 99_export_for_transfer.py --dest out --manifest-only          # 只出清单
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent       # Cq500_dataset
CODE = HERE.parent                           # code 根

SKIP_NAMES = {"__pycache__"}

# 旧版（按 study 划分）遗留的文件：不拷，避免新机器上误用。
#   rsna_study_map*.csv        已被 rsna_patient_map.csv 取代（study 级，会泄漏）
#   rsna_patient_labels.csv    旧版 study 级标签，名字有误导性
#   rsna_extract_*.py          一次性提取脚本，已被 rsna_extract_patient_map.py 取代
#   rsna_prepare_part1.py      同上
# （13/14_gradcam_*_rsna.py 已适配到 patient_id 索引，正常打包）
STALE = {
    "rsna_study_map.csv", "rsna_study_map_all.csv", "rsna_patient_labels.csv",
    "rsna_extract_studyuid.py", "rsna_extract_v2_all.py", "rsna_extract_v2_probe.py",
    "rsna_extract_fallback.py", "rsna_prepare_part1.py",
}

# 共享层：两个数据集都依赖它（位于 code 根目录）。任何打包都必须带上 ——
# RSNA 只靠 ich_common.py 就能独立运行，不再需要 Cq500_dataset 里的任何文件。
SHARED_FILES = ["ich_common.py", "compare_datasets.py"]

# 每个数据集要拷的代码与元数据（相对该数据集目录的 glob）
DATASET_GLOBS = {
    "Cq500_dataset": (["*.py"],
                      ["cq500_*.csv", "train_patients.csv", "val_patients.csv",
                       "test_patients.csv", "*.txt"]),
    "Rsna_dataset": (["*.py"],
                     ["rsna_*.csv", "train_patients.csv", "val_patients.csv",
                      "test_patients.csv", "*.txt"]),
    "PhysioNet": (["*.py"],
                  ["physionet_*.csv", "train_patients.csv", "val_patients.csv",
                   "test_patients.csv", "*.txt"]),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def add_dir(items, ds_name, with_png):
    """把某个数据集目录下的代码/元数据（可选 png）加入清单。"""
    base = CODE / ds_name
    if not base.is_dir():
        print(f"[warn] 缺少目录 {base}，跳过")
        return
    code_globs, meta_globs = DATASET_GLOBS[ds_name]
    skipped = []
    for g in code_globs + meta_globs:
        for p in sorted(base.glob(g)):
            if not p.is_file() or p.name in SKIP_NAMES:
                continue
            if p.name in STALE:
                skipped.append(p.name)
                continue
            items.append((p, Path(ds_name) / p.name))
    if skipped:
        print(f"[skip] {ds_name}: 排除 {len(skipped)} 个旧版/未适配文件 -> "
              f"{', '.join(sorted(skipped))}")
    if with_png and (base / "png").is_dir():
        for p in (base / "png").rglob("*"):
            if p.is_file():
                items.append((p, Path(ds_name) / "png" / p.relative_to(base / "png")))


def gather(datasets, with_png):
    items, seen = [], set()

    def push(src, rel):
        key = str(rel)
        if key not in seen:
            seen.add(key)
            items.append((src, rel))

    # 共享层（两个数据集都依赖；RSNA 只靠它就能独立运行，不碰 Cq500_dataset）
    for name in SHARED_FILES:
        p = CODE / name
        if p.is_file():
            push(p, Path(name))
        else:
            print(f"[warn] 缺少共享文件 {name}，新机器上会 import 失败")

    if "cq500" in datasets:
        add_dir(items, "Cq500_dataset", with_png)
    if "rsna" in datasets:
        add_dir(items, "Rsna_dataset", with_png)
    if "physionet" in datasets:
        add_dir(items, "PhysioNet", with_png)
    for _, rel in list(items):
        seen.add(str(rel))

    if (CODE / "weights").is_dir():
        for p in sorted((CODE / "weights").glob("*")):
            if p.is_file():
                push(p, Path("weights") / p.name)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="目标目录（外接盘 / 网络挂载点 / 打包目录）")
    ap.add_argument("--datasets", default="all",
                    choices=["all", "cq500", "rsna", "physionet"])
    ap.add_argument("--no-png", action="store_true", help="不拷 png 图像数据")
    ap.add_argument("--manifest-only", action="store_true", help="只生成清单，不拷文件")
    args = ap.parse_args()

    datasets = {"all": {"cq500", "rsna", "physionet"},
                "cq500": {"cq500"},
                "rsna": {"rsna"},
                "physionet": {"physionet"}}[args.datasets]
    dest = Path(args.dest).expanduser()
    with_png = not args.no_png
    items = gather(datasets, with_png)

    n_png = sum(1 for _, rel in items if "png" in rel.parts)
    total = sum(p.stat().st_size for p, _ in items)
    print(f"[plan] datasets={args.datasets}  png={'yes' if with_png else 'no'}")
    print(f"[plan] 小文件 {len(items) - n_png} 个"
          f" + png {n_png} 个 = {len(items)} 个，合计 {total / 1073741824:.2f} GB")

    dest.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        copied = 0
        for src, rel in items:
            dst = dest / rel
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                copied += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
            if copied % 20000 == 0:
                print(f"[copy] {copied}/{len(items)}", flush=True)
        print(f"[copy] 完成 {copied} 个文件")

    # ---- 清单 ----
    lines = ["# 流水线迁移清单", "", f"目标目录: {dest}",
             f"数据集: {args.datasets}", f"含 png: {with_png}",
             f"文件总数: {len(items)}", "",
             "## 代码 / 元数据 / 权重（含 sha256 前16位）"]
    for src, rel in items:
        if "png" in rel.parts:
            continue
        lines.append(f"{sha256(src)}  {src.stat().st_size:>12}  {rel.as_posix()}")
    if with_png:
        per_ds = {}
        for _, r in items:
            if "png" in r.parts:
                per_ds.setdefault(r.parts[0], 0)
                per_ds[r.parts[0]] += 1
        lines += ["", "## png 数据统计（逐文件未列，数量对齐即可）"]
        for k, v in sorted(per_ds.items()):
            lines.append(f"  {k}: {v}")
    (dest / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[manifest] {dest / 'MANIFEST.txt'}")

    ds_txt = {"all": "CQ500 + RSNA + PhysioNet", "cq500": "仅 CQ500",
              "rsna": "仅 RSNA", "physionet": "仅 PhysioNet"}[args.datasets]
    readme = f"""# 在目标服务器上恢复流水线（{ds_txt}）

本包由 99_export_for_transfer.py 生成，共 {len(items)} 个文件，
{total / 1073741824:.2f} GB。

## 1. 目录结构（**结构必须照搬**）

把本包内容放到任意路径 `$CODE`（例如 /home/user/ich）：

    $CODE/weights/                    3 个 timm 预训练权重（两套数据集共用）
    $CODE/Cq500_dataset/*.py          脚本
    $CODE/Cq500_dataset/cq500_*.csv   索引 / 划分 / 标签
    $CODE/Cq500_dataset/png/          224x224 PNG
    $CODE/Rsna_dataset/*.py           脚本
    $CODE/Rsna_dataset/rsna_*.csv     索引 / 划分 / 标签 / 患者映射
    $CODE/Rsna_dataset/png/           224x224 PNG
    $CODE/PhysioNet/*.py              脚本（含 physionet_run_cv.py / aggregate_cv.py）
    $CODE/PhysioNet/physionet_*.csv   索引 / 划分 / 逐层标签 / 5 折划分
    $CODE/PhysioNet/png*/             224x224 PNG（含 png_fold{{k}} 交叉验证图）

**两条硬性要求：**

1. **不要改目录名。** 脚本之间用相对路径互相引用。
2. **`ich_common.py` 必须与数据集目录同级**（都在同一个 `$CODE` 下）。
   它是两个数据集共用的计算层（脑窗、噪声、滤波、判别器类别、预训练加载、
   患者级投票、filter 穷举、Grad-CAM）。放在同级后：
     - **RSNA 不依赖 CQ500**：只靠 `ich_common.py` + `Rsna_dataset/` 就能单独跑
       （自检 G2 会强制验证这一点）
     - **CQ500 不依赖 RSNA**
     - 两边共用同一份计算定义，结果可以直接并排比较

## 2. 环境

    conda create -n ich python=3.11 -y
    conda activate ich
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm opencv-python numpy pandas scikit-learn scikit-image scipy pillow matplotlib seaborn tqdm

## 3. 自检（**先做这一步，再跑几小时的训练**）

    cd $CODE/Cq500_dataset && python 17_verify_cq500_pipeline.py   # 若打包了 CQ500
    cd $CODE/Rsna_dataset  && python rsna_verify.py                # 若打包了 RSNA

期望两者都 FAIL=0。若 CQ500 的 F1/F2/F3 或 RSNA 的 A1/E1 报错，说明 png 或索引
没拷全；若 RSNA 的 G2 报错，说明 ich_common.py 没在同级。
尚未训练时 E2/J1 会 SKIP，属正常。

## 4. 预训练权重

`weights/` 已含 3 个 backbone，**无需联网**：
    resnet50.a1_in1k.bin   swin_tiny_patch4_window7_224.ms_in1k.bin   efficientnet_b0.ra_in1k.bin
若缺失，跑 `python 00_fetch_pretrained_cq500.py`（走 curl + hf-mirror；
本机 Python 的 TLS 会被重置，requests/huggingface_hub 都下不动）。

## 5. 跑流水线

    cd $CODE/Cq500_dataset && python run_cq500_pipeline.py --dry-run && python run_cq500_pipeline.py
    cd $CODE/Rsna_dataset  && python run_rsna_pipeline.py  --dry-run && python run_rsna_pipeline.py

顺序：02 → 04 → 09 → 10(6class/7class) → 05 → 20 → 07 → 11(6class/7class)
      → 19(TTA 消融) → 16(6class/7class) → 18 → 12 → 15 → visualize
（两套数据集各自一份，互不干扰；结果分别落在 results/*_cq500 与 results/*_rsna）

Grad-CAM 图属于可选步骤，需要先加 `--save-images`：
    python 11_adaptive_unknown_pipeline_rsna.py --variant 7class --save-images
    python 13_gradcam_recovery_rsna.py --variant 7class
    python 14_gradcam_known_noise_recovery_rsna.py

## 6. PhysioNet 用 5 折患者级交叉验证（不是单次划分）

PhysioNet 只有 82 例，单次 70/15/15 的 test 仅 14 例（准确率步长 7.1%，配对检验几乎
没有功效）。正式做法是 5 折患者级交叉验证 —— 原论文也是这么做的。

折的构造：折 k 作 test，折 (k+1)%5 作 val，其余 3 折作 train（按患者严格互斥）。
每折产物靠环境变量 PHYSIONET_FOLD=k 隔离到 png_fold{{k}}/、results_fold{{k}}/ 等，
因此 5 折互不覆盖。

    cd $CODE/PhysioNet
    python physionet_make_split.py --n-folds 5     # 生成 physionet_split_fold{{0..4}}.csv
    python physionet_run_cv.py                     # 依次跑完 5 折（每折自动建图+自检+跑流水线）
    python physionet_aggregate_cv.py               # 汇总 mean±std 与合并评估

若只想跑单次 70/15/15（与 CQ500/RSNA 口径一致），不设 PHYSIONET_FOLD 即可：
    python physionet_images_to_png.py && python run_physionet_pipeline.py

## 7. 三个数据集的并排对比

三者都跑完后（在 `$CODE` 下）：

    python compare_datasets.py

输出 results/cross_dataset_comparison/（表 + 图），直接用于论文里并排展示。

## 8. 若某天要从原始数据重新生成

    export CQ500_RAW_DIR=/path/to/cq500        # CQ500（DICOM，需 pydicom）
    export RSNA_RAW_DIR=/path/to/rsna          # RSNA（DICOM，需 pydicom）
    export PHYSIONET_RAW_DIR=/path/to/physionet  # PhysioNet（已是 JPEG，无需 pydicom）
脚本顶部不再硬编码盘符，设这三个环境变量即可。
"""
    (dest / "TRANSFER_README.md").write_text(readme, encoding="utf-8")
    print(f"[readme] {dest / 'TRANSFER_README.md'}")

    print("\n[提示] 走 rsync 一次传即可（保留目录结构）：")
    print(f"    rsync -av --progress {dest}/ user@server:/home/user/ich/")


if __name__ == "__main__":
    main()
