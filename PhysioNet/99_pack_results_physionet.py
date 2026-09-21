# -*- coding: utf-8 -*-
"""
打包 PhysioNet 数据集的关键结果文件（供结果分析用）

只收**小而关键**的 CSV/表/日志，不含图片、权重、png 数据集 —— 整包通常几百 KB。
可选 --with-figures 把 Grad-CAM / 对比图一起收进来。

与 CQ500/RSNA 版本的区别：**PhysioNet 用 5 折交叉验证**，结果分散在
results_fold{0..4}/ 里（单次划分则落在 results/）。本脚本自动发现有哪些折，
逐折收集并保留折目录层级，最后把 results_cv_physionet/ 的跨折汇总一并收入。

它会顺带做三件对分析很关键的事：
  1. 现场跑一遍 physionet_verify.py（单次划分 + 每个存在的折）并把输出存进包里
  2. 记录运行环境（python/torch/CUDA/GPU）
  3. 从训练日志里抽出实际 epoch 数
—— 没有这三样，别人无法判断这组数字是否可解释。

用法:
    python 99_pack_results_physionet.py                     # 默认打到 PhysioNet_results.zip
    python 99_pack_results_physionet.py --dest D:\\out
    python 99_pack_results_physionet.py --with-figures
    python 99_pack_results_physionet.py --with-indexes      # 连切片索引一起收（每折 120KB）
    python 99_pack_results_physionet.py --folds 0,1         # 只打包部分折
    python 99_pack_results_physionet.py --no-verify         # 不跑自检
"""
import argparse
import csv as _csv
import hashlib
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEAD = HERE.parent  # code 根目录

_ap = argparse.ArgumentParser()
_ap.add_argument("--dest", default=str(HEAD / "PhysioNet_results"),
                 help="输出目录（会打成同名 .zip）")
_ap.add_argument("--with-figures", action="store_true", help="连 PNG 图一起打")
_ap.add_argument("--with-indexes", action="store_true",
                 help="连 physionet_slices_index*.csv 一起收（默认不收：那是输入不是结果）")
_ap.add_argument("--folds", default="",
                 help="逗号分隔，只打包这些折（默认自动发现磁盘上存在的全部）")
_ap.add_argument("--results-root", default="",
                 help="从别处的结果树打包（该目录下应有 results_fold*/ 或 results/）")
_ap.add_argument("--no-verify", action="store_true", help="跳过自检")
args = _ap.parse_args()

# 变体列表优先从共享层取；取不到就退回默认，保证打包脚本本身不会因环境失败
# 注意：ich_common.py 在 code 根目录（本文件的上一级），不在 PhysioNet 里
try:
    sys.path.insert(0, str(HEAD))
    from ich_common import DETECTOR_VARIANTS
    VARIANTS = sorted(DETECTOR_VARIANTS)
except Exception as e:  # pragma: no cover
    print(f"[warn] 无法从 ich_common 读取变体列表（{e}），用默认 {['6class', '7class']}")
    VARIANTS = ["6class", "7class"]

# (类别, 相对某一折结果目录的路径模板, 说明, 由哪一步产生)
# {v} 会被展开成 6class / 7class
ITEMS = [
    ("A_核心_Q1Q2", "optimality_physionet/{v}/oracle_summary_physionet.csv",
     "Q1(增强是否超过加噪) + Q2(选的filter是否最优) —— 最重要", "16"),
    ("A_核心_Q1Q2", "optimality_physionet/{v}/filter_sweep_unknown_mixed_physionet.csv",
     "复合噪声下 10 个 filter 的患者级指标全表（oracle 的原始证据）", "16"),
    ("A_核心_Q1Q2", "optimality_physionet/{v}/per_patient_oracle_physionet.csv",
     "逐患者 oracle 最优 filter 与 adaptive 是否一致（配对检验的原始数据）", "16"),
    ("A_核心_Q1Q2", "optimality_physionet/{v}/detector_accuracy_physionet.csv",
     "判别器在 test 各噪声条件上的准确率", "16"),

    ("B_判别器", "noise_detector_physionet/{v}/all_condition_behavior.csv",
     "判别器在 7 种条件上的准确率/in-vocab/置信度（含复合噪声）", "10"),
    ("B_判别器", "noise_detector_physionet/{v}/per_class_accuracy.csv",
     "判别器逐类准确率", "10"),
    ("B_判别器", "noise_detector_physionet/{v}/confusion_matrix.csv",
     "判别器混淆矩阵", "10"),
    ("B_判别器", "noise_detector_physionet/{v}/training_log.csv",
     "判别器训练日志（含实际 epoch 数、最佳验证准确率）", "10"),

    ("C_变体对比", "detector_comparison_physionet/detector_comparison_physionet.csv",
     "6class vs 7class 并排对比（核心结论）", "18"),
    ("C_变体对比", "detector_comparison_physionet/detector_variant_delta_physionet.csv",
     "加复合类带来的增量", "18"),
    ("C_变体对比", "detector_comparison_physionet/latex_detector_comparison_physionet.tex",
     "对比表 LaTeX", "18"),

    ("D_自适应", "adaptive_unknown_physionet/{v}/adaptive_unknown_results_physionet.csv",
     "raw / unknown_noisy / adaptive 三条件患者级指标", "11"),
    ("D_自适应", "adaptive_unknown_physionet/{v}/unknown_filter_selection_log_physionet.csv",
     "逐切片选了哪个 filter（看 filters_used 是否退化成单一 filter）", "11"),
    ("D_自适应", "adaptive_unknown_physionet/{v}/unknown_noise_detection_log_physionet.csv",
     "逐切片判别的噪声类别与置信度", "11"),
    ("D_自适应", "final_unknown_comparison_physionet/{v}/final_unknown_comparison_summary_physionet.csv",
     "最终三条件汇总", "12"),

    ("E_已知噪声基线", "noisy_evaluation_physionet/all_noisy_results_physionet.csv",
     "5 类已知噪声 + 复合噪声的退化基线", "05"),
    ("E_已知噪声基线", "enhanced_known_noise_physionet/val_grid_physionet.csv",
     "val 全网格（filter 映射就是在这上面拟合的）", "07"),
    ("E_已知噪声基线", "enhanced_known_noise_physionet/all_enhanced_known_noise_physionet.csv",
     "test 全网格 (噪声 x filter)", "07"),
    ("E_已知噪声基线", "enhanced_known_noise_physionet/best_filter_map_physionet.csv",
     "val 拟合出的 噪声->最优filter 映射（注意是 val 拟合、test 评估）", "07"),

    ("F_上下文", "quality_metrics_physionet/quality_summary_physionet.csv",
     "PSNR/SSIM 及相对加噪图的增益", "08"),
    ("F_上下文", "physionet_train_summary.csv",
     "各模型训练汇总（含实际 epoch 数、最佳患者级 F1）", "02"),

    ("G_TTA消融", "tta_unknown_physionet/tta_unknown_results_physionet.csv",
     "TTA 结果：condition x tta 的患者级指标（模型侧 bn/tent + 决策层 prior/priorq）", "19"),
    ("G_TTA消融", "tta_unknown_physionet/tta_adaptation_info_physionet.csv",
     "TTA 自适应过程信息（BN 更新批次数、熵前后对比、可调参数量）", "19"),
    ("G_TTA消融", "tta_unknown_physionet/norm_layers_physionet.csv",
     "归一化层构成（解释 bn 对 swin 为何是空操作）", "19"),

    ("H_边界校正", "boundary_correction_physionet/boundary_all_noise_physionet.csv",
     "★ 全部 7 个噪声条件下的决策边界校正（含阈值上限与相对 none 的配对检验）", "20"),
]

# 跨折汇总（results_cv_physionet/）—— 交叉验证下论文的主结果
CV_ITEMS = [
    ("CV_跨折汇总", "cv_summary.csv",
     "★ 5 折 mean±std —— 论文主表"),
    ("CV_跨折汇总", "cv_pooled.csv",
     "★ 5 折 test 合并（全体 82 例，每人恰好一次）—— 与 CQ500/RSNA 并排就用这张"),
    ("CV_跨折汇总", "cv_per_fold.csv", "逐折明细"),
    ("CV_跨折汇总", "cv_detector_behavior.csv", "判别器在复合噪声上的逐折行为"),
    ("CV_跨折汇总", "cv_detector_behavior_per_fold.csv", "同上，按折展开"),
    ("CV_跨折汇总", "latex_cv_summary.tex", "主表 LaTeX"),
    ("CV_跨折汇总", "cv_tta_summary.csv", "19 TTA 消融：逐折 mean±std"),
    ("CV_跨折汇总", "cv_tta_pooled.csv",
     "★ 19 TTA 消融：合并 82 例 + 合并 McNemar（b/c 相加后重检验）"),
    ("CV_跨折汇总", "cv_boundary_summary.csv", "20 边界校正：逐折 mean±std"),
    ("CV_跨折汇总", "cv_boundary_pooled.csv",
     "★ 20 边界校正：合并 82 例 + 合并 McNemar"),
]

TABLE_GLOB = "final_tables_physionet/*"
FIG_GLOBS = [
    "gradcam_recovery_physionet_*/*/*.png",
    "gradcam_known_noise_recovery_physionet/*/*/*.png",
    "detector_comparison_physionet/*.png",
    "cross_dataset_comparison/*.png",
    "enhanced_known_noise_physionet/*.png",
]
COHORT_FILES = ["physionet_folds.csv", "physionet_split_summary.csv",
                "physionet_dataset_provenance.csv", "physionet_vs_others_cohort.csv",
                "physionet_sampling_frame.csv"]


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def csv_shape(p):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            n = sum(1 for _ in fh)
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            cols = len(fh.readline().split(","))
        return f"{n - 1} 行 x {cols} 列"
    except Exception:
        return "?"


def discover_sources(root):
    """在 root 下找出所有结果目录：results_fold{k}/ 与（若像 PhysioNet 结果的）results/。

    返回 [(标签, 目录, 包内相对前缀), ...]，标签用于打印，前缀用于在包里保留层级。
    """
    out = []
    folds = []
    for p in sorted(Path(root).glob("results_fold*")):
        m = re.fullmatch(r"results_fold(\d+)", p.name)
        if m and p.is_dir():
            folds.append((int(m.group(1)), p))
    want = {int(x) for x in args.folds.split(",") if x.strip()} if args.folds else None
    for k, p in sorted(folds):
        if want is not None and k not in want:
            continue
        out.append((f"fold{k}", p, f"results_fold{k}"))

    hold = Path(root) / "results"
    if hold.is_dir() and any(hold.glob("*_physionet")):
        if want is None:
            out.append(("holdout", hold, "results"))
    return out


def run_verify(sources):
    """跑自检：单次划分 + 每个存在的折。换机器后先看这段就知道数据是否完整。"""
    if args.no_verify:
        return "（已用 --no-verify 跳过）\n"
    import os
    script = HERE / "physionet_verify.py"
    env0 = dict(os.environ, PYTHONIOENCODING="utf-8")
    env0.pop("PHYSIONET_FOLD", None)
    runs = [("holdout (无 PHYSIONET_FOLD)", env0)]
    for label, _, _ in sources:
        if label.startswith("fold"):
            runs.append((label, dict(env0, PHYSIONET_FOLD=label[4:])))
    chunks = []
    for label, env in runs:
        try:
            r = subprocess.run([sys.executable, str(script)], cwd=str(HERE),
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=900, env=env)
            chunks.append(f"===== {label} : exit={r.returncode} =====\n"
                          f"{r.stdout}\n--- stderr ---\n{r.stderr}\n")
        except Exception as e:
            chunks.append(f"===== {label} : 自检未能运行: {e} =====\n")
    return "\n".join(chunks)


def env_report():
    lines = [f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"平台: {platform.platform()}",
             f"Python: {sys.version.split()[0]} ({sys.executable})"]
    try:
        import torch
        lines.append(f"torch: {torch.__version__}  CUDA可用={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}  "
                         f"显存={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    except Exception as e:
        lines.append(f"torch: 不可用 ({e})")
    for mod in ("timm", "cv2", "numpy", "pandas", "sklearn", "skimage", "scipy"):
        try:
            m = __import__(mod)
            lines.append(f"{mod}: {getattr(m, '__version__', '?')}")
        except Exception:
            lines.append(f"{mod}: 缺失")
    return "\n".join(lines) + "\n"


def main():
    dest = Path(args.dest)
    if dest.suffix == ".zip":
        dest = dest.with_suffix("")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    root = Path(args.results_root) if args.results_root else HEAD
    sources = discover_sources(root)
    if not sources:
        print(f"[err] 在 {root} 下没找到任何 PhysioNet 结果目录"
              f"（results_fold*/ 或 results/）。先跑流水线再来打包。")
        raise SystemExit(2)
    print(f"[扫描] {root}")
    for label, d, _ in sources:
        print(f"       {label:8s} <- {d}")

    collected, missing = [], []

    def take(src, rel):
        if src.is_file():
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(src.read_bytes())
            collected.append((rel, out))
            return True
        return False

    for label, rdir, prefix in sources:
        for cat, tmpl, desc, step in ITEMS:
            names = [tmpl.format(v=v) for v in VARIANTS] if "{v}" in tmpl else [tmpl]
            for rel in names:
                if not take(rdir / rel, Path(prefix) / rel):
                    missing.append((f"{prefix}/{rel}", desc, step))
        for p in sorted(rdir.glob(TABLE_GLOB)):
            take(p, Path(prefix) / "final_tables_physionet" / p.name)

    # 跨折汇总：只在没有指定 --folds 时才收（部分折时它不完整，收了会误导）
    cv_dir = root / "results_cv_physionet"
    if not args.folds:
        for cat, rel, desc in CV_ITEMS:
            if not take(cv_dir / rel, Path("results_cv_physionet") / rel):
                missing.append((f"results_cv_physionet/{rel}", desc, "aggregate"))

    for f in COHORT_FILES:
        take(HERE / f, Path("cohort") / f)
    for k, _, _ in sources:
        if k.startswith("fold"):
            take(HERE / f"physionet_split_fold{k[4:]}.csv", Path("cohort") / f"physionet_split_fold{k[4:]}.csv")
    take(HERE / "physionet_split.csv", Path("cohort") / "physionet_split.csv")

    if args.with_indexes:
        for p in sorted(HERE.glob("physionet_slices_index*.csv")):
            take(p, Path("cohort") / p.name)

    if args.with_figures:
        for label, rdir, prefix in sources:
            for g in FIG_GLOBS:
                for p in sorted(rdir.glob(g)):
                    take(p, Path("figures") / prefix / p.relative_to(rdir))

    meta = dest / "_run_metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "environment.txt").write_text(env_report(), encoding="utf-8")
    (meta / "verify_output.txt").write_text(run_verify(sources), encoding="utf-8")

    rows = [{"file": rel.as_posix(), "size_kb": round(out.stat().st_size / 1024, 1),
             "sha256_16": sha16(out),
             "shape": csv_shape(out) if out.suffix == ".csv" else "-"}
            for rel, out in collected]
    with open(meta / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["file", "size_kb", "sha256_16", "shape"])
        w.writeheader()
        w.writerows(rows)

    with open(dest / "README_FOR_ANALYSIS.md", "w", encoding="utf-8") as fh:
        fh.write("# PhysioNet 结果包\n\n")
        fh.write(f"生成于 {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        fh.write(f"包含 {len(rows)} 个文件；缺失 {len(missing)} 个。\n\n")
        fh.write("## 纳入的折\n\n")
        for label, d, _ in sources:
            fh.write(f"- {label}\n")
        fh.write("\n## 阅读顺序\n\n")
        fh.write("1. `results_cv_physionet/cv_summary.csv` —— 5 折 mean±std，论文主表\n")
        fh.write("2. `results_cv_physionet/cv_pooled.csv` —— 合并 82 例，与另两套数据集并排用这张\n")
        fh.write("3. `results_fold{k}/optimality_physionet/{v}/oracle_summary_physionet.csv` —— Q1/Q2 核心\n")
        fh.write("4. `results_fold{k}/optimality_physionet/{v}/filter_sweep_unknown_mixed_physionet.csv` —— filter 排名原始证据\n")
        fh.write("5. `results_fold{k}/boundary_correction_physionet/boundary_all_noise_physionet.csv` —— 各噪声条件下的边界校正收益（20）\n")
        fh.write("6. `_run_metadata/verify_output.txt` —— 各折自检输出，先看它确认数据没坏\n\n")
        if missing:
            fh.write("## 缺失（对应步骤可能没跑）\n\n")
            for rel, desc, step in missing:
                fh.write(f"- `{rel}` — {desc}（由 {step} 产生）\n")
            fh.write("\n")
        fh.write("## 清单\n\n见 `_run_metadata/manifest.csv`（含每文件 sha256 前 16 位与行x列）。\n")

    print("=" * 68)
    print("PhysioNet 结果打包")
    print("=" * 68)
    print(f"  已收 {len(rows)} 个文件，共 {sum(r['size_kb'] for r in rows) / 1024:.2f} MB")
    if args.folds:
        print(f"  （只含折 {args.folds}，故未收跨折汇总）")
    if missing:
        print(f"\n  缺失 {len(missing)} 个（对应步骤可能未运行）：")
        for rel, desc, step in missing[:25]:
            print(f"    - [{step:>8}] {rel}")
        if len(missing) > 25:
            print(f"    ... 另有 {len(missing) - 25} 个，详见 README_FOR_ANALYSIS.md")
    print(f"\n  输出目录: {dest}")

    zpath = dest.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(dest))
    print(f"  压缩包  : {zpath}  ({zpath.stat().st_size / 1024:.0f} KB)")
    print("\n把这个 .zip 发出去即可。")


if __name__ == "__main__":
    main()
