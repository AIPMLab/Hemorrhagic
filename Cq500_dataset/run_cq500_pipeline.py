# -*- coding: utf-8 -*-
"""
CQ500 流水线一键驱动脚本（依赖驱动的顺序执行器）

所有 *_cq500 路径与已废弃的 dataset1 完全隔离。

两个判别器变体（论文中互相对比）：
    6class —— 只在 6 类单一退化上训练（clean + 5 种单噪声），复合噪声对其分布外
    7class —— 额外把复合未知噪声作为第 7 类学习
二者结果分别落在 results/*_cq500/{variant}/，最后由 18 汇总对比。

依赖关系（→ 表示前置步骤）:
    00 →(预训练权重)→ 02 / 10
    02 →(分类器权重)→ 05 / 07 / 11 / 13 / 14 / 16
    04 →(加噪图)→ 05 / 06 / 07 / 11 / 16
    09 →(判别器数据集)→ 10
    10 --variant X →(判别器权重)→ 11 --variant X / 16 --variant X
    07 →(val 拟合的 类别->filter 表)→ 11
    11 --variant X →(逐切片决策日志)→ 16 --variant X
    16 (两个变体都完成) → 18 →(对比表/图)
    06 →(增强图，可选)→ 08 / 14
    12 / 15 / visualize → 汇总与出图

推荐顺序: 00 → 02 → 04 → 09 → 10(A/B) → 05 → 07 → 11(A/B) → 16(A/B) → 18 → 12 → 15 → visualize

用法:
    python run_cq500_pipeline.py --dry-run
    python run_cq500_pipeline.py --from 07
    python run_cq500_pipeline.py --with-optional
    python run_cq500_pipeline.py --force
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 本脚本不 import cq500_commons，故需自带这一保护：
# Windows 控制台默认 cp1252/GBK，中文输出会抛 UnicodeEncodeError。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

BASE = Path(__file__).resolve().parent
CODE = BASE.parent  # code 根（models/、noisy_tests/、results/ 等所在）
sys.path.insert(0, str(BASE))

# 路径一律取自 cq500_commons —— 由它统一决定（含 CQ500_SMOKE 等开关）。
# 不要在这里硬编码：一旦两边的路径规则不一致（例如各步骤写了 *_smoke 而本脚本
# 去无后缀位置找），每一步都会误报"前置产物缺失"。
from cq500_commons import (  # noqa: E402
    ADAPTIVE_ROOT, DATA_ROOT, DETECTOR_DATASET_DIR, ENHANCED_ROOT, INDEX_CSV,
    MODEL_DIR, NOISY_ROOT, RESULT_ROOT, WEIGHTS_DIR,
)

DETECT_DS = DETECTOR_DATASET_DIR
CLS_RESNET = MODEL_DIR / "resnet50_cq500_best.pth"
CLS_SWIN = MODEL_DIR / "swin_tiny_patch4_window7_224_cq500_best.pth"
VARIANTS = ["6class", "7class"]


def det_weight(v):
    return MODEL_DIR / f"noise_detector_efficientnetb0_cq500_{v}_best.pth"


def adaptive_out(v):
    return RESULT_ROOT / "adaptive_unknown_cq500" / v / "adaptive_unknown_results_cq500.csv"


def adaptive_sel(v):
    """11 的逐切片 filter 决策日志 —— 19（TTA 消融）靠它重建 adaptive 输入。"""
    return RESULT_ROOT / "adaptive_unknown_cq500" / v / \
        "unknown_filter_selection_log_cq500.csv"


def optimality_out(v):
    return RESULT_ROOT / "optimality_cq500" / v / "oracle_summary_cq500.csv"


# (步名, 脚本, 附加参数, 前置产物, 后置产物[存在则跳过], 是否可选)
STEPS = [
    ("00 预训练权重", "00_fetch_pretrained_cq500.py", [],
     [INDEX_CSV], [WEIGHTS_DIR], False),

    ("02 分类器训练", "02_train_cq500.py", [],
     [INDEX_CSV, DATA_ROOT, WEIGHTS_DIR], [CLS_RESNET, CLS_SWIN], False),

    ("04 噪声生成", "04_generate_noisy_test_cq500.py", [],
     [INDEX_CSV, DATA_ROOT], [NOISY_ROOT / "unknown_mixed"], False),

    ("09 判别器数据集", "09_build_noise_detector_dataset_cq500.py", [],
     [INDEX_CSV, DATA_ROOT], [DETECT_DS / "manifest.csv"], False),

    ("10 判别器训练 6class", "10_train_noise_detector_cq500.py", ["--variant", "6class"],
     [DETECT_DS / "manifest.csv", WEIGHTS_DIR], [det_weight("6class")], False),

    ("10 判别器训练 7class", "10_train_noise_detector_cq500.py", ["--variant", "7class"],
     [DETECT_DS / "manifest.csv", WEIGHTS_DIR], [det_weight("7class")], False),

    ("05 加噪评测", "05_evaluate_noisy_test_cq500.py", [],
     [CLS_RESNET, NOISY_ROOT / "gaussian"],
     [RESULT_ROOT / "noisy_evaluation_cq500" / "all_noisy_results_cq500.csv"], False),

    # 边界校正铺到全部噪声条件：只依赖 02 的权重 + 04 的加噪图，与判别器无关
    ("20 边界校正(全噪声)", "20_boundary_correction_cq500.py", [],
     [CLS_RESNET, CLS_SWIN, NOISY_ROOT / "unknown_mixed"],
     [RESULT_ROOT / "boundary_correction_cq500" / "boundary_all_noise_cq500.csv"],
     False),

    ("07 已知噪声最优 filter", "07_evaluate_enhanced_known_noise_cq500.py", [],
     [CLS_RESNET, NOISY_ROOT / "gaussian"],
     [RESULT_ROOT / "enhanced_known_noise_cq500" / "best_filter_map_cq500.csv"], False),

    ("11 自适应 6class", "11_adaptive_unknown_pipeline_cq500.py", ["--variant", "6class"],
     [CLS_RESNET, det_weight("6class"), NOISY_ROOT / "unknown_mixed",
      RESULT_ROOT / "enhanced_known_noise_cq500" / "best_filter_map_cq500.csv"],
     [adaptive_out("6class")], False),

    ("11 自适应 7class", "11_adaptive_unknown_pipeline_cq500.py", ["--variant", "7class"],
     [CLS_RESNET, det_weight("7class"), NOISY_ROOT / "unknown_mixed",
      RESULT_ROOT / "enhanced_known_noise_cq500" / "best_filter_map_cq500.csv"],
     [adaptive_out("7class")], False),

    # TTA 消融：只依赖两个 11 的决策日志，和 16 相互独立
    ("19 TTA 消融", "19_tta_unknown_cq500.py", [],
     [adaptive_sel("6class"), adaptive_sel("7class")],
     [RESULT_ROOT / "tta_unknown_cq500" / "tta_unknown_results_cq500.csv"], False),

    ("16 最优性 6class", "16_optimality_analysis_cq500.py", ["--variant", "6class"],
     [CLS_RESNET, det_weight("6class"),
      RESULT_ROOT / "adaptive_unknown_cq500" / "6class"
      / "unknown_filter_selection_log_cq500.csv"],
     [optimality_out("6class")], False),

    ("16 最优性 7class", "16_optimality_analysis_cq500.py", ["--variant", "7class"],
     [CLS_RESNET, det_weight("7class"),
      RESULT_ROOT / "adaptive_unknown_cq500" / "7class"
      / "unknown_filter_selection_log_cq500.csv"],
     [optimality_out("7class")], False),

    ("18 判别器变体对比", "18_compare_detectors_cq500.py", [],
     [optimality_out("6class"), optimality_out("7class")],
     [RESULT_ROOT / "detector_comparison_cq500" / "detector_comparison_cq500.csv"], False),

    ("12 最终对比 6class", "12_final_unknown_comparison_cq500.py", ["--variant", "6class"],
     [adaptive_out("6class")],
     [RESULT_ROOT / "final_unknown_comparison_cq500" / "6class"
      / "final_unknown_comparison_summary_cq500.csv"], False),

    ("12 最终对比 7class", "12_final_unknown_comparison_cq500.py", ["--variant", "7class"],
     [adaptive_out("7class")],
     [RESULT_ROOT / "final_unknown_comparison_cq500" / "7class"
      / "final_unknown_comparison_summary_cq500.csv"], False),

    ("15 论文表格", "15_generate_final_tables_cq500.py", [],
     [RESULT_ROOT / "detector_comparison_cq500" / "detector_comparison_cq500.csv"],
     [RESULT_ROOT / "final_tables_cq500" / "table9_detector_comparison.csv"], False),

    ("可视化", "visualize_cq500.py", [],
     [RESULT_ROOT / "enhanced_known_noise_cq500" / "all_enhanced_known_noise_cq500.csv"],
     [RESULT_ROOT / "enhanced_known_noise_cq500" / "fig_bar_cq500.png"], False),

    # ---- 可选：落盘量大，仅供出图与质量指标 ----
    ("06 增强图生成(可选)", "06_generate_enhanced_images_cq500.py", [],
     [NOISY_ROOT / "unknown_mixed"], [ENHANCED_ROOT / "unknown_mixed" / "median"], True),

    ("08 质量指标(可选)", "08_quality_metrics_psnr_ssim_cq500.py", [],
     [NOISY_ROOT / "gaussian", ENHANCED_ROOT / "gaussian" / "median"],
     [RESULT_ROOT / "quality_metrics_cq500" / "quality_summary_cq500.csv"], True),

    ("14 Grad-CAM 已知噪声(可选)", "14_gradcam_known_noise_recovery_cq500.py", [],
     [CLS_RESNET, RESULT_ROOT / "enhanced_known_noise_cq500" / "best_filter_map_cq500.csv",
      ENHANCED_ROOT / "gaussian" / "median"],
     [RESULT_ROOT / "gradcam_known_noise_recovery_cq500"], True),

    ("13 Grad-CAM 未知噪声(可选)", "13_gradcam_recovery_cq500.py",
     ["--variant", "7class", "--model", "resnet50"],
     [CLS_RESNET, adaptive_out("7class"),
      ADAPTIVE_ROOT / "7class" / "resnet50"],
     [RESULT_ROOT / "gradcam_recovery_cq500_7class_resnet50"], True),
]


def missing(paths):
    return [str(p) for p in paths if not p.exists()]


ENV_NAME = "TTA-py3.11-cu118"


def find_env_python(env_name=ENV_NAME):
    """在常见 conda 安装位置中查找指定环境的 python.exe；找不到返回 None。"""
    cands = []
    prefix = Path(sys.prefix)
    if (prefix / "conda-meta").exists():
        root = prefix.parent.parent if prefix.parent.name == "envs" else prefix
        cands.append(root / "envs" / env_name / "python.exe")
    home = Path.home()
    for r in (home / "anaconda3", home / "miniconda3", home / "Anaconda3",
              home / "Miniconda3", Path("C:/ProgramData/anaconda3"),
              Path("C:/ProgramData/miniconda3")):
        cands.append(r / "envs" / env_name / "python.exe")
    conda = shutil.which("conda")
    if conda:
        try:
            out = subprocess.run([conda, "env", "list", "--json"],
                                 capture_output=True, text=True, timeout=15).stdout
            for e in json.loads(out).get("envs", []):
                if os.path.basename(e) == env_name:
                    cands.insert(0, Path(e) / "python.exe")
        except Exception:
            pass
    for c in cands:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return None


def resolve_python(requested=None):
    return (requested or os.environ.get("TTA_PYTHON")
            or find_env_python() or sys.executable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start",
                    help="从哪一步开始：步骤号(如 09 / 11 / 16) 或步名子串(如 判别器)")
    ap.add_argument("--dry-run", action="store_true", help="仅打印执行计划")
    ap.add_argument("--force", action="store_true", help="忽略产物存在性，全部重跑")
    ap.add_argument("--with-optional", action="store_true", help="同时执行可选步骤")
    ap.add_argument("--python", dest="python", help="指定解释器（默认自动查找 %s）" % ENV_NAME)
    args = ap.parse_args()

    steps = [s for s in STEPS if args.with_optional or not s[5]]
    names = [s[0] for s in steps]
    start_idx = 0
    if args.start:
        key = args.start.strip()
        # 先按"步骤号前缀"匹配（步名形如 "09 判别器数据集"）——这是文档承诺的语义。
        # 旧实现把纯数字一律当成**序号**（1..N），于是 `--from 09` 落到第 9 步
        # = "11 自适应"，静默跳过 09/10，下游全部因缺依赖而跳过。这个坑必须堵掉。
        hit = next((i for i, n in enumerate(names) if n.startswith(key + " ")), None)
        if hit is None:
            hit = next((i for i, n in enumerate(names) if key in n), None)
        if hit is None and key.isdigit() and 1 <= int(key) <= len(names):
            hit = int(key) - 1                      # 退路：按序号解释
        if hit is None:
            raise SystemExit("--from 无法解析: %r\n可用步骤:\n  %s"
                             % (args.start, "\n  ".join(names)))
        start_idx = hit
    plan = steps[start_idx:]

    py = resolve_python(args.python)
    print(f"环境: {py}")
    if args.dry_run:
        print("== 执行计划（DRY RUN） ==")
    for name, script, extra, reqs, outs, optional in plan:
        status = []
        if missing(reqs):
            status.append(f"依赖缺失:{len(missing(reqs))} 项")
        if not args.force and outs and not missing(outs):
            status.append("产物已存在(将跳过)")
        if optional:
            status.append("可选")
        print(f"  {script:46s} {name:26s} {' | '.join(status)}")

    if args.dry_run:
        return

    failed = False
    for name, script, extra, reqs, outs, optional in plan:
        no_deps = missing(reqs)
        if no_deps:
            msg = f"[跳过] {name} 前置产物缺失: {no_deps}"
            print(msg if optional else f"{msg}  <-- 非可选步骤，请检查")
            continue
        if not args.force and outs and not missing(outs):
            print(f"[跳过] {name} 产出已存在")
            continue
        cmd = [py, str(BASE / script)] + list(extra)
        print(f"[运行] {name} -> {' '.join([script] + list(extra))}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run(cmd, cwd=str(BASE), env=env)
        if r.returncode != 0:
            print(f"[失败] {name} 退出码 {r.returncode}")
            failed = True
            break
    print("[完成]" if not failed else "[中断：存在失败步骤]")


if __name__ == "__main__":
    main()
