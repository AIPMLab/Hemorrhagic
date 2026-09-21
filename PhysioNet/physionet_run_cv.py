# -*- coding: utf-8 -*-
"""
PhysioNet 5 折患者级交叉验证驱动

为什么用交叉验证：
  本数据集只有 82 例。单次 70/15/15 划分的 test 仅 14 例 —— 患者级准确率的步长是
  7.1%，McNemar 配对检验几乎没有功效（实测同一模型换一折就能差 20 个百分点）。
  5 折交叉验证让**每个患者都被评估一次**，可报告 mean ± std，统计上稳妥得多，
  也正是原论文（Hssayeni et al.）采用的方案。

折的构造（见 physionet_make_split.py --n-folds 5）：
  折 k 作 test，折 (k+1)%5 作 val（用于拟合 类别->filter 映射、选最优模型），
  其余 3 折作 train。三者按患者严格互斥，每折 test 约 16-17 例。

每折的产物靠环境变量 PHYSIONET_FOLD=k 隔离：
  png_fold{k}/  physionet_slices_index_fold{k}.csv
  noisy_tests/physionet_fold{k}/  results_fold{k}/  models_fold{k}/
  noise_detector_dataset_physionet_fold{k}/
因此 5 折互不覆盖，跑完由 physionet_aggregate_cv.py 汇总。

用法:
    python physionet_run_cv.py                     # 跑全部 5 折
    python physionet_run_cv.py --folds 0 1         # 只跑指定折
    python physionet_run_cv.py --dry-run           # 只看计划
    python physionet_run_cv.py --skip-build        # 跳过建图（已建好）
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

BASE = Path(__file__).resolve().parent
N_FOLDS = 5


def run(cmd, env, dry, passthrough_dry=False):
    """执行一步。

    dry=True 时默认只打印不执行；但 passthrough_dry=True 时会真的调用子进程，
    只是把 --dry-run 透传给它 —— 这样能看到它按 PHYSIONET_FOLD 解析出的**真实路径**，
    用来确认折隔离确实生效（否则 dry-run 只是一行字，看不到环境变量的效果）。
    """
    cmd = list(cmd)
    if dry and passthrough_dry:
        cmd = cmd + ["--dry-run"]
    print(f"    $ {' '.join(cmd)}", flush=True)
    if dry and not passthrough_dry:
        return 0
    return subprocess.run(cmd, cwd=str(BASE), env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-build", action="store_true",
                    help="跳过建图与检查（已建好 png_fold{k} 时用）")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    # 折划分文件必须先生成
    missing = [k for k in args.folds
               if not (BASE / f"physionet_split_fold{k}.csv").exists()]
    if missing:
        raise SystemExit(
            f"缺少折划分文件 {['physionet_split_fold%d.csv' % k for k in missing]}\n"
            f"请先运行: python physionet_make_split.py --n-folds {N_FOLDS}")

    print(f"折数: {len(args.folds)}  折: {args.folds}")
    failed = []
    for k in args.folds:
        print(f"\n{'='*60}\n=== fold {k} ===\n{'='*60}")
        env = dict(os.environ, PHYSIONET_FOLD=str(k), PYTHONIOENCODING="utf-8")

        if not args.skip_build:
            # 1) 按该折的划分生成图片与索引
            if run([args.python, "physionet_images_to_png.py", "--clean"], env,
                   args.dry_run) != 0:
                failed.append((k, "images_to_png")); continue
            # 2) 自检（早失败，别等训练几小时）
            if run([args.python, "physionet_verify.py"], env, args.dry_run) != 0:
                failed.append((k, "verify")); continue

        # 3) 跑整条流水线（dry-run 时把 --dry-run 透传，以便看到该折解析出的路径）
        rc = run([args.python, "run_physionet_pipeline.py"], env, args.dry_run,
                 passthrough_dry=True)
        if rc != 0:
            failed.append((k, "pipeline")); continue

    print(f"\n{'='*60}")
    if failed:
        print(f"[失败] {len(failed)} 折未完成: {failed}")
    else:
        print(f"[完成] {len(args.folds)} 折全部跑完"
              f"{'（dry-run）' if args.dry_run else ''}")
    print("\n[下一步] 汇总各折结果：")
    print("    python physionet_aggregate_cv.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
