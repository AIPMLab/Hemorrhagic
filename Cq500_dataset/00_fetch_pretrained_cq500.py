# -*- coding: utf-8 -*-
"""
CQ500 预训练权重获取（离线可用）

背景：本机环境下 huggingface_hub / requests 走 TLS 时会被重置
      （SSLEOFError: UNEXPECTED_EOF_WHILE_READING），
      且 hf-mirror.com 会把请求 308 重定向回 huggingface.co，
      使 huggingface_hub 抛 FileMetadataError（拒绝非镜像来源）。
      实测 curl.exe 可以正常完成整条重定向链并拿到完整文件，
      因此本脚本改用 curl 下载，再在本地校验权重能否严格加载进对应 timm 模型。

下载后 02/10 会优先使用本地文件（cq500_commons.create_model_with_pretrained），
不再依赖网络，也不依赖 HF 缓存。

用法:
    python 00_fetch_pretrained_cq500.py            # 缺失的才下
    python 00_fetch_pretrained_cq500.py --force    # 全部重下
    python 00_fetch_pretrained_cq500.py --verify-only
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timm
import torch

from cq500_commons import PRETRAINED_FILE, WEIGHTS_DIR, load_pretrained_into

# 多镜像依次尝试：先国内镜像，再官方源
ENDPOINTS = [
    "https://hf-mirror.com/{repo}/resolve/main/{fname}",
    "https://huggingface.co/{repo}/resolve/main/{fname}",
]

NEEDED = {
    "resnet50": ("timm/resnet50.a1_in1k", "pytorch_model.bin"),
    "swin_tiny_patch4_window7_224": ("timm/swin_tiny_patch4_window7_224.ms_in1k",
                                     "pytorch_model.bin"),
    "efficientnet_b0": ("timm/efficientnet_b0.ra_in1k", "pytorch_model.bin"),
}


def curl_download(url, out_path):
    """用 curl 下载（跟随重定向）。返回 (ok, 说明)。"""
    if not shutil.which("curl"):
        return False, "找不到 curl"
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    cmd = ["curl", "-s", "-L", "--max-time", "900",
           "-o", str(tmp), "-w", "%{http_code} %{size_download}", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=960)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "curl 超时"
    code = (r.stdout or "").strip()
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False, f"curl 退出码 {r.returncode} {code}"
    if not tmp.exists() or tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        return False, f"文件过小或缺失 (http/size = {code})"
    tmp.replace(out_path)
    return True, f"{out_path.stat().st_size / 1048576:.1f} MB ({code})"


def verify(arch, path):
    """按与 create_model_with_pretrained **完全相同**的加载路径校验
    （直接复用 load_pretrained_into，避免校验口径与真实加载不一致）：
    主干张量必须全部命中；state_dict 里多出的 buffer（如 swin 的 attn_mask /
    relative_position_index，新版 timm 改为动态计算）属正常版本差异；
    分类头因类别数不同而形状不符，会被形状过滤跳过。
    """
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        return False, f"torch.load 失败: {type(e).__name__}: {e}"
    if not isinstance(sd, dict) or not sd:
        return False, f"不是非空 state_dict（{type(sd).__name__}）"
    if isinstance(sd.get("model"), dict):
        sd = sd["model"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    try:
        model = timm.create_model(arch, pretrained=False)
        backbone_missing, skipped, unexpected = load_pretrained_into(model, sd)
    except Exception as e:  # noqa: BLE001
        return False, f"加载失败: {type(e).__name__}: {str(e)[:220]}"
    if backbone_missing:
        return False, (f"主干缺失 {len(backbone_missing)} 个张量，例如 {backbone_missing[:4]}")
    n = sum(p.numel() for p in model.parameters())
    note = ""
    if unexpected:
        note += f"，忽略 {len(unexpected)} 个随版本变化的 buffer"
    if skipped:
        note += f"，按形状跳过 {len(skipped)} 个分类头张量"
    return True, f"{len(sd)} tensors, {n/1e6:.1f}M params, 主干完整{note}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[dir] {WEIGHTS_DIR}")
    print(f"[env] HF_ENDPOINT={__import__('os').environ.get('HF_ENDPOINT')}")

    results = {}
    for arch, (repo, fname) in NEEDED.items():
        out = WEIGHTS_DIR / PRETRAINED_FILE[arch]
        print(f"\n=== {arch} ===")
        print(f"    期望文件: {out.name}")

        if out.exists() and not args.force:
            print(f"    已存在 ({out.stat().st_size/1048576:.1f} MB)，跳过下载")
        elif args.verify_only:
            print("    --verify-only：文件缺失，跳过")
            results[arch] = "missing"
            continue
        else:
            ok = False
            for tpl in ENDPOINTS:
                url = tpl.format(repo=repo, fname=fname)
                host = url.split("/")[2]
                print(f"    尝试 {host} ...", flush=True)
                ok, msg = curl_download(url, out)
                print(f"      -> {'OK' if ok else 'FAIL'}: {msg}")
                if ok:
                    break
            if not ok:
                results[arch] = "download-failed"
                continue

        ok, msg = verify(arch, out)
        print(f"    校验: {'OK  ' if ok else 'FAIL'} {msg}")
        results[arch] = "ok" if ok else f"verify-failed: {msg}"

    print("\n" + "=" * 62)
    for arch, st in results.items():
        print(f"  {arch:32s} {st}")
    bad = [a for a, s in results.items() if s != "ok"]
    if bad:
        print(f"\n[警告] 以下权重不可用: {bad}")
        print("       02/10 将回退到 timm 在线下载（本机可能失败）")
    else:
        print("\n[完成] 全部预训练权重就绪，02/10 将直接使用本地文件")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
