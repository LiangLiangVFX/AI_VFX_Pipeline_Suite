# -*- coding: utf-8 -*-
"""
BiRefNet_HR-Matting 全自动高清万物/人像+道具发丝抠像执行器
- 原生支持 1024x1024 (高速) 与 2048x2048 (超清发丝) 推理
- 自动过滤 Thumbs.db 等非图片文件，跑完自动弹出输出目录
"""
import os
import re
import sys
import ctypes
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

py_root = os.path.dirname(os.path.abspath(sys.executable))
torch_lib_dir = os.path.join(py_root, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib_dir):
    os.environ["PATH"] = torch_lib_dir + os.pathsep + py_root + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib_dir)
            os.add_dll_directory(py_root)
        except Exception:
            pass
    for dll_name in ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "libiomp5md.dll", "c10.dll"]:
        dll_path = os.path.join(torch_lib_dir, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(os.path.normpath(dll_path))
            except Exception:
                pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import transforms
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="实拍拆帧图片序列文件夹")
    parser.add_argument("--output", required=True, help="输出文件夹路径")
    parser.add_argument("--prefix", default="birefnet", help="输出文件名前缀")
    parser.add_argument("--res", type=int, default=1024, help="推理分辨率 (1024=高速省显存, 2048=极致发丝)")
    args = parser.parse_args()

    input_dir = args.input.strip().strip('"')
    output_dir = args.output.strip().strip('"')
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print("     BiRefNet-HR-Matting 全自动高清万物/发丝抠像引擎启动")
    print("=================================================================")
    print(f"• 输入序列目录: {input_dir}")
    print(f"• 输出目标目录: {output_dir}")
    print(f"• 推理内部精度: {args.res}x{args.res}")

    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
    files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(valid_exts)
    ])
    if not files:
        raise RuntimeError(f"未在目录中找到有效图片序列: {input_dir}")

    m_first = re.search(r'(\d+)\.[^.]+$', files[0])
    start_frame = int(m_first.group(1)) if m_first else 1001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = os.path.join(CURRENT_DIR, "models", "BiRefNet_HR-matting")
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
        raise RuntimeError(f"未检测到 BiRefNet 权重文件，请先下载至: {model_dir}")

    print(f">> [1/2] 正在加载 BiRefNet_HR-matting 模型至 GPU ({device})...")
    torch.set_float32_matmul_precision("high")
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    if device == "cuda":
        birefnet.half()

    transform_image = transforms.Compose([
        transforms.Resize((args.res, args.res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for old_f in os.listdir(output_dir):
        if old_f.startswith(f"{args.prefix}_alpha_"):
            try:
                os.remove(os.path.join(output_dir, old_f))
            except Exception:
                pass

    print(f">> [2/2] 开始全序列高清抠像，共 {len(files)} 帧...")
    for idx, fname in enumerate(tqdm(files, desc="BiRefNet 实时进度", ascii=False)):
        fpath = os.path.join(input_dir, fname)
        pil_img = Image.open(fpath).convert("RGB")
        orig_size = pil_img.size  # (W, H)

        inp_t = transform_image(pil_img).unsqueeze(0).to(device)
        if device == "cuda":
            inp_t = inp_t.half()

        preds = birefnet(inp_t)[-1].sigmoid().cpu().float()
        pred_pil = transforms.ToPILImage()(preds[0].squeeze(0)).resize(orig_size, Image.BILINEAR)

        f_num = start_frame + idx
        alpha_out = os.path.join(output_dir, f"{args.prefix}_alpha_{f_num:04d}.png")
        pred_pil.save(alpha_out, "PNG")

    print("-----------------------------------------------------------------")
    print(f">> [OK] BiRefNet 发丝 Alpha 输出完成: {output_dir}")
    print("-----------------------------------------------------------------")

    if os.name == "nt" and os.path.exists(output_dir):
        try:
            subprocess.Popen(["explorer.exe", os.path.normpath(output_dir)])
        except Exception:
            pass


if __name__ == "__main__":
    main()