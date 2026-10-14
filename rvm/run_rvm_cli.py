# -*- coding: utf-8 -*-
"""
RVM 序列帧全自动抠像批处理执行器 (修复 PyTorch 2.9+ WinError 1114 顺序冲突)
"""
import os
import sys
import ctypes

# 1. 在加载任何其他库之前，优先设置 DLL 搜索路径并预加载 c10.dll
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
    # 显式预加载底层运行时与 c10.dll
    for dll_name in ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "libiomp5md.dll", "c10.dll"]:
        dll_path = os.path.join(torch_lib_dir, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(os.path.normpath(dll_path))
            except Exception:
                pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 【严格顺序】必须最先导入 torch 和 torchvision，绝不能先导入 PIL 或 cv2！
import torch
from torchvision import transforms

import argparse
from PIL import Image
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model import MattingNetwork
except ImportError:
    print(f">> [错误] 未在 {current_dir} 找到 model 模块，请确认 RVM 源码已放入！")
    sys.exit(1)

def run_matting(input_dir, output_dir, model_path, downsample_ratio=0.5):
    if not os.path.exists(output_dir):
        try: os.makedirs(output_dir)
        except: pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">> [RVM Engine] 载入权重: {model_path} (运行设备: {device})")
    
    model = MattingNetwork('resnet50').to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr', '.dpx')
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(valid_exts)]
    if not files:
        print(f">> [错误] 目录未扫描到有效图像: {input_dir}")
        sys.exit(1)

    print(f">> [RVM Engine] 开始推理，共 {len(files)} 帧...")
    rec = [None] * 4
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for filename in tqdm(files, desc="RVM Processing"):
            src_path = os.path.join(input_dir, filename)
            src_img = Image.open(src_path).convert('RGB')
            src_tensor = to_tensor(src_img).unsqueeze(0).to(device)

            fgr, pha, *rec = model(src_tensor, *rec, downsample_ratio=downsample_ratio)

            alpha_np = (pha[0, 0].cpu().numpy() * 255).astype('uint8')
            alpha_img = Image.fromarray(alpha_np, mode='L')

            rgba_img = src_img.copy()
            rgba_img.putalpha(alpha_img)

            base_name = os.path.splitext(filename)[0]
            out_file = os.path.join(output_dir, f"{base_name}.png")
            rgba_img.save(out_file, "PNG")

    print(">> [RVM Engine] 全序列处理完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--downsample", type=float, default=0.5)
    args = parser.parse_args()
    run_matting(args.input, args.output, args.model, args.downsample)