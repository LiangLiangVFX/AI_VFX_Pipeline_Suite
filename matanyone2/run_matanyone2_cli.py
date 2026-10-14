# -*- coding: utf-8 -*-
"""
MatAnyone 2 (CVPR 2026) 拆帧序列抠像执行器 (支持首帧 Roto 增删/排除/合并模块区域)
"""
import os
import re
import sys
import ctypes

# 1. 优先预加载底层 DLL，杜绝 WinError 1114
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, CURRENT_DIR)


def get_rvm_first_frame_mask(first_img_path, device):
    """调用本地 RVM 提取第 1 帧人物初始遮罩 (返回 uint8 numpy 数组)"""
    rvm_dir = os.path.join(SUITE_ROOT, "rvm")
    rvm_ckpt = os.path.join(rvm_dir, "models", "rvm_resnet50.pth")
    sys.path.insert(0, rvm_dir)
    from model import MattingNetwork

    print(">> [MatAnyone2] 正在使用 RVM 提取第 1 帧人形基础遮罩...")
    model = MattingNetwork('resnet50').to(device).eval()
    model.load_state_dict(torch.load(rvm_ckpt, map_location=device))

    img = Image.open(first_img_path).convert("RGB")
    src_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pha, *_ = model(src_tensor, None, None, None, None, downsample_ratio=0.5)
    mask_np = ((pha[0, 0].cpu().numpy() > 0.25) * 255).astype(np.uint8)
    return mask_np, img.size


def build_guided_first_mask(first_img_path, user_mask_path, mask_mode, save_mask_path, device):
    """
    根据用户在 Nuke 中选择的模式，合成第 1 帧最终引导遮罩：
    - auto      : 仅使用 RVM 自动识人
    - intersect : 只保留 Roto 圈内的人 (排除圈外杂物/路人)
    - subtract  : 从人身上挖掉 Roto 区域 (排除不需要的某块)
    - union     : 人 + Roto 模块区域 (把人和特定模块/道具合并一起抠)
    - replace   : 完全使用用户画的 Roto 遮罩
    """
    has_user_mask = bool(user_mask_path and os.path.exists(user_mask_path))

    if mask_mode == "replace" and has_user_mask:
        src_w, src_h = Image.open(first_img_path).size
        u_img = Image.open(user_mask_path).convert("L").resize((src_w, src_h), Image.NEAREST)
        u_np = ((np.array(u_img) > 127) * 255).astype(np.uint8)
        Image.fromarray(u_np, mode="L").save(save_mask_path, "PNG")
        print(">> [MatAnyone2] 首帧模式: [完全按 Nuke Roto 区域]")
        return save_mask_path

    # 其余模式均先获取 RVM 首帧人物遮罩
    rvm_np, (src_w, src_h) = get_rvm_first_frame_mask(first_img_path, device)

    if not has_user_mask or mask_mode == "auto":
        Image.fromarray(rvm_np, mode="L").save(save_mask_path, "PNG")
        print(">> [MatAnyone2] 首帧模式: [全自动只抠人]")
        return save_mask_path

    u_img = Image.open(user_mask_path).convert("L").resize((src_w, src_h), Image.NEAREST)
    u_np = ((np.array(u_img) > 127) * 255).astype(np.uint8)

    if mask_mode == "intersect":
        # 交集：只保留 Roto 圈内的人
        final_np = np.minimum(rvm_np, u_np)
        print(">> [MatAnyone2] 首帧模式: [只保留 Roto 圈内的人 (排除圈外)]")
    elif mask_mode == "subtract":
        # 差集：挖掉 Roto 划定的区域
        final_np = np.where(u_np > 127, 0, rvm_np).astype(np.uint8)
        print(">> [MatAnyone2] 首帧模式: [从人身上挖掉 Roto 区域 (局部排除)]")
    elif mask_mode == "union":
        # 并集：人 + Roto 模块区域合并
        final_np = np.maximum(rvm_np, u_np)
        print(">> [MatAnyone2] 首帧模式: [人 + Roto 模块区域 (合并跟踪)]")
    else:
        final_np = rvm_np

    Image.fromarray(final_np, mode="L").save(save_mask_path, "PNG")
    return save_mask_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="实拍拆帧图片序列文件夹")
    parser.add_argument("--output", required=True, help="输出文件夹路径")
    parser.add_argument("--mask", default="", help="Nuke 导出的第 1 帧 Roto 遮罩路径")
    parser.add_argument("--mask_mode", default="auto", choices=["auto", "intersect", "subtract", "union", "replace"])
    parser.add_argument("--prefix", default="matanyone2", help="输出文件名前缀")
    args = parser.parse_args()

    input_dir = args.input.strip().strip('"')
    output_dir = args.output.strip().strip('"')
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    if not files:
        raise RuntimeError(f"未在目录中找到拆帧图片: {input_dir}")

    m = re.search(r'(\d+)\.[^.]+$', files[0])
    start_frame = int(m.group(1)) if m else 1001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1. 生成合成后的第 1 帧引导遮罩
    first_frame_path = os.path.join(input_dir, files[0])
    guided_mask_file = os.path.join(output_dir, "_guided_first_frame_mask.png")
    user_mask = args.mask.strip().strip('"') if args.mask else ""
    mask_path = build_guided_first_mask(first_frame_path, user_mask, args.mask_mode, guided_mask_file, device)

    # 2. 清理旧文件，保证重跑直接覆盖
    for old_f in os.listdir(output_dir):
        if old_f.startswith(f"{args.prefix}_alpha_") or old_f.startswith(f"{args.prefix}_rgba_"):
            try:
                os.remove(os.path.join(output_dir, old_f))
            except Exception:
                pass

    # 3. 调用 MatAnyone 2 执行发丝精修与全序列记忆跟踪
    ckpt_path = os.path.join(CURRENT_DIR, "pretrained_models", "matanyone2.pth")
    raw_out_dir = os.path.join(output_dir, "_raw_matanyone2")
    os.makedirs(raw_out_dir, exist_ok=True)

    infer_script = os.path.join(CURRENT_DIR, "inference_matanyone2.py")
    if not os.path.exists(infer_script):
        raise FileNotFoundError(f"未找到 {infer_script}，请确认已克隆 MatAnyone2 仓库！")

    import subprocess
    cmd = [
        sys.executable, infer_script,
        "-i", input_dir,
        "-m", mask_path,
        "-o", raw_out_dir,
        "--save-image"
    ]
    if os.path.exists(ckpt_path):
        cmd.extend(["-c", ckpt_path])

    print(f">> [MatAnyone2] 开始全序列发丝跟踪，共 {len(files)} 帧...")
    subprocess.run(cmd, check=True, cwd=CURRENT_DIR)

    # 4. 封装为 Nuke 标准序列帧命名
    seq_name = os.path.basename(os.path.normpath(input_dir))
    pha_dir = os.path.join(raw_out_dir, seq_name, "pha")
    if not os.path.exists(pha_dir):
        for root_d, dirs, _ in os.walk(raw_out_dir):
            if "pha" in dirs:
                pha_dir = os.path.join(root_d, "pha")
                break

    pha_files = sorted(os.listdir(pha_dir)) if os.path.exists(pha_dir) else []
    for idx, fname in enumerate(tqdm(files, desc="封装 Nuke 标准序列")):
        f_num = start_frame + idx
        src_img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        if idx < len(pha_files):
            alpha_img = Image.open(os.path.join(pha_dir, pha_files[idx])).convert("L")
        else:
            alpha_img = Image.new("L", src_img.size, 0)

        if alpha_img.size != src_img.size:
            alpha_img = alpha_img.resize(src_img.size, Image.BILINEAR)

        alpha_out = os.path.join(output_dir, f"{args.prefix}_alpha_{f_num:04d}.png")
        alpha_img.save(alpha_out, "PNG")

        rgba_img = src_img.copy()
        rgba_img.putalpha(alpha_img)
        rgba_out = os.path.join(output_dir, f"{args.prefix}_rgba_{f_num:04d}.png")
        rgba_img.save(rgba_out, "PNG")

    print(f">> [MatAnyone2] ✅ 全部序列输出完成: {output_dir}")
    if os.name == "nt" and os.path.exists(output_dir):
        try:
            os.startfile(os.path.normpath(output_dir))
        except Exception:
            pass


if __name__ == "__main__":
    main()