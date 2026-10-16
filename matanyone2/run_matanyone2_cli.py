# -*- coding: utf-8 -*-
"""
MatAnyone 2 (CVPR 2026) 拆帧序列抠像执行器 (BiRefNet + RVM 双引导保底版)
- 100% 保留原有 RVM 首帧引导逻辑，支持随时切换或异常自动回退保底
- 新增 BiRefNet_HR-matting 首帧引导，连人带手中道具/武器一次性高清提取
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
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import argparse
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, CURRENT_DIR)


def get_rvm_first_frame_mask(first_img_path, device):
    """【原版保底】调用本地 RVM 提取引导帧人物初始遮罩"""
    rvm_dir = os.path.join(SUITE_ROOT, "rvm")
    rvm_ckpt = os.path.join(rvm_dir, "models", "rvm_resnet50.pth")
    if rvm_dir not in sys.path:
        sys.path.insert(0, rvm_dir)
    from model import MattingNetwork

    print(">> [1/3] 正在使用 [RVM 原版引擎] 提取引导帧人形基础遮罩...")
    model = MattingNetwork('resnet50').to(device).eval()
    model.load_state_dict(torch.load(rvm_ckpt, map_location=device))

    img = Image.open(first_img_path).convert("RGB")
    src_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pha, *_ = model(src_tensor, None, None, None, None, downsample_ratio=0.5)
    mask_np = ((pha[0, 0].cpu().numpy() > 0.25) * 255).astype(np.uint8)
    return mask_np, img.size


def get_birefnet_first_frame_mask(first_img_path, device):
    """【升级引擎】调用本地 BiRefNet_HR-matting 提取引导帧高清人物+随身道具遮罩（失败自动回退 RVM）"""
    biref_dir = os.path.join(SUITE_ROOT, "birefnet", "models", "BiRefNet_HR-matting")
    safetensor_path = os.path.join(biref_dir, "model.safetensors")
    if not os.path.exists(safetensor_path):
        print(">> [提示] 未检测到 BiRefNet 权重，自动无缝回退至原版 RVM 引擎...")
        return get_rvm_first_frame_mask(first_img_path, device)

    try:
        from transformers import AutoModelForImageSegmentation
        print(">> [1/3] 正在使用 [BiRefNet 高清引擎] 提取引导帧前景 (人+随身道具)...")
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            biref_dir, trust_remote_code=True, local_files_only=True
        ).to(device).eval()
        if device == "cuda":
            birefnet.half()

        img = Image.open(first_img_path).convert("RGB")
        orig_size = img.size
        tfm = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        inp_t = tfm(img).unsqueeze(0).to(device)
        if device == "cuda":
            inp_t = inp_t.half()

        with torch.no_grad():
            preds = birefnet(inp_t)[-1].sigmoid().cpu().float()
        pred_pil = transforms.ToPILImage()(preds[0].squeeze(0)).resize(orig_size, Image.BILINEAR)
        mask_np = ((np.array(pred_pil) > 64) * 255).astype(np.uint8)
        del birefnet
        if device == "cuda":
            torch.cuda.empty_cache()
        return mask_np, orig_size
    except Exception as e:
        print(f">> [提示] BiRefNet 引导发生异常 ({e})，自动无缝回退至原版 RVM 引擎...")
        return get_rvm_first_frame_mask(first_img_path, device)


def build_guided_first_mask(first_img_path, user_mask_path, mask_mode, guide_engine, save_mask_path, device):
    """根据所选模式与所选引导引擎合成引导帧遮罩"""
    has_user_mask = bool(user_mask_path and os.path.exists(user_mask_path))

    if mask_mode == "replace" and has_user_mask:
        src_w, src_h = Image.open(first_img_path).size
        u_img = Image.open(user_mask_path).convert("L").resize((src_w, src_h), Image.NEAREST)
        u_np = ((np.array(u_img) > 127) * 255).astype(np.uint8)
        Image.fromarray(u_np, mode="L").save(save_mask_path, "PNG")
        print(">> [引导模式] 完全按 Nuke Roto 区域")
        return u_np

    if guide_engine == "birefnet":
        base_np, (src_w, src_h) = get_birefnet_first_frame_mask(first_img_path, device)
    else:
        base_np, (src_w, src_h) = get_rvm_first_frame_mask(first_img_path, device)

    if not has_user_mask or mask_mode == "auto":
        Image.fromarray(base_np, mode="L").save(save_mask_path, "PNG")
        print(f">> [引导模式] 全自动提取主体 (引擎: {guide_engine})")
        return base_np

    u_img = Image.open(user_mask_path).convert("L").resize((src_w, src_h), Image.NEAREST)
    u_np = ((np.array(u_img) > 127) * 255).astype(np.uint8)

    if mask_mode == "intersect":
        final_np = np.minimum(base_np, u_np)
        print(">> [引导模式] 只保留 Roto 圈内的主体 (排除圈外)")
    elif mask_mode == "subtract":
        final_np = np.where(u_np > 127, 0, base_np).astype(np.uint8)
        print(">> [引导模式] 从主体上挖掉 Roto 区域 (局部排除)")
    elif mask_mode == "union":
        final_np = np.maximum(base_np, u_np)
        print(">> [引导模式] 主体 + Roto 模块区域 (合并跟踪)")
    else:
        final_np = base_np

    Image.fromarray(final_np, mode="L").save(save_mask_path, "PNG")
    return final_np


def gen_dilate(alpha, min_kernel_size, max_kernel_size):
    kernel_size = np.random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilated = cv2.dilate(fg_and_unknown, kernel, iterations=1) * 255
    return dilated.astype(np.float32)


def gen_erosion(alpha, min_kernel_size, max_kernel_size):
    kernel_size = np.random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    eroded = cv2.erode(fg, kernel, iterations=1) * 255
    return eroded.astype(np.float32)


@torch.inference_mode()
@torch.amp.autocast("cuda")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="实拍拆帧图片序列文件夹")
    parser.add_argument("--output", required=True, help="输出文件夹路径")
    parser.add_argument("--mask", default="", help="Nuke 导出的引导帧 Roto 遮罩路径")
    parser.add_argument("--mask_mode", default="auto", choices=["auto", "intersect", "subtract", "union", "replace"])
    parser.add_argument("--guide_engine", default="birefnet", choices=["birefnet", "rvm"], help="首帧引导引擎 (birefnet 或 rvm)")
    parser.add_argument("--prefix", default="matanyone2", help="输出文件名前缀")
    parser.add_argument("--warmup", type=int, default=10, help="首帧发丝精修迭代次数")
    parser.add_argument("--erode_kernel", type=int, default=10)
    parser.add_argument("--dilate_kernel", type=int, default=10)
    parser.add_argument("--start_at", type=int, default=0, help="从指定帧号开始往后重算补救 (0代表从头跑全片)")
    parser.add_argument("--save_rgba", action="store_true", help="是否额外保存 RGBA 四通道图")
    args = parser.parse_args()

    input_dir = args.input.strip().strip('"')
    output_dir = args.output.strip().strip('"')
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print("        MatAnyone 2 (CVPR 2026) 发丝级AI抠像引擎启动")
    print("=================================================================")
    print(f"• 输入序列目录: {input_dir}")
    print(f"• 输出目标目录: {output_dir}")

    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
    all_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(valid_exts)
    ])
    if not all_files:
        raise RuntimeError(f"未在目录中找到有效拆帧图片: {input_dir}")

    m_first = re.search(r'(\d+)\.[^.]+$', all_files[0])
    seq_first_frame = int(m_first.group(1)) if m_first else 1001

    offset_idx = 0
    if args.start_at > seq_first_frame:
        offset_idx = min(args.start_at - seq_first_frame, len(all_files) - 1)
    files = all_files[offset_idx:]
    start_frame = seq_first_frame + offset_idx
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if offset_idx > 0:
        print(f"• [局部补救模式] 保留 {seq_first_frame}~{start_frame - 1} 帧，仅从第 {start_frame} 帧开始往后重算！")

    # 1. 生成引导遮罩 (支持 BiRefNet 或 RVM，且失败自动回退保底)
    first_frame_path = os.path.join(input_dir, files[0])
    guided_mask_file = os.path.join(output_dir, f"_guided_frame_{start_frame}_mask.png")
    user_mask = args.mask.strip().strip('"') if args.mask else ""
    mask_np = build_guided_first_mask(
        first_frame_path, user_mask, args.mask_mode, args.guide_engine, guided_mask_file, device
    )

    # 2. 如果是从头跑全片，清理旧文件；如果是局部补救，直接覆盖对应帧号的文件
    if offset_idx == 0:
        for old_f in os.listdir(output_dir):
            if old_f.startswith(f"{args.prefix}_alpha_") or old_f.startswith(f"{args.prefix}_rgba_"):
                try:
                    os.remove(os.path.join(output_dir, old_f))
                except Exception:
                    pass

    # 3. 加载 MatAnyone 2 核心引擎
    ckpt_path = os.path.join(CURRENT_DIR, "pretrained_models", "matanyone2.pth")
    from matanyone2.utils.get_default_model import get_matanyone2_model
    from matanyone2.inference.inference_core import InferenceCore

    print(f">> [2/3] 正在加载 MatAnyone 2 模型权重至 GPU ({device})...")
    matanyone2_model = get_matanyone2_model(ckpt_path, device=device)
    processor = InferenceCore(matanyone2_model, cfg=matanyone2_model.cfg, device=device)

    mask_f = mask_np.astype(np.float32)
    if args.dilate_kernel > 0:
        mask_f = gen_dilate(mask_f, args.dilate_kernel, args.dilate_kernel + 1)
    if args.erode_kernel > 0:
        mask_f = gen_erosion(mask_f, args.erode_kernel, args.erode_kernel + 1)
    mask_tensor = torch.from_numpy(mask_f).to(device)
    objects = [1]

    inference_indices = [0] * args.warmup + list(range(len(files)))
    print(f">> [3/3] 开始发丝跟踪: 帧范围 {start_frame} ~ {start_frame + len(files) - 1} (共 {len(files)} 帧)...")

    for step_idx, frame_idx in enumerate(tqdm(inference_indices, desc="MatAnyone 2 实时进度", ascii=False)):
        fpath = os.path.join(input_dir, files[frame_idx])
        pil_img = Image.open(fpath).convert("RGB")
        np_rgb = np.array(pil_img)
        image_t = torch.from_numpy(np_rgb).permute(2, 0, 1).float().to(device) / 255.0

        if step_idx == 0:
            output_prob = processor.step(image_t, mask_tensor, objects=objects)
            output_prob = processor.step(image_t, first_frame_pred=True)
        elif step_idx < args.warmup:
            output_prob = processor.step(image_t, first_frame_pred=True)
        else:
            output_prob = processor.step(image_t)

        if step_idx >= args.warmup:
            pha = processor.output_prob_to_mask(output_prob)
            pha_np = (pha.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            f_num = start_frame + frame_idx

            alpha_img = Image.fromarray(pha_np, mode="L")
            alpha_out = os.path.join(output_dir, f"{args.prefix}_alpha_{f_num:04d}.png")
            alpha_img.save(alpha_out, "PNG")

            if args.save_rgba:
                rgba_img = pil_img.copy()
                rgba_img.putalpha(alpha_img)
                rgba_out = os.path.join(output_dir, f"{args.prefix}_rgba_{f_num:04d}.png")
                rgba_img.save(rgba_out, "PNG")

    print("-----------------------------------------------------------------")
    print(f">> [OK] 发丝 Alpha 输出完成: {output_dir}")
    print("-----------------------------------------------------------------")

    if os.name == "nt" and os.path.exists(output_dir):
        try:
            subprocess.Popen(["explorer.exe", os.path.normpath(output_dir)])
        except Exception:
            try:
                os.startfile(os.path.normpath(output_dir))
            except Exception:
                pass


if __name__ == "__main__":
    main()