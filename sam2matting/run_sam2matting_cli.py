# -*- coding: utf-8 -*-
"""
SAM2Matting (ECCV 2026) 广义万物视频发丝级跟踪抠像执行器
- 激活 SAM2Matting 原生 Box + Mask 双重提示能力，画个粗略 Roto 圈即可由模型自身精准吸附物体边缘
- 全自动模式下默认采用 BiRefNet 高清引导（若无则自动静默回退 RVM）
"""
import os
import re
import sys
import shutil
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, CURRENT_DIR)


def get_rvm_first_frame_mask(first_img_path, device):
    """调用本地 RVM 提取引导帧基础遮罩 (自动保底方案)"""
    rvm_dir = os.path.join(SUITE_ROOT, "rvm")
    rvm_ckpt = os.path.join(rvm_dir, "models", "rvm_resnet50.pth")
    if rvm_dir not in sys.path:
        sys.path.insert(0, rvm_dir)
    from model import MattingNetwork

    print(">> [1/3] 正在使用 [RVM 引擎] 提取首帧基础区域...")
    model = MattingNetwork('resnet50').to(device).eval()
    model.load_state_dict(torch.load(rvm_ckpt, map_location=device))

    img = Image.open(first_img_path).convert("RGB")
    src_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pha, *_ = model(src_tensor, None, None, None, None, downsample_ratio=0.5)
    mask_np = ((pha[0, 0].cpu().numpy() > 0.25) * 255).astype(np.uint8)
    return mask_np, img.size


def get_best_auto_first_frame_mask(first_img_path, device):
    """自动优先调用 BiRefNet 提取首帧高清主体，若未安装权重则无缝回退 RVM"""
    biref_dir = os.path.join(SUITE_ROOT, "birefnet", "models", "BiRefNet_HR-matting")
    if not os.path.exists(os.path.join(biref_dir, "model.safetensors")):
        return get_rvm_first_frame_mask(first_img_path, device)

    try:
        from transformers import AutoModelForImageSegmentation
        print(">> [1/3] 正在使用 [BiRefNet 2K高清引擎] 自动锁定首帧主体与道具...")
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
        print(f">> [提示] BiRefNet 引导回退至 RVM ({e})...")
        return get_rvm_first_frame_mask(first_img_path, device)


def build_guided_first_mask(first_img_path, user_mask_path, mask_mode, save_mask_path, device):
    """合成首帧引导遮罩"""
    has_user_mask = bool(user_mask_path and os.path.exists(user_mask_path))

    if mask_mode == "replace" and has_user_mask:
        src_w, src_h = Image.open(first_img_path).size
        u_img = Image.open(user_mask_path).convert("L").resize((src_w, src_h), Image.NEAREST)
        u_np = ((np.array(u_img) > 127) * 255).astype(np.uint8)
        Image.fromarray(u_np, mode="L").save(save_mask_path, "PNG")
        print(">> [引导模式] 使用 SAM2Matting 原生目标锁定 (基于 Nuke Roto 范围)")
        return u_np

    base_np, (src_w, src_h) = get_best_auto_first_frame_mask(first_img_path, device)

    if not has_user_mask or mask_mode == "auto":
        Image.fromarray(base_np, mode="L").save(save_mask_path, "PNG")
        print(">> [引导模式] 全自动提取主体")
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


@torch.inference_mode()
@torch.amp.autocast("cuda")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="实拍拆帧图片序列文件夹")
    parser.add_argument("--output", required=True, help="输出文件夹路径")
    parser.add_argument("--mask", default="", help="Nuke 导出的引导帧 Roto 遮罩路径")
    parser.add_argument("--mask_mode", default="auto", choices=["auto", "intersect", "subtract", "union", "replace"])
    parser.add_argument("--guide_engine", default="birefnet", help="保留兼容参数")
    parser.add_argument("--variant", default="base_plus", choices=["base_plus", "tiny"], help="SAM2Matting 权重规格")
    parser.add_argument("--prefix", default="sam2mat", help="输出文件名前缀")
    parser.add_argument("--start_at", type=int, default=0, help="从指定帧号开始往后重算补救")
    args = parser.parse_args()

    input_dir = args.input.strip().strip('"')
    output_dir = args.output.strip().strip('"')
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print("      SAM2Matting (ECCV 2026) 万物发丝级视频跟踪抠像引擎启动")
    print("=================================================================")
    print(f"• 输入序列目录: {input_dir}")
    print(f"• 输出目标目录: {output_dir}")
    print(f"• 模型架构规格: SAM2.1-{args.variant}")

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

    # 1. 生成引导帧遮罩
    first_frame_path = os.path.join(input_dir, files[0])
    guided_mask_file = os.path.join(output_dir, f"_guided_frame_{start_frame}_mask.png")
    user_mask = args.mask.strip().strip('"') if args.mask else ""
    mask_np = build_guided_first_mask(first_frame_path, user_mask, args.mask_mode, guided_mask_file, device)

    if offset_idx == 0:
        for old_f in os.listdir(output_dir):
            if old_f.startswith(f"{args.prefix}_alpha_"):
                try:
                    os.remove(os.path.join(output_dir, old_f))
                except Exception:
                    pass

    # 2. 构建 SAM2Matting 视频预测器
    if args.variant == "tiny":
        ckpt_path = os.path.join(CURRENT_DIR, "checkpoints", "SAM2Matting-SAM2.1Tiny.pt")
        cfg_name = "configs/sam2matting-sam2.1tiny.yaml"
    else:
        ckpt_path = os.path.join(CURRENT_DIR, "checkpoints", "SAM2Matting-SAM2.1Base+.pt")
        cfg_name = "configs/sam2matting-sam2.1base+.yaml"

    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"未找到模型权重: {ckpt_path}\n请先运行 HuggingFace 下载命令获取权重！")

    print(f">> [2/3] 正在加载 SAM2Matting ({os.path.basename(ckpt_path)}) 至 GPU...")
    from sam2.build_sam import build_sam2matting_video_predictor
    predictor = build_sam2matting_video_predictor(cfg_name, ckpt_path, device=device)

    # 3. 准备标准数字命名的临时帧目录
    temp_frame_dir = os.path.join(output_dir, "_temp_sam2_frames")
    if os.path.exists(temp_frame_dir):
        shutil.rmtree(temp_frame_dir, ignore_errors=True)
    os.makedirs(temp_frame_dir, exist_ok=True)

    try:
        for idx, fname in enumerate(files):
            src_p = os.path.join(input_dir, fname)
            dst_p = os.path.join(temp_frame_dir, f"{idx:05d}.jpg")
            if fname.lower().endswith(('.jpg', '.jpeg')):
                shutil.copy2(src_p, dst_p)
            else:
                Image.open(src_p).convert("RGB").save(dst_p, "JPEG", quality=98)

        print(f">> [3/3] 正在使用 SAM2Matting 原生编码器锁定边缘并传播发丝 Alpha (共 {len(files)} 帧)...")
        inference_state = predictor.init_state(video_path=temp_frame_dir)
        predictor.reset_state(inference_state)

        # 若为模式 5 (replace) 且用户画了粗略 Roto 圈，优先使用 SAM2Matting 原生 Box 框选锁定能力，自动贴合物体真实边界！
        ys, xs = np.where(mask_np > 127)
        if args.mask_mode == "replace" and len(xs) > 10 and hasattr(predictor, "add_new_points_or_box"):
            box_xyxy = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            # 同时取 Roto 中心区域正样本点辅助锁定
            cy, cx = int(np.median(ys)), int(np.median(xs))
            pts = np.array([[cx, cy]], dtype=np.float32)
            lbls = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=pts,
                labels=lbls,
                box=box_xyxy
            )
        else:
            mask_bool_t = torch.from_numpy(mask_np > 127).to(device)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                mask=mask_bool_t
            )

        for out in tqdm(predictor.propagate_in_video(inference_state), total=len(files), desc="SAM2Matting 实时进度", ascii=False):
            out_frame_idx, _, _, alpha, _ = out
            alpha_2d = alpha.detach().cpu().squeeze().float().numpy().clip(0.0, 1.0)
            pha_np = (alpha_2d * 255.0).astype(np.uint8)

            f_num = start_frame + int(out_frame_idx)
            alpha_img = Image.fromarray(pha_np, mode="L")
            alpha_out = os.path.join(output_dir, f"{args.prefix}_alpha_{f_num:04d}.png")
            alpha_img.save(alpha_out, "PNG")
    finally:
        if os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir, ignore_errors=True)

    print("-----------------------------------------------------------------")
    print(f">> [OK] SAM2Matting 全部 {len(files)} 帧发丝 Alpha 输出完成: {output_dir}")
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