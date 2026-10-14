# -*- coding: utf-8 -*-
"""
一键部署 SAM 2 拆帧序列批量工作流（防强缓存 + 4层路径修正版）：
1. 强制绑定 bridge/sam2_workflow.json，打开或按 F5 刷新必定清空旧缓存并加载该工作流
2. 支持整个拆帧文件夹批量读取 (VFX_LoadSequence)
3. 跑完自动覆盖保存到输出目录，并自动弹出 Windows 文件夹窗口 (VFX_SaveSequence)
"""
import os
import json
import shutil

suite_root = os.path.dirname(os.path.abspath(__file__))
bridge_dir = os.path.join(suite_root, "bridge")
os.makedirs(bridge_dir, exist_ok=True)
bridge_wf_path = os.path.join(bridge_dir, "sam2_workflow.json")
comfy_dir = os.path.join(suite_root, "ComfyUI_windows_portable", "ComfyUI")
default_out_dir = os.path.join(comfy_dir, "output", "sam2_sequence").replace("\\", "/")

# 1. 生成标准的拆帧序列工作流并写入 bridge/sam2_workflow.json
sam2_seq_workflow = {
  "last_node_id": 8,
  "last_link_id": 9,
  "nodes": [
    {
      "id": 1,
      "type": "VFX_LoadSequence",
      "title": "1. 读取实拍拆帧序列文件夹 (粘贴文件夹路径或任一帧图片路径)",
      "pos": [60, 260],
      "size": [430, 130],
      "flags": {},
      "order": 0,
      "mode": 0,
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [3]},
        {"name": "START_FRAME", "type": "INT", "links": [5]}
      ],
      "properties": {"Node name for S&R": "VFX_LoadSequence"},
      "widgets_values": ["请粘贴拆帧图片所在文件夹路径", 0]
    },
    {
      "id": 2,
      "type": "SAM2ModelLoader (segment anything2)",
      "title": "2. SAM 2 模型加载器",
      "pos": [60, 40],
      "size": [430, 60],
      "flags": {},
      "order": 1,
      "mode": 0,
      "outputs": [
        {"name": "SAM2_MODEL", "type": "SAM2_MODEL", "links": [1]}
      ],
      "properties": {"Node name for S&R": "SAM2ModelLoader (segment anything2)"},
      "widgets_values": ["sam2_1_hiera_base_plus.pt"]
    },
    {
      "id": 3,
      "type": "GroundingDinoModelLoader (segment anything2)",
      "title": "3. 目标识别加载器",
      "pos": [60, 140],
      "size": [430, 60],
      "flags": {},
      "order": 2,
      "mode": 0,
      "outputs": [
        {"name": "GROUNDING_DINO_MODEL", "type": "GROUNDING_DINO_MODEL", "links": [2]}
      ],
      "properties": {"Node name for S&R": "GroundingDinoModelLoader (segment anything2)"},
      "widgets_values": ["GroundingDINO_SwinT_OGC (694MB)"]
    },
    {
      "id": 4,
      "type": "GroundingDinoSAM2Segment (segment anything2)",
      "title": "4. SAM 2 批量抠像 (默认 person 抠全身)",
      "pos": [530, 120],
      "size": [400, 160],
      "flags": {},
      "order": 3,
      "mode": 0,
      "inputs": [
        {"name": "sam_model", "type": "SAM2_MODEL", "link": 1},
        {"name": "grounding_dino_model", "type": "GROUNDING_DINO_MODEL", "link": 2},
        {"name": "image", "type": "IMAGE", "link": 3}
      ],
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [4, 7]},
        {"name": "MASK", "type": "MASK", "links": [6, 8]}
      ],
      "properties": {"Node name for S&R": "GroundingDinoSAM2Segment (segment anything2)"},
      "widgets_values": ["person", 0.3, False]
    },
    {
      "id": 5,
      "type": "VFX_SaveSequence",
      "title": "5. 自动保存序列并弹出文件夹 (重跑自动覆盖旧图)",
      "pos": [980, 60],
      "size": [450, 210],
      "flags": {},
      "order": 4,
      "mode": 0,
      "inputs": [
        {"name": "images", "type": "IMAGE", "link": 4},
        {"name": "mask", "type": "MASK", "link": 6},
        {"name": "start_frame", "type": "INT", "link": 5, "widget": {"name": "start_frame"}}
      ],
      "properties": {"Node name for S&R": "VFX_SaveSequence"},
      "widgets_values": [default_out_dir, "sam2", 1001, True, True, True]
    },
    {
      "id": 6,
      "type": "PreviewImage",
      "title": "预览：透明人物抠像结果",
      "pos": [980, 320],
      "size": [220, 250],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {"name": "images", "type": "IMAGE", "link": 7}
      ],
      "properties": {"Node name for S&R": "PreviewImage"}
    },
    {
      "id": 7,
      "type": "MaskToImage",
      "title": "遮罩转预览",
      "pos": [530, 330],
      "size": [210, 46],
      "flags": {},
      "order": 6,
      "mode": 0,
      "inputs": [
        {"name": "mask", "type": "MASK", "link": 8}
      ],
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [9]}
      ],
      "properties": {"Node name for S&R": "MaskToImage"}
    },
    {
      "id": 8,
      "type": "PreviewImage",
      "title": "预览：黑白实心遮罩 (Mask)",
      "pos": [1220, 320],
      "size": [220, 250],
      "flags": {},
      "order": 7,
      "mode": 0,
      "inputs": [
        {"name": "images", "type": "IMAGE", "link": 9}
      ],
      "properties": {"Node name for S&R": "PreviewImage"}
    }
  ],
  "links": [
    [1, 2, 0, 4, 0, "SAM2_MODEL"],
    [2, 3, 0, 4, 1, "GROUNDING_DINO_MODEL"],
    [3, 1, 0, 4, 2, "IMAGE"],
    [4, 4, 0, 5, 0, "IMAGE"],
    [5, 1, 1, 5, 2, "INT"],
    [6, 4, 1, 5, 1, "MASK"],
    [7, 4, 0, 6, 0, "IMAGE"],
    [8, 4, 1, 7, 0, "MASK"],
    [9, 7, 0, 8, 0, "IMAGE"]
  ],
  "groups": [],
  "config": {},
  "extra": {},
  "version": 0.4
}

with open(bridge_wf_path, "w", encoding="utf-8") as f:
    json.dump(sam2_seq_workflow, f, ensure_ascii=False, indent=2)

user_wf_dir = os.path.join(comfy_dir, "user", "default", "workflows")
os.makedirs(user_wf_dir, exist_ok=True)
shutil.copyfile(bridge_wf_path, os.path.join(user_wf_dir, "sam2_workflow.json"))

# 2. 写入自定义节点与前端强制加载脚本
plugin_dir = os.path.join(comfy_dir, "custom_nodes", "LiangLiang_SAM2_AutoLoad")
js_dir = os.path.join(plugin_dir, "js")
os.makedirs(js_dir, exist_ok=True)

init_py_code = '''# -*- coding: utf-8 -*-
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from aiohttp import web
import server

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SUITE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if not os.path.exists(os.path.join(SUITE_ROOT, "ComfyUI_windows_portable")):
    SUITE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

BRIDGE_WF_FILE = os.path.join(SUITE_ROOT, "bridge", "sam2_workflow.json")


@server.PromptServer.instance.routes.get("/liangliang/get_bridge_workflow")
async def get_bridge_workflow(request):
    if os.path.exists(BRIDGE_WF_FILE):
        with open(BRIDGE_WF_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        current_default_out = os.path.join(
            SUITE_ROOT, "ComfyUI_windows_portable", "ComfyUI", "output", "sam2_sequence"
        ).replace("\\\\", "/")

        for node in data.get("nodes", []):
            if node.get("type") == "VFX_SaveSequence" and node.get("widgets_values"):
                out_val = str(node["widgets_values"][0]).replace("\\\\", "/")
                if "ComfyUI/output/sam2_sequence" in out_val and not os.path.exists(out_val):
                    node["widgets_values"][0] = current_default_out

        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        return web.json_response(data, headers=headers)
    return web.json_response({"error": f"not found: {BRIDGE_WF_FILE}"}, status=404)


@server.PromptServer.instance.routes.post("/liangliang/notify_reload")
async def notify_reload(request):
    server.PromptServer.instance.send_sync("liangliang_reload_workflow", {"status": "reload"})
    return web.json_response({"status": "ok"})


class VFX_LoadSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "请粘贴拆帧图片所在文件夹路径", "multiline": False}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "START_FRAME")
    FUNCTION = "load_sequence"
    CATEGORY = "LiangLiangVFX"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def load_sequence(self, folder_path, max_frames=0):
        folder_path = folder_path.strip().strip('"').strip("'")
        if os.path.isfile(folder_path):
            folder_path = os.path.dirname(folder_path)

        if not os.path.isdir(folder_path):
            raise ValueError(f"找不到拆帧序列文件夹，请检查路径是否正确: {folder_path}")

        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        if not files:
            raise ValueError(f"文件夹内没有找到有效图片 (.png/.jpg/.tif): {folder_path}")

        if max_frames > 0:
            files = files[:max_frames]

        m = re.search(r'(\\d+)\\.[^.]+$', files[0])
        start_frame = int(m.group(1)) if m else 1001

        images = []
        target_size = None
        for fname in files:
            fpath = os.path.join(folder_path, fname)
            img = Image.open(fpath).convert("RGB")
            if target_size is None:
                target_size = img.size
            elif img.size != target_size:
                img = img.resize(target_size, Image.BILINEAR)
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(torch.from_numpy(arr))

        batch_tensor = torch.stack(images, dim=0)
        print(f">> [LiangLiangVFX] 已批量加载拆帧序列: 共 {len(files)} 帧 (起始帧: {start_frame})")
        return (batch_tensor, start_frame)


class VFX_SaveSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_folder": ("STRING", {"default": "output/sam2_sequence", "multiline": False}),
                "file_prefix": ("STRING", {"default": "sam2", "multiline": False}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 999999, "step": 1}),
                "save_bw_mask": ("BOOLEAN", {"default": True}),
                "save_rgba_cutout": ("BOOLEAN", {"default": True}),
                "auto_open_folder": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_sequence"
    OUTPUT_NODE = True
    CATEGORY = "LiangLiangVFX"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def save_sequence(self, images, output_folder, file_prefix="sam2", start_frame=1001,
                      save_bw_mask=True, save_rgba_cutout=True, auto_open_folder=True, mask=None):
        output_folder = output_folder.strip().strip('"').strip("'")
        if not output_folder or "请粘贴" in output_folder:
            output_folder = os.path.join(SUITE_ROOT, "ComfyUI_windows_portable", "ComfyUI", "output", "sam2_sequence")

        os.makedirs(output_folder, exist_ok=True)

        for old_f in os.listdir(output_folder):
            if old_f.startswith(f"{file_prefix}_mask_") or old_f.startswith(f"{file_prefix}_rgba_"):
                try:
                    os.remove(os.path.join(output_folder, old_f))
                except Exception:
                    pass

        batch_size = images.shape[0]
        for i in range(batch_size):
            frame_num = start_frame + i
            img_np = (images[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            mask_np = None
            if mask is not None and i < mask.shape[0]:
                m_tensor = mask[i].cpu().numpy()
                if m_tensor.ndim == 3:
                    m_tensor = m_tensor[0]
                mask_np = (m_tensor * 255.0).clip(0, 255).astype(np.uint8)

            if save_bw_mask and mask_np is not None:
                mask_img = Image.fromarray(mask_np, mode="L")
                mask_path = os.path.join(output_folder, f"{file_prefix}_mask_{frame_num:04d}.png")
                mask_img.save(mask_path, "PNG")

            if save_rgba_cutout:
                if img_np.shape[-1] == 4:
                    rgba_img = Image.fromarray(img_np, mode="RGBA")
                else:
                    rgb_img = Image.fromarray(img_np[:, :, :3], mode="RGB")
                    if mask_np is not None:
                        rgba_img = rgb_img.copy()
                        rgba_img.putalpha(Image.fromarray(mask_np, mode="L"))
                    else:
                        rgba_img = rgb_img
                rgba_path = os.path.join(output_folder, f"{file_prefix}_rgba_{frame_num:04d}.png")
                rgba_img.save(rgba_path, "PNG")

        print(f">> [LiangLiangVFX] 已保存并覆盖 {batch_size} 帧序列至: {output_folder}")

        if auto_open_folder and os.name == "nt" and os.path.exists(output_folder):
            try:
                os.startfile(os.path.normpath(output_folder))
            except Exception as e:
                print("打开输出文件夹失败:", e)

        return {"ui": {"text": [f"已输出 {batch_size} 帧至 {output_folder}"]}}


NODE_CLASS_MAPPINGS = {
    "VFX_LoadSequence": VFX_LoadSequence,
    "VFX_SaveSequence": VFX_SaveSequence
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VFX_LoadSequence": "1. 读取拆帧序列文件夹 (LiangLiangVFX)",
    "VFX_SaveSequence": "5. 自动保存序列并弹窗 (LiangLiangVFX)"
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
'''

with open(os.path.join(plugin_dir, "__init__.py"), "w", encoding="utf-8") as f:
    f.write(init_py_code)

# 3. 前端脚本：彻底清除浏览器 localStorage / sessionStorage 强缓存，强制加载最新 bridge/sam2_workflow.json
autoload_js_code = '''import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function purgeComfyWorkflowCache() {
    const keysToRemove = [
        "workflow",
        "Comfy.PreviousWorkflow",
        "Comfy.Workflow.Drafts"
    ];
    keysToRemove.forEach(k => {
        try { localStorage.removeItem(k); } catch (e) {}
        try { sessionStorage.removeItem(k); } catch (e) {}
    });
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
        const key = sessionStorage.key(i);
        if (key && key.toLowerCase().includes("workflow")) {
            try { sessionStorage.removeItem(key); } catch (e) {}
        }
    }
}

async function forceLoadBridgeWorkflow() {
    try {
        purgeComfyWorkflowCache();
        const resp = await fetch("/liangliang/get_bridge_workflow?t=" + Date.now(), {
            cache: "no-store",
            headers: {
                "Pragma": "no-cache",
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        });
        if (!resp.ok) {
            console.warn("[LiangLiangVFX] 未找到 bridge/sam2_workflow.json，状态码:", resp.status);
            return;
        }
        const wfData = await resp.json();
        await app.loadGraphData(wfData, true, true, null);

        if (wfData.nodes && app.graph && app.graph._nodes_by_id) {
            for (const nodeDef of wfData.nodes) {
                const liveNode = app.graph._nodes_by_id[nodeDef.id];
                if (liveNode && liveNode.widgets && Array.isArray(nodeDef.widgets_values)) {
                    for (let i = 0; i < liveNode.widgets.length && i < nodeDef.widgets_values.length; i++) {
                        liveNode.widgets[i].value = nodeDef.widgets_values[i];
                    }
                }
            }
            app.graph.setDirtyCanvas(true, true);
        }

        try {
            localStorage.setItem("workflow", JSON.stringify(wfData));
        } catch (e) {}

        console.log("[LiangLiangVFX] 已清除浏览器强缓存并强制加载最新 bridge/sam2_workflow.json！");
    } catch (err) {
        console.error("[LiangLiangVFX] 加载默认工作流失败:", err);
    }
}

app.registerExtension({
    name: "LiangLiang.SAM2.BridgeAutoLoad",
    async init() {
        purgeComfyWorkflowCache();
    },
    async setup() {
        await forceLoadBridgeWorkflow();
        setTimeout(forceLoadBridgeWorkflow, 600);

        api.addEventListener("liangliang_reload_workflow", () => {
            console.log("[LiangLiangVFX] 收到 Nuke 端更新通知，自动热重载工作流...");
            forceLoadBridgeWorkflow();
        });
    }
});
'''

with open(os.path.join(js_dir, "autoload_sam2.js"), "w", encoding="utf-8") as f:
    f.write(autoload_js_code)

print("✅ 升级完成！请重启 ComfyUI 黑框，并在浏览器按 Ctrl+F5 刷新！")