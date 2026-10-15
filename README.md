<div align="center">

# 🎬 AI_VFX_Pipeline_Suite

**面向 Nuke 与 Unreal Engine 影视合成管线的独立 AI 发丝级视频抠像算力套件**  
*Standalone GPU-Accelerated AI Video Matting Backend Suite for Nuke VFX Pipelines*

[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite/releases)
[![MatAnyone 2](https://img.shields.io/badge/Engine-MatAnyone%202%20(CVPR%202026)-success.svg)](https://github.com/pq-yang/MatAnyone2)
[![RVM](https://img.shields.io/badge/Engine-RVM%20(ResNet50)-orange.svg)](https://github.com/PeterL1n/RobustVideoMatting)
[![Python](https://img.shields.io/badge/Python-3.12%20(Embedded)-yellow.svg)](#)
[![CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-76b900.svg)](#)
[![DCC](https://img.shields.io/badge/DCC-Foundry%20Nuke-f5a623.svg)](#)

</div>

---

## 📖 项目概述 (Overview)

**`AI_VFX_Pipeline_Suite`** 是一套专为影视后期合成（Foundry Nuke）与虚幻引擎（Unreal Engine 5）制片管线打造的**纯后台、免网页端（ComfyUI-Free）、便携式 AI 视频抠像算力基座**。

本项目采用工业界标准的 **“前端 DCC 插件仓 + 后端独立算力仓”解耦架构**：
* **前端交互层**：由配套仓库 **[VFX_Pipeline_CamSync](https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync)**（`NukeWorkflowTool`）提供 Nuke 菜单、交互面板与四模块正交节点网自动构建。
* **后端算力层（本仓库）**：封装独立的 `python_embeded`（Python 3.12 + PyTorch + CUDA）物理隔离沙盒，集成 **MatAnyone 2 (CVPR 2026)** 与 **RVM (ResNet50)** 双引擎。彻底解决分割模型（如 SAM 2）边缘生硬如剪纸的痛点，实现真正的**发丝级半透明 Alpha（0.0~1.0 灰阶过渡）提取与跨帧目标锁定跟踪**。

---

## ✨ 核心技术亮点 (Key Features)

### 1. 彻底脱离 ComfyUI，GPU 显存直通极速推理
* **无浏览器依赖**：告别繁琐的网页端节点连线与缓存冲突，在 Nuke 内一键触发后台独立实时进度窗口，算完自动置顶弹出输出目录并回导 Nuke 节点树。
* **内存直通与脏数据免疫**：直接调用 `MatAnyone 2` 底层 `InferenceCore` 在 GPU 显存内完成序列帧推理，严格过滤有效图像后缀，100% 免疫公司文件服务器上的 `Thumbs.db`、`.DS_Store` 及中文路径引发的崩溃。

### 2. RVM + MatAnyone 2 双引擎协同与 5 种首帧布尔引导模式
* 针对影视镜头中的“多人干扰”、“局部杂物误抠”、“人物手持道具/机械模块连带抠像”等复杂场景，将 **RVM 的 0.2 秒首帧快速识人**与 **Nuke 单帧 Roto 遮罩**进行底层布尔运算合成，再交由 **MatAnyone 2** 进行首帧自适应膨胀腐蚀精修（Warmup）与全序列记忆传播：
  1. **`auto`（全自动只抠人）**：无需绘制任何 Roto，RVM 自动提取首帧人形并由 MatAnyone 2 完成全片发丝精修。
  2. **`intersect`（只保留 Roto 圈内的人）**：`RVM ∩ Roto`，在首帧随手画个大圈框住主角，自动剔除圈外路人与背景干扰。
  3. **`subtract`（从人身上挖掉 Roto 区域）**：`RVM - Roto`，在首帧圈出多抠的局部杂物，精准挖除后向后跟踪。
  4. **`union`（人 + Roto 模块区域合并跟踪）**：`RVM ∪ Roto`，无需画人，仅在首帧画出人物手中的武器/道具/机械模块，自动合二为一整体跟踪。
  5. **`replace`（完全按 Roto 区域抠像）**：跳过 RVM，完全依据首帧 Roto 区域抠取任意非人目标（车辆、动物、特效道具等），并自动将手绘硬边转化为柔顺边缘。

### 3. 中途故障帧局部补救 (`--start_at`)
* 当长镜头中途（如第 `1018` 帧）因剧烈运动模糊或遮挡导致跟踪丢失时，无需重跑整段镜头。只需在面板填入故障起始帧号（`1018`）并在该帧绘制引导 Roto，引擎将**完好保留 `1001~1017` 帧已有成果**，仅从 `1018` 帧开始向后重算并无缝覆盖。

### 4. 严密的 DCC 环境隔离与容错机制
* **杜绝 `WinError 1114` (DLL 冲突)**：自动清洗 Nuke 注入的 `PYTHONPATH`、`PYTHONHOME`、`QT_PLUGIN_PATH` 等污染变量，并在进程启动首行通过 `ctypes` 预加载 `vcruntime140`、`libiomp5md`、`c10.dll` 等核心动态库。
* **杜绝 `UnicodeEncodeError` (GBK 编码崩溃)**：强制重构子进程标准输出流为 `UTF-8` 并注入 `chcp 65001` 实时终端，确保中文路径与进度条稳定输出。

---

## 📂 仓库目录架构 (Directory Structure)

遵循工业级**代码与大体积二进制产物分离（Code & Artifacts Separation）**原则，Git 仓库仅托管轻量源码，数 GB 的运行环境与 `.pth` 权重通过 **GitHub Releases** 分发：

```text
AI_VFX_Pipeline_Suite/
├── matanyone2/                              # [引擎 2] MatAnyone 2 (CVPR 2026) 核心目录
│   ├── matanyone2/                          # 官方网络架构与 InferenceCore 源码
│   ├── pretrained_models/
│   │   └── matanyone2.pth                   # (由 Release 解压) MatAnyone 2 预训练权重
│   ├── inference_matanyone2.py              # 官方推理脚本
│   └── run_matanyone2_cli.py                # ★ Nuke 专用 GPU 显存直通 & 首帧布尔引导执行器
│
├── rvm/                                     # [引擎 1] Robust Video Matting 核心目录
│   ├── model/                               # RVM 网络架构源码
│   ├── models/
│   │   └── rvm_resnet50.pth                 # (由 Release 解压) RVM ResNet50 预训练权重
│   └── run_rvm_cli.py                       # ★ RVM 序列帧全自动抠像执行器
│
├── python_embeded/                          # (由 Release 解压) 独立 Python 3.12 + PyTorch + CUDA 沙盒
│   ├── Lib/site-packages/matanyone2.pth     # 相对路径寻址配置 (../../../matanyone2)
│   └── python.exe                           # 算力引擎解释器入口
│
├── 一键注册AI引擎环境(双击运行).bat           # ★ 工作站环境绑定与 LL_AI_SUITE_ROOT 注册脚本
├── .gitignore                               # 屏蔽大体积环境包与临时缓存
└── .gitattributes                           # 跨平台文本与脚本换行符规范
```

---

## 🚀 工作站首次部署指南 (Quick Start & Deployment)

在新工作站上只需执行一次以下 **3 步部署**，后续日常代码升级只需在 GitHub Desktop 中点击 **`Pull`** 即可秒级同步：

### 第一步：克隆双端仓库
使用 **GitHub Desktop** 或 Git 命令将本仓库与前端 Nuke 工具仓库克隆到本地（推荐放在同一父目录下）：
```bat
git clone [https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite.git](https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite.git)
git clone [https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync.git](https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync.git)
```

### 第二步：下载并解压 Release 运行环境与模型权重包
1. 前往本仓库右侧的 **[Releases 页面 (v1.0.0)](https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite/releases)**。
2. 下载 **Assets** 下的所有分卷压缩包（如 `.7z.001`, `.7z.002` 等）到同一目录。
3. 右键点击第 1 个分卷 **`.001`**，使用 7-Zip 或 WinRAR **解压到 `AI_VFX_Pipeline_Suite` 根目录**（确保解压后 `python_embeded/python.exe`、`matanyone2/pretrained_models/matanyone2.pth` 与 `rvm/models/rvm_resnet50.pth` 均就位）。

### 第三步：双击运行一键注册脚本
1. 在 `AI_VFX_Pipeline_Suite` 根目录下，双击运行 **`一键注册AI引擎环境(双击运行).bat`**：
   * 自动写入便携 Python 相对模块寻址路径 `../../../matanyone2`；
   * 自动注册 Windows 用户环境变量 `LL_AI_SUITE_ROOT`，使 Nuke 跨盘符 100% 自动定位算力套件。
2. 进入前端仓库 `VFX_Pipeline_CamSync\Nuke\NukeWorkflowTool\`，双击运行 **`一键安装(双击运行).bat`**，即可启动 Nuke 开始使用！

---

## 🖥️ Nuke 全流程实战指南 (Nuke Production Workflow)

### 1. 四大模块横向正交画布布局
管线在 Nuke 工作区中严格按照从左至右的工业正交顺序排布，模块间预留安全走廊，绝不重叠：

```text
┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌──────────────────────────┐
│ 1. 实拍预处理与输出底板  │   │ 2. 【AI 抠像与修补模块】 │   │ 3. UE 预合成主干区      │   │ 4. 分层 AOV 与 Cryptomatte│
│    (X: -60 ~ 580)      │   │  MatAnyone 2 (CVPR2026)│   │   (单层 / 多层自适应)   │   │    (X: 2060 ~ 2980)      │
│                        │   │     发丝级             │   │   (X: 1340 ~ 1960)     │   │                          │
│  Read_Plate   Roto     │   │    (X: 680 ~ 1240)     │   │                        │   │  Actor_ID   Material_ID  │
│      │         │       │   │                        │   │       Read_UE_BG       │   │     │            │       │
│  Switch ──► Keyer      │   │            Read_Alpha  │   │           │            │   │  Crypto     4x Shuffles  │
│      │         │       │   │                │       │   │     Grade / Merges     │   │                          │
│  Copy_Alpha ◄──┘       │   │            Shuffle     │   │           │            │   │                          │
│      │                 │   │                │       │   │           ▼            │   │                          │
│      ├─────────────────┼───┼─► Dot_B     Grade      │   │    Merge_Comp_Over ◄───┼───┘ (前景自动连自 Premult_AI)
│      ▼                 │   │     │          │       │   │           │            │
│  Premult_Plate         │   │     │      FilterErode │   │    HorizonWriteNode    │
│                        │   │     │          │       │   │           │            │
│                        │   │     ▼        Blur      │   │        Viewer1         │
│                        │   │  Copy_AI ◄─────┘       │   │                        │
│                        │   │     │                  │   │                        │
│                        │   │  Premult_MatAnyone2 ───┼───┼───────────┘            │
│                        │   │     │                  │   │                        │
│                        │   │  Write_Final_PNG       │   │                        │
└────────────────────────┘   └────────────────────────┘   └────────────────────────┘   └──────────────────────────┘
```

### 2. MatAnyone 2 面板 6 大模式速查表

在 Nuke 顶部菜单或左侧工具栏点击 **`LiangLiangVFX` -> `[AI] MatAnyone 2 发丝级目标锁定抠像 (CVPR 2026)`**：

| 面板模式选项 | 适用镜头场景 | Nuke 内操作方式（仅需在首帧/指定帧画 1 帧） | 底层引导遮罩合成逻辑 |
| :--- | :--- | :--- | :--- |
| **`1_全自动只抠人`** | 常规单人/清晰主体镜头 | **无需画任何 Roto**，直接点击运行 | 调用 RVM 0.2s 提取首帧人形 $\to$ MatAnyone 2 发丝跟踪 |
| **`2_只保留Roto圈内的人`** | 画面有多人/路人，只需抠主角 | 在第 1 帧双击 **`Holdout_Roto_KEEP`** 随手画大圈包住主角 | `RVM人形 ∩ Holdout_Roto`，圈外路人直接清零 |
| **`3_从人身上挖掉Roto区域`** | 人身旁被多抠了一块背景杂物 | 在第 1 帧双击 **`Garbage_Roto_REMOVE`** 圈出多余杂物 | `RVM人形 - Garbage_Roto`，局部挖除后向后跟踪 |
| **`4_人加上Roto模块区域`** | 需将人手里的道具/武器一起抠出 | 人不用画！在第 1 帧双击 **`Holdout_Roto_KEEP`** 仅画出道具 | `RVM人形 ∪ Holdout_Roto`，人与道具合并整体跟踪 |
| **`5_完全按Roto区域抠像`** | 不抠人，只抠特定物体/载具/道具 | 在第 1 帧双击 **`Holdout_Roto_KEEP`** 画出目标物体轮廓 | 完全跳过 RVM，将首帧 Roto 边缘软化精修后跟踪全片 |
| **`6_直接导回已生成的序列`** | 已算完镜头，重新导入或刷新节点网 | 直接选择并点击 OK（免重新运算） | 自动构建 AI 节点网并挂载 `Write_Final_PNG`、`Merge_Comp_Over` 与 `Viewer1` |

---

## 💻 命令行独立调用参考 (CLI Reference)

除在 Nuke 内一键调用外，`run_matanyone2_cli.py` 与 `run_rvm_cli.py` 也支持在终端或外部批处理脚本中独立运行：

### 调用 MatAnyone 2 引擎
```bat
"python_embeded\python.exe" "matanyone2\run_matanyone2_cli.py" ^
  --input "W:\project\shot_2220\proxy-jpg" ^
  --output "W:\project\shot_2220\roto\matanyone2" ^
  --mask_mode auto ^
  --prefix mat2_2220 ^
  --warmup 10 ^
  --erode_kernel 10 ^
  --dilate_kernel 10 ^
  --start_at 0
```

**参数说明：**
* `--input`：输入实拍拆帧图片序列文件夹路径（支持 `.png`, `.jpg`, `.tif`, `.bmp`, `.webp`）。
* `--output`：输出发丝 Alpha 序列的目标目录。
* `--mask`：*(可选)* 外部引导帧黑白遮罩 PNG 路径。
* `--mask_mode`：引导遮罩合成模式，可选 `auto` | `intersect` | `subtract` | `union` | `replace`（默认 `auto`）。
* `--prefix`：输出序列文件名前缀（默认生成 `{prefix}_alpha_%04d.png`）。
* `--warmup`：首帧发丝边缘精修预热迭代次数（默认 `10`）。
* `--erode_kernel` / `--dilate_kernel`：首帧遮罩硬边腐蚀与膨胀半径（默认 `10`）。
* `--start_at`：指定从第几帧开始向后重算补救（默认 `0` 代表从头跑全片）。
* `--save_rgba`：*(可选)* 附加此参数可额外输出预乘好的 4 通道 RGBA 序列。

---

## 👨‍💻 作者与致谢 (Author & Acknowledgements)

* **Author**: **LiangLiangArtist** (`LiangLiangVFX`)
* **Companion Repository**: [VFX_Pipeline_CamSync](https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync)
* **Algorithm References**:
  * [MatAnyone 2 (CVPR 2026)](https://github.com/pq-yang/MatAnyone2) — S-Lab, Nanyang Technological University
  * [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting)