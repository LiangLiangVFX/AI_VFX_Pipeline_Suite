<div align="center">

# 🎬 AI_VFX_Pipeline_Suite

**面向 Nuke 与 Unreal Engine 影视合成管线的多引擎 AI 发丝级视频抠像算力套件**  
*Standalone Multi-Engine GPU AI Video Matting Backend Suite for Nuke VFX Pipelines*

[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite/releases)
[![MatAnyone 2](https://img.shields.io/badge/CVPR%202026-MatAnyone%202-success.svg)](https://github.com/pq-yang/MatAnyone2)
[![SAM2Matting](https://img.shields.io/badge/ECCV%202026-SAM2Matting-9b59b6.svg)](https://github.com/FudanCVL/SAM2Matting)
[![BiRefNet](https://img.shields.io/badge/2K%20Matting-BiRefNet--HR-ff6b6b.svg)](https://github.com/ZhengPeng7/BiRefNet)
[![RVM](https://img.shields.io/badge/Fallback-RVM%20(ResNet50)-orange.svg)](https://github.com/PeterL1n/RobustVideoMatting)
[![DCC](https://img.shields.io/badge/DCC-Foundry%20Nuke-f5a623.svg)](https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync)

</div>

---

## 📖 项目概述 (Overview)

**`AI_VFX_Pipeline_Suite`** 是一套专为影视后期合成（Foundry Nuke）与虚幻引擎（Unreal Engine 5）制片管线打造的**纯后台、免网页端（ComfyUI-Free）、便携式多引擎 AI 视频抠像算力基座**。

本项目采用工业界标准的 **“前端 DCC 插件仓 + 后端独立算力仓”解耦架构**：
* **前端交互层**：由配套仓库 **[VFX_Pipeline_CamSync](https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync)**（`NukeWorkflowTool`）提供 Nuke 双栏菜单、交互面板与四模块横向正交节点网自动构建。
* **后端算力层（本仓库）**：封装独立的 `python_embeded`（Python + PyTorch + CUDA）物理隔离沙盒，集成 **BiRefNet-HR-Matting**、**MatAnyone 2 (CVPR 2026)** 与 **SAM2Matting (ECCV 2026)** 三大主力 SOTA 引擎（并以 **RVM** 作为底层静默容错保底），全面覆盖从**单帧 2K 万物全自动抠像**、**人像特写发丝精修**到**复杂运动/非人道具跨帧防跟丢抠像**的全场景制片需求。

---

## 🧠 四大 AI 引擎矩阵与协同分工 (Multi-Engine Matrix)

管线内部各引擎既可独立执行序列抠像并输出至各自专属子目录（互不覆盖），又可在底层协同完成“首帧高清识主体 + 跨帧发丝记忆跟踪”：

| 引擎名称 | 算法来源 | 核心定位与优势场景 | 引导机制 | Nuke 输出目录 |
| :--- | :---: | :--- | :--- | :--- |
| **1. BiRefNet-HR-Matting** | SOTA 2K Matting | **全自动高清万物抠像 & 首帧引导王牌**<br>原生支持 `1024` / `2048 (2K)` 推理，无需任何 Roto 自动提取人物+手持武器/道具/任意前景，同时作为跟踪引擎的默认首帧提取器 | 全自动无须引导 | `workarea/roto/birefnet` |
| **2. MatAnyone 2** | **CVPR 2026** | **影视级主角人像发丝天花板**<br>基于 240 万帧实拍人像训练，支持首帧 `Warmup` 迭代精修与 `Erode/Dilate` 边缘自适应软化，逆光乱发与薄纱表现极佳 | BiRefNet 首帧自动引导 + 5种 Roto 布尔模式 | `workarea/roto/matanyone2` |
| **3. SAM2Matting** | **ECCV 2026** | **万物防跟丢视频抠像之王 (`Base+` / `Tiny`)**<br>融合 SAM 2.1 超强防跟丢记忆库与多尺度渐进式发丝预测器，专治剧烈打斗、大幅遮挡、动物毛发、载具与细长武器 | 原生 `Box+Point` 框选自动贴边 + 5种 Roto 模式 | `workarea/roto/sam2matting` |
| **4. RVM (ResNet50)** | CVPR 2022 | **底层静默容错保底引擎**<br>在后台提供毫秒级人形提取保底，若新工作站尚未部署 BiRefNet 权重，管线将自动无缝回退至 RVM 保障生产不中断 | 后台自动触发 | `workarea/bg/ai_rvm_matte` |

---

## ✨ 核心技术特性 (Key Technical Features)

### 1. 脱离 ComfyUI，GPU 显存直通与独立实时进度终端
* **无浏览器依赖**：在 Nuke 内一键触发 Windows 独立实时进度窗口（`CREATE_NEW_CONSOLE`），实时展示模型加载阶段与 `tqdm` 帧进度条；运算结束后自动置顶弹出输出目录，并在 Nuke 中构建无损 `Copy -> Premult` 预乘节点树。
* **脏数据免疫**：严格过滤有效图片后缀（`.png`, `.jpg`, `.tif`, `.bmp`, `.webp`），100% 免疫文件服务器上的 `Thumbs.db`、`.DS_Store` 隐藏文件。

### 2. 五种首帧布尔引导模式 + 原生框选自动贴边
针对影视镜头中的“多人路人干扰”、“背景杂物误抠”、“手持武器/机械模块连带跟踪”等复杂情况，提供 5 种首帧引导模式：
1. **`1_全自动抠主体 (auto)`**：无需画任何 Roto，自动调用 **BiRefNet** 提取第 1 帧高清人物与随身道具，再交由 `MatAnyone 2` 或 `SAM2Matting` 向后全片跟踪发丝。
2. **`2_只保留Roto圈内的主体 (intersect)`**：`BiRefNet ∩ Holdout_Roto_KEEP`，在首帧随手画个大圈框住主角，自动剔除圈外路人。
3. **`3_从主体上挖掉Roto区域 (subtract)`**：`BiRefNet - Garbage_Roto_REMOVE`，在首帧圈出多抠的杂物并将其精准挖除。
4. **`4_主体加上Roto模块区域 (union)`**：`BiRefNet ∪ Holdout_Roto_KEEP`，仅在首帧补画漏掉的道具轮廓，合并后整体向后跟踪。
5. **`5_完全按Roto区域 / 原生框选锁定任意物体 (replace)`**：跳过首帧自动识别，直接抠取任意非人目标。在 `SAM2Matting` 下更会自动激活 **SAM 2.1 原生 `Box + Center Point` 提示编码器**，只需随手画个粗略外圈即可自动吸附物体真实边缘并生成柔顺毛发 Alpha。

### 3. 中途故障帧局部重算补救 (`--start_at`)
* 当长镜头中途（如第 `1018` 帧）因遮挡或剧烈运动模糊导致跟丢时，无需重跑整段镜头。在面板填入起始运算帧号 `1018` 并在该帧修正引导 Roto，引擎将**完好保留 `1001~1017` 帧已有序列**，仅从 `1018` 帧向后重算并无缝覆盖。

### 4. 严密的 DCC 环境隔离与容错机制
* **防 DLL 冲突 (`WinError 1114`)**：自动清洗 Nuke 注入的 `PYTHONPATH`、`PYTHONHOME`、`QT_PLUGIN_PATH`、`OCIO` 等环境变量，启动首行通过 `ctypes` 预加载 `vcruntime140`、`libiomp5md`、`c10.dll` 等核心动态库。
* **防 GBK 编码与批处理管道符崩溃**：强制重构子进程 `stdout/stderr` 为 `UTF-8` 并清洗终端标题特殊字符。

---

## 📂 仓库目录结构 (Directory Structure)

遵循工业级**代码与大体积二进制产物分离（Code & Artifacts Separation）**原则，Git 仓库仅托管轻量源码，数 GB 的运行环境与权重通过 **GitHub Releases** 分发：

```text
AI_VFX_Pipeline_Suite/
├── birefnet/                                # [引擎 1] BiRefNet-HR-Matting 2K 全自动万物抠像
│   ├── models/BiRefNet_HR-matting/          # 模型架构代码与配置文件 (model.safetensors 由 Release 解压)
│   └── run_birefnet_cli.py                  # ★ BiRefNet 序列帧 1K/2K 抠像执行器
│
├── matanyone2/                              # [引擎 2] MatAnyone 2 (CVPR 2026) 人像发丝抠像
│   ├── matanyone2/                          # 官方网络架构与 InferenceCore 源码
│   ├── pretrained_models/                   # (由 Release 解压) matanyone2.pth 权重
│   └── run_matanyone2_cli.py                # ★ MatAnyone 2 显存直通 + BiRefNet/RVM 双引导执行器
│
├── sam2matting/                             # [引擎 3] SAM2Matting (ECCV 2026) 万物跟踪抠像
│   ├── sam2/                                # SAM2Matting 核心网络与 configs 配置
│   ├── checkpoints/                         # (由 Release 解压) SAM2Matting-SAM2.1Base+.pt & Tiny.pt
│   └── run_sam2matting_cli.py               # ★ SAM2Matting 原生 Box+Mask 锁定与序列传播执行器
│
├── rvm/                                     # [引擎 4] Robust Video Matting 保底引擎
│   ├── model/                               # RVM 网络源码
│   ├── models/                              # (由 Release 解压) rvm_resnet50.pth 权重
│   └── run_rvm_cli.py                       # RVM 独立序列执行器
│
├── python_embeded/                          # (由 Release 解压) 独立 Python 3.12 + PyTorch + CUDA 沙盒
├── 一键注册AI引擎环境(双击运行).bat           # ★ 工作站环境绑定、依赖校验与 LL_AI_SUITE_ROOT 注册脚本
├── .gitignore                               # 屏蔽大体积环境包、模型权重与临时缓存
└── .gitattributes                           # 跨平台换行符规范
```

---

## 🚀 工作站首次部署指南 (Workstation Setup)

在新工作站上只需执行一次以下 **3 步部署**，后续日常代码升级只需在 GitHub Desktop 中点击 **`Pull`** 即可秒级同步：

1. **克隆双端仓库**：
   使用 **GitHub Desktop** 将本仓库与前端仓库克隆到本地：
   ```bat
   git clone [https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite.git](https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite.git)
   git clone [https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync.git](https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync.git)
   ```
2. **下载并解压 Release 环境与权重包**：
   * 前往本仓库 **[Releases 页面](https://github.com/LiangLiangVFX/AI_VFX_Pipeline_Suite/releases)**。
   * 下载基础运行环境分卷包（`AI_Env_Runtime.7z.001` 等）以及模型扩展包（`AI_Models_Addon_BiRefNet_SAM2Matting.7z`），全部**解压至 `AI_VFX_Pipeline_Suite` 根目录**。
3. **双击运行一键注册脚本**：
   * 双击运行 `AI_VFX_Pipeline_Suite` 根目录下的 **`一键注册AI引擎环境(双击运行).bat`**（自动写入相对路径、校验依赖并注册 `LL_AI_SUITE_ROOT` 环境变量）。
   * 进入 `VFX_Pipeline_CamSync\Nuke\NukeWorkflowTool\`，双击运行 **`一键安装(双击运行).bat`**，即可启动 Nuke 使用！

---

## 🎛️ 实战调参黄金口诀 (Parameter Tuning Guide)

在 85% 的常规镜头中，保持面板默认参数即可获得最佳效果；针对极端镜头可按以下准则微调：

* **常规人像镜头**：直接使用 **`MatAnyone 2`**，保持默认（`Warmup = 10`, `边缘软化半径 Erode/Dilate = 10`）。
* **迎风乱发 / 逆光蓬松长发 / 剧烈运动模糊**：使用 **`MatAnyone 2`**，将 `边缘软化半径` 调大至 **`15 ~ 25`**（扩大未知半透明发丝搜索带），`Warmup` 调至 **`15 ~ 20`**。
* **细长武器 / 手持道具 / 大幅度打斗遮挡 / 非人目标**：
  * 方案 A：直接使用 **`SAM2Matting`**（默认 `SAM2.1_Base+` 规格），利用 SAM 2.1 记忆特征防跟丢；
  * 方案 B：若使用 **`MatAnyone 2`**，需将 `边缘软化半径` 调小至 **`3 ~ 5`**，防止 10 像素腐蚀吃断细长武器。
* **无遮挡镜头一键全自动提取 2K 极致细节（人+随身装备）**：直接使用 **`BiRefNet`**，下拉框选择 **`2_极致发丝模式_2048x2048`**，以点对点原生 2K 分辨率推理。

---

## 👨‍💻 作者与致谢 (Author & Acknowledgements)

* **Author**: **LiangLiangArtist** (`LiangLiangVFX`) | Contact: `liangliangvfx@gmail.com`
* **Frontend Pipeline Repo**: [VFX_Pipeline_CamSync](https://github.com/LiangLiangVFX/VFX_Pipeline_CamSync)
* **Algorithm References**:
  * [MatAnyone 2 (CVPR 2026)](https://github.com/pq-yang/MatAnyone2) | [SAM2Matting (ECCV 2026)](https://github.com/FudanCVL/SAM2Matting) | [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | [RVM](https://github.com/PeterL1n/RobustVideoMatting)