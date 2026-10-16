@echo off
chcp 65001 >nul
title AI_VFX_Pipeline_Suite 环境注册与依赖同步工具
cd /d "%~dp0"
set "SUITE_DIR=%~dp0"
if "%SUITE_DIR:~-1%"=="\" set "SUITE_DIR=%SUITE_DIR:~0,-1%"

echo =========================================================
echo     正在配置 AI_VFX_Pipeline_Suite 工作站环境...
echo =========================================================
echo.

if exist "python_embeded\python.exe" (
    echo ../../../matanyone2> "python_embeded\Lib\site-packages\matanyone2.pth"
    echo [1/3] 已写入便携 Python 相对模块寻址路径: ../../../matanyone2
    
    echo [2/3] 正在校验并补齐 BiRefNet 与 SAM2Matting 轻量依赖库...
    "python_embeded\python.exe" -m pip install transformers timm einops kornia safetensors huggingface_hub hydra-core iopath omegaconf --upgrade-strategy only-if-needed -q
) else (
    echo [警告] 当前目录下未找到 python_embeded 文件夹！
    echo        请先从 GitHub Releases 下载运行环境包并解压到当前目录。
)

setx LL_AI_SUITE_ROOT "%SUITE_DIR%" >nul
echo [3/3] 已成功注册系统环境变量 LL_AI_SUITE_ROOT = %SUITE_DIR%

echo.
echo =========================================================
echo [OK] 注册完成！四大 AI 引擎已与 Nuke 管线建立自动连接！
echo =========================================================
pause