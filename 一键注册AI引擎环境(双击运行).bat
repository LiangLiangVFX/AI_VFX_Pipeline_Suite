@echo off
chcp 65001 >nul
title AI_VFX_Pipeline_Suite 环境注册工具
cd /d "%~dp0"
set "SUITE_DIR=%~dp0"
:: 去掉末尾反斜杠
if "%SUITE_DIR:~-1%"=="\" set "SUITE_DIR=%SUITE_DIR:~0,-1%"

echo =========================================================
echo     正在注册 AI_VFX_Pipeline_Suite 到当前工作站...
echo =========================================================
echo.

:: 1. 写入相对路径到 python_embeded，确保换任何电脑/盘符都能找到 matanyone2 模块
if exist "python_embeded\Lib\site-packages" (
    echo ../../../matanyone2> "python_embeded\Lib\site-packages\matanyone2.pth"
    echo [1/2] 已绑定便携 Python 相对模块路径: ../../../matanyone2
) else (
    echo [警告] 当前目录下未找到 python_embeded 文件夹！
    echo        请先从 GitHub Releases 下载运行环境包并解压到当前目录。
)

:: 2. 写入系统用户环境变量 LL_AI_SUITE_ROOT，让 VFX_Pipeline_CamSync (Nuke) 100%% 自动定位此目录
setx LL_AI_SUITE_ROOT "%SUITE_DIR%" >nul
echo [2/2] 已成功注册环境变量 LL_AI_SUITE_ROOT = %SUITE_DIR%

echo.
echo =========================================================
echo [OK] 注册完成！现在 Nuke 管线已与当前 AI 引擎建立自动连接！
echo =========================================================
pause