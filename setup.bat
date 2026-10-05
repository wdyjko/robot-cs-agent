@echo off
REM ===========================================================================
REM  智能扫地机器人客服 Agent —— Windows 一键环境安装脚本
REM
REM  作用：
REM    1) 在项目根目录创建 .venv 虚拟环境（所有依赖只装进 .venv，不污染全局）
REM    2) 安装 CPU 版 torch（体积更小，默认使用 CPU 推理）
REM    3) 安装 backend/requirements.txt 全部依赖
REM    4) 若 .env 不存在，则从 .env.example 复制一份
REM
REM  用法：
REM    setup.bat               仅安装环境
REM    setup.bat build-kb      安装环境后顺便构建知识库
REM    setup.bat --skip-torch  跳过 torch（已装过可跳过，节省时间）
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set BUILD_KB=0
set SKIP_TORCH=0
for %%A in (%*) do (
    if /I "%%A"=="build-kb" set BUILD_KB=1
    if /I "%%A"=="--build-kb" set BUILD_KB=1
    if /I "%%A"=="--skip-torch" set SKIP_TORCH=1
)

echo ============================================================
echo   智能扫地机器人客服 Agent - 环境安装
echo ============================================================
echo.

REM ---------- 1. 检查 Python ----------
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python 命令，请先安装 Python 3.10 - 3.12 并加入 PATH。
    echo        下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "delims=" %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo [1/5] 检测到 !PYVER!

python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [警告] 建议使用 Python 3.10 - 3.13，当前版本可能缺少部分依赖的预编译包。
)

REM ---------- 2. 创建虚拟环境 ----------
if exist ".venv\Scripts\python.exe" (
    echo [2/5] .venv 已存在，跳过创建
) else (
    echo [2/5] 正在创建虚拟环境 .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败。
        echo        若提示 ensurepip 失败，请确认 %%TEMP%% 目录可写，或改用管理员权限重试。
        pause
        exit /b 1
    )
)

set VPY=.venv\Scripts\python.exe
if not exist "%VPY%" (
    echo [错误] 未找到 %VPY%，虚拟环境创建异常。
    pause
    exit /b 1
)
echo       虚拟环境: %CD%\.venv

REM ---------- 3. 升级 pip ----------
echo [3/5] 升级 pip / setuptools / wheel ...
"%VPY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [错误] pip 升级失败，请检查网络。
    pause
    exit /b 1
)

REM ---------- 4. 安装依赖 ----------
if "%SKIP_TORCH%"=="0" (
    echo [4/5] 安装 CPU 版 torch（约 200MB，首次较慢）...
    "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [警告] CPU 版 torch 安装失败，回退到默认源再试一次...
        "%VPY%" -m pip install torch
    )
) else (
    echo [4/5] 按参数要求跳过 torch 安装
)

echo       安装 backend\requirements.txt ...
"%VPY%" -m pip install -r backend\requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试。
    pause
    exit /b 1
)

REM ---------- 5. 生成 .env ----------
if exist ".env" (
    echo [5/5] .env 已存在，保留原有配置
) else (
    copy /Y ".env.example" ".env" >nul
    echo [5/5] 已生成 .env（请打开填写 OPENAI_API_KEY 等信息）
)

REM ---------- 可选：构建知识库 ----------
if "%BUILD_KB%"=="1" (
    echo.
    echo 正在构建知识库 ...
    "%VPY%" scripts\build_kb.py
)

echo.
echo ============================================================
echo   安装完成！
echo ============================================================
echo.
echo 下一步：
echo   1) 编辑 .env，填写 OPENAI_API_KEY（不填也能跑，只是没有 LLM 总结）
echo   2) 构建知识库：
echo        "%VPY%" scripts\build_kb.py
echo   3) 启动后端：
echo        "%VPY%" -m uvicorn backend.app.main:app --reload --port 8000
echo   4) 启动前端（另开一个命令行）：
echo        "%VPY%" -m streamlit run frontend\streamlit_app.py
echo.
pause
