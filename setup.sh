#!/usr/bin/env bash
# ===========================================================================
#  智能扫地机器人客服 Agent —— macOS / Linux 一键环境安装脚本
#
#  作用：
#    1) 在项目根目录创建 .venv 虚拟环境（所有依赖只装进 .venv，不污染全局）
#    2) 安装 CPU 版 torch（体积更小，默认使用 CPU 推理）
#    3) 安装 backend/requirements.txt 全部依赖
#    4) 若 .env 不存在，则从 .env.example 复制一份
#
#  用法：
#    bash setup.sh               仅安装环境
#    bash setup.sh --build-kb    安装环境后顺便构建知识库
#    bash setup.sh --skip-torch  跳过 torch（已装过可跳过，节省时间）
# ===========================================================================

set -euo pipefail

cd "$(dirname "$0")"

BUILD_KB=0
SKIP_TORCH=0
for arg in "$@"; do
  case "$arg" in
    build-kb|--build-kb) BUILD_KB=1 ;;
    --skip-torch) SKIP_TORCH=1 ;;
    *) echo "[警告] 未知参数：$arg" ;;
  esac
done

echo "============================================================"
echo "  智能扫地机器人客服 Agent - 环境安装"
echo "============================================================"
echo

# ---------- 1. 检查 Python ----------
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "[错误] 未找到 python3，请先安装 Python 3.10 - 3.13。"
  exit 1
fi
echo "[1/5] 使用解释器：$($PYTHON_BIN --version 2>&1) ($(command -v "$PYTHON_BIN"))"

# ---------- 2. 创建虚拟环境 ----------
if [ -x ".venv/bin/python" ]; then
  echo "[2/5] .venv 已存在，跳过创建"
else
  echo "[2/5] 正在创建虚拟环境 .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

VPY=".venv/bin/python"
if [ ! -x "$VPY" ]; then
  echo "[错误] 未找到 $VPY，虚拟环境创建异常。"
  exit 1
fi
echo "      虚拟环境：$(pwd)/.venv"

# ---------- 3. 升级 pip ----------
echo "[3/5] 升级 pip / setuptools / wheel ..."
"$VPY" -m pip install --upgrade pip setuptools wheel

# ---------- 4. 安装依赖 ----------
if [ "$SKIP_TORCH" -eq 0 ]; then
  echo "[4/5] 安装 CPU 版 torch（约 200MB，首次较慢）..."
  "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    || "$VPY" -m pip install torch
else
  echo "[4/5] 按参数要求跳过 torch 安装"
fi

echo "      安装 backend/requirements.txt ..."
"$VPY" -m pip install -r backend/requirements.txt

# ---------- 5. 生成 .env ----------
if [ -f ".env" ]; then
  echo "[5/5] .env 已存在，保留原有配置"
else
  cp .env.example .env
  echo "[5/5] 已生成 .env（请填写 OPENAI_API_KEY 等信息）"
fi

# ---------- 可选：构建知识库 ----------
if [ "$BUILD_KB" -eq 1 ]; then
  echo
  echo "正在构建知识库 ..."
  "$VPY" scripts/build_kb.py
fi

echo
echo "============================================================"
echo "  安装完成！"
echo "============================================================"
echo
echo "下一步："
echo "  1) 编辑 .env，填写 OPENAI_API_KEY（不填也能跑，只是没有 LLM 总结）"
echo "  2) 构建知识库：  $VPY scripts/build_kb.py"
echo "  3) 启动后端：    $VPY -m uvicorn backend.app.main:app --reload --port 8000"
echo "  4) 启动前端：    $VPY -m streamlit run frontend/streamlit_app.py"
echo
