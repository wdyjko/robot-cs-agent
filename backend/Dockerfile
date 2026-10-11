# 智能扫地机器人客服 Agent —— 后端镜像
FROM python:3.11-slim

# 国内镜像源（构建加速）。
# 默认用阿里云「公网」地址，任何环境都能用；阿里云 ECS 内网可换成：
#   --build-arg PIP_INDEX_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
#   --build-arg PIP_TRUSTED_HOST=mirrors.cloud.aliyuncs.com
# 注意：mirrors.cloud.aliyuncs.com 只提供 HTTP，写成 https 会因证书主机名不匹配而失败。
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com
ARG TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cpu

# 基础环境：中文编码 + 不生成 pyc + 不缓冲日志
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

# 说明：这里刻意不安装 build-essential / gcc。
# chromadb 的原生依赖（chroma-hnswlib）以及 numpy / torch 在 amd64 与 arm64 上
# 都有现成的 manylinux wheel，pip 不会走源码编译，省掉约 88 MB 的编译工具链。
# 如果你的平台确实缺 wheel（构建时报 "gcc: not found"），把下面两行放回来即可：
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#       && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt /app/requirements.txt
# torch 走 CPU 专用源（体积远小于默认 CUDA 版）；镜像源不可用时回退官方源
RUN pip install --upgrade pip \
    && (pip install torch --index-url "${TORCH_INDEX_URL}" || pip install torch) \
    && pip install -r /app/requirements.txt

# 再拷代码（app 包位于 /app/app）
COPY . /app

# 知识库与向量库目录（会被 compose 的 volume 覆盖）
RUN mkdir -p /app/data/raw /app/data/processed /app/vectorstore

EXPOSE 8000

# 健康检查用 Python 自带的 urllib，避免为了一次探活额外安装 curl。
# 探活失败时 urlopen 会抛异常并打印堆栈，这里把 stderr 静默掉（只看退出码）。
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)" 2>/dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
