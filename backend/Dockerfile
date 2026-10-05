# 智能扫地机器人客服 Agent —— 后端镜像
FROM python:3.11-slim

# 基础环境：中文编码 + 不生成 pyc + 不缓冲日志
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

# 系统依赖：gcc/g++ 供 chromadb 等包编译，curl 供健康检查
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/requirements.txt

# 再拷代码（app 包位于 /app/app）
COPY . /app

# 知识库与向量库目录（会被 compose 的 volume 覆盖）
RUN mkdir -p /app/data/raw /app/data/processed /app/vectorstore

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
