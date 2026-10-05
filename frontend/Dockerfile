# 智能扫地机器人客服 Agent —— 前端镜像（Streamlit）
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

RUN pip install --upgrade pip && pip install streamlit requests

COPY streamlit_app.py /app/streamlit_app.py

# Streamlit 配置：关闭使用统计、允许容器外访问
RUN mkdir -p /root/.streamlit \
    && printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\n\n[browser]\ngatherUsageStats = false\n' \
       > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
