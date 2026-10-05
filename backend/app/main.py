"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, kb, tools
from .core.config import settings
from .core.logging import get_logger, setup_logging
from .models.db import init_db
from .models.schemas import HealthResponse
from .services.llm_service import is_llm_available
from .services.rag_service import get_status as get_kb_status

setup_logging(level=settings.log_level, log_file=settings.chroma_persist_path.parent / "app.log")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 关闭钩子。"""
    logger.info("=" * 68)
    logger.info("%s v%s 启动中……", settings.app_name, settings.app_version)
    init_db()
    settings.ensure_dirs()

    if not settings.llm_enabled:
        logger.warning("未配置 OPENAI_API_KEY：将只返回知识库检索结果（可在 .env 中配置后重启）")
    if not settings.weather_api_enabled:
        logger.warning("未配置 WEATHER_API_KEY：天气将使用 mock 数据")

    try:
        status = get_kb_status()
        if status.get("ready"):
            logger.info("知识库已就绪：collection=%s，条目数=%s", status["collection"], status["document_count"])
        else:
            logger.warning("知识库为空，请先执行：python scripts/build_kb.py")
    except Exception as exc:  # noqa: BLE001
        logger.warning("知识库状态检查失败: %s", exc)

    logger.info("API 文档: http://localhost:%s/docs", settings.api_port)
    logger.info("=" * 68)
    yield
    logger.info("%s 已关闭", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于 LangGraph + Chroma 的扫地/扫拖一体机器人智能客服 Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(kb.router)
app.include_router(tools.router)


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="健康检查")
async def health() -> HealthResponse:
    """健康检查：同时反映 LLM 与知识库是否可用。"""
    try:
        kb_ready = bool(get_kb_status().get("ready"))
    except Exception:  # noqa: BLE001
        kb_ready = False
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        llm_enabled=is_llm_available(),
        kb_ready=kb_ready,
    )


@app.get("/", tags=["system"], summary="服务信息")
async def root() -> dict:
    """返回服务基础信息与主要接口清单。"""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": [
            "POST /api/chat",
            "POST /api/chat/stream",
            "GET  /api/chat/history",
            "DELETE /api/chat/history",
            "POST /api/kb/rebuild",
            "GET  /api/kb/status",
            "GET  /api/weather",
            "GET  /api/location",
            "GET  /api/datetime",
            "GET  /api/profile",
            "POST /api/profile",
            "GET  /health",
        ],
    }
