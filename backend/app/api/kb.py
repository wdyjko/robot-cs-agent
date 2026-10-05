"""知识库接口：重建与状态查询。"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..core.logging import get_logger
from ..models.schemas import KBStatus, KBuildResponse
from ..services import rag_service

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])
logger = get_logger(__name__)


@router.post("/rebuild", response_model=KBuildResponse, summary="重建向量知识库")
async def rebuild() -> KBuildResponse:
    """重新加载 data/raw 下的知识库文件并重建向量库（耗时较长，放线程池执行）。"""
    started = time.time()
    try:
        count = await run_in_threadpool(rag_service.build_vectorstore)
    except Exception as exc:  # noqa: BLE001
        logger.exception("重建知识库失败")
        raise HTTPException(status_code=500, detail=f"重建知识库失败: {exc}") from exc

    elapsed = round(time.time() - started, 2)
    status = await run_in_threadpool(rag_service.get_status)
    return KBuildResponse(
        ok=count > 0,
        collection=status["collection"],
        document_count=count,
        persist_dir=status["persist_dir"],
        message=f"重建完成，共 {count} 条知识条目，耗时 {elapsed}s" if count else "未加载到任何知识库内容，请检查 data/raw 目录",
        elapsed_seconds=elapsed,
    )


@router.get("/status", response_model=KBStatus, summary="知识库状态")
async def status() -> KBStatus:
    """返回集合名称、条目数、持久化目录等信息。"""
    try:
        data = await run_in_threadpool(rag_service.get_status)
        return KBStatus(**data)
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取知识库状态失败")
        raise HTTPException(status_code=500, detail=f"读取知识库状态失败: {exc}") from exc
