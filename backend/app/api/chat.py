"""聊天接口：同步问答 + SSE 流式（先同步实现，流式按句推送）。"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..core.logging import get_logger
from ..models.schemas import ChatRequest, ChatResponse
from ..services import agent_service, memory_service

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = get_logger(__name__)


@router.post("", response_model=ChatResponse, summary="同步问答")
async def chat(req: ChatRequest) -> ChatResponse:
    """一次完整的客服问答：意图 → 工具 → 改写 → 检索 → 摘要 → 回答。"""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    try:
        return await agent_service.run(req)
    except Exception as exc:  # noqa: BLE001
        logger.exception("处理问答失败")
        raise HTTPException(status_code=500, detail=f"处理问答失败: {exc}") from exc


def _sse(event: str, data: dict) -> str:
    """打包成 SSE 帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream", summary="SSE 流式问答")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式问答。

    当前实现：先同步跑完整链路，再按句子切片推送（保证内容正确、前端体验接近流式）。
    后续要接真正的 token 级流式，只需把 answer_node 换成 `llm.astream`。
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    async def event_generator():
        yield _sse("start", {"session_id": req.session_id, "message": req.message})
        try:
            response = await agent_service.run(req)
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式问答失败")
            yield _sse("error", {"message": str(exc)})
            return

        # 按行/句切片推送，避免一次性把长文塞给前端
        buffer = ""
        for char in response.answer:
            buffer += char
            if char in "\n。！？!?；;":
                yield _sse("delta", {"text": buffer})
                buffer = ""
                await asyncio.sleep(0.01)
        if buffer:
            yield _sse("delta", {"text": buffer})

        yield _sse(
            "sources",
            {
                "sources": [item.model_dump() for item in response.sources],
                "tool_calls": [item.model_dump() for item in response.tool_calls],
            },
        )
        yield _sse("done", {"extra": response.extra})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history", summary="获取会话历史")
async def history(session_id: str = Query(default="default"), limit: int = Query(default=20, ge=1, le=200)) -> dict:
    """读取某个会话的对话历史。"""
    return {"ok": True, "session_id": session_id, "messages": memory_service.get_history(session_id, limit=limit)}


@router.delete("/history", summary="清空会话")
async def clear_history(session_id: str = Query(default="default")) -> dict:
    """清空某个会话的历史记录。"""
    deleted = memory_service.clear_history(session_id)
    return {"ok": True, "session_id": session_id, "deleted": deleted}
