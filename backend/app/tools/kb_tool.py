"""知识库检索工具：封装 rag_service.search，供 LangGraph 节点调用。"""

from __future__ import annotations

from langchain_core.tools import tool

from ..core.config import settings
from ..core.logging import get_logger
from ._common import fail, ok

logger = get_logger(__name__)


@tool
def search_knowledge_base(query: str, k: int = 5) -> dict:
    """在扫地机器人客服知识库中检索，返回最相关的知识条目（含来源元数据）。

    Args:
        query: 检索查询（建议为自包含的完整问题）。
        k: 返回条数，默认 5。
    """
    try:
        from ..services.rag_service import search as _search

        top_k = int(k) if k else settings.final_top_k
        results = _search(query, k=top_k)
        if not results:
            logger.warning("知识库检索结果为空，query=%s", query)
        return ok(query=query, count=len(results), results=results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("知识库检索异常")
        return fail(f"知识库检索失败: {exc}", query=query, count=0, results=[])
