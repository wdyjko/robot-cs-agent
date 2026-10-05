"""LLM 工厂与调用封装。

设计要点：
- 通过 OpenAI 兼容接口调用（OpenAI / DeepSeek / Qwen / GLM 均可，改 .env 即可切换）。
- 未配置 API Key 时不报错，返回 None，业务层自动走「纯知识库检索」兜底。
- 所有调用都带异常兜底与日志，不让 LLM 抖动影响主流程。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@lru_cache(maxsize=1)
def get_llm():
    """返回单例 ChatOpenAI；未配置 Key 时返回 None。"""
    if not settings.llm_enabled:
        logger.warning("未配置 OPENAI_API_KEY，LLM 功能已禁用，将只返回知识库检索结果。")
        return None
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        logger.info("LLM 已就绪: model=%s base_url=%s", settings.llm_model, settings.openai_base_url)
        return llm
    except Exception as exc:  # noqa: BLE001
        logger.error("初始化 LLM 失败: %s", exc)
        return None


def is_llm_available() -> bool:
    return get_llm() is not None


async def ainvoke_text(prompt: str) -> str | None:
    """异步调用 LLM，返回纯文本；失败返回 None。"""
    llm = get_llm()
    if llm is None:
        return None
    try:
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):  # 部分模型返回分段内容
            content = "".join(str(part) for part in content)
        text = str(content).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM 调用失败: %s", exc)
        return None


def invoke_text(prompt: str) -> str | None:
    """同步调用 LLM（供脚本 / 同步工具使用）。"""
    llm = get_llm()
    if llm is None:
        return None
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        text = str(content).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM 调用失败: %s", exc)
        return None


def extract_json(text: str | None) -> dict[str, Any] | None:
    """从模型输出里稳健地抠出 JSON 对象。"""
    if not text:
        return None

    candidates: list[str] = []
    block = _JSON_BLOCK_RE.search(text)
    if block:
        candidates.append(block.group(1))
    candidates.append(text)

    # 再尝试首尾大括号切片
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    logger.warning("无法从模型输出中解析 JSON: %s", (text or "")[:200])
    return None


async def ainvoke_json(prompt: str) -> dict[str, Any] | None:
    """异步调用 LLM 并解析为 JSON。"""
    return extract_json(await ainvoke_text(prompt))
