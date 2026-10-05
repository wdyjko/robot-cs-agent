"""摘要服务：把检索片段压缩成结构化 JSON（summary / steps / warnings / sources）。"""

from __future__ import annotations

import json
import re
from typing import Any

from ..core.logging import get_logger
from ..core.prompts import SAFETY_KEYWORDS, SAFETY_NOTICE, SUMMARY_PROMPT
from .llm_service import ainvoke_json, is_llm_available
from .rag_service import docs_to_context

logger = get_logger(__name__)


def detect_safety(question: str, docs: list[dict[str, Any]], doc_limit: int = 2) -> bool:
    """判断是否存在安全风险。

    只扫描**最相关的前 doc_limit 条**检索结果：如果扫描全部 Top5，很容易因为
    某条不相关的结果里出现「短路」二字，就给「开机无反应」这种普通问题误挂安全警告。
    """
    if any(keyword in (question or "") for keyword in SAFETY_KEYWORDS):
        return True

    for doc in (docs or [])[:doc_limit]:
        metadata = doc.get("metadata") or {}
        haystack = f"{doc.get('content', '')} {metadata.get('title', '')} {metadata.get('category', '')}"
        if any(keyword in haystack for keyword in SAFETY_KEYWORDS):
            return True
    return False


def _clean_points(content: str, title: str = "") -> list[str]:
    """把条目正文拆成可读要点。

    优先取 `- ` 开头的答案行（100问 类文件），其次按分号/句号拆分（故障排除类）。
    同时过滤掉「另一个问题」这种整句标题，避免在步骤里出现无关问题。
    """
    text = (content or "").strip()
    if not text:
        return []

    points: list[str] = []
    leftovers: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            candidate = line[2:].strip()
            if len(candidate) >= 6:
                points.append(candidate)
        else:
            leftovers.append(line)

    if not points and leftovers:
        merged = re.sub(r"^\s*\d{1,3}\s*[.、．)）]\s*", "", " ".join(leftovers))
        for segment in re.split(r"[；;]", merged):
            segment = segment.strip().strip("*").strip()
            if len(segment) >= 8:
                points.append(segment)

    # 去掉与条目标题雷同的内容（例如整句“XX怎么办？”）
    if title:
        cleaned_title = title.strip().strip("？?").replace("**", "")
        points = [point for point in points if cleaned_title not in point or len(point) > len(cleaned_title) + 8]

    # 去重且保持顺序
    unique: list[str] = []
    for point in points:
        if point not in unique:
            unique.append(point)
    return unique


def _fallback_summary(question: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """无 LLM 时的兜底摘要：不编造，只把检索原文要点整理出来。"""
    if not docs:
        return {
            "summary": "知识库中未检索到与该问题相关的内容，建议咨询品牌官方售后。",
            "steps": [],
            "warnings": [],
            "sources": [],
        }

    sources: list[str] = []
    all_points: list[str] = []
    summary_parts: list[str] = []

    for index, doc in enumerate(docs[:5]):
        metadata = doc.get("metadata") or {}
        title = str(metadata.get("title") or "").strip().replace("**", "")
        label = f"{metadata.get('source_file', '')} - {metadata.get('item_no', '')}. {title}".strip(" -")
        if label:
            sources.append(label)

        points = _clean_points(str(doc.get("content", "")), title)
        for point in points:
            if point not in all_points:
                all_points.append(point)

        # 前两条用于结论，避免结论过长
        if index < 2:
            head = points[0] if points else ""
            if title and head and head != title:
                summary_parts.append(f"{title}：{head}")
            elif title:
                summary_parts.append(title)
            elif head:
                summary_parts.append(head)

    summary = "；".join(summary_parts)[:400] or "知识库中未检索到明确结论，建议参考下方来源或咨询官方售后。"
    return {
        "summary": f"（未启用 LLM，以下为知识库原文摘录）{summary}",
        "steps": all_points[:6],
        "warnings": [],
        "sources": sources,
    }


async def summarize(question: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """调用 LLM 生成结构化摘要；无 LLM 或失败时走兜底摘要。"""
    docs = docs or []
    if not docs:
        logger.warning("检索结果为空，跳过 LLM 摘要")
        return _fallback_summary(question, docs)

    if not is_llm_available():
        return _fallback_summary(question, docs)

    prompt = SUMMARY_PROMPT.format(question=question, context=docs_to_context(docs))
    result = await ainvoke_json(prompt)
    if not result:
        logger.warning("摘要 JSON 解析失败，使用兜底摘要")
        return _fallback_summary(question, docs)

    summary = str(result.get("summary", "")).strip()
    steps = result.get("steps") or []
    warnings = result.get("warnings") or []
    sources = result.get("sources") or []

    # 归一化，避免模型返回字符串而不是数组
    def _as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [line.strip(" -•\t") for line in value.split("\n") if line.strip()]
        return []

    steps = _as_list(steps)
    warnings = _as_list(warnings)
    sources = _as_list(sources)

    if not sources:
        sources = [
            f"{doc.get('metadata', {}).get('source_file', '')} - {doc.get('metadata', {}).get('item_no', '')}. "
            f"{doc.get('metadata', {}).get('title', '')}".strip(" -")
            for doc in docs[:5]
        ]

    if detect_safety(question, docs) and not any("停用" in warning or "售后" in warning for warning in warnings):
        warnings.insert(0, SAFETY_NOTICE)

    return {
        "summary": summary or _fallback_summary(question, docs)["summary"],
        "steps": steps,
        "warnings": warnings,
        "sources": [source for source in sources if source],
    }


def summary_to_text(summary: dict[str, Any]) -> str:
    """把摘要 dict 转成给最终 Prompt 用的文本。"""
    if not summary:
        return "（无摘要）"
    lines = [f"结论摘要：{summary.get('summary', '')}"]
    steps = summary.get("steps") or []
    if steps:
        lines.append("可执行步骤：")
        lines.extend(f"- {step}" for step in steps)
    warnings = summary.get("warnings") or []
    if warnings:
        lines.append("注意事项：")
        lines.extend(f"- {warning}" for warning in warnings)
    sources = summary.get("sources") or []
    if sources:
        lines.append("来源：" + "；".join(str(source) for source in sources))
    return "\n".join(lines)


def dumps(summary: dict[str, Any]) -> str:
    """JSON 字符串形式，便于塞进 AgentState。"""
    try:
        return json.dumps(summary, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover
        return "{}"
