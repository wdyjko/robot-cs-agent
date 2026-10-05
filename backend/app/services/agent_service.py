"""Agent 服务：用 LangGraph 编排「意图 → 工具 → 改写 → 检索 → 摘要 → 回答」。

图流程：
    intent_node → tool_node → rewrite_node → retrieve_node → summarize_node → answer_node

设计原则：
- 规则优先：能靠关键词判断的意图不调用 LLM，省时省钱；
- 检索优先：回答只基于知识库检索结果与工具结果，不编造；
- 全程兜底：没配 API Key、网络失败、模型异常都能返回可用答案。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..core.config import settings
from ..core.logging import get_logger
from ..core.prompts import (
    DATETIME_KEYWORDS,
    FINAL_ANSWER_PROMPT,
    INTENT_PROMPT,
    LOCATION_KEYWORDS,
    QUERY_REWRITE_PROMPT,
    SAFETY_NOTICE,
    WEATHER_KEYWORDS,
)
from ..models.schemas import ChatRequest, ChatResponse, SourceItem, ToolCallItem
from ..tools.datetime_tool import get_current_datetime
from ..tools.kb_tool import search_knowledge_base
from ..tools.location import get_user_location
from ..tools.profile import get_user_profile
from ..tools.weather import build_env_advice, get_weather
from .llm_service import ainvoke_json, ainvoke_text, is_llm_available
from . import memory_service
from .summary_service import detect_safety, dumps as dump_summary, summarize, summary_to_text

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """Agent 在图中流转的状态。"""

    question: str
    session_id: str
    user_id: str
    location: dict
    datetime_info: dict
    weather: dict
    user_profile: dict
    retrieved_docs: list
    knowledge_summary: str
    tool_calls: list
    answer: str
    sources: list
    # 内部辅助字段
    client_ip: str
    intent: dict
    rewritten_query: str
    history: str
    summary_dict: dict
    safety: bool


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _rule_intent(question: str) -> dict[str, Any]:
    """规则命中的意图判断。"""
    text = question or ""
    need_location = any(keyword in text for keyword in LOCATION_KEYWORDS)
    need_datetime = any(keyword in text for keyword in DATETIME_KEYWORDS)
    need_weather = any(keyword in text for keyword in WEATHER_KEYWORDS)
    hit = need_location or need_datetime or need_weather
    return {
        "need_location": need_location,
        "need_datetime": need_datetime,
        "need_weather": need_weather,
        "reason": "规则命中" if hit else "规则未命中",
        "source": "rule" if hit else "none",
    }


def _format_environment(state: AgentState) -> str:
    """把位置 / 时间 / 天气 / 画像拼成 Prompt 里的实时环境段落。"""
    lines: list[str] = []

    location = state.get("location") or {}
    if location.get("city"):
        lines.append(f"- 用户所在城市：{location.get('city')}（来源：{location.get('source', '未知')}）")

    datetime_info = state.get("datetime_info") or {}
    if datetime_info.get("date"):
        lines.append(
            f"- 当前时间：{datetime_info.get('date')} {datetime_info.get('time', '')} "
            f"{datetime_info.get('weekday', '')}，季节：{datetime_info.get('season', '')}"
        )
        for hint in (datetime_info.get("hints") or [])[:2]:
            lines.append(f"  · 季节提示：{hint}")

    weather = state.get("weather") or {}
    if weather.get("ok") is False:
        lines.append(f"- 天气：获取失败（{weather.get('error', '未知原因')}）")
    elif weather.get("temperature") is not None:
        lines.append(
            f"- 天气：{weather.get('description', '')}，气温 {weather.get('temperature')}℃"
            f"（{weather.get('temp_min')}~{weather.get('temp_max')}℃），"
            f"湿度 {weather.get('humidity')}%，降水 {weather.get('precipitation')}mm，风力 {weather.get('wind', '')}"
            f"（数据来源：{weather.get('source', 'mock')}）"
        )
        for advice in (weather.get("env_advice") or [])[:5]:
            lines.append(f"  · 环境适配：{advice}")

    profile = state.get("user_profile") or {}
    if profile:
        traits = []
        if profile.get("model"):
            traits.append(f"机型 {profile['model']}")
        if profile.get("floor_type"):
            traits.append(f"地板 {profile['floor_type']}")
        traits.append("有宠物" if profile.get("has_pet") else "无宠物")
        traits.append("有婴儿" if profile.get("has_baby") else "无婴儿")
        lines.append("- 用户特征：" + "，".join(traits))

    return "\n".join(lines) if lines else "（无实时环境信息）"


def _format_profile(profile: dict) -> str:
    if not profile:
        return "（无画像）"
    return (
        f"城市：{profile.get('city') or '未填'}；机型：{profile.get('model') or '未填'}；"
        f"地板：{profile.get('floor_type') or '未填'}；宠物：{'有' if profile.get('has_pet') else '无'}；"
        f"婴儿：{'有' if profile.get('has_baby') else '无'}"
    )


def _make_sources(docs: list[dict], summary: dict | None = None) -> list[SourceItem]:
    """把检索结果转成结构化来源列表。"""
    sources: list[SourceItem] = []
    for doc in docs or []:
        meta = doc.get("metadata", {}) or {}
        sources.append(
            SourceItem(
                file=str(meta.get("source_file", "")),
                item_no=str(meta.get("item_no", "")),
                title=str(meta.get("title", "")),
                category=str(meta.get("category", "")),
                snippet=str(doc.get("content", ""))[:200],
                score=float(doc.get("score", 0.0) or 0.0),
            )
        )
    if sources:
        return sources[:6]

    # 检索为空但摘要里有来源描述时，退化为文本来源
    for item in (summary or {}).get("sources", []) or []:
        text = str(item)
        file_name, _, rest = text.partition(" - ")
        sources.append(SourceItem(file=file_name.strip(), title=rest.strip(), item_no="", snippet=""))
    return sources[:6]


def _fallback_answer(state: AgentState) -> str:
    """无 LLM 时的兜底回答：只呈现知识库事实，不做推理。"""
    summary = state.get("summary_dict") or {}
    question = state.get("question", "")
    parts: list[str] = []

    if state.get("safety"):
        parts.append(SAFETY_NOTICE)
        parts.append("")

    parts.append(f"关于「{question}」，知识库检索结果如下：")
    parts.append("")
    parts.append(f"**结论**：{summary.get('summary') or '知识库中未检索到相关内容，建议咨询品牌官方售后。'}")

    steps = summary.get("steps") or []
    if steps:
        parts.append("")
        parts.append("**可执行步骤 / 要点**：")
        parts.extend(f"- {step}" for step in steps)

    # 安全提示统一在开头给出，这里过滤掉重复项
    warnings = [w for w in (summary.get("warnings") or []) if SAFETY_NOTICE not in str(w)]
    if warnings:
        parts.append("")
        parts.append("**注意事项**：")
        parts.extend(f"- {warning}" for warning in warnings)

    env_text = _format_environment(state)
    if env_text and env_text != "（无实时环境信息）":
        parts.append("")
        parts.append("**结合你当前环境**：")
        parts.extend(f"- {line.strip('- ')}" for line in env_text.split("\n") if line.strip().startswith("-"))

    sources = state.get("sources") or []
    if sources:
        parts.append("")
        parts.append("**参考来源**：")
        for index, source in enumerate(sources, start=1):
            label = f"{source.file or '知识库'} - {source.item_no}. {source.title}".strip(" -")
            parts.append(f"[{index}] {label}")

    parts.append("")
    if is_llm_available():
        # 已配置 Key 但本次调用失败/超时（网络、额度、限流等），不能误报成「未配置」
        parts.append(
            "> 提示：本次 LLM 调用失败或超时（可能是网络、额度或限流问题），以上为知识库检索原文的整理结果。"
            "可稍后重试，或在 `.env` 中调小 `LLM_TIMEOUT`／`LLM_MAX_RETRIES` 以更快回退。"
        )
    else:
        parts.append(
            "> 提示：当前未配置 LLM API Key，以上为知识库检索原文的整理结果。"
            "配置 `.env` 中的 `OPENAI_API_KEY` 后可获得更自然的总结回答。"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
async def intent_node(state: AgentState) -> dict[str, Any]:
    """判断是否需要调用位置 / 时间 / 天气工具（规则优先，不确定再问 LLM）。"""
    question = state.get("question", "")
    intent = _rule_intent(question)

    if intent["source"] == "none" and is_llm_available():
        result = await ainvoke_json(INTENT_PROMPT.format(question=question))
        if result:
            intent = {
                "need_location": bool(result.get("need_location")),
                "need_datetime": bool(result.get("need_datetime")),
                "need_weather": bool(result.get("need_weather")),
                "reason": str(result.get("reason", "")),
                "source": "llm",
            }
        else:
            intent["reason"] = "规则未命中且 LLM 判断失败，按保守策略处理"

    # 需要天气必然需要城市；需要天气通常也需要时间（判断季节/梅雨）
    if intent.get("need_weather"):
        intent["need_location"] = True
        intent["need_datetime"] = True

    logger.info("意图判断: %s", intent)
    return {"intent": intent}


async def tool_node(state: AgentState) -> dict[str, Any]:
    """按需调用工具，记录 tool_calls。"""
    intent = state.get("intent") or {}
    user_id = state.get("user_id", "u1")
    tool_calls: list[dict[str, Any]] = []
    updates: dict[str, Any] = {}

    # 0) 用户画像：任何问题都可以用来做个性化（优先用前端传入的）
    profile_from_request = state.get("user_profile") or {}
    if profile_from_request.get("user_id"):
        profile = profile_from_request
    else:
        profile = get_user_profile.invoke({"user_id": user_id})
    tool_calls.append(
        ToolCallItem(
            name="get_user_profile",
            arguments={"user_id": user_id},
            result={key: value for key, value in profile.items() if key != "ok"},
            ok=bool(profile.get("ok", True)),
        ).model_dump()
    )
    updates["user_profile"] = profile

    # 1) 位置
    manual_city = ((state.get("location") or {}).get("city")) or None
    if intent.get("need_location") or manual_city:
        location = get_user_location.invoke(
            {
                "user_id": user_id,
                "client_ip": state.get("client_ip") or None,
                "manual_city": manual_city,
            }
        )
        tool_calls.append(
            ToolCallItem(name="get_user_location", arguments={"user_id": user_id, "manual_city": manual_city}, result=location).model_dump()
        )
        updates["location"] = location

    # 2) 时间
    if intent.get("need_datetime"):
        datetime_info = get_current_datetime.invoke({})
        tool_calls.append(ToolCallItem(name="get_current_datetime", arguments={}, result=datetime_info).model_dump())
        updates["datetime_info"] = datetime_info

    # 3) 天气
    if intent.get("need_weather"):
        city = (updates.get("location") or state.get("location") or {}).get("city") or settings.default_city
        weather = get_weather.invoke({"city": city, "date": "today"})
        # 结合画像细化环境建议（地板/宠物/婴儿）
        if weather.get("ok"):
            merged_advice = list(weather.get("env_advice") or [])
            for advice in build_env_advice(weather, profile):
                if advice not in merged_advice:
                    merged_advice.append(advice)
            weather["env_advice"] = merged_advice
        tool_calls.append(ToolCallItem(name="get_weather", arguments={"city": city, "date": "today"}, result=weather).model_dump())
        updates["weather"] = weather

    updates["tool_calls"] = tool_calls
    return updates


async def rewrite_node(state: AgentState) -> dict[str, Any]:
    """结合历史改写查询，提高检索命中率。"""
    question = state.get("question", "")
    history = state.get("history") or "（无历史对话）"
    rewritten = question

    if is_llm_available():
        text = await ainvoke_text(QUERY_REWRITE_PROMPT.format(history=history, question=question))
        if text:
            # 只取第一行，去掉可能的引号
            candidate = text.splitlines()[0].strip().strip("\"'“”")
            if 2 <= len(candidate) <= 200:
                rewritten = candidate

    # 命中环境类问题时，把环境要素补进检索词，帮助召回「环境适配」条目
    intent = state.get("intent") or {}
    if intent.get("need_weather"):
        extras = []
        weather = state.get("weather") or {}
        if weather.get("humidity") and float(weather["humidity"]) > 80:
            extras.append("潮湿 湿度高 拖地")
        if weather.get("temperature") is not None:
            temperature = float(weather["temperature"])
            if temperature > 35:
                extras.append("高温 充电")
            elif temperature < 5:
                extras.append("低温 开机 电池")
        if extras:
            rewritten = f"{rewritten} {' '.join(extras)}"

    logger.info("查询改写: %s -> %s", question, rewritten)
    return {"rewritten_query": rewritten}


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """检索知识库。"""
    query = state.get("rewritten_query") or state.get("question", "")
    result = search_knowledge_base.invoke({"query": query, "k": settings.final_top_k})
    docs = result.get("results", []) if isinstance(result, dict) else []

    # 改写后检索为空时，用原问题再试一次
    if not docs and query != state.get("question"):
        fallback = search_knowledge_base.invoke({"query": state.get("question", ""), "k": settings.final_top_k})
        docs = fallback.get("results", []) if isinstance(fallback, dict) else []

    logger.info("检索命中 %d 条", len(docs))
    return {"retrieved_docs": docs}


async def summarize_node(state: AgentState) -> dict[str, Any]:
    """把检索片段压缩成结构化 JSON。"""
    docs = state.get("retrieved_docs") or []
    question = state.get("question", "")
    summary = await summarize(question, docs)

    safety = detect_safety(question, docs)
    if safety and not summary.get("warnings"):
        summary["warnings"] = [SAFETY_NOTICE]

    return {
        "summary_dict": summary,
        "knowledge_summary": dump_summary(summary),
        "safety": safety,
    }


async def answer_node(state: AgentState) -> dict[str, Any]:
    """组装最终 Prompt 并生成回答。"""
    question = state.get("question", "")
    summary = state.get("summary_dict") or {}
    docs = state.get("retrieved_docs") or []
    sources = _make_sources(docs, summary)
    safety = bool(state.get("safety"))

    answer: str | None = None
    if is_llm_available():
        prompt = FINAL_ANSWER_PROMPT.format(
            question=question,
            history=state.get("history") or "（无历史对话）",
            user_profile=_format_profile(state.get("user_profile") or {}),
            environment=_format_environment(state),
            knowledge_summary=summary_to_text(summary),
            sources="；".join(f"{source.file} - {source.item_no}. {source.title}".strip(" -") for source in sources) or "（无）",
        )
        answer = await ainvoke_text(prompt)

    if not answer:
        answer = _fallback_answer({**state, "sources": sources})

    # 安全兜底：无论模型怎么写，风险场景都必须出现在回答里
    if safety and "立即停止使用" not in answer and "停用" not in answer:
        answer = f"{SAFETY_NOTICE}\n\n{answer}"

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# 构图
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_graph():
    """构建并编译 LangGraph 状态图（单例）。"""
    graph = StateGraph(AgentState)
    graph.add_node("intent_node", intent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("answer_node", answer_node)

    graph.set_entry_point("intent_node")
    graph.add_edge("intent_node", "tool_node")
    graph.add_edge("tool_node", "rewrite_node")
    graph.add_edge("rewrite_node", "retrieve_node")
    graph.add_edge("retrieve_node", "summarize_node")
    graph.add_edge("summarize_node", "answer_node")
    graph.add_edge("answer_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
async def run(req: ChatRequest) -> ChatResponse:
    """执行一次完整的客服问答。"""
    session_id = req.session_id or "default"
    user_id = req.user_id or "u1"

    # 1) 落库用户消息
    memory_service.save_message(session_id, "user", req.message)

    # 2) 前端传入画像时先写回数据库
    if req.user_profile:
        profile_payload = dict(req.user_profile)
        profile_payload.setdefault("user_id", user_id)
        memory_service.upsert_user_profile(user_id, profile_payload)

    # 3) 准备初始状态
    history = memory_service.get_history_text(session_id, limit=6)
    initial_state: AgentState = {
        "question": req.message,
        "session_id": session_id,
        "user_id": user_id,
        "location": dict(req.location or {}),
        "user_profile": dict(req.user_profile or {}),
        "history": history,
        "tool_calls": [],
        "retrieved_docs": [],
        "sources": [],
        "client_ip": req.client_ip or "",
    }

    try:
        final_state = await get_graph().ainvoke(initial_state)
    except Exception as exc:  # noqa: BLE001 - 图执行失败也要给出可读回答
        logger.exception("Agent 图执行失败")
        final_state = {
            **initial_state,
            "answer": f"抱歉，处理你的问题时出现了内部错误：{exc}。请稍后重试，或直接联系品牌官方售后。",
            "sources": [],
            "tool_calls": [],
        }

    answer = final_state.get("answer") or "抱歉，暂时无法生成回答，请稍后重试。"
    sources = final_state.get("sources") or []
    tool_calls = final_state.get("tool_calls") or []

    # 4) 落库助手回复
    memory_service.save_message(session_id, "assistant", answer)

    # 5) 组装响应
    source_items = [item if isinstance(item, SourceItem) else SourceItem(**item) for item in sources]
    tool_items = [item if isinstance(item, ToolCallItem) else ToolCallItem(**item) for item in tool_calls]

    return ChatResponse(
        answer=answer,
        sources=source_items,
        tool_calls=tool_items,
        extra={
            "rewritten_query": final_state.get("rewritten_query", ""),
            "intent": final_state.get("intent", {}),
            "environment": _format_environment(final_state),
            "summary": final_state.get("summary_dict", {}),
            "llm_enabled": is_llm_available(),
        },
    )


async def run_simple(question: str, session_id: str = "default", user_id: str = "u1") -> ChatResponse:
    """便于脚本 / 测试使用的简化入口。"""
    return await run(ChatRequest(session_id=session_id, message=question, user_id=user_id))
