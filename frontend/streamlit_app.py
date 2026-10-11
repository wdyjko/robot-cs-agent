"""智能扫地机器人客服 Agent —— Streamlit 前端。

启动：
    .venv/Scripts/python.exe -m streamlit run frontend/streamlit_app.py

界面说明：
- 侧边栏：用户画像（用户 ID / 城市 / 机型 / 地板 / 宠物 / 婴儿）、知识库状态与重建、会话管理、实时环境
- 主聊天区：逐步流式显示回答（SSE），并提供「工具调用」与「参考来源」折叠面板
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterator
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_TIMEOUT = 600

st.set_page_config(
    page_title="智能扫地机器人客服",
    page_icon=":material/robot_2:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CITY_OPTIONS = [
    "不指定", "杭州", "北京", "上海", "广州", "深圳", "成都", "重庆", "武汉", "西安",
    "南京", "苏州", "天津", "长沙", "青岛", "沈阳", "郑州", "昆明", "哈尔滨", "福州", "厦门",
]
FLOOR_OPTIONS = ["未指定", "木地板", "瓷砖", "大理石", "地毯", "混合地面"]
MODEL_OPTIONS = [
    "未指定", "科沃斯 T 系列", "科沃斯 X 系列", "石头 G 系列", "石头 P 系列",
    "云鲸 J 系列", "追觅 X 系列", "小米米家", "其他品牌",
]
EXAMPLE_QUESTIONS = [
    "机器人开机无反应怎么办？",
    "今天我这里湿度高，适合拖地吗？",
    "这个月怎么保养扫地机器人？",
    "宠物家庭选什么扫地机器人？",
    "滤网多久换一次？",
    "机器人不回充是什么原因？",
    "木地板拖地需要注意什么？",
    "电池鼓包了还能用吗？",
]


# ---------------------------------------------------------------------------
# 会话状态（统一初始化）
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", f"s-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("user_id", "u1")
    st.session_state.setdefault("backend_url", DEFAULT_BACKEND)
    st.session_state.setdefault("kb_status", None)
    st.session_state.setdefault("pending_question", None)


init_state()


# ---------------------------------------------------------------------------
# 后端调用
# ---------------------------------------------------------------------------
def _url(path: str) -> str:
    return st.session_state.backend_url.rstrip("/") + path


def _connection_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"无法连接后端 {st.session_state.backend_url}。"
            "请先启动后端：.venv/Scripts/python.exe -m uvicorn backend.app.main:app --reload --port 8000"
        )
    return f"请求失败：{exc}"


def api_get(path: str, params: dict[str, Any] | None = None) -> tuple[bool, dict | None, str]:
    try:
        response = requests.get(_url(path), params=params, timeout=30)
        response.raise_for_status()
        return True, response.json(), ""
    except Exception as exc:  # noqa: BLE001
        return False, None, _connection_error(exc)


def api_post(path: str, payload: dict[str, Any], timeout: int = CHAT_TIMEOUT) -> tuple[bool, dict | None, str]:
    try:
        response = requests.post(_url(path), json=payload, timeout=timeout)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", "")
            except Exception:  # noqa: BLE001
                detail = response.text[:200]
            return False, None, f"后端返回 {response.status_code}：{detail}"
        return True, response.json(), ""
    except Exception as exc:  # noqa: BLE001
        return False, None, _connection_error(exc)


def api_delete(path: str, params: dict[str, Any] | None = None) -> tuple[bool, dict | None, str]:
    try:
        response = requests.delete(_url(path), params=params, timeout=30)
        response.raise_for_status()
        return True, response.json(), ""
    except Exception as exc:  # noqa: BLE001
        return False, None, _connection_error(exc)


def stream_answer(payload: dict[str, Any], sink: dict[str, Any]) -> Iterator[str]:
    """调用 SSE 接口并逐段产出文本；来源与工具调用写入 sink。"""
    try:
        with requests.post(_url("/api/chat/stream"), json=payload, stream=True, timeout=CHAT_TIMEOUT) as response:
            if response.status_code >= 400:
                sink["error"] = f"后端返回 {response.status_code}"
                return
            event = ""
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("event:"):
                    event = raw_line[len("event:") :].strip()
                    continue
                if not raw_line.startswith("data:"):
                    continue

                try:
                    data = json.loads(raw_line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue

                if event == "delta":
                    yield data.get("text", "")
                elif event == "sources":
                    sink["sources"] = data.get("sources", [])
                    sink["tool_calls"] = data.get("tool_calls", [])
                elif event == "done":
                    sink["extra"] = data.get("extra", {})
                elif event == "error":
                    sink["error"] = data.get("message", "未知错误")
    except Exception as exc:  # noqa: BLE001
        sink["error"] = _connection_error(exc)


def refresh_kb_status() -> dict | None:
    ok, data, _ = api_get("/api/kb/status")
    if ok:
        st.session_state.kb_status = data
    return st.session_state.kb_status


# ---------------------------------------------------------------------------
# Markdown 兜底规范化
#
# 背景：模型常输出 `**结论：**知识库中……`（`**` 后面直接跟正文）。
# CommonMark 规定 `**` 作为右定界符时**后面紧跟非空白字符则不加粗**，
# 于是页面上会显示成字面星号。prompts.py 已在提示词里要求加空格，
# 这里再做一层防御，兼容任意模型 / 历史消息。
#
# 两种常见写法都要处理：
#   1) `**结论：**知识库……`  → 星号后缺空格
#   2) `**结论**：知识库……`  → 冒号在加粗外面，星号后紧跟冒号，同样不加粗
# ---------------------------------------------------------------------------
# ① 把「加粗标记后面的冒号」挪进加粗内部：**结论**： → **结论：**
#    注意必须把冒号一并捕获吃掉，否则会残留成 `**结论：**：`
_BOLD_OUTER_COLON_RE = re.compile(r"\*\*([^*\n]{1,20})\*\*([：:])")
# ② 给「**小标题：**」与后续正文之间补一个空格（标题类，含冒号）
_BOLD_HEADING_RE = re.compile(r"\*\*([^*\n]{1,20}[：:])\*\*(?=\S)")
# ③ 兜底：行首（含 `- ` 列表标记）的短加粗块后面紧跟正文时补空格。
#    刻意锚定行首，避免给「请**立即停止使用**该机器人」这类句中加粗插入空格。
#    这里不带 re.MULTILINE —— 行首与否由调用方逐行判断（见 normalize_markdown）。
_BOLD_TAIL_RE = re.compile(r"^(\s*(?:[-*]\s+)?)\*\*([^*\n]{1,20})\*\*(?=[^\s，。；、！？：,.;:!?])")


def normalize_markdown(text: str, at_line_start: bool = True) -> str:
    """规范化 `**小标题**` 的写法，使其能正常渲染为加粗。

    at_line_start 表示 text 的首字符是否处于「真正的行首」。
    流式渲染时每个分块会单独调用本函数，分块起点未必是行首，
    因此必须由调用方告知，否则会给句中加粗误插空格。
    """
    if not text or "**" not in text:
        return text
    text = _BOLD_OUTER_COLON_RE.sub(r"**\1\2**", text)
    text = _BOLD_HEADING_RE.sub(r"**\1** ", text)

    if not _BOLD_TAIL_RE.search(text):
        return text

    # 逐行处理：第 0 行只在确实处于行首时才套用规则③
    lines = text.split("\n")
    fixed = [
        line if (index == 0 and not at_line_start) else _BOLD_TAIL_RE.sub(r"\1**\2** ", line)
        for index, line in enumerate(lines)
    ]
    return "\n".join(fixed)


def _safe_cut(buffer: str, at_line_start: bool = True) -> int:
    """返回可以安全输出的前缀长度。

    必须同时满足两点，否则流式结果会和整体规范化不一致：
    1. 前缀里的 `**` 成对出现（不能把加粗标记切一半）；
    2. 前缀**规范化之后**不以 `**` 结尾 —— 否则「`**标题：**` 后面要不要补空格」
       取决于尚未到达的下一个字符，必须等它到齐再决定。
    """
    cut = len(buffer)
    stripped = buffer.rstrip("*")
    if len(stripped) != len(buffer):
        cut = len(stripped)

    prefix = buffer[:cut]
    if prefix.count("**") % 2 == 1:
        cut = prefix.rfind("**")

    # 条件 2：回退到「以 ** 结尾的加粗块」之前
    while cut > 0:
        candidate = buffer[:cut]
        if not normalize_markdown(candidate, at_line_start=at_line_start).endswith("**"):
            break
        opening = candidate.rfind("**", 0, len(candidate) - 2)
        cut = opening if opening > 0 else 0

    return max(cut, 0)


def normalize_stream(chunks: Iterator[str]) -> Iterator[str]:
    """流式规范化：按块产出，块与块之间不会截断加粗标记。

    同时跟踪「当前块是否处于真正的行首」，让规则③只在行首生效。
    """
    buffer = ""
    emitted = ""
    at_line_start = True

    for chunk in chunks:
        buffer += chunk or ""
        cut = _safe_cut(buffer, at_line_start=at_line_start)
        if cut <= 0:
            continue
        pending, buffer = buffer[:cut], buffer[cut:]
        yield normalize_markdown(pending, at_line_start=at_line_start)
        emitted += pending
        at_line_start = emitted.endswith("\n")

    if buffer:
        yield normalize_markdown(buffer, at_line_start=at_line_start)


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(":material/robot_2: 扫地机器人客服")
    st.caption("LangGraph + 双路混合检索 · 结合天气与季节给建议")
    st.divider()

    st.session_state.backend_url = st.text_input("后端地址", value=st.session_state.backend_url)

    st.subheader(":material/person: 用户信息")
    user_id = st.text_input("用户 ID", value=st.session_state.user_id)
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.session_state.messages = []

    city = st.selectbox("所在城市", CITY_OPTIONS, index=1)
    model = st.selectbox("机型", MODEL_OPTIONS, index=0)
    floor_type = st.selectbox("地板类型", FLOOR_OPTIONS, index=0)
    with st.container(horizontal=True):
        has_pet = st.checkbox("有宠物", value=False)
        has_baby = st.checkbox("有婴儿", value=False)

    if st.button(":material/save: 保存用户画像", width="stretch"):
        ok, _data, err = api_post(
            "/api/profile",
            {
                "user_id": st.session_state.user_id,
                "city": "" if city == "不指定" else city,
                "model": "" if model == "未指定" else model,
                "floor_type": "" if floor_type == "未指定" else floor_type,
                "has_pet": has_pet,
                "has_baby": has_baby,
            },
        )
        if ok:
            st.success("画像已保存，后续回答会带上这些信息")
        else:
            st.error(err)

    st.divider()

    # ------------------------- 知识库 -------------------------
    st.subheader(":material/database: 知识库")
    if st.session_state.kb_status is None:
        refresh_kb_status()
    kb = st.session_state.kb_status or {}

    if kb:
        st.metric("知识条目", kb.get("document_count", 0))
        st.caption(
            f"集合：`{kb.get('collection', '-')}`　向量库：`{kb.get('vector_backend', '-')}`\n\n"
            f"Embedding：`{kb.get('embedding_backend', '-')}`"
        )
        if kb.get("last_build"):
            st.caption(f"最近构建：{kb['last_build']}")
        if kb.get("needs_rebuild"):
            st.warning("建库配置与当前配置不一致，请重建知识库", icon=":material/sync_problem:")
        elif kb.get("ready"):
            st.success("知识库已就绪", icon=":material/check_circle:")
        else:
            st.warning("知识库为空，请先构建", icon=":material/warning:")
    else:
        st.info("无法获取知识库状态，请确认后端已启动")

    with st.container(horizontal=True):
        if st.button(":material/refresh: 刷新状态", width="stretch"):
            refresh_kb_status()
            st.rerun()
        if st.button(":material/build_circle: 重建知识库", width="stretch"):
            with st.spinner("正在重建知识库（首次构建需加载 Embedding 模型，可能较久）…"):
                ok, data, err = api_post("/api/kb/rebuild", {}, timeout=3600)
            if ok:
                st.success(data.get("message", "重建完成"))
                refresh_kb_status()
                st.rerun()
            else:
                st.error(err)

    st.divider()

    # ------------------------- 会话 -------------------------
    st.subheader(":material/forum: 会话")
    st.caption(f"会话 ID：`{st.session_state.session_id}`")
    with st.container(horizontal=True):
        if st.button(":material/delete_sweep: 清空会话", width="stretch"):
            api_delete("/api/chat/history", {"session_id": st.session_state.session_id})
            st.session_state.messages = []
            st.session_state.session_id = f"s-{uuid.uuid4().hex[:8]}"
            st.rerun()
        if st.button(":material/add_comment: 新建会话", width="stretch"):
            st.session_state.messages = []
            st.session_state.session_id = f"s-{uuid.uuid4().hex[:8]}"
            st.rerun()

    with st.expander(":material/wb_sunny: 查看当前环境", expanded=False):
        if st.button("获取实时环境", width="stretch"):
            _, dt_data, _ = api_get("/api/datetime")
            target_city = city if city != "不指定" else "杭州"
            _, weather_data, _ = api_get("/api/weather", {"city": target_city})
            if dt_data:
                data = dt_data.get("data", {})
                st.write(f"**日期**：{data.get('date')} {data.get('weekday')}　**季节**：{data.get('season')}")
            if weather_data:
                data = weather_data.get("data", {})
                st.write(
                    f"**{weather_data.get('city')}**：{data.get('description')} "
                    f"{data.get('temperature')}℃　湿度 {data.get('humidity')}%"
                )
                for advice in data.get("env_advice", [])[:4]:
                    st.caption(f"· {advice}")


# ---------------------------------------------------------------------------
# 渲染工具
# ---------------------------------------------------------------------------
def render_extras(message: dict[str, Any]) -> None:
    """渲染工具调用与参考来源折叠面板。"""
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        with st.expander(f"工具调用（{len(tool_calls)} 次）", icon=":material/handyman:", expanded=False):
            for index, call in enumerate(tool_calls, start=1):
                state_icon = ":material/check_circle:" if call.get("ok", True) else ":material/error:"
                st.markdown(f"{state_icon} **{index}. `{call.get('name')}`**")
                if call.get("arguments"):
                    st.caption(f"入参：{call['arguments']}")
                with st.container():
                    st.json(call.get("result"), expanded=False)
                if index < len(tool_calls):
                    st.divider()

    sources = message.get("sources") or []
    if sources:
        with st.expander(f"参考来源（{len(sources)} 条）", icon=":material/menu_book:", expanded=False):
            for index, source in enumerate(sources, start=1):
                label = f"{source.get('file', '知识库')}"
                if source.get("item_no"):
                    label += f" - {source['item_no']}"
                if source.get("title"):
                    label += f". {source['title']}"
                st.markdown(f"**[{index}] {label}**")
                if source.get("category"):
                    st.caption(f"分类：{source['category']}")
                if source.get("snippet"):
                    st.caption(f"片段：{source['snippet']}…")
                if index < len(sources):
                    st.divider()

    extra = message.get("extra") or {}
    if extra.get("rewritten_query") or extra.get("environment"):
        with st.expander("检索与意图细节", icon=":material/travel_explore:", expanded=False):
            if extra.get("rewritten_query") and extra["rewritten_query"] != message.get("question"):
                st.write(f"**改写后的检索词**：{extra['rewritten_query']}")
            if extra.get("intent"):
                st.write(f"**意图判断**：{extra['intent']}")
            if extra.get("environment"):
                st.text(extra["environment"])


def render_message(message: dict[str, Any]) -> None:
    """渲染一条历史消息。"""
    avatar = ":material/person:" if message["role"] == "user" else ":material/robot_2:"
    with st.chat_message(message["role"], avatar=avatar):
        content = message["content"]
        st.markdown(normalize_markdown(content) if message["role"] == "assistant" else content)
        if message["role"] == "assistant":
            render_extras(message)


def build_payload(question: str) -> dict[str, Any]:
    return {
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
        "message": question,
        "location": {"city": city} if city != "不指定" else None,
        "user_profile": {
            "user_id": st.session_state.user_id,
            "city": "" if city == "不指定" else city,
            "model": "" if model == "未指定" else model,
            "floor_type": "" if floor_type == "未指定" else floor_type,
            "has_pet": has_pet,
            "has_baby": has_baby,
        },
    }


def ask(question: str) -> None:
    """发送问题：流式渲染回答，并记录来源与工具调用。"""
    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    sink: dict[str, Any] = {}
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        with st.status(":shimmer[正在检索知识库并生成回答]", type="compact", expanded=False) as status:
            st.caption("意图识别 → 工具调用 → 混合检索（向量 + BM25）→ 摘要 → 生成回答")
        answer = st.write_stream(normalize_stream(stream_answer(build_payload(question), sink)))

        if sink.get("error") and not answer:
            st.error(sink["error"])
            status.update(label="请求失败", state="error")
        else:
            if sink.get("error"):
                st.warning(sink["error"])
            status.update(label="已基于知识库生成回答", state="complete")

        message = {
            "role": "assistant",
            "content": answer or "",
            "sources": sink.get("sources", []),
            "tool_calls": sink.get("tool_calls", []),
            "extra": sink.get("extra", {}),
            "question": question,
        }
        render_extras(message)

    st.session_state.messages.append(message)


# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
st.title("智能扫地机器人客服 Agent")
st.caption("故障排查 · 维护保养 · 使用技巧 · 选购建议 —— 回答全部基于知识库检索结果，不编造内容")

# 历史消息
for msg in st.session_state.messages:
    render_message(msg)

# 空会话时给出建议问题
if not st.session_state.messages:
    st.markdown("**你可以这样问：**")
    selected = st.pills(
        "示例问题",
        EXAMPLE_QUESTIONS,
        selection_mode="single",
        label_visibility="collapsed",
    )
    if selected:
        st.session_state.pending_question = selected

# 侧边栏示例问题点选后立即提问
if st.session_state.pending_question:
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    ask(pending)

# 输入框：回答生成期间禁用，避免打断流式输出
if prompt := st.chat_input("请输入你的问题，例如：今天湿度高适合拖地吗？", submit_mode="disable"):
    ask(prompt)

st.divider()
st.caption(
    "提示：所有回答均来自 `data/raw` 知识库检索结果与实时工具数据；"
    "未配置 LLM API Key 时返回知识库原文整理结果，天气未配置 Key 时使用 mock 数据。"
)
