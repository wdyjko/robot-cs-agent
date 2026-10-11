"""Pydantic 请求 / 响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """聊天请求。"""

    session_id: str = Field(default="default", description="会话 ID，用于读取历史")
    message: str = Field(..., description="用户问题")
    user_id: str = Field(default="u1", description="用户 ID，用于读取画像")
    location: dict[str, Any] | None = Field(default=None, description="前端传入的位置，如 {'city': '杭州'}")
    user_profile: dict[str, Any] | None = Field(default=None, description="前端传入的画像，会写回数据库")
    client_ip: str | None = Field(default=None, description="客户端 IP，用于 IP 定位兜底")


class SourceItem(BaseModel):
    """引用来源。"""

    file: str = ""
    item_no: str = ""
    title: str = ""
    category: str = ""
    snippet: str = ""
    score: float = 0.0


class ToolCallItem(BaseModel):
    """一次工具调用记录，用于前端展示调用过程。"""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True


class ChatResponse(BaseModel):
    """聊天响应。"""

    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    tool_calls: list[ToolCallItem] = Field(default_factory=list)
    # 便于调试 / 前端展示的额外信息（改写后的查询、环境信息等）
    extra: dict[str, Any] = Field(default_factory=dict)


class KBStatus(BaseModel):
    """知识库状态。

    允许额外字段（extra="allow"）：`rag_service.get_status()` 会返回一些条件出现的键
    （如 `built_backend`、`built_vector_backend`），未来还可能继续增加。
    这里放宽约束，避免状态接口因为多出一个字段就 500。
    """

    model_config = ConfigDict(extra="allow")

    collection: str
    document_count: int
    persist_dir: str
    # 以下为增强字段，不影响设计文档约定的三个核心字段
    ready: bool = False
    embedding_model: str = ""
    embedding_backend: str = ""
    vector_backend: str = ""
    last_build: str = ""
    # 建库时用的向量后端与当前配置不一致时为 True（需要重建知识库）
    needs_rebuild: bool = False


class KBuildResponse(BaseModel):
    """重建知识库响应。"""

    ok: bool
    collection: str
    document_count: int
    persist_dir: str
    message: str
    elapsed_seconds: float = 0.0


class WeatherResponse(BaseModel):
    ok: bool
    city: str
    data: dict[str, Any] = Field(default_factory=dict)
    source: str = ""


class LocationResponse(BaseModel):
    ok: bool
    user_id: str
    data: dict[str, Any] = Field(default_factory=dict)


class DatetimeResponse(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)


class ProfileRequest(BaseModel):
    """用户画像保存请求。"""

    user_id: str = "u1"
    city: str = ""
    model: str = ""
    floor_type: str = ""
    has_pet: bool = False
    has_baby: bool = False


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    llm_enabled: bool
    kb_ready: bool
