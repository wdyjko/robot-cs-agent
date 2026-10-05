"""工具集合导出。"""

from __future__ import annotations

from .datetime_tool import get_current_datetime
from .kb_tool import search_knowledge_base
from .location import get_user_location
from .profile import get_user_profile
from .weather import get_weather

ALL_TOOLS = [
    get_user_location,
    get_weather,
    get_current_datetime,
    get_user_profile,
    search_knowledge_base,
]

# 名称 -> 工具对象，便于按名字调用
TOOL_REGISTRY = {item.name: item for item in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOL_REGISTRY",
    "get_current_datetime",
    "get_user_location",
    "get_user_profile",
    "get_weather",
    "search_knowledge_base",
]
