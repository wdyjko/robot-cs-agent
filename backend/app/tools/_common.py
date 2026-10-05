"""工具公共辅助函数。"""

from __future__ import annotations

from typing import Any


def ok(**kwargs: Any) -> dict[str, Any]:
    """统一的工具返回结构：ok + 数据字段。"""
    result: dict[str, Any] = {"ok": True}
    result.update(kwargs)
    return result


def fail(message: str, **kwargs: Any) -> dict[str, Any]:
    """失败返回结构，保证工具永远返回 dict 而不是抛异常。"""
    result: dict[str, Any] = {"ok": False, "error": message}
    result.update(kwargs)
    return result
