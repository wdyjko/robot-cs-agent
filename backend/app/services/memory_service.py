"""记忆服务：对话历史 + 用户画像（SQLite 原型）。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from ..core.logging import get_logger
from ..models.db import ChatHistory, SessionLocal, UserProfile, init_db

logger = get_logger(__name__)

_init_done = False


def _ensure_init() -> None:
    """惰性建表，保证单独调用服务方法时也不会报 no such table。"""
    global _init_done
    if not _init_done:
        init_db()
        _init_done = True


# ---------------------------------------------------------------------------
# 对话历史
# ---------------------------------------------------------------------------
def save_message(session_id: str, role: str, content: str) -> None:
    """保存一条对话消息。"""
    _ensure_init()
    if not content:
        return
    with SessionLocal() as db:
        try:
            db.add(ChatHistory(session_id=session_id or "default", role=role or "user", content=content))
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error("保存对话失败: %s", exc)


def get_history(session_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """获取最近 limit 条消息（按时间正序返回，便于拼 Prompt）。"""
    _ensure_init()
    with SessionLocal() as db:
        try:
            rows = (
                db.execute(
                    select(ChatHistory)
                    .where(ChatHistory.session_id == (session_id or "default"))
                    .order_by(ChatHistory.id.desc())
                    .limit(max(1, int(limit)))
                )
                .scalars()
                .all()
            )
            return [row.to_dict() for row in reversed(rows)]
        except Exception as exc:
            logger.error("读取对话历史失败: %s", exc)
            return []


def get_history_text(session_id: str, limit: int = 6) -> str:
    """把最近若干轮对话拼成纯文本，用于 Prompt。"""
    history = get_history(session_id, limit=limit)
    if not history:
        return "（无历史对话）"
    lines = []
    for item in history:
        speaker = "用户" if item["role"] == "user" else "客服"
        lines.append(f"{speaker}：{item['content']}")
    return "\n".join(lines)


def clear_history(session_id: str) -> int:
    """清空某个会话的历史，返回删除条数。"""
    _ensure_init()
    with SessionLocal() as db:
        try:
            result = db.execute(delete(ChatHistory).where(ChatHistory.session_id == (session_id or "default")))
            db.commit()
            return int(result.rowcount or 0)
        except Exception as exc:
            db.rollback()
            logger.error("清空历史失败: %s", exc)
            return 0


# ---------------------------------------------------------------------------
# 用户画像
# ---------------------------------------------------------------------------
def get_user_profile(user_id: str) -> dict[str, Any]:
    """读取用户画像；不存在时返回带默认值的空画像。"""
    _ensure_init()
    default = {
        "user_id": user_id or "u1",
        "city": "",
        "model": "",
        "floor_type": "",
        "has_pet": False,
        "has_baby": False,
        "updated_at": "",
    }
    with SessionLocal() as db:
        try:
            row = db.get(UserProfile, user_id or "u1")
            return row.to_dict() if row else default
        except Exception as exc:
            logger.error("读取用户画像失败: %s", exc)
            return default


def upsert_user_profile(user_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    """新增或更新用户画像，返回最新画像。"""
    _ensure_init()
    with SessionLocal() as db:
        try:
            row = db.get(UserProfile, user_id or "u1")
            if row is None:
                row = UserProfile(user_id=user_id or "u1")
                db.add(row)

            if "city" in profile and profile["city"] is not None:
                row.city = str(profile["city"])
            if "model" in profile and profile["model"] is not None:
                row.model = str(profile["model"])
            if "floor_type" in profile and profile["floor_type"] is not None:
                row.floor_type = str(profile["floor_type"])
            if "has_pet" in profile and profile["has_pet"] is not None:
                row.has_pet = 1 if _to_bool(profile["has_pet"]) else 0
            if "has_baby" in profile and profile["has_baby"] is not None:
                row.has_baby = 1 if _to_bool(profile["has_baby"]) else 0

            db.commit()
            db.refresh(row)
            return row.to_dict()
        except Exception as exc:
            db.rollback()
            logger.error("更新用户画像失败: %s", exc)
            return get_user_profile(user_id)


def _to_bool(value: Any) -> bool:
    """把前端传来的各种写法统一成 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "有", "on"}
