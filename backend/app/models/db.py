"""数据库模型与会话管理（SQLAlchemy + SQLite）。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """ORM 基类。"""


class ChatHistory(Base):
    """对话历史表。"""

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class UserProfile(Base):
    """用户画像表。"""

    __tablename__ = "user_profile"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    city: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(128), default="")          # 机型
    floor_type: Mapped[str] = mapped_column(String(64), default="")      # 地板类型
    has_pet: Mapped[int] = mapped_column(Integer, default=0)             # 是否有宠物
    has_baby: Mapped[int] = mapped_column(Integer, default=0)            # 是否有婴儿
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "city": self.city or "",
            "model": self.model or "",
            "floor_type": self.floor_type or "",
            "has_pet": bool(self.has_pet),
            "has_baby": bool(self.has_baby),
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }


# ---------------------------------------------------------------------------
# 引擎 / 会话工厂
# ---------------------------------------------------------------------------
_connect_args = {"check_same_thread": False} if settings.sqlite_url_resolved.startswith("sqlite") else {}

engine = create_engine(
    settings.sqlite_url_resolved,
    connect_args=_connect_args,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    """建表。应用启动时调用；可重复调用。"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("数据库初始化完成: %s", settings.sqlite_url_resolved)
    except Exception as exc:  # pragma: no cover - 启动阶段不因数据库失败而挂掉
        logger.error("数据库初始化失败: %s", exc)


def get_db() -> Iterator[Session]:
    """FastAPI 依赖注入用的会话生成器。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """脚本 / 服务层使用的上下文管理器，自动提交与回滚。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
