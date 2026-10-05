"""用户画像工具：从数据库读取用户机型 / 地板类型 / 宠物 / 婴儿等信息。"""

from __future__ import annotations

from langchain_core.tools import tool

from ..core.logging import get_logger
from ._common import fail, ok

logger = get_logger(__name__)


@tool
def get_user_profile(user_id: str = "u1") -> dict:
    """读取用户画像（城市、机型、地板类型、是否有宠物/婴儿），用于个性化建议。

    Args:
        user_id: 用户 ID。
    """
    try:
        from ..services.memory_service import get_user_profile as _get_profile

        profile = _get_profile(user_id)
        return ok(**profile)
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取用户画像异常")
        return fail(f"读取用户画像失败: {exc}", user_id=user_id)
