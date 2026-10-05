"""时间工具：返回当前日期、月份、季节、星期（中国时区）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from ..core.logging import get_logger
from ._common import ok

logger = get_logger(__name__)

TZ = ZoneInfo("Asia/Shanghai")
_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _season_hints(month: int, day: int, city: str = "") -> list[str]:
    """根据月份给出季节相关的保养/环境提示。"""
    hints: list[str] = []
    if 3 <= month <= 5:
        hints.append("春季花粉与扬尘多：建议增加 HEPA 滤网检查频次，清扫前适当关闭门窗。")
    if month == 6 or (month == 7 and day <= 15):
        hints.append("梅雨季（6 月—7 月中）：空气潮湿，减少湿拖，重点做拖布烘干与除湿。")
    if 2 <= month <= 4:
        hints.append("回南天/返潮季（2—4 月，南方明显）：地面易返潮，建议干拖为主。")
    if 6 <= month <= 8:
        hints.append("夏季高温：避免机器人长时间在阳光直射处充电，充电环境要通风。")
    if month in (9, 10, 11):
        hints.append("秋季干燥、灰尘多：可提高清扫频次，定期清洗滤网保持吸力。")
    if month in (12, 1, 2):
        hints.append("冬季低温：机器人从室外搬入室内后先静置 30 分钟再开机；注意电池保温。")
    return hints


def get_datetime_info() -> dict[str, Any]:
    """当前时间信息的字典形式。"""
    now = datetime.now(TZ)
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "time": now.strftime("%H:%M:%S"),
        "weekday": _WEEKDAYS[now.weekday()],
        "weekday_index": now.weekday(),
        "season": _season_of(now.month),
        "is_weekend": now.weekday() >= 5,
        "timezone": "Asia/Shanghai",
        "hints": _season_hints(now.month, now.day),
    }


def _season_of(month: int) -> str:
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    if month in (9, 10, 11):
        return "秋季"
    return "冬季"


@tool
def get_current_datetime() -> dict:
    """获取当前日期、月份、季节、星期与时间，用于「今天」「这个月」「最近」这类问题。"""
    try:
        return ok(**get_datetime_info())
    except Exception as exc:  # noqa: BLE001
        logger.exception("时间工具异常")
        return ok(date=datetime.now().strftime("%Y-%m-%d"), error=str(exc))
