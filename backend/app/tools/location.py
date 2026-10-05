"""位置工具：优先手动指定 → 用户画像 → IP 定位（可 mock） → 默认城市。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..core.config import settings
from ..core.logging import get_logger
from ._common import ok

logger = get_logger(__name__)

# 常见城市纬度经度兜底表（无法联网时使用，保证建议不因定位失败而中断）
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "杭州": (30.2741, 120.1551),
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "成都": (30.5728, 104.0668),
    "重庆": (29.5630, 106.5516),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "南京": (32.0603, 118.7969),
    "苏州": (31.2989, 120.5853),
    "天津": (39.3434, 117.3616),
    "长沙": (28.2282, 112.9388),
    "青岛": (36.0671, 120.3826),
    "沈阳": (41.8057, 123.4315),
    "郑州": (34.7466, 113.6254),
    "昆明": (25.0389, 102.7183),
    "哈尔滨": (45.8038, 126.5350),
    "福州": (26.0745, 119.2965),
    "厦门": (24.4798, 118.0894),
}


def _ip_locate(client_ip: str | None = None) -> dict[str, Any] | None:
    """通过 IP 定位城市。失败返回 None（不抛异常）。"""
    try:
        import requests

        url = f"http://ip-api.com/json/{client_ip or ''}?lang=zh-CN&fields=status,country,regionName,city,lat,lon"
        response = requests.get(url, timeout=5)
        payload = response.json()
        if payload.get("status") == "success":
            return {
                "city": payload.get("city") or payload.get("regionName") or "",
                "province": payload.get("regionName", ""),
                "country": payload.get("country", ""),
                "lat": payload.get("lat"),
                "lon": payload.get("lon"),
                "source": "ip",
            }
    except Exception as exc:  # noqa: BLE001 - 定位失败属于预期情况
        logger.warning("IP 定位失败（将使用默认城市）: %s", exc)
    return None


def _coord_for(city: str) -> tuple[float | None, float | None]:
    coords = _CITY_COORDS.get((city or "").strip())
    return coords if coords else (None, None)


def resolve_location(
    user_id: str = "u1",
    client_ip: str | None = None,
    manual_city: str | None = None,
) -> dict[str, Any]:
    """位置解析主逻辑（工具函数与内部调用共用）。"""
    # 1) 手动指定
    if manual_city and str(manual_city).strip():
        city = str(manual_city).strip()
        lat, lon = _coord_for(city)
        return ok(city=city, lat=lat, lon=lon, source="manual", user_id=user_id)

    # 2) 用户画像
    try:
        from ..services.memory_service import get_user_profile

        profile = get_user_profile(user_id)
        if profile.get("city"):
            city = str(profile["city"]).strip()
            lat, lon = _coord_for(city)
            return ok(city=city, lat=lat, lon=lon, source="profile", user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取用户画像位置失败: %s", exc)

    # 3) IP 定位
    located = _ip_locate(client_ip)
    if located and located.get("city"):
        return ok(user_id=user_id, **located)

    # 4) 默认城市
    city = settings.default_city
    lat, lon = _coord_for(city)
    return ok(city=city, lat=lat, lon=lon, source="default", user_id=user_id, note="未能定位，已使用默认城市")


@tool
def get_user_location(user_id: str = "u1", client_ip: str | None = None, manual_city: str | None = None) -> dict:
    """获取用户所在城市。优先级：手动指定 > 用户画像 > IP 定位 > 默认城市。

    Args:
        user_id: 用户 ID。
        client_ip: 客户端 IP，可选，用于 IP 定位。
        manual_city: 用户在前端手动选择的城市，可选，优先级最高。
    """
    return resolve_location(user_id=user_id, client_ip=client_ip, manual_city=manual_city)
