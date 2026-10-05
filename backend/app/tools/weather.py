"""天气工具：有 WEATHER_API_KEY 走真实 API（OpenWeatherMap 兼容），否则返回稳定的 mock 数据。

无论是否配置 Key，返回结构一致，包含温度、湿度、降水、风力、天气描述，
并附带环境适配建议（湿度/降雨/温度 → 扫地机维护策略）。
"""

from __future__ import annotations

import hashlib
from datetime import date as date_cls
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool

from ..core.config import settings
from ..core.logging import get_logger
from ._common import fail, ok

logger = get_logger(__name__)

_SEASON_DESCRIPTIONS = {
    "春": ["多云", "小雨", "阴", "晴间多云"],
    "夏": ["雷阵雨", "晴", "多云", "阵雨"],
    "秋": ["晴", "多云", "阴", "小雨"],
    "冬": ["阴", "小雨", "多云", "晴冷"],
}
_SEASON_TEMP = {"春": (12, 24), "夏": (26, 38), "秋": (14, 26), "冬": (-6, 10)}


def _season_of(month: int) -> str:
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    if month in (9, 10, 11):
        return "秋"
    return "冬"


def _resolve_date(date: str | None) -> date_cls:
    """把 today / tomorrow / YYYY-MM-DD 解析成日期。"""
    today = datetime.now().date()
    raw = (date or "today").strip().lower()
    if raw in {"today", "今天", "now", ""}:
        return today
    if raw in {"tomorrow", "明天"}:
        return today + timedelta(days=1)
    if raw in {"yesterday", "昨天"}:
        return today - timedelta(days=1)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return today


def _seed(*parts: str) -> int:
    """根据城市 + 日期生成稳定随机种子，保证同一天多次查询结果一致。"""
    digest = hashlib.md5("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _mock_weather(city: str, target: date_cls) -> dict[str, Any]:
    """生成合理且稳定的 mock 天气。"""
    seed = _seed(city, target.isoformat())
    season = _season_of(target.month)
    low, high = _SEASON_TEMP[season]

    temperature = round(low + (seed % 1000) / 1000.0 * (high - low), 1)
    humidity = 45 + (seed >> 3) % 51          # 45% ~ 95%
    precipitation = round(((seed >> 5) % 100) / 10.0, 1)  # 0 ~ 9.9 mm
    wind_level = 1 + (seed >> 7) % 5          # 1 ~ 5 级
    description = _SEASON_DESCRIPTIONS[season][(seed >> 9) % len(_SEASON_DESCRIPTIONS[season])]

    if precipitation > 5:
        description = "中雨" if season != "冬" else "小雨"
    elif precipitation > 0.5 and "晴" in description:
        description = "阵雨"

    return {
        "city": city,
        "date": target.isoformat(),
        "season": season,
        "temperature": temperature,
        "temp_min": round(temperature - 3, 1),
        "temp_max": round(temperature + 4, 1),
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_level": wind_level,
        "wind": f"{wind_level}级",
        "description": description,
        "source": "mock",
        "note": "未配置 WEATHER_API_KEY，当前为演示用 mock 数据",
    }


def _fetch_real_weather(city: str, target: date_cls) -> dict[str, Any] | None:
    """调用 OpenWeatherMap（兼容接口）获取实时天气。失败返回 None。"""
    try:
        import requests

        key = settings.weather_api_key
        # 1) 城市名 -> 经纬度
        geo = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": city, "limit": 1, "appid": key},
            timeout=8,
        ).json()
        if not geo:
            logger.warning("天气 API 未能解析城市: %s", city)
            return None
        lat, lon = geo[0].get("lat"), geo[0].get("lon")

        # 2) 实时天气
        payload = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": key, "units": "metric", "lang": "zh_cn"},
            timeout=8,
        ).json()

        main = payload.get("main", {})
        wind = payload.get("wind", {})
        rain = payload.get("rain", {}) or {}
        weather_desc = (payload.get("weather") or [{}])[0].get("description", "")

        return {
            "city": city,
            "date": target.isoformat(),
            "season": _season_of(target.month),
            "temperature": round(float(main.get("temp", 0.0)), 1),
            "temp_min": round(float(main.get("temp_min", 0.0)), 1),
            "temp_max": round(float(main.get("temp_max", 0.0)), 1),
            "humidity": int(main.get("humidity", 0)),
            "precipitation": round(float(rain.get("1h", 0.0)), 1),
            "wind_level": round(float(wind.get("speed", 0.0)) / 2.0, 1),
            "wind": f"{wind.get('speed', 0)} m/s",
            "description": weather_desc,
            "source": "openweathermap",
        }
    except Exception as exc:  # noqa: BLE001 - 网络/额度问题都要能降级
        logger.error("天气 API 调用失败（降级为 mock）: %s", exc)
        return None


def build_env_advice(weather: dict[str, Any], profile: dict[str, Any] | None = None) -> list[str]:
    """把天气 + 画像映射成扫地机维护建议（对应设计文档「天气映射」）。"""
    advice: list[str] = []
    profile = profile or {}

    temperature = weather.get("temperature")
    humidity = weather.get("humidity")
    precipitation = weather.get("precipitation") or 0
    description = str(weather.get("description", ""))

    if isinstance(humidity, (int, float)) and humidity > 80:
        advice.append("湿度偏高（>80%）：减少湿拖、以干拖为主，拖布用后必须烘干，防止发霉异味。")
    if precipitation > 0 or any(word in description for word in ("雨", "雪")):
        advice.append("当前有降水：尽量少拖地，可开空调除湿；拖完加强通风，避免地面返潮打滑。")
    if any(word in description for word in ("回南天", "梅雨", "返潮")):
        advice.append("回南天/梅雨：减少拖地频次，重点做除湿与拖布烘干。")
    if isinstance(temperature, (int, float)) and temperature > 35:
        advice.append("高温（>35℃）：避免阳光直射充电，充电处保持通风，防止电池过热。")
    if isinstance(temperature, (int, float)) and temperature < 5:
        advice.append("低温（<5℃）：机器人在室温静置 30 分钟后再开机充电，避免低温损伤电池。")
    if any(word in description for word in ("沙尘", "雾霾", "浮尘")):
        advice.append("沙尘/雾霾天：关窗清扫，使用大吸力并确认 HEPA 滤网到位，结束后清理尘盒。")

    floor_type = str(profile.get("floor_type", ""))
    if "木" in floor_type:
        advice.append("木地板：使用低出水量或干拖，拖布拧到不滴水，避免地板起翘。")
    if profile.get("has_pet"):
        advice.append("宠物家庭：使用防缠绕主刷，清扫后及时清理滚刷和尘盒毛发，必要时开启除螨。")
    if profile.get("has_baby"):
        advice.append("有婴儿：优先使用低噪音模式，清扫后保持地面干燥防滑，定期清洗拖布。")

    return advice


@tool
def get_weather(city: str = "杭州", date: str = "today") -> dict:
    """查询指定城市指定日期的天气，返回温度、湿度、降水、风力、天气描述与环境建议。

    Args:
        city: 城市名，例如「杭州」。
        date: today / tomorrow / YYYY-MM-DD，默认 today。
    """
    try:
        target = _resolve_date(date)
        city = (city or settings.default_city).strip() or settings.default_city

        weather: dict[str, Any] | None = None
        if settings.weather_api_enabled:
            weather = _fetch_real_weather(city, target)
        if weather is None:
            weather = _mock_weather(city, target)

        weather["env_advice"] = build_env_advice(weather)
        return ok(**weather)
    except Exception as exc:  # noqa: BLE001
        logger.exception("天气工具异常")
        return fail(f"天气查询失败: {exc}", city=city)
