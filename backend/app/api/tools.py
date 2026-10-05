"""工具接口：天气 / 位置 / 时间，方便前端与调试直接调用。"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import DatetimeResponse, LocationResponse, ProfileRequest, WeatherResponse
from ..services import memory_service
from ..tools.datetime_tool import get_current_datetime
from ..tools.location import get_user_location
from ..tools.weather import get_weather

router = APIRouter(prefix="/api", tags=["tools"])
logger = get_logger(__name__)


@router.get("/weather", response_model=WeatherResponse, summary="查询天气")
async def weather(
    city: str = Query(default="", description="城市名，留空使用默认城市"),
    date: str = Query(default="today", description="today / tomorrow / YYYY-MM-DD"),
) -> WeatherResponse:
    """返回温度、湿度、降水、风力、天气描述与环境适配建议。"""
    target_city = (city or "").strip() or settings.default_city
    result = get_weather.invoke({"city": target_city, "date": date})
    data = {key: value for key, value in result.items() if key != "ok"}
    return WeatherResponse(ok=bool(result.get("ok")), city=target_city, data=data, source=str(data.get("source", "")))


@router.get("/location", response_model=LocationResponse, summary="查询用户位置")
async def location(
    user_id: str = Query(default="u1"),
    manual_city: str = Query(default="", description="手动指定城市，优先级最高"),
) -> LocationResponse:
    """按「手动 > 画像 > IP > 默认城市」的顺序解析用户位置。"""
    result = get_user_location.invoke(
        {"user_id": user_id, "manual_city": (manual_city or "").strip() or None, "client_ip": None}
    )
    data = {key: value for key, value in result.items() if key != "ok"}
    return LocationResponse(ok=bool(result.get("ok")), user_id=user_id, data=data)


@router.get("/datetime", response_model=DatetimeResponse, summary="查询当前时间")
async def datetime_info() -> DatetimeResponse:
    """返回当前日期、月份、季节、星期与季节提示。"""
    result = get_current_datetime.invoke({})
    data = {key: value for key, value in result.items() if key != "ok"}
    return DatetimeResponse(ok=bool(result.get("ok")), data=data)


@router.get("/profile", summary="读取用户画像")
async def read_profile(user_id: str = Query(default="u1")) -> dict:
    """读取用户画像（城市、机型、地板类型、宠物、婴儿）。"""
    return {"ok": True, "profile": memory_service.get_user_profile(user_id)}


@router.post("/profile", summary="保存用户画像")
async def write_profile(req: ProfileRequest) -> dict:
    """保存用户画像，后续问答会自动带上做个性化建议。"""
    profile = memory_service.upsert_user_profile(req.user_id, req.model_dump())
    return {"ok": True, "profile": profile}
