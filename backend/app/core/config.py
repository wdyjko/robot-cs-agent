"""全局配置：使用 pydantic-settings 从 .env 读取，提供单例 settings。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[0]=core, [1]=app, [2]=backend, [3]=项目根目录
BASE_DIR: Path = Path(__file__).resolve().parents[3]
ENV_FILE: Path = BASE_DIR / ".env"


def _absolute(path_value: str) -> Path:
    """把配置里的相对路径统一解析成相对「项目根目录」的绝对路径。

    这样无论从哪个工作目录启动（uvicorn / streamlit / scripts），
    向量库、数据库、原始数据的落点都一致。
    """
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


class Settings(BaseSettings):
    """项目全部可配置项。字段名与 .env 中的大写变量一一对应（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "智能扫地机器人客服 Agent"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    backend_url: str = "http://localhost:8000"
    cors_origins: str = "*"

    # ---------- LLM（OpenAI 兼容接口）----------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_timeout: int = 60
    llm_max_retries: int = 2

    # ---------- Embedding ----------
    embedding_model: str = "BAAI/bge-m3"
    # 本地模型目录（可选，填了就不联网下载）
    embedding_model_path: str = ""
    # 是否允许联网下载 embedding 模型；关闭后直接使用内置轻量向量器
    embedding_allow_download: bool = True
    embedding_dim: int = 384

    # ---------- Chroma ----------
    chroma_persist_dir: str = "./vectorstore"
    chroma_collection: str = "robot_customer_service"
    # 向量库后端：auto（自动探测）/ chroma（强制 Chroma）/ local（强制内置 NumPy 向量库）
    vector_backend: str = "auto"

    # ---------- 数据 ----------
    data_raw_dir: str = "./data/raw"
    sqlite_url: str = "sqlite:///./robot_cs.db"

    # ---------- 工具 ----------
    weather_api_key: str = ""
    default_city: str = "杭州"

    # ---------- 重排（可选，预留接口）----------
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---------- 检索参数 ----------
    vector_top_k: int = 10
    bm25_top_k: int = 10
    final_top_k: int = 5
    chunk_size: int = 600
    chunk_overlap: int = 100

    # ================= 派生路径 =================
    @property
    def data_raw_path(self) -> Path:
        return _absolute(self.data_raw_dir)

    @property
    def chroma_persist_path(self) -> Path:
        return _absolute(self.chroma_persist_dir)

    @property
    def bm25_corpus_path(self) -> Path:
        """BM25 语料落盘文件（与向量库同级，重建时一起刷新）。"""
        return self.chroma_persist_path / "bm25_corpus.json"

    @property
    def sqlite_url_resolved(self) -> str:
        """把 sqlite:///./robot_cs.db 变成绝对路径，避免受 CWD 影响。"""
        prefix = "sqlite:///"
        if self.sqlite_url.startswith(prefix):
            raw = self.sqlite_url[len(prefix) :]
            if raw and raw != ":memory:":
                return prefix + str(_absolute(raw).as_posix())
        return self.sqlite_url

    @property
    def sqlite_db_path(self) -> Path | None:
        """SQLite 文件绝对路径；内存库或非 sqlite 返回 None。"""
        prefix = "sqlite:///"
        if self.sqlite_url_resolved.startswith(prefix):
            return Path(self.sqlite_url_resolved[len(prefix) :])
        return None

    @property
    def llm_enabled(self) -> bool:
        """是否具备真实 LLM 能力（没配 Key 就降级为纯检索模式）。"""
        key = (self.openai_api_key or "").strip()
        return bool(key) and not key.lower().startswith("sk-xxx") and key.lower() not in {"none", "null"}

    @property
    def weather_api_enabled(self) -> bool:
        return bool((self.weather_api_key or "").strip())

    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.cors_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    def ensure_dirs(self) -> None:
        """确保运行所需的目录存在。"""
        for path in (self.data_raw_path, self.chroma_persist_path):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:  # pragma: no cover - 权限异常时只提示不阻断
                pass
        db_path = self.sqlite_db_path
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """带缓存的配置加载器。"""
    loaded = Settings()
    loaded.ensure_dirs()
    return loaded


# 单例，业务代码统一 from ..core.config import settings
settings: Settings = get_settings()
