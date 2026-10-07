"""Embedding 工厂。

优先使用配置的 `EMBEDDING_MODEL`（默认 BAAI/bge-m3，走 sentence-transformers）；
当模型不可用（未联网 / 未下载 / 显存或内存不足）时，自动降级到内置的
`HashingEmbeddings`（确定性哈希向量，零依赖、零下载），保证项目始终可运行。

⚠️ 关于联网检查（实测踩坑）：
`huggingface_hub` 在加载模型时会对**每个文件**发一次 HTTP HEAD 做更新检查。
当 huggingface.co 不可达时，每次检查要重试 5 次（退避最长 16 秒），
多个文件累计会让启动**卡住数分钟**。所以在模型已经缓存到本地时，
本项目会自动切到离线模式（`HF_HUB_OFFLINE=1`）跳过这些检查。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+")


class HashingEmbeddings(Embeddings):
    """轻量兜底向量器：字符/词哈希 + L2 归一化。

    不依赖任何模型下载，检索效果弱于 bge-m3，但能保证全流程可跑通，
    并且对中文关键词（如「滤网」「鼓包」「拖布」）仍然有明显区分度。
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = int(dim)

    # -- 内部：把一段文本转成稀疏计数 -> 稠密哈希向量 --------------------
    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        text = (text or "").strip()
        if not text:
            return vector

        tokens = _TOKEN_RE.findall(text.lower())
        # 单字 + 双字组合，兼顾中文分词缺失的情况
        grams: list[str] = list(tokens)
        grams.extend(text[i : i + 2] for i in range(max(0, len(text) - 1)))
        grams.extend(tokens[i] + "_" + tokens[i + 1] for i in range(max(0, len(tokens) - 1)))
        if not grams:
            grams = [text]

        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _hf_cache_dir(model_name: str) -> Path:
    """HF 缓存中该模型的目录（与 huggingface_hub 的命名规则一致）。"""
    base = os.environ.get("HF_HOME")
    hub = Path(base) if base else Path.home() / ".cache" / "huggingface"
    return hub / "hub" / ("models--" + model_name.replace("/", "--"))


def _model_is_cached(model_name: str) -> bool:
    """模型是否已完整缓存在本地（有 config + 至少一个权重文件）。"""
    try:
        snapshots = _hf_cache_dir(model_name) / "snapshots"
        if not snapshots.exists():
            return False
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir() or not (snapshot / "config.json").exists():
                continue
            has_weights = any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin"))
            if has_weights:
                return True
    except OSError:
        return False
    return False


def apply_hf_env(model_name: str) -> bool:
    """按配置准备 HuggingFace 相关环境变量；返回是否切到了离线模式。

    必须在 `import huggingface_hub` / `transformers` **之前**调用，
    因为这些库在导入时就会读取 `HF_HUB_OFFLINE`。
    """
    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

    mode = (settings.hf_hub_offline or "auto").strip().lower()
    offline = False
    if mode in {"1", "true", "yes", "on"}:
        offline = True
    elif mode in {"0", "false", "no", "off"}:
        offline = False
    else:  # auto：本地已有完整缓存就离线，省掉每个文件的联网更新检查
        offline = _model_is_cached(model_name)

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return offline


def _build_hf_embeddings(model_name: str):
    """真正构造 HuggingFaceEmbeddings 并做一次编码自检。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("加载 Embedding 模型: %s", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    # 触发一次真实编码，确认模型可用（否则延迟到建库时才报错）
    probe = embeddings.embed_query("滤网多久更换一次")
    logger.info("Embedding 模型加载成功，向量维度=%d", len(probe))
    return embeddings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """返回单例 Embedding 实例（带降级）。"""
    model_name = (settings.embedding_model_path or "").strip() or settings.embedding_model

    # 用户显式关闭时直接使用内置向量器（例如环境里 torch 不可用，省去报错与等待）
    if not settings.embedding_allow_download:
        logger.warning("EMBEDDING_ALLOW_DOWNLOAD=false，直接使用内置 HashingEmbeddings（检索效果弱于 %s）", model_name)
        return HashingEmbeddings(dim=settings.embedding_dim)

    offline = apply_hf_env(model_name)
    if offline:
        logger.info("本地已有模型缓存，启用 HF 离线模式（跳过逐文件联网检查，启动更快）")

    try:
        return _build_hf_embeddings(model_name)
    except Exception as exc:  # noqa: BLE001
        # 离线模式失败通常意味着缓存不完整 —— 关掉离线再试一次（会尝试联网补齐）
        if offline:
            logger.warning("离线模式加载失败（%s），关闭离线模式重试一次……", exc)
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                return _build_hf_embeddings(model_name)
            except Exception as retry_exc:  # noqa: BLE001
                exc = retry_exc
        logger.warning(
            "Embedding 模型 %s 加载失败（%s），已降级为内置 HashingEmbeddings（检索效果会变弱）。"
            "如需完整效果：先联网下载模型，或设置 EMBEDDING_MODEL_PATH 指向本地模型，"
            "网络受限时可用 HF_ENDPOINT 指定镜像站。",
            model_name,
            exc,
        )
        return HashingEmbeddings(dim=settings.embedding_dim)


def get_embedding_backend_name() -> str:
    """当前实际生效的向量后端名称，用于状态展示。"""
    backend = get_embeddings()
    return type(backend).__name__


def get_embedding_description() -> dict[str, Any]:
    """给 /api/kb/status 和前端展示用的描述信息。"""
    backend = get_embeddings()
    return {
        "backend": type(backend).__name__,
        "model": (settings.embedding_model_path or "").strip() or settings.embedding_model,
        "degraded": isinstance(backend, HashingEmbeddings),
    }
