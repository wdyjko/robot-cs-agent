"""Embedding 工厂。

优先使用配置的 `EMBEDDING_MODEL`（默认 BAAI/bge-m3，走 sentence-transformers）；
当模型不可用（未联网 / 未下载 / 显存或内存不足）时，自动降级到内置的
`HashingEmbeddings`（确定性哈希向量，零依赖、零下载），保证项目始终可运行。
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """返回单例 Embedding 实例（带降级）。"""
    model_name = (settings.embedding_model_path or "").strip() or settings.embedding_model

    # 用户显式关闭时直接使用内置向量器（例如环境里 torch 不可用，省去报错与等待）
    if not settings.embedding_allow_download:
        logger.warning("EMBEDDING_ALLOW_DOWNLOAD=false，直接使用内置 HashingEmbeddings（检索效果弱于 %s）", model_name)
        return HashingEmbeddings(dim=settings.embedding_dim)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("加载 Embedding 模型: %s（首次使用需要下载，请耐心等待）", model_name)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )
        # 触发一次真实编码，确认模型可用（否则延迟到建库时才报错）
        probe = embeddings.embed_query("滤网多久更换一次")
        logger.info("Embedding 模型加载成功，向量维度=%d", len(probe))
        return embeddings
    except Exception as exc:  # noqa: BLE001 - 任何失败都要能降级
        logger.warning(
            "Embedding 模型 %s 加载失败（%s），已降级为内置 HashingEmbeddings（检索效果会变弱）。"
            "如需完整效果，请先联网下载模型或设置 EMBEDDING_MODEL_PATH 指向本地模型。",
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
