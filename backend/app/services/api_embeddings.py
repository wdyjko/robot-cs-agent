"""云端 Embedding（OpenAI 兼容接口 / DashScope 原生接口）。

为什么需要它
------------
项目默认用本地 `BAAI/bge-m3`（sentence-transformers）做向量化：
权重约 2.3GB，加载后进程内存超 3GB，**2 核 2G 的机器跑不动**（会被 OOM 杀掉），
只能降级成内置哈希向量器（检索语义能力弱）。

本模块提供第三条路：直接调用云端 Embedding 接口，
- 不需要 torch、不需要下载模型，内存占用可忽略；
- 语义检索质量反而更好（百炼 qwen3.7-text-embedding / text-embedding-v4 是生产级向量模型）；
- 代价：建库和每次提问需要联网，各一次 HTTP 请求（几百毫秒）。

配置（写在项目根目录 .env，Docker 与非 Docker 两种部署方式都生效）

    EMBEDDING_API_KEY=sk-你的百炼Key         # 必填；留空则不启用本模块
    EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
    EMBEDDING_API_MODEL=text-embedding-v4
    EMBEDDING_API_BATCH=10                   # 单次请求最多传几条文本
    EMBEDDING_API_DIM=                       # 留空（0）用模型默认维度
    EMBEDDING_API_TIMEOUT=60

关于 `EMBEDDING_API_BASE`：
- 通用兼容地址：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 百炼【业务空间】专属地址：形如 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`
  两种都行，但**一定要带 `/compatible-mode/v1` 后缀**，否则请求路径不对（直接拼 `/embeddings` 会 404）。

常用模型（百炼 OpenAI 兼容接口）：

| 模型名 | 默认维度 | 可选维度 | 单次最大行数 |
| --- | --- | --- | --- |
| `qwen3.7-text-embedding` | 1024 | 2560 / 2048 / 1536 / 1024 / 768 / 512 / 256 | 20 |
| `qwen3.7-text-embedding-flash` | 1024 | 1024 / 768 / 512 / 256 | 20 |
| `text-embedding-v4` | 1024 | 2048 / 1536 / 1024 / 768 / 512 / 256 / 128 / 64 | 10 |
| `text-embedding-v3` | 1024 | 1024 / 768 / 512 / 256 / 128 / 64 | 10 |

> 默认 `EMBEDDING_API_BATCH=10` 对以上所有模型都安全（v3/v4 单次上限就是 10 行）。

本模块**同时兼容两种响应结构**（这是踩过的坑）：
1) OpenAI 兼容：`{"data": [{"index": 0, "embedding": [...]}, ...]}`
2) DashScope 原生：`{"output": {"embeddings": [{"text_index": 0, "embedding": [...]}, ...]}}`

接入点：`embeddings.py` 的 `get_embeddings()` 会在最前面优先尝试本模块，
失败时自动回落到原有逻辑（本地模型 / 内置向量器），不会让服务起不来。
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
from langchain_core.embeddings import Embeddings

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# 单条文本截断长度（字符）：中文约等于 token 数，远低于各模型 8192 token 的上限
MAX_CHARS = 6000

# 触发重试的 HTTP 状态码（限流 / 网关抖动 / 服务端错误）
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _api_key() -> str:
    return (settings.embedding_api_key or "").strip()


def is_api_embedding_configured() -> bool:
    """是否配置了可用的云端 Embedding（Key 为空或还是占位符则视为未配置）。"""
    key = _api_key()
    if not key:
        return False
    if key.lower().startswith("sk-xxx") or key.lower() in {"none", "null"}:
        logger.warning("EMBEDDING_API_KEY 还是占位符（%s...），忽略云端 Embedding 配置", key[:8])
        return False
    if not (settings.embedding_api_base or "").strip():
        logger.warning("EMBEDDING_API_KEY 已配置但 EMBEDDING_API_BASE 为空，忽略云端 Embedding 配置")
        return False
    return True


class OpenAICompatEmbeddings(Embeddings):
    """调用云端 `/embeddings` 接口做向量化（兼容 OpenAI 与 DashScope 两种返回结构）。

    只依赖 requests，不引入额外 SDK；请求体只带 model / input /（可选）dimensions，
    避开各家兼容层的参数差异。内置分批、重试与按序号排序。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        batch_size: int = 10,
        dimensions: int = 0,
        timeout: int = 60,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.batch_size = max(1, int(batch_size or 10))
        self.dimensions = int(dimensions or 0)
        self.timeout = max(5, int(timeout or 60))
        self.max_retries = max(0, int(max_retries))
        self.url = f"{self.base_url}/embeddings"
        self.dim: int | None = None
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # 内部：准备文本
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize(texts: list[str]) -> list[str]:
        """空文本会被接口拒绝，超长文本会被截断。"""
        cleaned: list[str] = []
        for text in texts:
            value = (text or "").strip() or "空"
            cleaned.append(value[:MAX_CHARS])
        return cleaned

    # ------------------------------------------------------------------
    # 内部：解析返回（兼容两种结构）
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(payload: Any, expected: int) -> list[list[float]]:
        """把两种可能的返回结构都转成「按输入顺序排列的向量列表」。"""
        if not isinstance(payload, dict):
            raise ValueError(f"返回不是 JSON 对象：{str(payload)[:300]}")

        rows = payload.get("data")                       # ① OpenAI 兼容
        if not rows:
            output = payload.get("output") or {}         # ② DashScope 原生
            if isinstance(output, dict):
                rows = output.get("embeddings") or output.get("data")

        if not rows:
            raise ValueError(
                "无法解析返回结构（既没有 data，也没有 output.embeddings）："
                + json.dumps(payload, ensure_ascii=False)[:300]
            )

        # OpenAI 用 index，DashScope 用 text_index；统一后按序号排回输入顺序
        rows = sorted(rows, key=lambda item: item.get("index", item.get("text_index", 0)))
        vectors: list[list[float]] = []
        for item in rows:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if vector is None:
                raise ValueError(f"返回项里没有 embedding 字段：{str(item)[:200]}")
            vectors.append(vector)

        if len(vectors) != expected:
            raise ValueError(f"返回条数不匹配：期望 {expected}，实际 {len(vectors)}")
        return vectors

    # ------------------------------------------------------------------
    # 内部：一次 HTTP 请求（含重试）
    # ------------------------------------------------------------------
    def _post(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    self.url, headers=headers, json=payload, timeout=self.timeout
                )
                if response.status_code in RETRY_STATUS:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                if response.status_code >= 400:
                    # 4xx（Key 错、模型名错、额度用尽、路径不对等）重试没有意义，直接抛出
                    raise ValueError(
                        f"HTTP {response.status_code}（{self.url}）: {response.text[:400]}"
                    )
                return self._parse_response(response.json(), len(texts))

            except ValueError:
                # 参数/结构类问题重试无用，直接抛（避免重复消耗 token 额度）
                raise
            except Exception as exc:  # noqa: BLE001 - 网络抖动 / 限流，退避重试
                last_error = exc
                if attempt >= self.max_retries:
                    break
                wait = min(2**attempt, 8)
                logger.warning(
                    "云端 Embedding 请求失败（第 %d/%d 次：%s），%.1fs 后重试",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"云端 Embedding 调用失败：{last_error}")

    # ------------------------------------------------------------------
    # 内部：分批向量化
    # ------------------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        texts = self._sanitize(texts)
        results: list[list[float]] = []
        total = len(texts)

        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(self._post(batch))
            if total > self.batch_size:
                logger.info(
                    "云端 Embedding 进度：%d/%d", min(start + self.batch_size, total), total
                )

        if results and self.dim is None:
            self.dim = len(results[0])
            logger.info("云端 Embedding 就绪：model=%s，向量维度=%d", self.model, self.dim)
        return results

    # ------------------------------------------------------------------
    # LangChain Embeddings 接口
    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def build_api_embeddings() -> OpenAICompatEmbeddings:
    """按 settings 构造云端 Embedding 实例。"""
    return OpenAICompatEmbeddings(
        api_key=_api_key(),
        base_url=settings.embedding_api_base,
        model=(settings.embedding_api_model or "text-embedding-v4").strip(),
        batch_size=settings.embedding_api_batch,
        dimensions=settings.embedding_api_dim,
        timeout=settings.embedding_api_timeout,
    )
