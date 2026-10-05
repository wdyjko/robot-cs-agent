"""内置 NumPy 向量库（Chroma 不可用时的等价兜底实现）。

背景：部分 Windows 环境（受限沙箱 / 缺少运行库）下 chromadb 的原生引擎在写入时会
直接触发访问违例（0xC0000005）导致进程崩溃——这种崩溃无法用 try/except 捕获，
所以不能「先试 Chroma、失败再兜底」，必须在调用前就把后端选好。

本模块提供与 Chroma 用法等价的轻量实现：
    - add_documents(documents, ids)
    - similarity_search(query, k)
    - delete_collection()
    - count()

数据落盘为两个文件（放在 CHROMA_PERSIST_DIR 下）：
    <collection>.vectors.npy   向量矩阵 (float32)
    <collection>.records.json  文本与元数据

910 条 × 384 维仅约 1.4 MB，内存里做余弦相似度是毫秒级，足够本项目使用。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..core.logging import get_logger

logger = get_logger(__name__)


class LocalVectorStore:
    """基于 NumPy 的持久化向量库，接口对齐 langchain_chroma.Chroma 的常用子集。"""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path,
        embedding_function: Embeddings,
        **_ignored: Any,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_function

        self._vectors_path = self.persist_directory / f"{collection_name}.vectors.npy"
        self._records_path = self.persist_directory / f"{collection_name}.records.json"
        self._lock = threading.Lock()

        self._vectors: Any = None      # numpy.ndarray (N, dim)
        self._records: list[dict] = []  # [{"id", "content", "metadata"}]
        self._load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            import numpy as np

            if self._vectors_path.exists() and self._records_path.exists():
                self._vectors = np.load(self._vectors_path)
                self._records = json.loads(self._records_path.read_text(encoding="utf-8"))
                logger.info(
                    "已加载内置向量库 %s：%d 条，维度 %s",
                    self.collection_name,
                    len(self._records),
                    getattr(self._vectors, "shape", "-"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("加载内置向量库失败，将重新开始: %s", exc)
            self._vectors = None
            self._records = []

    def _save(self) -> None:
        import numpy as np

        if self._vectors is None:
            self._vectors = np.zeros((0, 0), dtype="float32")
        np.save(self._vectors_path, self._vectors)
        self._records_path.write_text(json.dumps(self._records, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add_documents(self, documents: list[Document], ids: list[str] | None = None, **_: Any) -> list[str]:
        """写入文档；返回写入的 id 列表。"""
        import numpy as np

        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        metadatas = [dict(doc.metadata) for doc in documents]
        ids = ids or [str(i) for i in range(len(documents))]

        vectors = np.asarray(self.embedding_function.embed_documents(texts), dtype="float32")

        with self._lock:
            if self._vectors is None or getattr(self._vectors, "size", 0) == 0:
                self._vectors = vectors
            else:
                if self._vectors.shape[1] != vectors.shape[1]:
                    raise ValueError(
                        f"向量维度不一致：已有 {self._vectors.shape[1]} 维，新写入 {vectors.shape[1]} 维。"
                        "请重建知识库（EMBEDDING 配置变更后必须重建）。"
                    )
                self._vectors = np.vstack([self._vectors, vectors])

            for index, doc_id in enumerate(ids):
                self._records.append(
                    {
                        "id": doc_id,
                        "content": documents[index].page_content,
                        "metadata": metadatas[index],
                    }
                )
            self._save()

        return list(ids)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 5, **_: Any) -> list[Document]:
        """余弦相似度 Top-k 检索。"""
        import numpy as np

        with self._lock:
            if self._vectors is None or len(self._records) == 0:
                return []
            vectors = self._vectors
            records = list(self._records)

        try:
            query_vector = np.asarray(self.embedding_function.embed_query(query), dtype="float32")
        except Exception as exc:  # noqa: BLE001
            logger.error("查询向量化失败: %s", exc)
            return []

        if query_vector.shape[0] != vectors.shape[1]:
            logger.warning(
                "查询向量维度（%d）与向量库维度（%d）不一致，请重建知识库", query_vector.shape[0], vectors.shape[1]
            )
            return []

        # 余弦相似度：向量已归一化时等价于点积；这里统一做一次归一化以防万一
        def _normalize(matrix: Any) -> Any:
            norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            return matrix / norms

        scores = _normalize(vectors) @ _normalize(query_vector.reshape(1, -1)).ravel()
        top_k = min(int(k), len(records))
        order = np.argsort(-scores)[:top_k]

        results: list[Document] = []
        for index in order:
            record = records[int(index)]
            metadata = dict(record.get("metadata", {}))
            metadata["similarity"] = round(float(scores[int(index)]), 6)
            results.append(Document(page_content=record.get("content", ""), metadata=metadata))
        return results

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def delete_collection(self) -> None:
        """清空向量库（等价于 Chroma 的 delete_collection）。"""
        with self._lock:
            self._vectors = None
            self._records = []
            for path in (self._vectors_path, self._records_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:  # pragma: no cover
                    logger.warning("删除 %s 失败: %s", path, exc)
        logger.info("已清空内置向量库集合: %s", self.collection_name)

    @property
    def _collection(self) -> "LocalVectorStore":
        """兼容 Chroma 的 `vs._collection.count()` 写法。"""
        return self
