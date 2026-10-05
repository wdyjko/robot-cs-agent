"""向量库后端选择与管理。

支持三种后端（VECTOR_BACKEND 配置）：

- `chroma`：强制使用 chromadb（设计文档指定的方案）
- `local` ：强制使用内置 NumPy 向量库
- `auto`  ：**默认**。先在一个子进程里探测 chromadb 是否可用，再决定用哪个。
  之所以要在子进程里探测，是因为 chromadb 原生引擎在某些 Windows 环境下写入时
  会直接让进程崩溃（0xC0000005 访问违例），这种崩溃无法被 try/except 捕获。

探测结果按 chromadb 版本缓存在向量库目录下的 `.chroma_probe.json`，只探测一次。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.config import settings
from ..core.logging import get_logger
from .embeddings import get_embeddings
from .local_vectorstore import LocalVectorStore

logger = get_logger(__name__)

_backend_cache: dict[str, Any] = {"name": None}

# 探测失败结果的缓存时长（秒）：过期后重新探测，避免环境修好后仍被锁在降级后端
FAILURE_TTL_SECONDS = 6 * 3600

_PROBE_CODE = (
    "import chromadb,tempfile,pathlib;"
    "d=tempfile.mkdtemp();"
    "c=chromadb.PersistentClient(path=d);"
    "col=c.get_or_create_collection('probe_kb');"
    "col.add(ids=['1','2'],documents=['滤网堵塞','电池鼓包'],embeddings=[[0.1]*8,[0.2]*8]);"
    "assert col.count()==2;"
    "c.delete_collection('probe_kb');"
    "print('CHROMA_PROBE_OK')"
)


def _chroma_version() -> str:
    try:
        import chromadb

        return str(getattr(chromadb, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unavailable"


def _probe_cache_path() -> Path:
    return settings.chroma_persist_path / ".chroma_probe.json"


def _probe_chroma_in_subprocess(timeout: int = 180) -> bool:
    """在子进程里验证 chromadb 能否完成一次写入+查询，避免主进程被拖崩。"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("chromadb 探测超时（%ss），改用内置向量库", timeout)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("chromadb 探测无法执行（%s），改用内置向量库", exc)
        return False

    ok = completed.returncode == 0 and "CHROMA_PROBE_OK" in (completed.stdout or "")
    if not ok:
        error_lines = [line for line in (completed.stderr or "").strip().splitlines() if line.strip()]
        detail = error_lines[-1][:180] if error_lines else "无错误输出（通常是原生崩溃）"
        code = completed.returncode
        code_text = f"{code} / 0x{code & 0xFFFFFFFF:08X}" if code < 0 else str(code)
        logger.warning(
            "chromadb 在本环境下不可用（退出码 %s，%s），自动改用内置 NumPy 向量库。"
            "如需强制使用 Chroma，请在 .env 中设置 VECTOR_BACKEND=chroma 并检查原生依赖。",
            code_text,
            detail,
        )
    return ok


def _chroma_usable() -> bool:
    """判断是否可用 chromadb（带缓存与探测结果落盘）。

    缓存策略：
    - 探测**成功**：按 chromadb 版本长期缓存（环境没变就一直用 Chroma）；
    - 探测**失败**：只缓存 FAILURE_TTL_SECONDS，过期后重新探测——
      否则用户装好 VC++ 运行库 / 换环境后，会一直被旧的失败结果锁在降级后端上。
    """
    version = _chroma_version()
    if version == "unavailable":
        return False

    cache_path = _probe_cache_path()
    try:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("chromadb_version") == version:
                if cached.get("ok"):
                    return True
                checked_at = float(cached.get("checked_at") or 0)
                if time.time() - checked_at < FAILURE_TTL_SECONDS:
                    return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取 Chroma 探测缓存失败: %s", exc)

    ok = _probe_chroma_in_subprocess()

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"chromadb_version": version, "ok": ok, "checked_at": time.time()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover
        logger.debug("写入 Chroma 探测缓存失败: %s", exc)
    return ok


def get_backend_name() -> str:
    """返回实际生效的向量库后端名称：chroma 或 local。"""
    if _backend_cache["name"]:
        return str(_backend_cache["name"])

    configured = (settings.vector_backend or "auto").strip().lower()
    if configured == "local":
        name = "local"
    elif configured == "chroma":
        if not _chroma_usable():
            logger.warning("VECTOR_BACKEND=chroma，但探测显示 chromadb 不可用，仍按配置强制使用（可能崩溃）")
        name = "chroma"
    else:
        name = "chroma" if _chroma_usable() else "local"

    _backend_cache["name"] = name
    logger.info("向量库后端: %s（配置值 VECTOR_BACKEND=%s）", name, configured)
    return name


def reset_backend_cache() -> None:
    """清除后端缓存（重建知识库 / 修改配置后调用）。"""
    _backend_cache["name"] = None


def get_vectorstore():
    """获取向量库实例（Chroma 或内置实现，用法一致）。"""
    settings.chroma_persist_path.mkdir(parents=True, exist_ok=True)
    backend = get_backend_name()

    if backend == "chroma":
        try:
            from langchain_chroma import Chroma

            return Chroma(
                collection_name=settings.chroma_collection,
                persist_directory=str(settings.chroma_persist_path),
                embedding_function=get_embeddings(),
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("初始化 Chroma 失败（%s），改用内置向量库", exc)
            _backend_cache["name"] = "local"

    return LocalVectorStore(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_path,
        embedding_function=get_embeddings(),
    )


def count_documents(vectorstore: Any) -> int:
    """统一获取条目数（兼容 Chroma 与内置实现）。"""
    try:
        if isinstance(vectorstore, LocalVectorStore):
            return int(vectorstore.count())
        return int(vectorstore._collection.count())  # noqa: SLF001 - Chroma 未暴露公共计数接口
    except Exception as exc:  # noqa: BLE001
        logger.debug("统计向量库条目失败: %s", exc)
        return 0


def clear_collection(vectorstore: Any) -> None:
    """清空集合（兼容两种实现）。"""
    if isinstance(vectorstore, LocalVectorStore):
        vectorstore.delete_collection()
        return
    vectorstore.delete_collection()
