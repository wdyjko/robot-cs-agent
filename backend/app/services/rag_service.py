"""RAG 服务：文档加载 → 清洗 → 按条目切分 → 向量化入库 → 混合检索。

切分策略（对应设计文档「不要简单按字数切」）：
- `扫地机器人100问2.txt` / `扫拖一体机器人100问.txt`：`### 分类` + `N. **问题**` 形式
- `故障排除.txt`：`N. 故障现象：...；检测：...；修复：...`
- `维护保养.txt` / `选购指南.txt`：`## 分类` + `N. 内容`
- PDF：按编号条目切，切不动再退化为递归字符切分

所有切分结果都会带 `source_file` / `category` / `item_no` / `title` / `tags` 元数据。
检索采用「向量 Top10 + BM25 Top10 → RRF 融合去重 → Top5」，并预留重排接口。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ..core.config import settings
from ..core.logging import get_logger
from .embeddings import get_embeddings, get_embedding_description

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 可选依赖探测
#
# `langchain_text_splitters` 与 `langchain_community.document_loaders` 在导入时会
# 连带导入 sentence-transformers / torch。在部分 Windows 环境（缺少 VC 运行库、
# 受限沙箱）下 torch 的 c10.dll 会初始化失败并抛 OSError，直接导致整个服务起不来。
# 因此这里全部改成「能导入就用官方实现，不能导入就用内置等价实现」，保证项目
# 在任何环境下都能运行（内置实现同样支持按条目切分与递归字符切分）。
# ---------------------------------------------------------------------------
_LC_SPLITTER_CLASS = None
_LC_SPLITTER_READY = False
_LC_LOADERS_READY: bool | None = None


def _load_langchain_splitter():
    """尝试获取 LangChain 的 RecursiveCharacterTextSplitter（失败返回 None）。"""
    global _LC_SPLITTER_CLASS, _LC_SPLITTER_READY
    if not _LC_SPLITTER_READY:
        _LC_SPLITTER_READY = True
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            _LC_SPLITTER_CLASS = RecursiveCharacterTextSplitter
        except Exception as exc:  # noqa: BLE001 - torch 缺失 / DLL 失败都在这里兜住
            logger.warning("langchain_text_splitters 不可用（%s），使用内置递归切分器", exc)
            _LC_SPLITTER_CLASS = None
    return _LC_SPLITTER_CLASS


def _lc_loaders_ready() -> bool:
    """探测 langchain_community 的 Loader 是否可用（会连带导入 torch）。"""
    global _LC_LOADERS_READY
    if _LC_LOADERS_READY is None:
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader  # noqa: F401

            _LC_LOADERS_READY = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("langchain_community 文档加载器不可用（%s），使用内置加载实现", exc)
            _LC_LOADERS_READY = False
    return bool(_LC_LOADERS_READY)


class SimpleRecursiveSplitter:
    """内置递归字符切分器（LangChain 不可用时的等价兜底）。

    行为与 RecursiveCharacterTextSplitter 基本一致：按分隔符优先级递归切分，
    再贪心合并到 chunk_size，并保留 chunk_overlap 重叠。
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        **_ignored: Any,
    ) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))
        self.separators = separators or ["\n\n", "\n", "。", "；", "，", " ", ""]

    # -- 对外接口，保持与 LangChain 一致 --------------------------------
    def split_text(self, text: str) -> list[str]:
        return self._split(text, 0)

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        documents: list[Document] = []
        metadatas = metadatas or [{} for _ in texts]
        for text, metadata in zip(texts, metadatas):
            for piece in self.split_text(text):
                documents.append(Document(page_content=piece, metadata=dict(metadata)))
        return documents

    # -- 内部实现 -------------------------------------------------------
    def _hard_cut(self, text: str) -> list[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step) if text[i : i + self.chunk_size].strip()]

    def _split(self, text: str, separator_index: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        if separator_index >= len(self.separators):
            return self._hard_cut(text)

        separator = self.separators[separator_index]
        if separator == "" or separator not in text:
            return self._split(text, separator_index + 1)

        raw_segments = text.split(separator)
        # 把分隔符贴回上一段，保持语义完整
        segments = [seg + separator for seg in raw_segments[:-1]] + [raw_segments[-1]]

        chunks: list[str] = []
        buffer = ""
        for segment in segments:
            if not segment.strip():
                continue
            if len(segment) > self.chunk_size:
                if buffer.strip():
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split(segment, separator_index + 1))
                continue

            if len(buffer) + len(segment) <= self.chunk_size:
                buffer += segment
                continue

            if buffer.strip():
                chunks.append(buffer)
            overlap = buffer[-self.chunk_overlap :] if (self.chunk_overlap and buffer) else ""
            buffer = overlap + segment
            while len(buffer) > self.chunk_size:
                chunks.append(buffer[: self.chunk_size])
                buffer = buffer[self.chunk_size - self.chunk_overlap :] if self.chunk_overlap else buffer[self.chunk_size :]

        if buffer.strip():
            chunks.append(buffer)
        return [chunk.strip() for chunk in chunks if chunk.strip()]


def build_text_splitter(chunk_size: int, chunk_overlap: int, separators: list[str]) -> Any:
    """返回可用的文本切分器（优先官方实现，失败用内置实现）。"""
    splitter_class = _load_langchain_splitter()
    if splitter_class is not None:
        try:
            return splitter_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        except Exception as exc:  # noqa: BLE001
            logger.warning("初始化 LangChain 切分器失败（%s），改用内置切分器", exc)
    return SimpleRecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

# ---------------------------------------------------------------------------
# 正则与常量
# ---------------------------------------------------------------------------
# 条目起始：行首的 "12. " / "12、" / "12．" / "12) "
_ITEM_START_RE = re.compile(r"^\s{0,6}(\d{1,3})\s*[.、．)）]\s*(?=\S)")
# Markdown 分类标题：## 或 ###
_HEADING_RE = re.compile(r"^\s{0,3}#{2,4}\s*(.+?)\s*$")
# PDF 页眉页脚 / 页码行
_NOISE_LINE_RE = re.compile(
    r"^\s*(第?\s*\d+\s*页|page\s*\d+|\d+\s*/\s*\d+|-\s*\d+\s*-|扫拖一体机器人100问|扫地机器人100问)\s*$",
    re.IGNORECASE,
)
_ESCAPE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b\ufeff]")

# 标签词表：命中即打标，用于元数据过滤与展示
TAG_KEYWORDS: list[str] = [
    "电池", "充电", "回充", "滤网", "HEPA", "尘盒", "集尘", "主刷", "边刷", "滚刷",
    "拖布", "水箱", "烘干", "自清洁", "洗拖布", "传感器", "激光", "避障", "WiFi", "APP",
    "噪音", "异响", "毛发", "缠绕", "地毯", "木地板", "瓷砖", "宠物", "婴儿", "过敏",
    "吸力", "建图", "地图", "越障", "异味", "保养", "选购", "故障", "湿度", "高温",
    "低温", "除螨", "漏水", "鼓包", "保修", "耗材",
]

# 分类兜底：文件名 -> 默认分类
_FILE_CATEGORY: dict[str, str] = {
    "故障排除.txt": "故障排除",
    "扫地机器人100问.pdf": "常见问题",
    "扫地机器人100问2.txt": "常见问题",
    "扫拖一体机器人100问.txt": "扫拖一体常见问题",
    "维护保养.txt": "维护保养",
    "选购指南.txt": "选购指南",
}

_build_lock = threading.Lock()
_bm25_cache: dict[str, Any] = {"key": None, "docs": [], "bm25": None}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 1. 加载
# ---------------------------------------------------------------------------
def _loader_for(path: Path):
    """按扩展名选择 LangChain Loader（仅在其可用时调用）。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path))
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8")


def _load_text_builtin(path: Path) -> list[Document]:
    """内置 TXT 加载：等价于 TextLoader。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={"source_file": path.name, "file_path": str(path), "loader": "builtin-text"},
        )
    ]


def _load_pdf_builtin(path: Path) -> list[Document]:
    """内置 PDF 加载：用 pypdf 逐页抽取，等价于 PyPDFLoader。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - 单页抽取失败不影响整本
            logger.warning("%s 第 %d 页解析失败: %s", path.name, page_index + 1, exc)
            text = ""
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": path.name,
                    "file_path": str(path),
                    "page": page_index,
                    "loader": "builtin-pypdf",
                },
            )
        )
    return documents


def load_documents(raw_dir: Path | str | None = None) -> list[Document]:
    """从 data/raw 加载全部知识库文件。

    优先使用 PyPDFLoader / TextLoader（设计文档要求）；当 langchain_community
    因 torch 等原生依赖不可用时，自动改用内置的 pypdf / 纯文本加载实现。
    """
    directory = Path(raw_dir) if raw_dir else settings.data_raw_path
    if not directory.exists():
        logger.error("原始数据目录不存在: %s", directory)
        return []

    files = sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".pdf", ".md"}])
    if not files:
        logger.error("原始数据目录为空: %s", directory)
        return []

    use_langchain = _lc_loaders_ready()
    documents: list[Document] = []
    for path in files:
        loaded: list[Document] = []
        if use_langchain:
            try:
                loader = _loader_for(path)
                loaded = loader.load()
            except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响其它文件
                logger.warning("LangChain Loader 加载 %s 失败（%s），改用内置加载器", path.name, exc)
                loaded = []
        if not loaded:
            try:
                loaded = _load_pdf_builtin(path) if path.suffix.lower() == ".pdf" else _load_text_builtin(path)
            except Exception as exc:  # noqa: BLE001
                logger.error("加载 %s 失败: %s", path.name, exc)
                continue

        for doc in loaded:
            doc.metadata["source_file"] = path.name
            doc.metadata.setdefault("file_path", str(path))
        documents.extend(loaded)
        logger.info("已加载 %s（%d 段）", path.name, len(loaded))

    logger.info("共加载 %d 个原始文档段", len(documents))
    return documents


# ---------------------------------------------------------------------------
# 2. 清洗
# ---------------------------------------------------------------------------
def clean_documents(docs: list[Document]) -> list[Document]:
    """清理 PDF 页眉页脚、转义符、多余空白。"""
    cleaned: list[Document] = []
    for doc in docs:
        text = doc.page_content or ""
        text = _ESCAPE_RE.sub("", text)
        text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")

        kept_lines: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                kept_lines.append("")
                continue
            if _NOISE_LINE_RE.match(stripped):
                continue
            kept_lines.append(stripped)

        text = "\n".join(kept_lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 10:  # 过滤空页
            continue
        cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
    logger.info("清洗后剩余 %d 个文档段", len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# 3. 切分
# ---------------------------------------------------------------------------
def _extract_tags(text: str, category: str) -> list[str]:
    """基于关键词表打标签。"""
    tags: list[str] = []
    if category:
        tags.append(category)
    for keyword in TAG_KEYWORDS:
        if keyword in text and keyword not in tags:
            tags.append(keyword)
    return tags[:12]


def _guess_title(first_line: str, item_no: str) -> str:
    """从条目首行猜标题。"""
    text = first_line.strip()
    # 去掉 "12. " 前缀
    text = re.sub(r"^\s{0,6}\d{1,3}\s*[.、．)）]\s*", "", text)
    # 去掉 markdown 加粗
    text = text.replace("**", "").strip()
    # 问答类：问题在首行，直接作为标题
    if text.endswith("？") or text.endswith("?"):
        return text[:80]
    # 故障类：取「故障现象：xxx」部分
    match = re.match(r"^([^；;]{0,60})", text)
    if match:
        candidate = match.group(1).strip("：: ")
        if candidate:
            return candidate[:80]
    return (text[:80] or f"第{item_no}条")


def _split_text_into_items(text: str, source_file: str, page: int | None = None) -> list[Document]:
    """把一个文件（或一页）的文本按编号条目切分。"""
    default_category = _FILE_CATEGORY.get(source_file, "")

    # PDF/长文：先把换行规整，避免 loader 把整页压成一行导致条目识别失败
    normalized = text
    if normalized.count("\n") < 3 and len(normalized) > 200:
        normalized = re.sub(r"(?<=[。；;！？!?])", "\n", normalized)

    items: list[Document] = []
    category = default_category
    buffer: list[str] = []
    current_no = ""
    current_title = ""
    seen_numbers: set[str] = set()

    def flush() -> None:
        nonlocal buffer, current_no, current_title
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        buffer = []
        if len(content) < 8:
            current_no, current_title = "", ""
            return
        metadata = {
            "source_file": source_file,
            "category": category or default_category,
            "file_category": default_category,
            "item_no": current_no,
            "title": current_title,
            "tags": _extract_tags(content, category or default_category),
        }
        if page is not None:
            metadata["page"] = page
        items.append(Document(page_content=content, metadata=metadata))
        current_no, current_title = "", ""

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            if buffer:
                buffer.append("")
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            category = heading.group(1).strip()
            continue

        start = _ITEM_START_RE.match(line)
        if start:
            number = start.group(1)
            # 同一文件里编号从 1 重新开始 => 视为新一章/新文件段
            flush()
            current_no = number
            current_title = _guess_title(line, number)
            buffer = [line.strip()]
            seen_numbers.add(number)
            continue

        if buffer:
            buffer.append(line.strip())
        # 条目外的前言（文件大标题等）直接丢弃，避免污染检索

    flush()
    return items


def split_documents(docs: list[Document]) -> list[Document]:
    """按条目/问答切分；切不出条目的文档退化为递归字符切分。"""
    recursive = build_text_splitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    # 合并同一文件的多个 Document（PDF 分页 / TXT 单文档）
    merged: dict[str, dict[str, Any]] = {}
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page")
        bucket = merged.setdefault(source, {"texts": [], "pages": []})
        bucket["texts"].append(doc.page_content)
        bucket["pages"].append(page)

    all_chunks: list[Document] = []
    for source, bucket in merged.items():
        full_text = "\n".join(bucket["texts"])
        items = _split_text_into_items(full_text, source)

        if len(items) < 3:
            # 条目识别失败（例如 PDF 版式特殊）→ 退化为递归切分
            logger.warning("%s 仅切出 %d 条，退化为递归字符切分", source, len(items))
            fallback = recursive.create_documents(
                [full_text],
                metadatas=[
                    {
                        "source_file": source,
                        "category": _FILE_CATEGORY.get(source, ""),
                        "item_no": "",
                        "title": Path(source).stem,
                        "tags": _extract_tags(full_text, _FILE_CATEGORY.get(source, "")),
                    }
                ],
            )
            all_chunks.extend(fallback)
            continue

        # 超长条目二次切分，保留元数据
        for item in items:
            if len(item.page_content) <= settings.chunk_size * 2:
                all_chunks.append(item)
                continue
            for piece in recursive.split_text(item.page_content):
                meta = dict(item.metadata)
                meta["title"] = item.metadata.get("title", "")
                meta["split"] = "continuation"
                all_chunks.append(Document(page_content=piece, metadata=meta))

    # 去重（同一文件里重复条目）+ 生成稳定 id
    unique: list[Document] = []
    seen: set[str] = set()
    for chunk in all_chunks:
        content = chunk.page_content.strip()
        if not content:
            continue
        key = hashlib.md5((chunk.metadata.get("source_file", "") + content).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        chunk.metadata["chunk_id"] = key
        unique.append(chunk)

    logger.info("切分完成：%d 条知识条目", len(unique))
    return unique


# ---------------------------------------------------------------------------
# 4. 向量库
# ---------------------------------------------------------------------------
def get_vectorstore():
    """获取向量库实例（委托给 vectorstore_service，支持 Chroma / 内置实现）。"""
    from .vectorstore_service import get_vectorstore as _get_vectorstore

    return _get_vectorstore()


def _doc_id(doc: Document, index: int) -> str:
    base = f"{doc.metadata.get('source_file', '')}-{doc.metadata.get('item_no', '')}-{index}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def enrich_for_index(doc: Document) -> Document:
    """构造「用于检索的文本」。

    向量化与 BM25 都使用增强后的文本，把文件名、分类、标题也纳入检索范围，
    这样「怎么保养」「选购要注意什么」这类问题才能命中对应文件里的条目
    （例如 维护保养.txt 的条目正文里往往并不出现「保养」二字）。

    原始正文保存在 metadata['raw_content'] 中，对外展示与拼 Prompt 时使用它。
    """
    meta = dict(doc.metadata)
    raw_content = doc.page_content
    header_parts = [
        str(meta.get("source_file", "")).replace(".txt", "").replace(".pdf", ""),
        str(meta.get("file_category", "")),
        str(meta.get("category", "")),
        str(meta.get("title", "")),
    ]
    # 去重但保持顺序（文件名与文件级分类常常相同）
    header_parts = list(dict.fromkeys(part for part in header_parts if part))
    header = " | ".join(header_parts)
    tags = meta.get("tags") or []
    tag_line = " ".join(str(tag) for tag in tags)
    index_text = f"{header}\n{tag_line}\n{raw_content}".strip()

    meta["raw_content"] = raw_content
    return Document(page_content=index_text, metadata=meta)


def index_documents(chunks: list[Document]) -> list[Document]:
    """批量构造检索用文档。"""
    return [enrich_for_index(chunk) for chunk in chunks]


def _save_bm25_corpus(docs: list[Document]) -> None:
    """把切分结果落盘，供 BM25 检索加载。"""
    payload = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    path = settings.bm25_corpus_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    meta_path = path.parent / "kb_meta.json"
    from .vectorstore_service import get_backend_name

    meta_path.write_text(
        json.dumps(
            {
                "collection": settings.chroma_collection,
                "document_count": len(docs),
                "built_at": datetime.now().isoformat(timespec="seconds"),
                "vector_backend": get_backend_name(),
                **get_embedding_description(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # 语料变了，缓存必须失效
    with _cache_lock:
        _bm25_cache.update({"key": None, "docs": [], "bm25": None})


def build_vectorstore() -> int:
    """全量重建向量库，返回条目数。串行执行，避免并发重建互相踩踏。"""
    with _build_lock:
        started = time.time()
        docs = load_documents()
        if not docs:
            logger.error("未加载到任何知识库文件，请检查 DATA_RAW_DIR 配置")
            return 0
        docs = clean_documents(docs)
        chunks = split_documents(docs)
        if not chunks:
            logger.error("切分后没有产生任何知识条目")
            return 0

        vectorstore = get_vectorstore()
        from .vectorstore_service import clear_collection

        try:
            clear_collection(vectorstore)
            logger.info("已清空旧集合: %s", settings.chroma_collection)
        except Exception as exc:  # noqa: BLE001 - 首次构建时集合不存在
            logger.info("无需清空旧集合（%s）", exc)
        vectorstore = get_vectorstore()

        # 构造检索用文本（文件名 + 分类 + 标题 + 标签 + 正文），原始正文存进 raw_content
        index_docs = index_documents(chunks)

        # 分批写入，避免一次性占用过多内存
        batch_size = 64
        for start in range(0, len(index_docs), batch_size):
            batch = index_docs[start : start + batch_size]
            ids = [_doc_id(doc, start + i) for i, doc in enumerate(batch)]
            vectorstore.add_documents(documents=batch, ids=ids)
            logger.info("已写入 %d/%d 条", min(start + batch_size, len(index_docs)), len(index_docs))

        _save_bm25_corpus(index_docs)
        elapsed = time.time() - started
        logger.info("知识库构建完成：%d 条，耗时 %.1fs", len(chunks), elapsed)
        return len(chunks)


# ---------------------------------------------------------------------------
# 5. 检索
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """轻量中文分词：双字组合 + 英文数字词。

    不加入单个汉字——单字几乎在所有文档里都出现，会把「保养」「滤网」这类
    真正有区分度的双字词的权重淹没掉（实测去掉单字后排序明显更准）。
    """
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", text)
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = [chars[i] + chars[i + 1] for i in range(max(0, len(chars) - 1))]
    return words + bigrams


def _load_bm25():
    """加载 BM25 索引（按语料文件指纹缓存）。"""
    path = settings.bm25_corpus_path
    if not path.exists():
        return [], None

    stat = path.stat()
    cache_key = f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    with _cache_lock:
        if _bm25_cache["key"] == cache_key:
            return _bm25_cache["docs"], _bm25_cache["bm25"]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        docs = [Document(page_content=item["content"], metadata=item.get("metadata", {})) for item in payload]
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(corpus) if corpus else None
    except Exception as exc:  # noqa: BLE001
        logger.error("加载 BM25 索引失败: %s", exc)
        return [], None

    with _cache_lock:
        _bm25_cache.update({"key": cache_key, "docs": docs, "bm25": bm25})
    return docs, bm25


def _vector_search(query: str, k: int) -> list[Document]:
    """向量检索。库为空时返回空列表。"""
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=k)
        return results
    except Exception as exc:  # noqa: BLE001
        logger.error("向量检索失败: %s", exc)
        return []


def _bm25_search(query: str, k: int) -> list[Document]:
    """BM25 关键词检索。"""
    docs, bm25 = _load_bm25()
    if not docs or bm25 is None:
        return []
    try:
        scores = bm25.get_scores(tokenize(query))
    except Exception as exc:  # noqa: BLE001
        logger.error("BM25 检索失败: %s", exc)
        return []

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits: list[Document] = []
    for index in ranked[:k]:
        if scores[index] <= 0:
            continue
        doc = docs[index]
        hits.append(Document(page_content=doc.page_content, metadata=dict(doc.metadata)))
    return hits


def _rerank(query: str, docs: list[Document]) -> list[Document]:
    """重排接口（预留）。启用 RERANK_ENABLED 且模型可用时生效。"""
    if not settings.rerank_enabled or not docs:
        return docs
    try:
        from sentence_transformers import CrossEncoder

        model_name = settings.rerank_model
        if not hasattr(_rerank, "_model"):
            _rerank._model = CrossEncoder(model_name)  # type: ignore[attr-defined]
        model = _rerank._model  # type: ignore[attr-defined]
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
        return [docs[i] for i in order]
    except Exception as exc:  # noqa: BLE001
        logger.warning("重排模型不可用，跳过重排: %s", exc)
        return docs


def _rrf_fuse(result_lists: list[list[Document]], k: int = 60, weights: list[float] | None = None) -> list[Document]:
    """RRF 融合多路检索结果，按 chunk_id/内容去重。

    weights 允许给不同检索路不同权重：内置哈希向量器语义能力弱时，会让 BM25 占更高权重，
    避免弱向量结果把 BM25 的准确命中挤出 Top-k。
    """
    scores: dict[str, float] = {}
    store: dict[str, Document] = {}
    weights = weights or [1.0] * len(result_lists)

    for list_index, results in enumerate(result_lists):
        weight = weights[list_index] if list_index < len(weights) else 1.0
        for rank, doc in enumerate(results):
            key = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank + 1))
            store.setdefault(key, doc)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[Document] = []
    for key, score in ordered:
        doc = store[key]
        metadata = dict(doc.metadata)
        metadata["score"] = round(float(score), 6)
        fused.append(Document(page_content=doc.page_content, metadata=metadata))
    return fused


def _discriminative_units(text: str) -> set[str]:
    """抽取区分度较高的检索单元：中文双字组合 + 英文数字词。"""
    lowered = (text or "").lower()
    chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = {chars[i] + chars[i + 1] for i in range(max(0, len(chars) - 1))}
    words = set(re.findall(r"[a-z0-9]{2,}", lowered))
    return bigrams | words


def _query_coverage(query: str, content: str) -> float:
    """查询词在文档中的覆盖率（0~1），用于抑制「关键词几乎不沾边」的结果。"""
    query_units = _discriminative_units(query)
    if not query_units:
        return 0.0
    content_units = _discriminative_units(content)
    return len(query_units & content_units) / len(query_units)


def _apply_coverage_boost(query: str, docs: list[Document]) -> list[Document]:
    """融合「RRF 名次 + 词面覆盖度」重新打分。

    词面覆盖度包含三路信号：
    - content_coverage：查询词在正文（含增强头部）中的覆盖率
    - title_coverage  ：查询词在条目标题中的覆盖率（标题命中通常最相关）
    - tag_coverage    ：查询词命中条目标签的比例（标签是人工归纳的关键词）

    这一步相当于一个廉价的「重排」：不需要额外模型，就能明显提升关键词类问题的精度。
    """
    query_units = _discriminative_units(query)
    scored: list[tuple[float, Document]] = []

    for doc in docs:
        base = float(doc.metadata.get("score", 0.0) or 0.0)
        metadata = dict(doc.metadata)

        content_coverage = _query_coverage(query, doc.page_content)
        title_text = " ".join(
            str(metadata.get(key, "")) for key in ("title", "category", "file_category")
        )
        title_coverage = _query_coverage(query, title_text) if title_text.strip() else 0.0
        tags = metadata.get("tags") or []
        tag_units = _discriminative_units(" ".join(str(tag) for tag in tags))
        tag_coverage = (len(query_units & tag_units) / len(query_units)) if query_units else 0.0

        score = base + 0.006 * content_coverage + 0.010 * title_coverage + 0.010 * tag_coverage
        metadata["coverage"] = round(content_coverage, 4)
        metadata["title_coverage"] = round(title_coverage, 4)
        metadata["tag_coverage"] = round(tag_coverage, 4)
        metadata["score"] = round(score, 6)
        scored.append((score, Document(page_content=doc.page_content, metadata=metadata)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored]


def _embedding_is_degraded() -> bool:
    """当前是否在用内置哈希向量器（语义能力弱，需要给 BM25 更高权重）。"""
    try:
        from .embeddings import HashingEmbeddings, get_embeddings

        return isinstance(get_embeddings(), HashingEmbeddings)
    except Exception:  # noqa: BLE001
        return False


class HybridRetriever:
    """混合检索器：向量 Top10 + BM25 Top10 → RRF 融合 → Top5。"""

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.final_top_k

    def invoke(self, query: str, **_: Any) -> list[Document]:
        return self.retrieve(query, self.top_k)

    # 兼容 LangChain 的 Runnable 调用方式
    def __call__(self, query: str) -> list[Document]:
        return self.invoke(query)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        query = (query or "").strip()
        k = k or self.top_k
        if not query:
            return []

        vector_hits = _vector_search(query, settings.vector_top_k)
        bm25_hits = _bm25_search(query, settings.bm25_top_k)

        # 内置哈希向量器语义弱 → 提高 BM25 权重，避免弱向量结果挤掉关键词精确命中
        if _embedding_is_degraded():
            weights = [0.35, 1.0]
        else:
            weights = [1.0, 1.0]

        fused = _rrf_fuse([vector_hits, bm25_hits], weights=weights)
        fused = _apply_coverage_boost(query, fused)
        fused = _rerank(query, fused)
        return fused[:k]


def get_retriever(top_k: int | None = None) -> HybridRetriever:
    """返回混合检索器。"""
    return HybridRetriever(top_k=top_k)


def search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """检索知识库，返回带 metadata 的字典列表。"""
    k = k or settings.final_top_k
    docs = get_retriever(top_k=k).retrieve(query, k)
    results: list[dict[str, Any]] = []
    for doc in docs:
        metadata = dict(doc.metadata)
        # 展示 / 拼 Prompt 使用原始正文（检索用的是增强文本）
        content = str(metadata.get("raw_content") or doc.page_content)
        results.append(
            {
                "content": content,
                "metadata": metadata,
                "source_file": metadata.get("source_file", ""),
                "category": metadata.get("category", ""),
                "item_no": metadata.get("item_no", ""),
                "title": metadata.get("title", ""),
                "tags": metadata.get("tags", []),
                "score": metadata.get("score", 0.0),
            }
        )
    return results


def docs_to_context(docs: list[dict[str, Any]]) -> str:
    """把检索结果拼成 Prompt 里的资料片段。"""
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {}) or {}
        label = f"{meta.get('source_file', '未知文件')} - {meta.get('item_no', '')}. {meta.get('title', '')}".strip()
        blocks.append(f"[片段{index}] 来源：{label}\n{doc.get('content', '')}")
    return "\n\n".join(blocks) if blocks else "（未检索到相关内容）"


def get_status() -> dict[str, Any]:
    """知识库状态：集合名、条目数、持久化目录。"""
    status: dict[str, Any] = {
        "collection": settings.chroma_collection,
        "document_count": 0,
        "persist_dir": str(settings.chroma_persist_path),
        "ready": False,
        "embedding_model": (settings.embedding_model_path or "").strip() or settings.embedding_model,
        "embedding_backend": "",
        "vector_backend": "",
        "last_build": "",
        "needs_rebuild": False,
    }

    # 优先从向量库读取真实条目数
    try:
        from .vectorstore_service import count_documents, get_backend_name

        vectorstore = get_vectorstore()
        status["document_count"] = count_documents(vectorstore)
        status["ready"] = status["document_count"] > 0
        status["vector_backend"] = get_backend_name()
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取向量库状态失败: %s", exc)
        # 退化用 BM25 语料条数
        docs, _ = _load_bm25()
        status["document_count"] = len(docs)
        status["ready"] = bool(docs)

    try:
        meta_path = settings.chroma_persist_path / "kb_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status["last_build"] = meta.get("built_at", "")
            status["embedding_backend"] = meta.get("backend", "")
            status["built_backend"] = meta.get("backend", "")
            status["built_vector_backend"] = meta.get("vector_backend", "")
            if not status["document_count"]:
                status["document_count"] = int(meta.get("document_count", 0))
                status["ready"] = status["document_count"] > 0

            # 建库时的向量后端与当前配置不一致 → 需要重建（否则向量维度不匹配会检索失败）
            current = get_embedding_description()
            from .vectorstore_service import get_backend_name as _current_vector_backend

            if meta.get("vector_backend") and meta.get("vector_backend") != _current_vector_backend():
                status["needs_rebuild"] = True
                logger.warning(
                    "知识库是用向量库后端 %s 构建的，当前为 %s，请重建知识库：python scripts/build_kb.py",
                    meta.get("vector_backend"),
                    _current_vector_backend(),
                )
            elif meta.get("backend") and meta.get("backend") != current["backend"]:
                status["needs_rebuild"] = True
                logger.warning(
                    "知识库是用 %s 构建的，当前生效的是 %s，向量维度不一致，请重建知识库："
                    "python scripts/build_kb.py",
                    meta.get("backend"),
                    current["backend"],
                )
            elif meta.get("model") and meta.get("model") != current["model"]:
                status["needs_rebuild"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取知识库元信息失败: %s", exc)

    if not status["embedding_backend"]:
        try:
            status["embedding_backend"] = get_embedding_description()["backend"]
        except Exception:  # noqa: BLE001
            status["embedding_backend"] = ""

    return status
