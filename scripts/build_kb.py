#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立的知识库构建脚本。

用法（在项目根目录 robot-cs-agent 下执行）：

    # Windows
    .venv\\Scripts\\python.exe scripts\\build_kb.py

    # macOS / Linux
    .venv/bin/python scripts/build_kb.py

可选参数：
    --raw-dir      指定 data/raw 目录
    --collection   指定 Chroma 集合名
    --query        构建完成后用一句话验证检索效果
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 让脚本能直接 import backend/app 里的包（app.*）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
for path in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.services import rag_service  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建扫地机器人客服知识库")
    parser.add_argument("--raw-dir", default=None, help="原始知识库目录，默认读取 DATA_RAW_DIR")
    parser.add_argument("--collection", default=None, help="Chroma 集合名，默认读取 CHROMA_COLLECTION")
    parser.add_argument("--query", default="滤网多久换一次", help="构建完成后用于验证检索的查询")
    parser.add_argument("--no-verify", action="store_true", help="跳过检索验证")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=settings.log_level)

    if args.collection:
        settings.chroma_collection = args.collection
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else settings.data_raw_path

    print("=" * 72)
    print("扫地机器人客服 —— 知识库构建")
    print("=" * 72)
    print(f"原始数据目录 : {raw_dir}")
    print(f"向量库目录   : {settings.chroma_persist_path}")
    print(f"集合名称     : {settings.chroma_collection}")
    print(f"Embedding    : {(settings.embedding_model_path or '').strip() or settings.embedding_model}")
    print("-" * 72)

    if not raw_dir.exists():
        print(f"[错误] 原始数据目录不存在：{raw_dir}")
        return 1

    files = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in {".txt", ".pdf", ".md"}])
    if not files:
        print(f"[错误] 目录中没有 .txt/.pdf 文件：{raw_dir}")
        return 1
    print(f"发现 {len(files)} 个知识库文件：")
    for path in files:
        print(f"  - {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    print("-" * 72)

    started = time.time()
    documents = rag_service.load_documents(raw_dir)
    if not documents:
        print("[错误] 未加载到任何文档")
        return 1

    cleaned = rag_service.clean_documents(documents)
    chunks = rag_service.split_documents(cleaned)
    if not chunks:
        print("[错误] 切分后没有产生知识条目")
        return 1

    # 统计各来源条目数，便于确认 6 个文件都被正确切分
    by_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        by_source[source] = by_source.get(source, 0) + 1
    print("各文件切分结果：")
    for source, count in sorted(by_source.items(), key=lambda item: -item[1]):
        print(f"  - {source}: {count} 条")
    print("-" * 72)

    count = rag_service.build_vectorstore()
    elapsed = time.time() - started
    print(f"构建完成：{count} 条知识条目，耗时 {elapsed:.1f}s")
    print(f"向量库位置：{settings.chroma_persist_path}")

    status = rag_service.get_status()
    print(f"状态校验  ：collection={status['collection']} count={status['document_count']} ready={status['ready']}")

    if not args.no_verify and args.query:
        print("-" * 72)
        print(f"检索验证：「{args.query}」")
        for index, doc in enumerate(rag_service.search(args.query, k=3), start=1):
            meta = doc.get("metadata", {})
            print(f"  [{index}] {meta.get('source_file')} - {meta.get('item_no')}. {meta.get('title')}")
            print(f"      得分: {doc.get('score')}")
            print(f"      内容: {str(doc.get('content', ''))[:80]}...")

    print("=" * 72)
    print("下一步：")
    print("  1) 启动后端：.venv/Scripts/python.exe -m uvicorn backend.app.main:app --reload --port 8000")
    print("  2) 启动前端：.venv/Scripts/python.exe -m streamlit run frontend/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
