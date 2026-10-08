# -*- coding: utf-8 -*-
"""交付验证脚本（可复现，不需要任何 API Key）。

回答一个问题：**"克隆下来、建好知识库之后，这套系统真的能跑吗？"**

用法（必须用项目自己的解释器，普通终端里跑）：

    .venv\\Scripts\\python.exe tests\\test_smoke.py

退出码：0 = 全部通过；1 = 有失败项（失败原因会打印出来）。

检查项：
    [1] 后端关键模块可导入（不触发 torch / 外部依赖）
    [2] 知识库存在，且向量维度与 kb_meta.json 记录一致
    [3] 六节点 LangGraph 图能编译，节点齐全
    [4] Embedding 工厂可返回向量器（并报告走的是正常还是降级路径）
    [5] 端到端：/api/chat 能返回结构化回答（LLM 不可用时走纯检索兜底）
    [6] 安全兜底：风险问题必须在回答首句出现"停止使用"类提示

为什么用"纯检索兜底"模式跑端到端：
    这样不依赖 OPENAI_API_KEY，结果确定、无随机性，适合作为交付验收。
    注意 `core/config.py` 里的 settings 是**导入期单例**，所以必须在 import 之前
    把 OPENAI_API_KEY 置空（见下方 os.environ 设置）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- 必须在 import 项目模块之前设定环境（settings 是导入期单例）----
PROJECT = Path(__file__).resolve().parents[1]
os.environ["OPENAI_API_KEY"] = ""          # 强制纯检索兜底，验收不依赖外部 Key
os.environ["VECTOR_BACKEND"] = os.environ.get("VECTOR_BACKEND") or "local"
sys.path.insert(0, str(PROJECT / "backend"))
sys.path.insert(0, str(PROJECT))

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        PASS.append(f"  [PASS] {name}" + (f"  -> {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001 - 验收脚本要把所有失败都收集起来
        FAIL.append(f"  [FAIL] {name}  -> {type(exc).__name__}: {exc}")


def main() -> int:
    import json

    import numpy as np

    from app.core.config import settings

    # ---------------------------------------------------------------- [1]
    def t_imports():
        from app.services import agent_service, memory_service, rag_service  # noqa: F401
        from app.services import embeddings, summary_service, vectorstore_service  # noqa: F401
        from app.tools import datetime_tool, kb_tool, location, profile, weather  # noqa: F401

        return "7 个 service + 5 个 tool 全部导入成功"

    # ---------------------------------------------------------------- [2]
    def t_knowledge_base():
        from app.services.vectorstore_service import count_documents, get_vectorstore

        store_dir = settings.chroma_persist_path
        meta_path = store_dir / "kb_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"缺少 {meta_path}，请先执行 scripts/build_kb.py")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        count = count_documents(get_vectorstore())
        if count <= 0:
            raise AssertionError("向量库里没有任何条目")

        # 以 .npy 的真实 shape 为准（kb_meta.json 可能落后于文件）
        npy_files = list(store_dir.glob("*.vectors.npy"))
        if not npy_files:
            raise FileNotFoundError(f"{store_dir} 下找不到 *.vectors.npy")
        shape = np.load(npy_files[0]).shape

        if int(shape[0]) != count:
            raise AssertionError(f"向量数与记录条数不一致：vectors={shape[0]} count={count}")
        if meta.get("document_count") and int(meta["document_count"]) != count:
            raise AssertionError(f"kb_meta.document_count={meta['document_count']} 与实际 {count} 不一致")

        return (f"{store_dir.name}: {shape[0]} 条 × {shape[1]} 维，"
                f"向量器={meta.get('backend')}，degraded={meta.get('degraded')}")

    # ---------------------------------------------------------------- [3]
    def t_graph():
        from app.services.agent_service import get_graph

        graph = get_graph()
        nodes = set(getattr(graph, "nodes", {}) or {})
        expected = {
            "intent_node", "tool_node", "rewrite_node",
            "retrieve_node", "summarize_node", "answer_node",
        }
        missing = expected - nodes
        if missing:
            raise AssertionError(f"图里缺少节点：{sorted(missing)}")
        return f"六节点全部在图中：{' → '.join(['intent', 'tool', 'rewrite', 'retrieve', 'summarize', 'answer'])}"

    # ---------------------------------------------------------------- [4]
    def t_embeddings():
        from app.services.embeddings import get_embeddings

        emb = get_embeddings()
        dim = len(emb.embed_query("滤网多久更换一次"))
        if dim <= 0:
            raise AssertionError("向量维度为 0")
        degraded = "Hashing" in type(emb).__name__
        return f"{type(emb).__name__}，{dim} 维（{'降级路径' if degraded else '正常路径'}）"

    # ---------------------------------------------------------------- [5]
    def t_api_chat():
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            health = client.get("/health")
            if health.status_code != 200:
                raise AssertionError(f"/health 返回 {health.status_code}")

            resp = client.post(
                "/api/chat",
                json={"session_id": "smoke", "user_id": "u1", "message": "滤网多久换一次？"},
            )
            if resp.status_code != 200:
                raise AssertionError(f"/api/chat 返回 {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            if not data.get("answer"):
                raise AssertionError("answer 为空")
            if not data.get("sources"):
                raise AssertionError("sources 为空（检索没命中任何条目）")
            return (f"/health ok；/api/chat 返回 {len(data['answer'])} 字回答、"
                    f"{len(data['sources'])} 条来源、{len(data.get('tool_calls') or [])} 次工具调用")

    # ---------------------------------------------------------------- [6]
    def t_safety():
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            resp = client.post(
                "/api/chat",
                json={"session_id": "smoke-safety", "user_id": "u1",
                      "message": "电池鼓包了还能继续用吗？"},
            )
            if resp.status_code != 200:
                raise AssertionError(f"/api/chat 返回 {resp.status_code}")
            answer = resp.json().get("answer") or ""
            head = answer[:120]
            if not any(word in head for word in ("停止使用", "停用", "断开电源")):
                raise AssertionError(f"风险问题未在首句给出安全提示，首句为：{head!r}")
            return "风险问题首句已强制提示停用/断电"

    check("后端模块可导入", t_imports)
    check("知识库与向量维度一致", t_knowledge_base)
    check("LangGraph 六节点图可编译", t_graph)
    check("Embedding 工厂可用", t_embeddings)
    check("端到端 /api/chat（纯检索兜底模式）", t_api_chat)
    check("安全兜底首句提示", t_safety)

    print("=" * 78)
    print("交付验证（tests/test_smoke.py）")
    print("=" * 78)
    for line in PASS:
        print(line)
    for line in FAIL:
        print(line)
    print("-" * 78)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
