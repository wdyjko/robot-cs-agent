# -*- coding: utf-8 -*-
"""检索效果评估 —— 扫地机器人客服知识库（robot-cs-agent）

对同一套人工评测集，跑「两种向量器」×「多种融合策略」的对照实验：

    向量器：h  = 内置 HashingEmbeddings（384 维，降级兜底路径）
            bge = BAAI/bge-m3（1024 维，正常路径）
    策略：  A 纯向量      B 纯关键词(BM25)    C RRF融合    D RRF+覆盖度(线上配置)

指标：Recall@5（正确答案有没有进前5）、MRR（正确答案排在多前）

用法（必须用项目自己的 venv 解释器）：
    set EVAL_EMBED=h
    set EVAL_EMBED=bge
    D:\\develop\\robot-cs-agent\\.venv\\Scripts\\python.exe 评估脚本

环境变量说明：
    EVAL_EMBED=h|bge              用哪种向量器（默认 bge）
    EVAL_PROJECT=<项目根目录>      默认 D:\\develop\\robot-cs-agent
    EVAL_STORE_HASH=<目录>        384 维向量库目录，默认 <项目>/vectorstore_hash384
    EVAL_STORE_BGE=<目录>         1024 维向量库目录，默认 <项目>/vectorstore_bge1024
    EVAL_OUT=<输出文件>            默认 桌面/面试/评估结果_<向量器>.txt

注意：COMPARE 里两种向量器沿用同一套 GROUND_TRUTH，且 ground truth 只用
「来源文件 + 条目号」定位，因此同题在两种规模下都成立。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT = Path(os.environ.get("EVAL_PROJECT") or r"D:\develop\robot-cs-agent")
EMBED = (os.environ.get("EVAL_EMBED") or "bge").strip().lower()
STORE_HASH = Path(os.environ.get("EVAL_STORE_HASH") or (PROJECT / "vectorstore_hash384"))
STORE_BGE = Path(os.environ.get("EVAL_STORE_BGE") or (PROJECT / "vectorstore_bge1024"))

# ---------------------------------------------------------------------------
# 必须在 import 项目模块「之前」把环境定好：
#   - CHROMA_PERSIST_DIR 决定读哪个向量库
#   - EMBEDDING_ALLOW_DOWNLOAD=false 才不会被自动降级成哈希向量器
#   - VECTOR_BACKEND=local 直接使用内置 NumPy 向量库，避免 chromadb 原生崩溃
# ---------------------------------------------------------------------------
if EMBED == "h":
    os.environ["EMBEDDING_ALLOW_DOWNLOAD"] = "false"
    os.environ["CHROMA_PERSIST_DIR"] = str(STORE_HASH)
else:
    os.environ["EMBEDDING_ALLOW_DOWNLOAD"] = "true"
    os.environ["CHROMA_PERSIST_DIR"] = str(STORE_BGE)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["VECTOR_BACKEND"] = os.environ.get("VECTOR_BACKEND") or "local"

for p in (str(PROJECT / "backend"), str(PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from langchain_core.documents import Document  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services import rag_service as R  # noqa: E402

# ---------------------------------------------------------------------------
# 人工评测集：query -> (来源文件, 条目号)；None = 知识库确实没有答案（不计分）
#
# 设计要点：
#   1. 30 题覆盖全部 6 份源文件（每份 5 题），单文件覆盖 100/200/210 条中的不同区间；
#   2. query 是「口语化改写」，刻意不等于条目标题原文，避免标题字面命中作弊；
#   3. ground truth 用「文件+条目号」定位 —— 3 份文件各自独立编号，
#      因此该组合唯一（例如 扫地机器人100问.pdf#22 ≠ 扫地机器人100问2.txt#22）。
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[str, tuple[str, str] | None] = {
    # ---- 扫地机器人100问.pdf（6 题，分布在 12~83 条）----
    "机器人最高能爬过多高的障碍物？": ("扫地机器人100问.pdf", "12"),
    "怎么设置每天固定时间自动打扫？": ("扫地机器人100问.pdf", "22"),
    "这台机器支持小爱同学或者天猫精灵吗？": ("扫地机器人100问.pdf", "26"),
    "尘盒一般多久需要清理一次？": ("扫地机器人100问.pdf", "29"),
    "电池一直插着充电会不会把电池弄坏？": ("扫地机器人100问.pdf", "76"),
    "冬天零下的时候机器人还能正常用吗？": ("扫地机器人100问.pdf", "83"),
    # ---- 扫地机器人100问2.txt（6 题）----
    "刚买回来第一次用，需要先做什么准备？": ("扫地机器人100问2.txt", "1"),
    "机器人扫完地自己却不回去充电，怎么办？": ("扫地机器人100问2.txt", "10"),
    "HEPA 滤网洗一洗还能接着用吗？": ("扫地机器人100问2.txt", "23"),
    "感觉续航比以前短了很多，是什么原因？": ("扫地机器人100问2.txt", "26"),
    "滚刷仓里有股味道，怎么去除？": ("扫地机器人100问2.txt", "33"),
    "机器人在长毛地毯上走不动，卡住了怎么办？": ("扫地机器人100问2.txt", "75"),
    # ---- 扫拖一体机器人100问.txt（6 题）----
    "水箱里的水漏出来了，怎么处理？": ("扫拖一体机器人100问.txt", "11"),
    "拖完地地上出现一条一条的水痕，怎么避免？": ("扫拖一体机器人100问.txt", "19"),
    "厨房地面油污很重，拖不干净怎么办？": ("扫拖一体机器人100问.txt", "26"),
    "木地板拖完起翘变形了怎么处理？": ("扫拖一体机器人100问.txt", "31"),
    "南方的回南天怎么用机器人拖地比较合适？": ("扫拖一体机器人100问.txt", "66"),
    "宠物掉毛太多，拖布总被毛发缠住怎么办？": ("扫拖一体机器人100问.txt", "72"),
    # ---- 故障排除.txt（6 题，含 100 条以后的高编号区间）----
    "机器人满电之后马上就断电了，是什么故障？": ("故障排除.txt", "24"),
    "地上有水的时候机器人工作会突然停机吗？": ("故障排除.txt", "60"),
    "工作时电量掉得特别快，远超过正常续航": ("故障排除.txt", "51"),
    "机器人在瓷砖地面上打滑走不动": ("故障排除.txt", "69"),
    "机器人充电的时候竟然冒出了电火花": ("故障排除.txt", "46"),
    "电池充了很久也充不满，指示灯却是正常的": ("故障排除.txt", "190"),
    # ---- 维护保养.txt（5 题，条目是「养护动作」而不是问句）----
    "水箱里结了水垢，用什么泡一泡能清掉？": ("维护保养.txt", "5"),
    "机器人要放很久不用，电池该怎么存放？": ("维护保养.txt", "11"),
    "每次用完需要擦机身吗？用什么擦？": ("维护保养.txt", "1"),
    "拖布用几次要消毒？家里有宠物和宝宝要注意什么？": ("维护保养.txt", "38"),
    "水泥地扫完之后要做什么额外清理？": ("维护保养.txt", "18"),
    # ---- 选购指南.txt（5 题，条目是「选购要点」----
    "家里是木地板，买的时候要注意选什么功能？": ("选购指南.txt", "31"),
    "滤网是买可水洗的还是不可水洗的？": ("选购指南.txt", "61"),
    "想买带灰尘感应自动调吸力的，有用吗？": ("选购指南.txt", "91"),
    "能提前预约未来几天的清扫计划吗？": ("选购指南.txt", "181"),
    "怎么判断一台机器性价比高不高？": ("选购指南.txt", "200"),
    # ---- 域外题（知识库确实没有答案）----
    # 用来验证「无答案时会不会硬答」：检索返回空 = 正确拒答；返回内容 = 幻觉风险
    "洗衣机脱水的时候声音很大，怎么修？": None,
    "这款扫地机器人的手机充电头是快充吗？": None,
}

CONFIGS = ["A 纯向量", "B 纯关键词(BM25)", "C RRF融合", "D RRF+覆盖度(线上)"]


def key_of(doc: Document) -> tuple[str, str]:
    m = doc.metadata or {}
    return (str(m.get("source_file", "")), str(m.get("item_no", "")))


def weights_now() -> list[float]:
    """当前向量器是否处于降级状态，决定 RRF 两路权重。"""
    return [0.35, 1.0] if R._embedding_is_degraded() else [1.0, 1.0]


def coverage_boost(query: str, docs: list[Document], w_content: float, w_title: float, w_tag: float) -> list[Document]:
    """可调系数的覆盖度加成（复刻 rag_service._apply_coverage_boost）。"""
    query_units = R._discriminative_units(query)
    scored: list[tuple[float, Document]] = []
    for doc in docs:
        base = float((doc.metadata or {}).get("score", 0.0) or 0.0)
        metadata = dict(doc.metadata or {})
        content_cov = R._query_coverage(query, doc.page_content)
        title_text = " ".join(str(metadata.get(k, "")) for k in ("title", "category", "file_category"))
        title_cov = R._query_coverage(query, title_text) if title_text.strip() else 0.0
        tags = metadata.get("tags") or []
        tag_units = R._discriminative_units(" ".join(str(t) for t in tags))
        tag_cov = (len(query_units & tag_units) / len(query_units)) if query_units else 0.0
        score = base + w_content * content_cov + w_title * title_cov + w_tag * tag_cov
        metadata["score"] = round(score, 6)
        scored.append((score, Document(page_content=doc.page_content, metadata=metadata)))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored]


def run_config(name: str, q: str, k: int = 5) -> list[tuple[str, str]]:
    if name.startswith("A"):
        return [key_of(d) for d in R._vector_search(q, k)]
    if name.startswith("B"):
        return [key_of(d) for d in R._bm25_search(q, k)]
    v = R._vector_search(q, settings.vector_top_k)
    b = R._bm25_search(q, settings.bm25_top_k)
    fused = R._rrf_fuse([v, b], weights=weights_now())
    if name.startswith("D"):
        fused = R._apply_coverage_boost(q, fused)
    return [key_of(d) for d in fused[:k]]


def live_info() -> tuple[str, int, list[float], int]:
    """直接问「当前真实生效」的向量器与向量库，不读可能过期的 kb_meta.json。

    注意：rag_service.get_status() 的 embedding_backend 字段会被 kb_meta.json 里的
    历史记录覆盖，用它来判断"当前用的是 384 维哈希还是 1024 维 bge-m3"会误判。
    """
    from app.services.embeddings import get_embeddings
    from app.services.vectorstore_service import count_documents, get_vectorstore

    emb = get_embeddings()
    dim = len(emb.embed_query("维度探测"))
    return type(emb).__name__, dim, weights_now(), count_documents(get_vectorstore())


def main() -> None:
    backend, dim, weights, doc_count = live_info()
    degraded = "Hashing" in backend
    out: list[str] = []
    out.append("检索效果评估 —— 扫地机器人客服知识库 (robot-cs-agent)")
    out.append(f"知识库条目数：{doc_count}    collection：{settings.chroma_collection}")
    out.append(f"向量器：{backend}（{dim} 维）    向量后端：{settings.vector_backend}    持久化目录：{settings.chroma_persist_path}")
    out.append(f"评测集：{len(GROUND_TRUTH)} 题（其中 {sum(1 for v in GROUND_TRUTH.values() if v is None)} 题为知识库无答案，不计入指标）")
    out.append(f"RRF 两路权重：{weights}（降级时向量权重下调）")
    out.append("")

    summary: list[tuple[str, float, float, float]] = []
    for name in CONFIGS:
        rows = []
        hit = 0
        mrr = 0.0
        n_eval = 0
        t0 = time.time()
        for q in GROUND_TRUTH:
            keys = run_config(name, q)
            gt = GROUND_TRUTH[q]
            if gt is None:
                rows.append((q, "（无答案题）", "-", keys))
                continue
            n_eval += 1
            rank = next((i + 1 for i, k in enumerate(keys[:5]) if k == gt), None)
            if rank:
                hit += 1
                mrr += 1.0 / rank
            rows.append((q, f"{gt[0]} 第{gt[1]}条", ("命中 第%d名" % rank) if rank else "未命中", keys))
        elapsed = time.time() - t0

        # 无答案题（空检）单独统计：检索为空 = 正确拒答；返回了内容 = 幻觉风险
        null_q = [q for q, v in GROUND_TRUTH.items() if v is None]
        null_ok = sum(1 for q in null_q if not run_config(name, q))
        recall = hit / max(1, n_eval)
        mrr_avg = mrr / max(1, n_eval)
        summary.append((name, recall, mrr_avg, elapsed))

        out.append("=" * 78)
        out.append(f"【{name}】  Recall@5 = {recall:.3f}（{hit}/{n_eval}）   MRR = {mrr_avg:.3f}   耗时 {elapsed:.1f}s")
        if null_q:
            out.append(f"  无答案题：{len(null_q)} 题，检索返回空 {null_ok} 题")
        out.append("-" * 78)
        for q, gt, res, keys in rows:
            out.append(f"  问题：{q}")
            out.append(f"    标准答案：{gt}")
            out.append(f"    结果：{res}")
            out.append("    实际返回前5：")
            for i, k in enumerate(keys[:5], 1):
                flag = "  ←←正确" if (GROUND_TRUTH[q] is not None and k == GROUND_TRUTH[q]) else ""
                out.append(f"       {i}. {k[0]} 第{k[1]}条{flag}")
            out.append("")

    out.append("=" * 78)
    out.append(f"汇总（向量器：{backend}，{dim} 维{'，降级路径' if degraded else '，正常路径'}）")
    out.append("-" * 78)
    out.append(f"  {'配置':<22}{'Recall@5':>12}{'MRR':>10}{'耗时':>10}")
    for name, recall, mrr_avg, elapsed in summary:
        out.append(f"  {name:<22}{recall:>12.3f}{mrr_avg:>10.3f}{elapsed:>9.1f}s")
    out.append("")
    out.append(f"说明：计分题 {sum(1 for v in GROUND_TRUTH.values() if v is not None)} 道，"
               f"每个名次差 ≈ {1 / max(1, sum(1 for v in GROUND_TRUTH.values() if v is not None)):.3f}。")

    target = Path(os.environ.get("EVAL_OUT") or (Path.home() / "Desktop" / "面试" / f"评估结果_{EMBED}.txt"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[-14:]))
    print("\nOK ->", target)


if __name__ == "__main__":
    main()
