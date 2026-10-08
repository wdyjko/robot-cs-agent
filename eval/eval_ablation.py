# -*- coding: utf-8 -*-
"""严格消融对照：同一套 34 道计分题，比较「向量器 × 融合策略」。
用法：
    python eval_ablation.py            # 主进程：spawn 两个子进程分别跑 hash384 / bge1024
    python eval_ablation.py hash384    # 子进程：只跑一种向量器

为什么必须分进程：项目里 `settings` 是导入期创建的单例、
`get_embeddings()` 也带 lru_cache，同进程内改环境变量切不动向量器。

对照组：
    A 纯向量              B 纯 BM25            C RRF 融合（权重随降级调整）
    D C + 覆盖度（线上配置，系数 0.006/0.010/0.010）
    E C + 覆盖度系数×10（0.06/0.10/0.10）      F 纯向量 + 覆盖度
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 默认假设脚本位于 <项目根>/eval/ 下；也可以用环境变量覆盖，便于在别处跑。
PROJECT = Path(os.environ.get("EVAL_PROJECT") or Path(__file__).resolve().parents[1])
PYTHON = Path(os.environ.get("EVAL_PYTHON") or (PROJECT / ".venv" / "Scripts" / "python.exe"))
OUT_DIR = Path(os.environ.get("EVAL_RESULT_DIR") or (Path(__file__).resolve().parent / "results"))
STORE_HASH = Path(os.environ.get("EVAL_STORE_HASH") or (PROJECT / "vectorstore"))
STORE_BGE = Path(os.environ.get("EVAL_STORE_BGE") or (PROJECT / "vectorstore_bge1024"))

GROUND_TRUTH: dict[str, tuple[str, str]] = {
    "机器人最高能爬过多高的障碍物？": ("扫地机器人100问.pdf", "12"),
    "怎么设置每天固定时间自动打扫？": ("扫地机器人100问.pdf", "22"),
    "这台机器支持小爱同学或者天猫精灵吗？": ("扫地机器人100问.pdf", "26"),
    "尘盒一般多久需要清理一次？": ("扫地机器人100问.pdf", "29"),
    "电池一直插着充电会不会把电池弄坏？": ("扫地机器人100问.pdf", "76"),
    "冬天零下的时候机器人还能正常用吗？": ("扫地机器人100问.pdf", "83"),
    "刚买回来第一次用，需要先做什么准备？": ("扫地机器人100问2.txt", "1"),
    "机器人扫完地自己却不回去充电，怎么办？": ("扫地机器人100问2.txt", "10"),
    "HEPA 滤网洗一洗还能接着用吗？": ("扫地机器人100问2.txt", "23"),
    "感觉续航比以前短了很多，是什么原因？": ("扫地机器人100问2.txt", "26"),
    "滚刷仓里有股味道，怎么去除？": ("扫地机器人100问2.txt", "33"),
    "机器人在长毛地毯上走不动，卡住了怎么办？": ("扫地机器人100问2.txt", "75"),
    "水箱里的水漏出来了，怎么处理？": ("扫拖一体机器人100问.txt", "11"),
    "拖完地地上出现一条一条的水痕，怎么避免？": ("扫拖一体机器人100问.txt", "19"),
    "厨房地面油污很重，拖不干净怎么办？": ("扫拖一体机器人100问.txt", "26"),
    "木地板拖完起翘变形了怎么处理？": ("扫拖一体机器人100问.txt", "31"),
    "南方的回南天怎么用机器人拖地比较合适？": ("扫拖一体机器人100问.txt", "66"),
    "宠物掉毛太多，拖布总被毛发缠住怎么办？": ("扫拖一体机器人100问.txt", "72"),
    "机器人满电之后马上就断电了，是什么故障？": ("故障排除.txt", "24"),
    "地上有水的时候机器人工作会突然停机吗？": ("故障排除.txt", "60"),
    "工作时电量掉得特别快，远超过正常续航": ("故障排除.txt", "51"),
    "机器人在瓷砖地面上打滑走不动": ("故障排除.txt", "69"),
    "机器人充电的时候竟然冒出了电火花": ("故障排除.txt", "46"),
    "电池充了很久也充不满，指示灯却是正常的": ("故障排除.txt", "190"),
    "水箱里结了水垢，用什么泡一泡能清掉？": ("维护保养.txt", "5"),
    "机器人要放很久不用，电池该怎么存放？": ("维护保养.txt", "11"),
    "每次用完需要擦机身吗？用什么擦？": ("维护保养.txt", "1"),
    "拖布用几次要消毒？家里有宠物和宝宝要注意什么？": ("维护保养.txt", "38"),
    "水泥地扫完之后要做什么额外清理？": ("维护保养.txt", "18"),
    "家里是木地板，买的时候要注意选什么功能？": ("选购指南.txt", "31"),
    "滤网是买可水洗的还是不可水洗的？": ("选购指南.txt", "61"),
    "想买带灰尘感应自动调吸力的，有用吗？": ("选购指南.txt", "91"),
    "能提前预约未来几天的清扫计划吗？": ("选购指南.txt", "181"),
    "怎么判断一台机器性价比高不高？": ("选购指南.txt", "200"),
}

MODES = {
    "hash384": (str(STORE_HASH), "false"),
    "bge1024": (str(STORE_BGE), "true"),
}


def key_of(doc) -> tuple[str, str]:
    m = doc.metadata or {}
    return (str(m.get("source_file", "")), str(m.get("item_no", "")))


def run_one(mode: str) -> None:
    """子进程：跑一种向量器，把结果打印成制表符分隔的汇总行。"""
    store, allow = MODES[mode]
    os.environ["EMBEDDING_ALLOW_DOWNLOAD"] = allow
    os.environ["CHROMA_PERSIST_DIR"] = store
    os.environ["VECTOR_BACKEND"] = "local"
    for p in (str(PROJECT / "backend"), str(PROJECT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from app.core.config import settings
    from app.services import rag_service as R
    from app.services.embeddings import get_embeddings
    from langchain_core.documents import Document

    emb = get_embeddings()
    dim = len(emb.embed_query("探测"))
    degraded = "Hashing" in type(emb).__name__
    weights = [0.35, 1.0] if degraded else [1.0, 1.0]

    def cov(query: str, docs: list, wc: float, wt: float, wg: float) -> list:
        qu = R._discriminative_units(query)
        scored = []
        for doc in docs:
            base = float((doc.metadata or {}).get("score", 0.0) or 0.0)
            md = dict(doc.metadata or {})
            cc = R._query_coverage(query, doc.page_content)
            title_text = " ".join(str(md.get(k, "")) for k in ("title", "category", "file_category"))
            tc = R._query_coverage(query, title_text) if title_text.strip() else 0.0
            tags = md.get("tags") or []
            tu = R._discriminative_units(" ".join(str(t) for t in tags))
            gc = (len(qu & tu) / len(qu)) if qu else 0.0
            s = base + wc * cc + wt * tc + wg * gc
            md["score"] = round(s, 6)
            scored.append((s, Document(page_content=doc.page_content, metadata=md)))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored]

    def all_paths(q: str) -> dict[str, list]:
        v = R._vector_search(q, settings.vector_top_k)
        b = R._bm25_search(q, settings.bm25_top_k)
        fused = R._rrf_fuse([v, b], weights=weights)
        return {
            "A 纯向量": v,
            "B 纯BM25": b,
            "C RRF融合": fused,
            "D RRF+覆盖度(线上系数)": R._apply_coverage_boost(q, list(fused)),
            "E RRF+覆盖度(x10系数)": cov(q, list(fused), 0.06, 0.10, 0.10),
            "F 纯向量+覆盖度(线上系数)": cov(q, list(v), 0.006, 0.010, 0.010),
        }
    names = list(all_paths("测试").keys())
    hits = {n: 0 for n in names}
    mrr = {n: 0.0 for n in names}
    ranks = {n: [] for n in names}
    for q, gt in GROUND_TRUTH.items():
        res = all_paths(q)
        for n in names:
            keys = [key_of(d) for d in res[n][:5]]
            rank = next((i + 1 for i, k in enumerate(keys) if k == gt), None)
            if rank:
                hits[n] += 1
                mrr[n] += 1.0 / rank
                ranks[n].append(rank)

    total = len(GROUND_TRUTH)
    print("META\t%s\t%s\t%d\t%s" % (mode, type(emb).__name__, dim, weights))
    for n in names:
        avg = (sum(ranks[n]) / len(ranks[n])) if ranks[n] else 0.0
        print("ROW\t%s\t%s\t%.4f\t%.4f\t%.2f\t%d" % (mode, n, hits[n] / total, mrr[n] / total, avg, hits[n]))
    print("TOTAL\t%d" % total)


def main() -> None:
    lines: list[str] = []
    for mode in MODES:
        # 关键：每个向量器一个独立子进程（settings / lru_cache 都在导入期固化）
        proc = subprocess.run(
            [str(PYTHON), str(Path(__file__).resolve()), mode],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_ENDPOINT": ""},
        )
        lines.append("=" * 88)
        lines.append(f"##### 向量器：{mode}  (子进程 exit={proc.returncode})")
        lines.append("=" * 88)
        for ln in (proc.stdout or "").splitlines():
            if ln.startswith("META\t"):
                _, m, cls, dim, w = ln.split("\t")
                lines.append(f"向量器实现：{cls}（{dim} 维）    RRF 两路权重：{w}    计分题：{len(GROUND_TRUTH)}")
                lines.append("-" * 88)
                lines.append(f"  {'配置':<22}{'Recall@5':>10}{'MRR':>9}{'平均命中名次':>14}{'命中数':>8}")
            elif ln.startswith("ROW\t"):
                _, m, name, rec, mrr, avg, hit = ln.split("\t")
                lines.append(f"  {name:<22}{float(rec):>10.3f}{float(mrr):>9.3f}{float(avg):>14.2f}{hit:>8}")
        if proc.stderr and proc.returncode != 0:
            lines.append("  [stderr] " + proc.stderr.strip().splitlines()[-1][:180])
        lines.append("")

    text = "\n".join(lines)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "ablation_对照表.txt"
    target.write_text(text, encoding="utf-8")
    print("OK ->", target)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in MODES:
        run_one(sys.argv[1])
    else:
        main()
