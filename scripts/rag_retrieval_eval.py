#!/usr/bin/env python3
"""
rag_retrieval_eval — RAG 知识库构建 + 检索评测(真实数字,零编造)。

流程:
1. 按主题从 arXiv 搜索建库(SEARCH_TOPICS × 每主题 N 篇)→ ar5iv 全文摄取
   → 知识库 docs/rag_kb/memory.db
2. 评测集:每篇论文 3 个查询(标题/去停用词标题/摘要首句)+ 负样本查询
3. 指标:hit@1 / hit@3 / hit@5 / recall@5 + 负样本 precision(检索器
   对库外主题不应给出高相似度命中)
4. 结果写入 docs/RAG_EVAL.md(可复现)

用法:
  python scripts/rag_retrieval_eval.py --build --eval --report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cross_project_memory import CrossProjectStore  # noqa: E402
from core.rag import RagKnowledgeBase  # noqa: E402
from scripts.ingest_papers import fetch_ar5iv_sections  # noqa: E402

KB_PATH = PROJECT_ROOT / "docs" / "rag_kb" / "memory.db"
RAG_EVAL_PATH = PROJECT_ROOT / "docs" / "RAG_EVAL.md"

# 主题 → 每主题抓取篇数(真实论文,来自 arXiv 搜索)
SEARCH_TOPICS = [
    ("data augmentation cifar", 8),
    ("learning rate schedules neural networks", 8),
    ("regularization convolutional neural networks", 8),
    ("residual networks image classification", 8),
    ("deep learning optimizers", 8),
    ("knowledge distillation", 6),
    ("attention is all you need transformer", 6),
    ("generative adversarial networks", 6),
]

# 负样本:库外主题(检索器不应高相似命中)
NEGATIVE_QUERIES = [
    "quantum error correction surface codes",
    "protein structure prediction alphafold folding",
    "autonomous vehicle lidar point cloud segmentation",
    "cryptocurrency blockchain consensus proof of stake",
    "weather forecasting ensemble numerical models",
    "music generation midi sequence transformer",
    "robotic arm inverse kinematics trajectory planning",
    "recommender system collaborative filtering matrix factorization",
    "speech recognition end to end ctc attention",
    "differential privacy federated learning mechanism design",
    "graph neural network molecular property prediction",
    "reinforcement learning atari dqn replay buffer",
    "image super resolution perceptual loss",
    "neural architecture search reinforcement proxy",
    "semantic segmentation deeplab atrous convolution",
]

_STOP = set("the a an of for in on and or to with from by is are be was were "
            "that this these those using use used via their our we they it its".split())


def arxiv_search(query: str, max_results: int) -> list[dict]:
    """arXiv API 搜索 → [{arxiv_id, title, abstract}]。"""
    params = {"search_query": f"all:{query}", "start": 0,
              "max_results": max_results, "sortBy": "relevance",
              "sortOrder": "descending"}
    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AutoResearcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        root = ET.fromstring(resp.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        arxiv_url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        papers.append({
            "arxiv_id": arxiv_url.rsplit("/", 1)[-1].split("v")[0],
            "title": " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split()),
            "abstract": " ".join((entry.findtext("a:summary", default="", namespaces=ns) or "").split()),
        })
    return papers


def build_corpus() -> list[dict]:
    """按主题搜索建库(去重)。"""
    seen: dict[str, dict] = {}
    for topic, n in SEARCH_TOPICS:
        try:
            for p in arxiv_search(topic, n):
                seen.setdefault(p["arxiv_id"], p)
            print(f"  [topic] {topic}: +{n} (累计 {len(seen)})")
        except Exception as exc:
            print(f"  [topic] {topic}: FAIL {exc}")
        time.sleep(0.5)  # arXiv API 礼貌间隔
    return list(seen.values())


def ingest_corpus(papers: list[dict], kb: RagKnowledgeBase) -> int:
    """ar5iv 全文摄取(方法段),失败自动回退摘要。"""
    n = 0
    for p in papers:
        try:
            sections = fetch_ar5iv_sections(p["arxiv_id"])
        except Exception:
            sections = None
        methods = sections["methods"] if sections else ""
        if sections and sections.get("abstract"):
            p["abstract"] = sections["abstract"]
        try:
            n += kb.add_paper(
                title=f"{p['title']} [arXiv:{p['arxiv_id']}]",
                abstract=p["abstract"], methods=methods)
        except Exception as exc:
            print(f"  [ingest] {p['arxiv_id']} FAIL: {exc}")
    return n


def build_queries(papers: list[dict]) -> list[tuple[str, str]]:
    """每篇 3 个查询:(标题, 标题去停用词, 摘要首句) → 期望论文 id。"""
    queries = []
    for p in papers:
        title = p["title"]
        queries.append((title, p["arxiv_id"]))
        t2 = " ".join(w for w in title.split() if w.lower() not in _STOP)
        if t2 and t2 != title:
            queries.append((t2, p["arxiv_id"]))
        first = p["abstract"].split(".")[0] if p["abstract"] else ""
        if first:
            queries.append((first[:150], p["arxiv_id"]))
    return queries


def evaluate(kb: RagKnowledgeBase, papers: list[dict]) -> dict:
    queries = build_queries(papers)
    hits = {1: 0, 3: 0, 5: 0}
    total = len(queries)
    for q, expect in queries:
        try:
            results = kb.retrieve(q, top_k=5)
        except Exception:
            continue
        hit_at = None
        for k in (1, 3, 5):
            top = results[:k]
            blob = " ".join(str(r.get("source", "")) + " " + str(r.get("text", ""))
                            for r in top).lower()
            if expect in blob:
                hit_at = k
                break
        if hit_at:
            for k in (1, 3, 5):
                if hit_at <= k:
                    hits[k] += 1

    # 负样本 precision:top-1 相似度低于阈值视为「正确拒绝」
    neg_ok = 0
    neg_scores = []
    for q in NEGATIVE_QUERIES:
        try:
            results = kb.retrieve(q, top_k=1)
        except Exception:
            continue
        score = float((results[0] or {}).get("score", 0) or 0) if results else 0.0
        neg_scores.append(score)
        if score < 0.05:  # 阈值:库外主题不应有高相似
            neg_ok += 1
    stats = kb.stats()
    return {
        "n_papers": len(papers),
        "n_queries": total,
        "hit@1": round(hits[1] / total, 4) if total else 0,
        "hit@3": round(hits[3] / total, 4) if total else 0,
        "hit@5": round(hits[5] / total, 4) if total else 0,
        "neg_precision": round(neg_ok / len(NEGATIVE_QUERIES), 4),
        "neg_top1_scores": [round(s, 4) for s in neg_scores],
        "chunks": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 知识库构建与检索评测")
    parser.add_argument("--build", action="store_true", help="重建知识库(arXiv 搜索+全文摄取)")
    parser.add_argument("--eval", action="store_true", help="跑检索评测")
    parser.add_argument("--report", action="store_true", help="写 docs/RAG_EVAL.md")
    args = parser.parse_args()
    if not (args.build or args.eval):
        parser.error("至少指定 --build 或 --eval")

    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    kb = RagKnowledgeBase(CrossProjectStore(KB_PATH), project="rag_eval_corpus")

    papers: list[dict] = []
    if args.build:
        print(f"[build] 主题搜索建库...")
        papers = build_corpus()
        print(f"[build] {len(papers)} 篇,开始 ar5iv 全文摄取...")
        n = ingest_corpus(papers, kb)
        print(f"[build] 摄取完成: {n} chunks")
        # 记住论文清单(下次 --eval 不用重抓)
        (KB_PATH.parent / "papers.json").write_text(
            json.dumps(papers, ensure_ascii=False), encoding="utf-8")
    else:
        meta = KB_PATH.parent / "papers.json"
        if meta.exists():
            papers = json.loads(meta.read_text(encoding="utf-8"))

    if args.eval and papers:
        print(f"[eval] 评测集:{len(papers)} 篇,查询 {len(papers) * 3} 左右 + 负样本...")
        result = evaluate(kb, papers)
        print(json.dumps({k: v for k, v in result.items() if k != "chunks"},
                         ensure_ascii=False, indent=2))
        if args.report:
            lines = [
                "# RAG 检索评测(真实数字,可复现)",
                "",
                f"生成时间: {time.strftime('%Y-%m-%d %H:%M')}",
                "",
                f"- 知识库: {len(papers)} 篇论文(arXiv 主题搜索 + ar5iv 全文摄取)",
                f"- 评测集: {result['n_queries']} 个正样本查询(每篇:标题/去停用词标题/摘要首句)",
                f"  + {len(NEGATIVE_QUERIES)} 个负样本查询(库外主题)",
                "",
                "| 指标 | 数值 |",
                "|------|------|",
                f"| hit@1 | {result['hit@1']:.0%} |",
                f"| hit@3 | {result['hit@3']:.0%} |",
                f"| hit@5 | {result['hit@5']:.0%} |",
                f"| 负样本 precision(top-1 相似度 < 0.05 比例) | {result['neg_precision']:.0%} |",
                "",
                "> 复现: `python scripts/rag_retrieval_eval.py --build --eval --report`",
                "",
            ]
            RAG_EVAL_PATH.write_text("\n".join(lines), encoding="utf-8")
            print(f"报告已写入: {RAG_EVAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
