#!/usr/bin/env python3
"""
scidocs_eval — 标准基准检索评测(MTEB SciDocs-reranking)。

数据:ModelScope 下载 test.jsonl.gz(每行 {query, positive[], negative[]})。
检索器:与项目 RAG 知识库同族的 hash 词袋 + 余弦(零 embedding 依赖,
sentence-transformers 未安装且 PyPI 不可达 —— 数字是 BOW 基线,诚实标注)。
指标:MRR@10 / NDCG@10(标准 reranking 口径)。

用法:
  python scripts/scidocs_eval.py [--max N] [--download] [--report]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "docs" / "scidocs_data"
DATA_FILE = DATA_DIR / "test.jsonl.gz"
REPORT = PROJECT_ROOT / "docs" / "SCIDOCS_EVAL.md"
URL = ("https://modelscope.cn/api/v1/datasets/MTEB/scidocs-reranking/"
       "repo?Revision=master&FilePath=test.jsonl.gz")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def bow_cosine(a: str, b: str) -> float:
    """与知识库同族的词袋余弦相似度。"""
    ta, tb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ta or not tb:
        return 0.0
    num = sum((ta & tb).values())
    den = math.sqrt(sum(v * v for v in ta.values())) * \
        math.sqrt(sum(v * v for v in tb.values()))
    return num / den if den else 0.0


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    DATA_FILE.write_bytes(raw)
    print(f"[download] {len(raw)} bytes -> {DATA_FILE}")


def load(max_n: int | None = None) -> list[dict]:
    rows = []
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_n is not None and i >= max_n:
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def evaluate(rows: list[dict]) -> dict:
    mrrs, ndcgs = [], []
    n_pos_miss = 0
    for row in rows:
        q = row.get("query", "")
        pos = row.get("positive", []) or []
        neg = row.get("negative", []) or []
        if not pos:
            continue
        # 候选池 = 正样本 + 负样本(标准 reranking 设定)
        candidates = [(p, True) for p in pos] + [(n, False) for n in neg]
        scored = sorted(
            ((bow_cosine(q, d), rel) for d, rel in candidates),
            key=lambda x: -x[0])
        # MRR@10 / NDCG@10
        k = 10
        top = scored[:k]
        ranks = [i + 1 for i, (s, rel) in enumerate(top) if rel]
        if ranks:
            mrrs.append(1.0 / ranks[0])
        else:
            mrrs.append(0.0)
            n_pos_miss += 1
        # NDCG@10
        dcg = sum((1.0 / math.log2(i + 2)) for i, (s, rel) in enumerate(top) if rel)
        idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(len(pos), k)))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        "n_queries": len(rows),
        "mrr@10": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0,
        "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0,
        "positive_missed_in_top10": n_pos_miss,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MTEB SciDocs-reranking 检索评测(BOW 基线)")
    parser.add_argument("--download", action="store_true", help="下载测试集")
    parser.add_argument("--max", type=int, default=None, help="最多评测 N 条(默认全部)")
    parser.add_argument("--report", action="store_true", help="写 docs/SCIDOCS_EVAL.md")
    args = parser.parse_args()

    if args.download or not DATA_FILE.exists():
        download()
    rows = load(args.max)
    print(f"[eval] {len(rows)} 条查询")
    result = evaluate(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        REPORT.write_text(
            "# MTEB SciDocs-reranking 检索评测(BOW 基线)\n\n"
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"- 数据: ModelScope MTEB/scidocs-reranking test.jsonl.gz\n"
            f"- 评测查询数: {result['n_queries']}\n"
            f"- 检索器: hash 词袋 + 余弦(与项目 RAG 知识库同族,零 embedding 依赖;\n"
            f"  sentence-transformers 未安装且 PyPI 不可达,数字为 BOW 基线,诚实标注)\n\n"
            "| 指标 | 数值 |\n|------|------|\n"
            f"| MRR@10 | {result['mrr@10']:.4f} |\n"
            f"| NDCG@10 | {result['ndcg@10']:.4f} |\n"
            f"| 正样本未进 top10 的查询数 | {result['positive_missed_in_top10']} |\n\n"
            "> 复现: `python scripts/scidocs_eval.py --download --report`\n",
            encoding="utf-8")
        print(f"报告已写入: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
