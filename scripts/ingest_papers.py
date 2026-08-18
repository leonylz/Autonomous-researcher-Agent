#!/usr/bin/env python3
"""
论文/文献批量摄取 — 把论文灌进项目的 RAG 知识库(零 LLM 成本)。

用法:
  python scripts/ingest_papers.py --project examples/eval_tasks/T4_paper_repro \
      --arxiv 1708.04552,1805.09501
  python scripts/ingest_papers.py --project <proj> --dir literature/

说明:
- --arxiv:arXiv id 列表(逗号分隔),从 arXiv API 抓标题+摘要入库;
- --dir:扫描目录下的 .md/.txt 文件,整篇入库(适合已下载的论文笔记);
- 存储:项目 workspace/memory.db(与 agent 的 RagKnowledgeBase 同库同命名空间),
  所以先 ingest、后启动 agent 即可检索;
- 零 LLM 成本、离线可测(测试 mock 网络层)。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cross_project_memory import CrossProjectStore  # noqa: E402
from core.rag import RagKnowledgeBase  # noqa: E402

# section 标题 → 类型(方法/实验段是 idea agent 决策假设的关键,优先注入)
# 用「包含匹配」而非前缀匹配:真实 ar5iv 标题常带编号或引导词
# ("3. Method"、"3.1 The Mixup Algorithm"、"A. Experimental Setup"),
# 前缀匹配会漏掉大半论文(T6 实测 15 篇仅 2 篇命中)。
_METHOD_HEADING = re.compile(
    r"(method|approach|proposed|procedure|algorithm|architecture|training "
    r"(strategy|details|setup|procedure)|implementation|setup|hyperparameter"
    r"|learning rate|lr schedule)", re.IGNORECASE)
_EXPERIMENT_HEADING = re.compile(
    r"(experiment|result|evaluation|ablation|comparison|analysis|benchmark"
    r"|error rate|accuracy)", re.IGNORECASE)
_SKIP_HEADING = re.compile(
    r"(introduction|related work|background|conclusion|acknowledg|reference"
    r"|appendix|limitation|future work|notation|preliminar|overview|abstract"
    r"|bibliograph|acknowledgment|supplementary)", re.IGNORECASE)
_SECTION_MAX_CHARS = 6000  # 单 section 最多保留的字符(防超大论文)


class _Ar5ivParser(HTMLParser):
    """ar5iv(arXiv 官方 HTML 渲染)解析:按标题切 section,分类收集文本。"""

    def __init__(self):
        super().__init__()
        self._headings = ("h1", "h2", "h3", "h4")
        self._current_tag = ""
        self._current_heading = ""
        self._current_text = []
        self.sections = []  # [(heading, text)]

    def handle_starttag(self, tag, attrs):
        self._current_tag = tag.lower()

    def handle_endtag(self, tag):
        if tag.lower() in self._headings:
            heading = " ".join(self._current_text).strip()
            self.sections.append((heading, ""))
            self._current_text = []
        elif tag.lower() in ("p", "div", "li"):
            text = " ".join(self._current_text).strip()
            if text:
                if self.sections and self.sections[-1][1] == "":
                    self.sections[-1] = (self.sections[-1][0], text)
                elif self.sections:
                    prev = self.sections[-1]
                    if len(prev[1]) < _SECTION_MAX_CHARS:
                        self.sections[-1] = (prev[0], prev[1] + "\n" + text)
            self._current_text = []

    def handle_data(self, data):
        if self._current_tag not in ("script", "style"):
            self._current_text.append(data)

    def text(self) -> str:
        """未命中任何标题的剩余文本(通常开头段落)。"""
        return " ".join(self._current_text).strip()


def _classify(heading: str) -> str:
    h = heading.strip().lower()
    if not h:
        return "other"
    if _SKIP_HEADING.search(h):
        return "skip"
    if _METHOD_HEADING.search(h) or _EXPERIMENT_HEADING.search(h):
        return "methods"
    return "other"


def fetch_ar5iv_sections(arxiv_id: str, timeout: int = 30) -> dict | None:
    """抓取 ar5iv HTML 全文并按 section 分类。

    返回 {"abstract": str, "methods": str, "other": str} 或 None(失败)。
    方法/实验段是假设决策的关键;intro/related/conclusion 直接丢弃(省存储)。
    """
    arxiv_id = arxiv_id.split("v")[0]  # 去掉版本号(ar5iv 用无版本 id)
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoResearcher/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    parser = _Ar5ivParser()
    try:
        parser.feed(raw)
    except Exception:
        return None

    abstract = ""
    methods_parts = []
    other_parts = []
    seen_abstract = False
    for heading, text in parser.sections:
        if not text:
            continue
        kind = _classify(heading)
        if kind == "skip":
            continue
        if "abstract" in heading.lower() and not seen_abstract:
            abstract = text
            seen_abstract = True
        elif kind == "methods":
            methods_parts.append(f"{heading}: {text}")
        else:
            other_parts.append(f"{heading}: {text}")

    if not abstract:
        # 开头无标题段落常是摘要(arXiv 布局)
        head = parser.text()
        if head and not seen_abstract:
            abstract = head[:1500]
    return {
        "abstract": abstract[:3000],
        "methods": "\n\n".join(methods_parts)[: _SECTION_MAX_CHARS * 3],
        "other": "\n\n".join(other_parts)[:_SECTION_MAX_CHARS],
    }


def fetch_arxiv_papers(arxiv_ids: list[str]) -> list[dict]:
    """从 arXiv API 批量抓取元数据(标题/摘要)。单请求,零依赖。"""
    ids = ",".join(arxiv_ids)
    params = {"id_list": ids, "max_results": len(arxiv_ids)}
    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AutoResearcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        root = ET.fromstring(resp.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        arxiv_url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        papers.append({
            "arxiv_id": arxiv_url.rsplit("/", 1)[-1],
            "title": " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split()),
            "abstract": " ".join((entry.findtext("a:summary", default="", namespaces=ns) or "").split()),
        })
    return papers


def ingest_arxiv(kb: RagKnowledgeBase, arxiv_ids: list[str],
                 save_md_to: Path | None = None, fulltext: bool = False) -> int:
    papers = fetch_arxiv_papers(arxiv_ids)
    total = 0
    for p in papers:
        methods = ""
        if fulltext:
            # 全文模式:ar5iv 抓方法/实验段(idea agent 决策假设的关键);
            # 失败自动 fallback 到摘要模式,不阻塞
            sections = fetch_ar5iv_sections(p["arxiv_id"])
            if sections:
                methods = sections["methods"]
                if sections["abstract"]:
                    p["abstract"] = sections["abstract"]
                print(f"    (fulltext: methods={len(methods)} chars, "
                      f"abstract={len(p['abstract'])} chars)")
        n = kb.add_paper(
            title=f"{p['title']} [arXiv:{p['arxiv_id']}]",
            abstract=p["abstract"],
            methods=methods,
        )
        total += n
        print(f"  + {p['arxiv_id']}: {p['title'][:60]} ({n} chunks)")
        if save_md_to is not None:
            md = (
                f"# {p['title']}\n\n"
                f"- arXiv: {p['arxiv_id']}\n"
                f"- URL: https://arxiv.org/abs/{p['arxiv_id']}\n\n"
                f"## Abstract\n\n{p['abstract']}\n"
            )
            if methods:
                md += f"\n## Methods / Experiments\n\n{methods}\n"
            md_path = save_md_to / f"{p['arxiv_id']}.md"
            md_path.write_text(md, encoding="utf-8")
            print(f"    文献落盘: {md_path}")
    return total


def ingest_dir(kb: RagKnowledgeBase, lit_dir: Path) -> int:
    total = 0
    for f in sorted(lit_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".md", ".txt"):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        n = kb.add_document(text, source=f.name)
        total += n
        print(f"  + {f.name} ({n} chunks)")
    return total


def main():
    parser = argparse.ArgumentParser(description="论文/文献批量摄取到 RAG 知识库")
    parser.add_argument("--project", required=True, help="项目目录(workspace/memory.db)")
    parser.add_argument("--arxiv", default="", help="arXiv id 列表,逗号分隔")
    parser.add_argument("--dir", default="", help="本地文献目录(.md/.txt)")
    parser.add_argument("--save-md", default="",
                        help="--arxiv 抓取结果同时落盘为 .md 的目录(默认不落盘)")
    parser.add_argument("--fulltext", action="store_true",
                        help="--arxiv 模式抓取 ar5iv 全文(方法/实验段),失败自动回退摘要")
    args = parser.parse_args()

    project_dir = Path(args.project).resolve()
    workspace = project_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # 与 ResearchGraph 相同的库名与命名空间,确保 agent 能检索到
    project_name = project_dir.name
    kb = RagKnowledgeBase(
        CrossProjectStore(workspace / "memory.db"),
        project=f"rag_{project_name[:40]}",
    )

    total = 0
    save_md_to = Path(args.save_md).resolve() if args.save_md else None
    if save_md_to is not None:
        save_md_to.mkdir(parents=True, exist_ok=True)
    if args.arxiv:
        ids = [i.strip() for i in args.arxiv.split(",") if i.strip()]
        print(f"[arxiv] 抓取 {len(ids)} 篇论文(fulltext={args.fulltext})...")
        total += ingest_arxiv(kb, ids, save_md_to=save_md_to,
                              fulltext=args.fulltext)
    if args.dir:
        lit_dir = Path(args.dir).resolve()
        print(f"[dir] 扫描 {lit_dir} ...")
        total += ingest_dir(kb, lit_dir)
    if not args.arxiv and not args.dir:
        parser.error("至少指定 --arxiv 或 --dir")

    stats = kb.stats()
    print(f"\n完成: 共 {total} chunks 入库。")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n知识库: {workspace / 'memory.db'}\n"
          f"现在启动 agent 即可在 think/idea 阶段检索到这些论文。")


if __name__ == "__main__":
    main()
