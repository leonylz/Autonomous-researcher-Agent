#!/usr/bin/env python3
"""
clean_papers — 论文库清洗与合并(ar5iv 全文 ↔ 手写摘要)。

背景:ingest_papers.py --fulltext 的原始抽取质量参差(表格残渣/重复表头/
部分论文 methods=0)。本脚本:
1. 读取 staging 目录的 ar5iv 抽取结果;
2. 清洗:去连续重复行(表格表头残渣)、折叠空白、剔除非正文行;
3. 合并:清洗后 Methods 段 ≥ MIN_METHODS 字符 → 追加到手写 literature/*.md
   (手写文件是准确基线:标题/arXiv/URL/摘要/Key Method 人工核验过);
   不足阈值(如 mixup 只有摘要)→ 保持手写文件不动,绝不降级。

用法:
  python scripts/clean_papers.py \
      --staging docs/t6_fulltext_staging \
      --library examples/eval_tasks/T6_innovation/literature
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MIN_METHODS = 1500  # 清洗后方法段低于此字符 → 不合并(手写为准)

_NAV_NOISE = re.compile(
    r"^(cite this paper|download|bibliographic|bibtex|fig\.|table \d+|"
    r"view previous|view next|search arxiv|about ar5iv|arxiv\.org|"
    r"the current browser|javascript|skip to|acknowledg)", re.IGNORECASE)


def clean_section(text: str) -> str:
    """清洗 ar5iv 抽取文本:去导航/表格残渣、重复行、空白折叠。"""
    lines = []
    prev = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _NAV_NOISE.match(line):
            continue
        if line == prev:  # 连续重复(表格表头残渣)
            continue
        if len(line) < 3 and not line.replace(".", "").isdigit():
            continue
        # 折叠内部空白(HTML 换行残渣)
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
        prev = line
    return "\n".join(lines)


def split_paper(md_text: str) -> tuple[str, str]:
    """拆出 (head, methods):methods 为 '## Methods / Experiments' 之后的内容。"""
    marker = "\n## Methods / Experiments\n"
    if marker in md_text:
        head, methods = md_text.split(marker, 1)
        return head.strip(), methods.strip()
    return md_text.strip(), ""


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗并合并 ar5iv 全文到论文库")
    parser.add_argument("--staging", required=True, help="ar5iv 抽取结果目录")
    parser.add_argument("--library", required=True, help="论文库目录(literature/)")
    args = parser.parse_args()

    staging = Path(args.staging)
    library = Path(args.library)
    if not staging.is_dir() or not library.is_dir():
        print("[FAIL] staging/library 目录无效")
        return 1

    # staging 文件名形如 {arxiv_id}v{n}.md → 归一为无版本 id
    staged = {}
    for f in staging.glob("*.md"):
        arxiv_id = f.stem.split("v")[0]
        staged.setdefault(arxiv_id, f)

    merged = kept = 0
    for lib_file in sorted(library.glob("*.md")):
        arxiv_id = lib_file.stem.split("v")[0]
        staged_file = staged.get(arxiv_id)
        if staged_file is None:
            print(f"  = {lib_file.name}: 无对应抽取,保持手写")
            continue
        head, methods = split_paper(staged_file.read_text(encoding="utf-8", errors="replace"))
        cleaned = clean_section(methods)
        if len(cleaned) < MIN_METHODS:
            print(f"  = {lib_file.name}: 清洗后方法段 {len(cleaned)} 字符 "
                  f"(<{MIN_METHODS}),保持手写")
            kept += 1
            continue
        lib_text = lib_file.read_text(encoding="utf-8", errors="replace")
        if "## Methods / Experiments (ar5iv fulltext)" in lib_text:
            print(f"  = {lib_file.name}: 已合并过,跳过")
            continue
        appendix = (
            f"\n## Methods / Experiments (ar5iv fulltext)\n\n"
            f"{cleaned}\n"
        )
        lib_file.write_text(lib_text.rstrip() + "\n" + appendix, encoding="utf-8")
        print(f"  + {lib_file.name}: 合并 ar5iv 全文 {len(cleaned)} 字符")
        merged += 1

    print(f"\n完成: {merged} 篇合并全文, {kept} 篇保持手写(抽取不足,不降级)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
