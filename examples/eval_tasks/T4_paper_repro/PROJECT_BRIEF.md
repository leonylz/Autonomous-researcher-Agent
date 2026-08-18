# T4 — Reproduce a paper method (Mixup + cosine schedule)

## Goal
Reproduce a published training recipe on CIFAR-10: **Mixup (α=0.2) + cosine
annealing + warmup**, reaching test accuracy ≥ 88% — inside the range reported
by the original papers for a ResNet-18-style model.

## Codebase
- Training script: `train.py` (agent creates it from `core/train_template.py`)
- Paper library: `literature/` contains two papers relevant to this task
  (Mixup paper + cosine schedule reference). **Search the paper library (RAG)
  before proposing a hypothesis and cite the paper id `[arxiv_id]` in your plan.**

## What to Try
1. FIRST: retrieve papers from the knowledge base (`search_papers`/`search_arxiv`
   for the latest; the local library already has the key ones).
2. Baseline ResNet-18 with plain cosine schedule; then add Mixup α=0.2.
3. Only deviate from the paper recipe (e.g. LR range, epochs) when the budget
   demands it — document the deviation and why.

## Constraints
- PyTorch, GPU 0, max 80 epochs per run.
- MUST use the framework training template and dry-run gate.
- **Every experiment hypothesis MUST carry the paper citation it is based on
  (`[arxiv_id]` or `[paper:title]`), so results can be traced back to sources.**

## Success Criteria
Test accuracy ≥ 88% AND at least one cited paper reference in the hypothesis of
the final experiment, recorded in the ledger.
