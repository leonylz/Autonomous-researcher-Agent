<p align="center">
  <img src="assets/banner.png" alt="Auto Deep Researcher Nodes" width="700"/>
</p>

<h1 align="center">Auto Deep Researcher</h1>
<h3 align="center">24/7 Autonomous Deep Learning Research Agent (LangGraph StateGraph)</h3>

<p align="center">
  <strong>From one sentence to a finished research loop — the agent proposes its own ideas (paper-grounded), designs and runs the experiments, reflects on what worked, and iterates autonomously, 24/7.</strong>
</p>

---

## What it is

An autonomous research agent that runs deep learning experiments end-to-end
without human supervision: it plans, writes code, launches training, monitors
zero-LLM, reflects on results, and iterates — 24/7, multi-project, with every
decision and cost traceable.

Control flow is a LangGraph `StateGraph` (supervisor routing, zero-LLM), with a
hardened component layer (execution, monitor, ledger, hypotheses, RAG,
approval, rollback, dashboard). Real-run evidence lives in
`docs/smoke_evidence/` and `docs/eval_runs/`.

## Closed loop

```
one-line goal
  → ensure_brief (LLM draft → PROJECT_BRIEF.md: goal / decision tree / constraints)
  → think_node     (leader: Tune vs. Innovate routing → code | idea | writing)
  → supervisor     (deterministic rule routing, zero LLM)
  → execute_node   (worker tool loop: read/write/launch, max_turns cap)
      ├─ launch_experiment(dry_run=true)  → system dry-run gate (interpreter + script hash + deps fingerprint)
      └─ launch_experiment(real)          → env-consistent launch
  → monitor_node   (zero LLM: METRIC_JSON contract parsing, divergence detection, early kill)
  → reflect_node   (leader: milestone / decision / innovation verdict, evidence-strength settlement)
  → loop / finish  (final report with [arXiv:xxx] citations)
```

## Key features

- **Multi-agent orchestration** — LangGraph supervisor + workers (code / idea /
  writing / review); Plan-then-Execute with replan; checkpoint resume with
  `pending_writes` self-healing.
- **Paper-driven innovation** — the idea agent inventories a local paper library
  (`literature/*.md`), reads papers, writes `IDEA_NOTES.md`, and the leader
  routes experiments by a Tune-vs-Innovate procedure; borrowed methods carry
  `[arXiv:xxxx.xxxxx]` citations.
- **Zero-LLM monitoring** — system polling parses the `METRIC_JSON` contract
  line (regex fallback), detects divergence (loss-rising with magnitude
  threshold + accuracy guard) and terminates early.
- **Memory as a scientific ledger** — four namespaces
  (episodic / semantic / procedural / preference) in a LangGraph Store, plus a
  hypothesis lifecycle state machine (proposed → testing → confirmed / refuted /
  inconclusive) settled by **evidence strength** (a single small negative run
  is `inconclusive`, not a refutation); cross-project shared memory (SQLite WAL).
- **Reliability** — four-tier training interpreter resolution (config pin →
  probe → async create → borrow); system dry-run gate; LLM graded retry;
  crash self-healing; agent locks; orphan training detection.
- **Safety** — shell allow/block list, shlex injection guard, API keys stripped
  from child processes, tool-layer prompt-injection filtering, HITL approval
  (off / exception / all), daily budget cap.
- **Observability & cost** — FastAPI + SSE dashboard; per-call token cost
  attributed to actor/action in `workspace/costs.jsonl`; full event replay
  (`events.jsonl`).

## Quick start

```bash
# 1) install dependencies (requires Python 3.10+, torch/torchvision)
pip install -e ".[llm,agent,dashboard]"

# 2) run with a one-line goal (auto-generates PROJECT_BRIEF.md via LLM)
export DEEPSEEK_API_KEY=...
python -m core.nodes --project examples/demo_goal --goal "Train a small CNN on MNIST to 99.5% test accuracy, CPU-reproducible"

# 3) or with a prepared project brief
python -m core.nodes --project examples/smoke_project --max-cycles 3
```

> Training interpreter: set `execution.python` in `config.yaml` (or let the
> system probe/create an environment with torch+torchvision).

## Eval tasks

`examples/eval_tasks/` contains 6 benchmark tasks (T1–T6) with machine-readable
targets and budgets:

| ID | Task | Target | Notes |
|----|------|--------|-------|
| T1 | MNIST CNN from scratch | test_acc ≥ 99.5% | CPU, data = local cache |
| T2 | CIFAR-10 ResNet-18 | test_acc ≥ 85% | needs data |
| T3 | Fix a diverging config | loss converges | needs data |
| T4 | Reproduce a paper method | paper-level accuracy | paper library + RAG |
| T5 | Crash detection & recovery | resume after kill | needs data |
| T6 | Innovation: paper-driven CIFAR-10 | test_acc ≥ 83% | 15-paper library, requires innovation |

```bash
python scripts/run_eval.py --dry            # validate task configs
python scripts/run_eval.py --real --tasks T1 --provider deepseek --model deepseek-chat
python scripts/run_eval.py --report         # aggregate results
```

> **Datasets are NOT in this repository** (MNIST / CIFAR-10 caches stay local —
> see each brief for the expected layout). The `literature/` paper libraries in
> T4/T6 ARE committed (offline-fulltext md files).

## Project structure

```
core/            graph nodes, tools, execution, monitor, ledger, hypotheses,
                 RAG, approval, rollback, dashboard, prompts (agents/*.md)
agents/          worker/leader prompt files (single source of truth, six-section
                 template with few-shot examples bootstrapped from real runs)
scripts/         eval driver, paper ingest, fulltext cleaning, trace extraction
examples/        eval tasks + smoke/demo projects
docs/            architecture, user guide, smoke report, run evidence
tests/           ~470 tests (regression for every real incident)
```

## Documentation

- [Architecture](docs/architecture.md)
- [User guide](docs/USER_GUIDE.md)
- [Smoke report (real-run validation)](docs/SMOKE_REPORT.md)
- [MCP bridge](docs/MCP_BRIDGE.md)
- [Prompt changelog](docs/PROMPT_CHANGELOG.md)
- [Production practices](docs/PRODUCTION_PRACTICES.md)
- [DSH pattern attribution](docs/DSH_ATTRIBUTION.md)

## License

Licensed under [Apache-2.0](LICENSE). Component-level attribution:
[docs/DSH_ATTRIBUTION.md](docs/DSH_ATTRIBUTION.md).
