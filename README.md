# Agentic Benchmark Harness

This project is a small benchmark harness for comparing tool-using research agents on a narrow domain.

The first domain is recommender systems and retrieval/ranking. The goal is not just to build one agent. The goal is to run multiple agent designs on the same tasks, score them consistently, and inspect the tradeoffs.

## What This Project Does

You define:

- a task set
- a document corpus or tools the agents can use
- several agent variants
- an evaluator

Then the harness:

1. runs each agent on each task
2. records traces
3. scores the answers
4. writes comparable results

## Concrete Example

Example task:

> Find two common approaches for cold-start recommendation and explain one tradeoff between them.

The harness can compare:

- `BaselineRAGAgent`: retrieve once and answer
- `ReActAgent`: retrieve, inspect, decide whether to retrieve again, then answer
- `PlannerExecutorAgent`: plan subquestions first, then retrieve for each step, then synthesize

For each run, the harness stores:

- final answer
- citations
- retrieved documents
- agent steps
- metric scores

## Initial Project Scope

This scaffold is intentionally small:

- 2 sample tasks
- a tiny local corpus
- 3 deterministic example agents
- a simple rule-based evaluator

It is enough to make the architecture concrete. Later you can replace the placeholder agent logic with real LLM calls, web search, reranking, and richer evals.

## Project Layout

```text
.
├── data
│   ├── corpus
│   │   └── recsys_docs.jsonl
│   └── tasks
│       └── sample_tasks.jsonl
├── pyproject.toml
├── README.md
├── scripts
│   └── run_benchmark.py
└── src
    └── agentic_bench
        ├── agents
        │   ├── base.py
        │   ├── baseline.py
        │   ├── planner_executor.py
        │   └── react_agent.py
        ├── evaluator.py
        ├── runner.py
        ├── schemas.py
        ├── tasks.py
        └── tools.py
```

## Quick Start

Use Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_benchmark.py
```

The script writes a JSON report to `results/`.

## What To Replace Next

The next upgrades should be:

1. swap deterministic agent logic for real LLM-backed agents
2. expand the task set to 25 to 50 tasks
3. add retrieval and reranking ablations
4. add richer evaluation and analysis notebooks

## What "Good" Looks Like

A strong finished version of this project should answer:

- which agent architecture is more accurate on your task family
- whether better retrieval matters more than more complex agent planning
- what the cost and latency tradeoffs look like
- what failure modes appear in traces
