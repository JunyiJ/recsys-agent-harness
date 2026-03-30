# Agentic Benchmark Harness

This project is a small benchmark harness for comparing tool-using research agents on a narrow domain.

The first domain is recommender systems. The goal is not just to build one agent. The goal is to build and run multiple agent designs on the same tasks, score them consistently, and inspect the tradeoffs.

## What This Project Does

Define:

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

The corpus is stored as JSONL, one chunk per line. Each corpus row should include:

- `doc_id`
- `title`
- `source`
- `year`
- `url`
- `text`

A larger verified corpus can be generated with:

```bash
python scripts/build_recent_recsys_corpus.py
```

This writes `data/corpus/recsys_recent_2024_2025_300.jsonl` using DBLP table-of-contents records for RecSys, KDD, and WWW (2024-2025), plus OpenAlex abstracts for the `text` field.

## Project Layout

```text
.
├── configs
│   ├── default.yaml
│   ├── default_llm.yaml
│   └── large_llm.yaml
├── data
│   ├── corpus
│   │   ├── recsys_docs.jsonl
│   │   └── recsys_recent_2024_2025_300.jsonl
│   └── tasks
│       ├── recsys_recent_2024_2025_tasks.jsonl
│       └── sample_tasks.jsonl
├── notebooks
│   ├── analyze_benchmark_report.ipynb
│   └── plot_results
│       ├── large_llm_run0
│       │   ├── snapshot_1_summary_table.png
│       │   ├── snapshot_1_aggregate_agent_performance.png
│       │   ├── per_task_score_by_agent_heatmap.png
│       │   └── bottleneck_distribution_by_agent_pct.png
│       └── large_llm_run1
├── pyproject.toml
├── README.md
├── results
│   ├── benchmark_report.json
│   ├── benchmark_report_llm.json
│   ├── benchmark_report_large_llm.json
│   ├── run_manifest.json
│   ├── run_manifest_llm.json
│   └── run_manifest_large_llm.json
├── scripts
│   ├── build_recent_recsys_corpus.py
│   └── run_benchmark.py
└── src
    └── agentic_bench
        ├── agents
        │   ├── __init__.py
        │   ├── base.py
        │   ├── baseline.py
        │   ├── planner_executor.py
        │   └── react_agent.py
        ├── config.py
        ├── evaluator.py
        ├── llm_utils.py
        ├── runner.py
        ├── schemas.py
        ├── tasks.py
        └── tools.py
```

## Quick Start

Use Python 3.11+.

```bash
pip install -e .
PYTHONPATH=src python3 scripts/run_benchmark.py --config configs/default.yaml
```

The script writes:

- a benchmark report to `results/`
- a run manifest with the resolved config and git metadata

For the LLM-judged run:

```bash
PYTHONPATH=src python3 scripts/run_benchmark.py --config configs/default_llm.yaml
```

## Results Summary

Main benchmark setup:

- curated local corpus of 300 recent recommender-systems papers
- 30 research-style QA tasks
- 3 agents: `baseline_rag`, `react_agent`, `planner_executor`
- actor model: `gpt-4.1-mini`
- judge model: `gpt-5`
- evaluator: LLM-based

This corresponds to the run manifest in `results/run_manifest_large_llm.json` and the report in `results/benchmark_report_large_llm.json`.

### Aggregate Results

| Agent | Avg Score | Fact Coverage | Retrieved Gold Doc Hit Rate | Citation Coverage | Support Quality | Avg Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `planner_executor` | 0.753 | 0.667 | 0.872 | 0.872 | 0.817 | 8.6 |
| `react_agent` | 0.732 | 0.611 | 0.889 | 0.856 | 0.850 | 6.4 |
| `baseline_rag` | 0.632 | 0.578 | 0.756 | 0.739 | 0.650 | 2.0 |


![Aggregate Agent Performance](notebooks/plot_results/large_llm_run0/snapshot_1_aggregate_agent_performance.png)

### Main Takeaways

- `planner_executor` is the strongest overall agent in this benchmark. It leads on average score, fact coverage, and citation coverage.
- `react_agent` is the strongest retrieval-grounding compromise. It has the best retrieved gold doc hit rate and the best support quality while using fewer steps than `planner_executor`.
- `baseline_rag` is meaningfully weaker than both multi-step agents, which suggests that for these research QA tasks, a single retrieval pass is often not enough.

### Hardest Task

The hardest task in this run was `cold_start_multimodal_vs_social_ripple`:

> Compare SiBraR and SocRipple as solutions for new-item recommendation. What signal does each rely on first, and in what product setting would each be most natural?

Its mean score across agents was only `0.139`, which suggests that tasks requiring fine-grained cross-paper comparison remain difficult even with a curated local corpus.

![Per-Task Score by Agent](notebooks/plot_results/large_llm_run0/per_task_score_by_agent_heatmap.png)

### Discussion

The high-level pattern is:

- `planner_executor` performs best overall because these questions often decompose naturally into subproblems such as identifying methods, comparing mechanisms, and extracting tradeoffs.
- `react_agent` appears more stable across tasks. In this run, its score variance is lower than `planner_executor` (`0.193` vs `0.238` population standard deviation over task scores), and it avoids some of the catastrophic failures where the planner commits to a poor decomposition and ends up with a very low score.
- `react_agent` is also likely cheaper in practice than `planner_executor`, at least by execution depth: it uses `6.4` steps on average versus `8.6` for the planner. This is only a proxy for cost, not a direct token measurement, but it still matters operationally.

So the practical interpretation is not simply "planner is better." A more accurate reading is:

- if the workload is dominated by structured comparison questions over a reasonably complete corpus, `planner_executor` is a strong choice
- if you want a more robust and cheaper default, `react_agent` is also a very reasonable solution

That interpretation matches the task family here. Many of the benchmark questions ask the agent to compare multiple methods, extract contrasts, and synthesize tradeoffs. Those are naturally compatible with planner-executor because the decomposition is often knowable upfront. In this setting, dynamic retrieval decisions from ReAct are still useful, but they are not always the deciding factor.

![Bottleneck Distribution by Agent](notebooks/plot_results/large_llm_run0/bottleneck_distribution_by_agent_pct.png)

## Analysis Notebook

The notebook at `notebooks/analyze_benchmark_report.ipynb`:

- compares agents across aggregate metrics
- plots per-task score heatmaps
- analyzes bottleneck distributions
- generates an automatic narrative summary

It also exports report-ready artifacts to `notebooks/plot_results/` when run.
