from pathlib import Path

from agentic_bench.agents.baseline import BaselineRAGAgent
from agentic_bench.agents.planner_executor import PlannerExecutorAgent
from agentic_bench.agents.react_agent import ReActAgent
from agentic_bench.runner import BenchmarkRunner
from agentic_bench.tasks import load_corpus, load_tasks
from agentic_bench.tools import LocalKeywordSearchTool


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    corpus = load_corpus(project_root / "data" / "corpus" / "recsys_recent_2024_2025_300.jsonl")
    tasks = load_tasks(project_root / "data" / "tasks" / "sample_tasks.jsonl")

    search_tool = LocalKeywordSearchTool(corpus=corpus)
    agents = [
        BaselineRAGAgent(search_tool=search_tool),
        ReActAgent(search_tool=search_tool),
        PlannerExecutorAgent(search_tool=search_tool),
    ]

    runner = BenchmarkRunner(agents=agents, tasks=tasks, corpus=corpus)
    report = runner.run()

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "benchmark_report.json"
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print(f"Wrote benchmark report to {output_path}")
    for summary in report.summaries:
        print(
            f"{summary.agent_name}: avg_score={summary.average_score:.2f}, "
            f"avg_steps={summary.average_steps:.2f}, avg_citations={summary.average_citations:.2f}"
        )


if __name__ == "__main__":
    main()
