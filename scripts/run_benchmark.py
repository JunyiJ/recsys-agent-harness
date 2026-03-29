import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from agentic_bench.agents.baseline import BaselineRAGAgent
from agentic_bench.agents.planner_executor import PlannerExecutorAgent
from agentic_bench.agents.react_agent import ReActAgent
from agentic_bench.config import RunConfig, load_run_config
from agentic_bench.runner import BenchmarkRunner
from agentic_bench.tasks import load_corpus, load_tasks
from agentic_bench.tools import LocalKeywordSearchTool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic benchmark harness from a YAML config.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the run config YAML, relative to the project root unless absolute.",
    )
    return parser.parse_args()


def _resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _build_agents(
    agent_names: list[str],
    search_tool: LocalKeywordSearchTool,
    agent_settings: dict[str, dict[str, int | float | str | bool]],
) -> list:
    agent_registry = {
        "baseline_rag": {
            "cls": BaselineRAGAgent,
            "allowed_settings": {"top_k"},
        },
        "react_agent": {
            "cls": ReActAgent,
            "allowed_settings": {"top_k"},
        },
        "planner_executor": {
            "cls": PlannerExecutorAgent,
            "allowed_settings": {"top_k"},
        },
    }

    agents = []
    for agent_name in agent_names:
        agent_spec = agent_registry.get(agent_name)
        if agent_spec is None:
            raise ValueError(f"Unknown agent '{agent_name}' in run config.")
        agent_cls = agent_spec["cls"]
        allowed_settings = agent_spec["allowed_settings"]
        raw_settings = agent_settings.get(agent_name, {})
        unknown_settings = set(raw_settings) - allowed_settings
        if unknown_settings:
            unknown_settings_str = ", ".join(sorted(unknown_settings))
            allowed_settings_str = ", ".join(sorted(allowed_settings)) or "none"
            raise ValueError(
                f"Unknown settings for agent '{agent_name}': {unknown_settings_str}. "
                f"Allowed settings: {allowed_settings_str}."
            )

        settings = dict(raw_settings)
        if "top_k" in settings:
            top_k = settings["top_k"]
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(
                    f"Invalid top_k for agent '{agent_name}': {top_k!r}. "
                    "Expected a positive integer."
                )

        agents.append(agent_cls(search_tool=search_tool, **settings))
    return agents


def _get_git_metadata(project_root: Path) -> dict[str, str | None]:
    def _run_git(*args: str) -> str | None:
        try:
            output = subprocess.check_output(
                ["git", *args],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return output or None

    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
    }


def _write_manifest(
    path: Path,
    *,
    config: RunConfig,
    config_path: Path,
    report_path: Path,
    corpus_path: Path,
    tasks_path: Path,
    git_metadata: dict[str, str | None],
) -> None:
    manifest = {
        "run_name": config.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "report_path": str(report_path),
        "corpus_path": str(corpus_path),
        "tasks_path": str(tasks_path),
        "git": git_metadata,
        "resolved_config": config.to_dict(),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(project_root, args.config)
    config = load_run_config(config_path)

    os.environ["OPENAI_MODEL"] = config.models.actor
    os.environ["OPENAI_JUDGE_MODEL"] = config.models.judge

    corpus_path = _resolve_path(project_root, config.data.corpus_path)
    tasks_path = _resolve_path(project_root, config.data.tasks_path)
    corpus = load_corpus(corpus_path)
    tasks = load_tasks(tasks_path)

    search_tool = LocalKeywordSearchTool(corpus=corpus)
    agents = _build_agents(config.agents, search_tool, config.agent_settings)

    runner = BenchmarkRunner(
        agents=agents,
        tasks=tasks,
        evaluator_type=config.evaluator.type,
        corpus=corpus,
    )
    report = runner.run()

    output_dir = _resolve_path(project_root, config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output.report_filename
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / config.output.manifest_filename
    _write_manifest(
        manifest_path,
        config=config,
        config_path=config_path,
        report_path=output_path,
        corpus_path=corpus_path,
        tasks_path=tasks_path,
        git_metadata=_get_git_metadata(project_root),
    )

    print(f"Wrote benchmark report to {output_path}")
    print(f"Wrote run manifest to {manifest_path}")
    for summary in report.summaries:
        print(
            f"{summary.agent_name}: avg_score={summary.average_score:.2f}, "
            f"avg_steps={summary.average_steps:.2f}, avg_citations={summary.average_citations:.2f}"
        )


if __name__ == "__main__":
    main()
