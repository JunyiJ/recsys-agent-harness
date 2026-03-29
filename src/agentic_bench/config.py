from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class DataConfig:
    corpus_path: str
    tasks_path: str


@dataclass(slots=True)
class ModelConfig:
    actor: str
    judge: str


@dataclass(slots=True)
class EvaluatorConfig:
    type: str


@dataclass(slots=True)
class OutputConfig:
    output_dir: str
    report_filename: str
    manifest_filename: str


@dataclass(slots=True)
class RunConfig:
    run_name: str
    data: DataConfig
    agents: list[str]
    agent_settings: dict[str, dict[str, int | float | str | bool]]
    models: ModelConfig
    evaluator: EvaluatorConfig
    output: OutputConfig

    def to_dict(self) -> dict:
        return asdict(self)


def load_run_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return RunConfig(
        run_name=str(raw.get("run_name", "default")).strip() or "default",
        data=DataConfig(**raw.get("data", {})),
        agents=list(raw.get("agents", [])),
        agent_settings=dict(raw.get("agent_settings", {})),
        models=ModelConfig(**raw.get("models", {})),
        evaluator=EvaluatorConfig(**raw.get("evaluator", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
