from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    source: str
    year: int
    url: str
    text: str


@dataclass(slots=True)
class Task:
    task_id: str
    question: str
    domain: str
    required_facts: list[str]
    required_citations: list[str]


@dataclass(slots=True)
class TraceStep:
    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlanStep:
    type: str
    question: str


@dataclass(slots=True)
class AgentResult:
    agent_name: str
    task_id: str
    answer: str
    citations: list[str]
    retrieved_doc_ids: list[str]
    steps: list[TraceStep]


@dataclass(slots=True)
class EvaluationResult:
    task_id: str
    agent_name: str
    score: float
    fact_coverage: float
    citation_coverage: float
    notes: list[str]


@dataclass(slots=True)
class RunRecord:
    task: Task
    result: AgentResult
    evaluation: EvaluationResult


@dataclass(slots=True)
class AgentSummary:
    agent_name: str
    average_score: float
    average_steps: float
    average_citations: float


@dataclass(slots=True)
class BenchmarkReport:
    records: list[RunRecord]
    summaries: list[AgentSummary]

    def model_dump_json(self, indent: int = 2) -> str:
        import json
        from dataclasses import asdict

        return json.dumps(asdict(self), indent=indent)
