from __future__ import annotations

from agentic_bench.agents.base import BaseAgent
from agentic_bench.evaluator import LLMBasedEvaluator, RuleBasedEvaluator
from agentic_bench.schemas import AgentSummary, BenchmarkReport, Document, RunRecord, Task


class BenchmarkRunner:
    def __init__(
        self,
        agents: list[BaseAgent],
        tasks: list[Task],
        *,
        evaluator_type: str = "rule",
        corpus: list[Document] | None = None,
    ) -> None:
        self.agents = agents
        self.tasks = tasks
        evaluator_type = evaluator_type.strip().lower()
        if evaluator_type == "llm":
            if corpus is None:
                raise ValueError("LLM-based evaluation requires the corpus to look up cited document text.")
            self.evaluator = LLMBasedEvaluator(corpus=corpus)
        else:
            self.evaluator = RuleBasedEvaluator()

    def run(self) -> BenchmarkReport:
        records: list[RunRecord] = []

        for agent in self.agents:
            for task in self.tasks:
                result = agent.run(task)
                evaluation = self.evaluator.evaluate(task, result)
                records.append(RunRecord(task=task, result=result, evaluation=evaluation))

        summaries: list[AgentSummary] = []
        for agent in self.agents:
            agent_records = [record for record in records if record.result.agent_name == agent.name]
            summaries.append(
                AgentSummary(
                    agent_name=agent.name,
                    average_score=round(
                        sum(record.evaluation.score for record in agent_records) / len(agent_records), 3
                    ),
                    average_fact_coverage=round(
                        sum(record.evaluation.fact_coverage for record in agent_records) / len(agent_records), 3
                    ),
                    average_retrieved_doc_hit_rate=round(
                        sum(record.evaluation.retrieved_doc_hit_rate for record in agent_records) / len(agent_records), 3
                    ),
                    average_citation_coverage=round(
                        sum(record.evaluation.citation_coverage for record in agent_records) / len(agent_records), 3
                    ),
                    average_support_quality=round(
                        sum(record.evaluation.support_quality for record in agent_records) / len(agent_records), 3
                    ),
                    average_steps=round(
                        sum(len(record.result.steps) for record in agent_records) / len(agent_records), 3
                    ),
                    average_citations=round(
                        sum(len(record.result.citations) for record in agent_records) / len(agent_records), 3
                    ),
                )
            )

        return BenchmarkReport(records=records, summaries=summaries)
