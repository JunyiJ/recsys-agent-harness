from __future__ import annotations

from agentic_bench.schemas import AgentResult, EvaluationResult, Task


def _fraction_matched(expected: list[str], actual_text: str) -> float:
    if not expected:
        return 1.0
    actual_lower = actual_text.lower()
    matched = 0
    for fact in expected:
        tokens = [token for token in fact.lower().split() if len(token) > 3]
        if tokens and all(token in actual_lower for token in tokens[:2]):
            matched += 1
    return matched / len(expected)


class RuleBasedEvaluator:
    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        fact_coverage = _fraction_matched(task.required_facts, result.answer)
        citation_hits = sum(1 for citation in task.required_citations if citation in result.citations)
        citation_coverage = citation_hits / len(task.required_citations) if task.required_citations else 1.0
        score = round((0.7 * fact_coverage) + (0.3 * citation_coverage), 3)

        notes: list[str] = []
        if fact_coverage < 1.0:
            notes.append("Missing at least one required fact.")
        if citation_coverage < 1.0:
            notes.append("Missing at least one required citation.")

        return EvaluationResult(
            task_id=task.task_id,
            agent_name=result.agent_name,
            score=score,
            fact_coverage=fact_coverage,
            citation_coverage=citation_coverage,
            notes=notes,
        )
