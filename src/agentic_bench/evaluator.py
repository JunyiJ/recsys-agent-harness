from __future__ import annotations

import os

from agentic_bench.llm_utils import extract_first_json_object, get_openai_client
from agentic_bench.schemas import AgentResult, Document, EvaluationResult, Task


def _fact_is_matched(expected_fact: str, actual_text: str) -> bool:
    actual_lower = actual_text.lower()
    tokens = [token for token in expected_fact.lower().split() if len(token) > 3]
    return bool(tokens) and all(token in actual_lower for token in tokens[:2])


def _fraction_matched(expected: list[str], actual_text: str) -> float:
    if not expected:
        return 1.0
    matched = 0
    for fact in expected:
        if _fact_is_matched(fact, actual_text):
            matched += 1
    return matched / len(expected)


def _coverage_rate(expected_ids: list[str], actual_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    hits = sum(1 for item in expected_ids if item in actual_ids)
    return hits / len(expected_ids)


def llm_judge(
    question: str,
    gold_required_facts: list[str],
    answer: str,
    cited_documents: list[Document],
) -> dict[str, object]:
    cited_references = "\n\n".join(
        f"{doc.doc_id} {doc.title} ({doc.source}, {doc.year}): {doc.text}" for doc in cited_documents
    )
    prompt = f"""You are an evaluator for retrieval-augmented QA.
Return only valid JSON with this exact schema:
{{
  "fact_coverage": [
    {{"fact": "string", "covered": "yes" | "no"}}
  ],
  "support_quality": "supported" | "partially supported" | "unsupported"
}}

Instructions:
- For each required fact, decide whether the answer explicitly contains that fact or a clear equivalent.
- Judge support_quality from the cited reference texts provided below.
- Use "supported" when the cited references directly support the answer's claims.
- Use "partially supported" when the cited references support only part of the answer or the answer overclaims beyond the evidence.
- Use "unsupported" when the cited references do not support the answer.
- Do not include any text outside the JSON object.

Question:
{question}

Required facts:
{chr(10).join(f"- {fact}" for fact in gold_required_facts) or "- None"}

Answer:
{answer}

Cited references:
{cited_references or "None"}
"""
    model = os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.1"))
    response = get_openai_client().responses.create(model=model, input=prompt)
    parsed, _ = extract_first_json_object(response.output_text.strip())

    fallback_fact_coverage = [
        {"fact": fact, "covered": "yes" if _fact_is_matched(fact, answer) else "no"}
        for fact in gold_required_facts
    ]
    fallback_support_quality = "supported" if cited_documents else "unsupported"
    if not isinstance(parsed, dict):
        return {
            "fact_coverage": fallback_fact_coverage,
            "support_quality": fallback_support_quality,
        }

    raw_fact_coverage = parsed.get("fact_coverage", [])
    fact_coverage: list[dict[str, str]] = []
    for idx, fact in enumerate(gold_required_facts):
        covered: str | None = None
        if isinstance(raw_fact_coverage, list) and idx < len(raw_fact_coverage):
            item = raw_fact_coverage[idx]
            if isinstance(item, dict):
                llm_covered = str(item.get("covered", "")).strip().lower()
                if llm_covered in {"yes", "no"}:
                    covered = llm_covered
        if covered is None:
            covered = "yes" if _fact_is_matched(fact, answer) else "no"
        fact_coverage.append({"fact": fact, "covered": covered})

    support_quality = str(parsed.get("support_quality", fallback_support_quality)).strip().lower()
    if support_quality not in {"supported", "partially supported", "unsupported"}:
        support_quality = fallback_support_quality

    return {
        "fact_coverage": fact_coverage,
        "support_quality": support_quality,
    }



class RuleBasedEvaluator:
    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        fact_coverage = _fraction_matched(task.required_facts, result.answer)
        retrieved_doc_hit_rate = _coverage_rate(task.required_citations, result.retrieved_doc_ids)
        citation_coverage = _coverage_rate(task.required_citations, result.citations)
        support_quality = citation_coverage
        score = round((0.6 * fact_coverage) + (0.2 * citation_coverage) + (0.2 * support_quality), 3)

        notes: list[str] = []
        if fact_coverage < 1.0:
            notes.append("Missing at least one required fact.")
        if citation_coverage < 1.0:
            notes.append("Missing at least one required citation.")
        notes.append("Support quality is a rule-based proxy in this evaluator.")

        return EvaluationResult(
            task_id=task.task_id,
            agent_name=result.agent_name,
            score=score,
            fact_coverage=fact_coverage,
            retrieved_doc_hit_rate=retrieved_doc_hit_rate,
            citation_coverage=citation_coverage,
            support_quality=support_quality,
            notes=notes,
        )


class LLMBasedEvaluator:
    def __init__(self, corpus: list[Document]) -> None:
        self.doc_lookup = {doc.doc_id: doc for doc in corpus}

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        cited_documents = [self.doc_lookup[citation] for citation in result.citations if citation in self.doc_lookup]
        llm_eval_results = llm_judge(
            question=task.question,
            gold_required_facts=task.required_facts,
            answer=result.answer,
            cited_documents=cited_documents,
        )
        judged_fact_coverage = llm_eval_results.get("fact_coverage", [])
        if isinstance(judged_fact_coverage, list) and judged_fact_coverage:
            covered_facts = sum(
                1
                for item in judged_fact_coverage
                if isinstance(item, dict) and str(item.get("covered", "no")).strip().lower() == "yes"
            )
            fact_coverage = covered_facts / len(judged_fact_coverage)
        else:
            fact_coverage = 1.0 if not task.required_facts else 0.0

        support_quality = str(llm_eval_results.get("support_quality", "unsupported")).strip().lower()
        support_quality = {
            "supported": 1.0,
            "partially supported": 0.5,
            "unsupported": 0.0,
        }.get(support_quality, 0.0)
        retrieved_doc_hit_rate = _coverage_rate(task.required_citations, result.retrieved_doc_ids)
        citation_coverage = _coverage_rate(task.required_citations, result.citations)
        score = round((0.5 * fact_coverage) + (0.2 * citation_coverage) + (0.3 * support_quality), 3)

        notes: list[str] = []
        if fact_coverage < 1.0:
            notes.append("Missing at least one required fact.")
        if citation_coverage < 1.0:
            notes.append("Citations do not fully support the answer.")
        if support_quality < 1.0:
            notes.append("Support quality is not fully grounded according to the LLM judge.")
        if not cited_documents:
            notes.append("No cited document text was available for grounding evaluation.")

        return EvaluationResult(
            task_id=task.task_id,
            agent_name=result.agent_name,
            score=score,
            fact_coverage=fact_coverage,
            retrieved_doc_hit_rate=retrieved_doc_hit_rate,
            citation_coverage=citation_coverage,
            support_quality=support_quality,
            notes=notes,
        )
