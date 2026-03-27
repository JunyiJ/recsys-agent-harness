from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from agentic_bench.agents.base import BaseAgent
from agentic_bench.schemas import AgentResult, Task, TraceStep

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _extract_first_json_object(response_text: str) -> tuple[dict | None, str]:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = None
    else:
        return (parsed, "full_json") if isinstance(parsed, dict) else (None, "non_object_json")

    decoder = json.JSONDecoder()
    start = response_text.find("{")
    while start != -1:
        try:
            candidate, _ = decoder.raw_decode(response_text[start:])
        except json.JSONDecodeError:
            start = response_text.find("{", start + 1)
            continue
        if isinstance(candidate, dict):
            return candidate, "embedded_json"
        start = response_text.find("{", start + 1)
    return None, "no_json_found"


def _normalize_answer(answer_value: object, fallback: str) -> tuple[str, bool]:
    if isinstance(answer_value, str):
        answer = answer_value.strip()
    elif isinstance(answer_value, list):
        answer = " ".join(str(part).strip() for part in answer_value if part is not None).strip()
    elif answer_value is None:
        answer = ""
    else:
        answer = str(answer_value).strip()

    if answer:
        return answer, True
    return fallback, False


class BaselineRAGAgent(BaseAgent):
    """
    RAG Agent that retrieve once and generate once
    """
    name = "baseline_rag"

    def run(self, task: Task) -> AgentResult:
        docs = self.search_tool.search(task.question, top_k=2)
        ref_ids = [doc.doc_id for doc in docs]
        if not docs:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                answer="No supporting references were retrieved from the corpus.",
                citations=[],
                retrieved_doc_ids=[],
                steps=[
                    TraceStep(kind="retrieve", content=task.question, metadata={"doc_ids": []}),
                    TraceStep(
                        kind="answer",
                        content="No supporting references were retrieved from the corpus.",
                        metadata={"output_contract_ok": False, "failure_reason": "no_references"},
                    ),
                ],
            )

        ref_parts = [f"{doc.doc_id} {doc.title} ({doc.source}, {doc.year}): {doc.text}" for doc in docs]
        ref = "\n\n".join(ref_parts)
        prompt = f"""You are a retrieval-augmented question answering assistant.
Use only the provided references to answer the question.
Return only valid JSON with this exact schema:
{{
  "answer": "string",
  "citations": [
    {{"doc_id": "string", "doc_title": "string"}}
  ]
}}

Requirements:
- "answer" must be concise and grounded in the references.
- "citations" must include only references used in the answer.
- If the references are insufficient, say so in "answer" and return only the citations you actually used.
- Do not include markdown, commentary, or any text outside the JSON object.

Question:
{task.question}

References:
{ref}
        """
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        response = _get_openai_client().responses.create(model=model, input=prompt)
        response_text = response.output_text.strip()
        parsed, parse_mode = _extract_first_json_object(response_text)

        output_contract_ok = True
        failure_reason = ""
        if parsed is None:
            parsed = {"answer": response_text, "citations": []}
            output_contract_ok = False
            failure_reason = parse_mode

        answer, answer_type_ok = _normalize_answer(parsed.get("answer", response_text), response_text)
        if not answer_type_ok:
            output_contract_ok = False
            failure_reason = failure_reason or "invalid_answer_type"

        raw_citations = parsed.get("citations", [])
        if isinstance(raw_citations, dict):
            raw_citations = [raw_citations]
        elif not isinstance(raw_citations, list):
            raw_citations = []
            output_contract_ok = False
            failure_reason = failure_reason or "invalid_citations_type"

        doc_lookup = {doc.doc_id: doc for doc in docs}
        citations: list[str] = []
        for citation in raw_citations:
            if not isinstance(citation, dict):
                output_contract_ok = False
                failure_reason = failure_reason or "invalid_citation_entry"
                continue
            doc_id = citation.get("doc_id")
            if doc_id in doc_lookup and doc_id not in citations:
                citations.append(doc_id)

        steps = [
            TraceStep(kind="retrieve", content=task.question, metadata={"doc_ids": ref_ids}),
            TraceStep(
                kind="answer",
                content=answer,
                metadata={
                    "citations": citations,
                    "output_contract_ok": output_contract_ok,
                    "parse_mode": parse_mode,
                    "failure_reason": failure_reason,
                },
            ),
        ]
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            answer=answer,
            citations=citations,
            retrieved_doc_ids=ref_ids,
            steps=steps,
        )
