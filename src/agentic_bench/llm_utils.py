from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from agentic_bench.schemas import Document, PlanStep

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_openai_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def extract_first_json_object(response_text: str) -> tuple[dict | None, str]:
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


def extract_first_json_array(response_text: str) -> tuple[list | None, str]:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = None
    else:
        return (parsed, "full_json") if isinstance(parsed, list) else (None, "non_array_json")

    decoder = json.JSONDecoder()
    start = response_text.find("[")
    while start != -1:
        try:
            candidate, _ = decoder.raw_decode(response_text[start:])
        except json.JSONDecodeError:
            start = response_text.find("[", start + 1)
            continue
        if isinstance(candidate, list):
            return candidate, "embedded_json"
        start = response_text.find("[", start + 1)
    return None, "no_json_found"


def normalize_answer(answer_value: object, fallback: str) -> tuple[str, bool]:
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


def normalize_questions(question_value: object, fallback: str) -> list[str]:
    if isinstance(question_value, str):
        questions = [question_value.strip()]
    elif isinstance(question_value, list):
        questions = [str(question).strip() for question in question_value if question is not None]
    else:
        questions = []

    normalized_questions = [question for question in questions if question]
    if normalized_questions:
        return normalized_questions

    fallback_question = fallback.strip()
    return [fallback_question] if fallback_question else []


def parse_grounded_qa_response(response_text: str, docs: list[Document]) -> dict[str, object]:
    parsed, parse_mode = extract_first_json_object(response_text)

    output_contract_ok = True
    failure_reason = ""
    if parsed is None:
        parsed = {"answer": response_text, "citations": []}
        output_contract_ok = False
        failure_reason = parse_mode

    answer, answer_type_ok = normalize_answer(parsed.get("answer", response_text), response_text)
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

    return {
        "answer": answer,
        "citations": citations,
        "output_contract_ok": output_contract_ok,
        "parse_mode": parse_mode,
        "failure_reason": failure_reason,
    }


def build_grounded_qa_prompt(question: str, docs: list[Document]) -> str:
    ref_parts = [f"{doc.doc_id} {doc.title} ({doc.source}, {doc.year}): {doc.text}" for doc in docs]
    references = "\n\n".join(ref_parts)
    return f"""You are a retrieval-augmented question answering assistant.
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
{question}

References:
{references}
"""


def build_plan_prompt(question: str) -> str:
    return f"""You are a planning assistant for a retrieval agent.
Return only valid JSON as an array with at most 4 items.
Each item must be an object with this exact schema:
{{
  "type": "search" | "think",
  "question": "string"
}}

Rules:
- Break the user question into the smallest useful sequence of steps.
- Keep each "question" concise and actionable.
- Use "search" for information that should be retrieved externally.
- Use "think" for synthesis or comparison steps.
- Do not include any text outside the JSON array.

User question:
{question}
"""


def build_plan(question: str) -> list[PlanStep]:
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    response = get_openai_client().responses.create(
        model=model,
        input=build_plan_prompt(question),
    )
    response_text = response.output_text.strip()
    parsed, _ = extract_first_json_array(response_text)

    if not isinstance(parsed, list):
        fallback_question = question.strip()
        return [PlanStep(type="search", question=fallback_question)] if fallback_question else []

    steps: list[PlanStep] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        step_type_value = item.get("type")
        question_value = item.get("question")
        if not isinstance(step_type_value, str):
            continue
        if not isinstance(question_value, str):
            continue
        step_type = step_type_value.strip().lower()
        step_question = question_value.strip()
        if step_type not in {"search", "think"}:
            continue
        if step_question:
            steps.append(PlanStep(type=step_type, question=step_question))
        if len(steps) == 4:
            break

    if steps:
        return steps

    fallback_question = question.strip()
    return [PlanStep(type="search", question=fallback_question)] if fallback_question else []
