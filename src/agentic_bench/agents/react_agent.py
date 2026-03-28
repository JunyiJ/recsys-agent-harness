from __future__ import annotations

import os

from agentic_bench.agents.base import BaseAgent
from agentic_bench.llm_utils import (
    build_grounded_qa_prompt,
    get_openai_client,
    normalize_questions,
    parse_grounded_qa_response,
)
from agentic_bench.schemas import AgentResult, Task, TraceStep


class ReActAgent(BaseAgent):
    name = "react_agent"
    max_initial_questions = 3
    search_top_k = 5

    def _build_no_references_result(self, task: Task, steps: list[TraceStep]) -> AgentResult:
        steps.append(
            TraceStep(
                kind="answer",
                content="No supporting references were retrieved from the corpus.",
                metadata={"output_contract_ok": False, "failure_reason": "no_references"},
            )
        )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            answer="No supporting references were retrieved from the corpus.",
            citations=[],
            retrieved_doc_ids=[],
            steps=steps,
        )

    def run(self, task: Task) -> AgentResult:
        question_plan_prompt = f"""Break the user question into the set of 2-3 sub-questions needed to answer it.
        Return only valid JSON with this exact schema:
        {{
        "question": ["string"]
        }}

        Question:
        {task.question}
        """
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        question_plan_response = get_openai_client().responses.create(
            model=model,
            input=question_plan_prompt,
        )
        question_plan_text = question_plan_response.output_text.strip()
        question_plan_json, question_plan_parse_mode = extract_first_json_object(question_plan_text)
        questions = normalize_questions(
            question_plan_json.get("question") if question_plan_json else None,
            task.question,
        )[: self.max_initial_questions]
        steps: list[TraceStep] = [
            TraceStep(
                kind="thought",
                content="Plan a small set of search queries from the user question.",
                metadata={
                    "planning_prompt": question_plan_prompt,
                    "questions": questions,
                    "parse_mode": question_plan_parse_mode,
                },
            ),
        ]

        initial_docs = []
        for question in questions:
            question_docs = self.search_tool.search(question, top_k=self.search_top_k)
            initial_docs.extend(question_docs)
            steps.append(
                TraceStep(
                    kind="retrieve",
                    content=question,
                    metadata={"doc_ids": [doc.doc_id for doc in question_docs]},
                )
            )
        initial_docs = list({doc.doc_id: doc for doc in initial_docs}.values())
        initial_references = "\n\n".join(
            f"{doc.doc_id} {doc.title} ({doc.source}, {doc.year}): {doc.text}" for doc in initial_docs
        )
        follow_up_prompt = f"""Suggest one concise follow-up search question that could help answer the user's question better.
        Return only the follow-up question as plain text.

        User question:
        {task.question}

        Retrieved references:
        {initial_references}
        """
        follow_up_response = get_openai_client().responses.create(
            model=model,
            input=follow_up_prompt,
        )
        follow_up = follow_up_response.output_text.strip() or task.question
        steps.append(
            TraceStep(
                kind="thought",
                content="Use the first-pass evidence to choose one follow-up search.",
                metadata={"follow_up_prompt": follow_up_prompt, "follow_up_query": follow_up},
            )
        )
        extra_docs = self.search_tool.search(follow_up, top_k=self.search_top_k)
        steps.append(
            TraceStep(kind="retrieve", content=follow_up, metadata={"doc_ids": [doc.doc_id for doc in extra_docs]})
        )

        all_docs = list({doc.doc_id: doc for doc in (initial_docs + extra_docs)}.values())
        if not all_docs:
            return self._build_no_references_result(task, steps)

        prompt = build_grounded_qa_prompt(question=task.question, docs=all_docs)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        response = get_openai_client().responses.create(model=model, input=prompt)
        response_text = response.output_text.strip()
        parsed_response = parse_grounded_qa_response(response_text=response_text, docs=all_docs)

        steps.append(
            TraceStep(
                kind="answer",
                content=parsed_response["answer"],
                metadata={
                    "citations": parsed_response["citations"],
                    "output_contract_ok": parsed_response["output_contract_ok"],
                    "parse_mode": parsed_response["parse_mode"],
                    "failure_reason": parsed_response["failure_reason"],
                },
            )
        )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            answer=parsed_response["answer"],
            citations=parsed_response["citations"],
            retrieved_doc_ids=[doc.doc_id for doc in all_docs],
            steps=steps,
        )
