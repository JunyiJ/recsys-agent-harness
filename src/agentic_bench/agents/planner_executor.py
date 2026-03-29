from __future__ import annotations

import os

from agentic_bench.agents.base import BaseAgent
from agentic_bench.llm_utils import (
    build_grounded_qa_prompt,
    build_plan,
    get_openai_client,
    parse_grounded_qa_response,
)
from agentic_bench.schemas import AgentResult, PlanStep, Task, TraceStep


class PlannerExecutorAgent(BaseAgent):
    name = "planner_executor"

    def __init__(self, search_tool, top_k: int = 5) -> None:
        super().__init__(search_tool=search_tool)
        self.search_top_k = top_k

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
        plan = self._build_plan(task.question)
        steps = [
            TraceStep(
                kind="plan",
                content=step.question,
                metadata={"step_type": step.type, "step_index": idx + 1},
            )
            for idx, step in enumerate(plan)
        ]

        docs = []
        seen_doc_ids: set[str] = set()
        for step in plan:
            if step.type == "think":
                steps.append(
                    TraceStep(
                        kind="thought",
                        content=step.question,
                        metadata={"source": "planner_step"},
                    )
                )
                continue

            step_docs = self.search_tool.search(step.question, top_k=self.search_top_k)
            steps.append(
                TraceStep(
                    kind="retrieve",
                    content=step.question,
                    metadata={"doc_ids": [doc.doc_id for doc in step_docs]},
                )
            )
            for doc in step_docs:
                if doc.doc_id not in seen_doc_ids:
                    docs.append(doc)
                    seen_doc_ids.add(doc.doc_id)

        if not docs:
            return self._build_no_references_result(task, steps)

        client = get_openai_client()
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        prompt = build_grounded_qa_prompt(question=task.question, docs=docs)
        response = client.responses.create(model=model, input=prompt)
        response_text = response.output_text.strip()
        parsed_response = parse_grounded_qa_response(response_text=response_text, docs=docs)

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
            retrieved_doc_ids=[doc.doc_id for doc in docs],
            steps=steps,
        )

    def _build_plan(self, question: str) -> list[PlanStep]:
        return build_plan(question)
