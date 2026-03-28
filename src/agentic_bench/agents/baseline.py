from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from agentic_bench.agents.base import BaseAgent
from agentic_bench.llm_utils import (
    build_grounded_qa_prompt,
    get_openai_client,
    parse_grounded_qa_response,
)
from agentic_bench.schemas import AgentResult, Task, TraceStep


class BaselineRAGAgent(BaseAgent):
    """
    RAG Agent that retrieve once and generate once
    """
    name = "baseline_rag"

    def run(self, task: Task) -> AgentResult:
        docs = self.search_tool.search(task.question, top_k=8)
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
        prompt = build_grounded_qa_prompt(question=task.question, docs=docs)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        response = get_openai_client().responses.create(model=model, input=prompt)
        response_text = response.output_text.strip()
        parsed_response = parse_grounded_qa_response(response_text=response_text, docs=docs)

        steps = [
            TraceStep(kind="retrieve", content=task.question, metadata={"doc_ids": ref_ids}),
            TraceStep(
                kind="answer",
                content=parsed_response["answer"],
                metadata={
                    "citations": parsed_response["citations"],
                    "output_contract_ok": parsed_response["output_contract_ok"],
                    "parse_mode": parsed_response["parse_mode"],
                    "failure_reason": parsed_response["failure_reason"],
                },
            ),
        ]
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            answer=parsed_response["answer"],
            citations=parsed_response["citations"],
            retrieved_doc_ids=ref_ids,
            steps=steps,
        )
