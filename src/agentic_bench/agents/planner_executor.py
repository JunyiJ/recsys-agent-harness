from __future__ import annotations

from agentic_bench.agents.base import BaseAgent
from agentic_bench.llm_utils import build_plan
from agentic_bench.schemas import AgentResult, Task, TraceStep


class PlannerExecutorAgent(BaseAgent):
    name = "planner_executor"

    def run(self, task: Task) -> AgentResult:
        plan = self._build_plan(task.question)
        steps = [TraceStep(kind="plan", content=step) for step in plan]

        docs = []
        seen_doc_ids: set[str] = set()
        for step in plan:
            step_docs = self.search_tool.search(step, top_k=5)
            if not step_docs:
                continue
            steps.append(
                TraceStep(
                    kind="execute",
                    content=step,
                    metadata={"doc_ids": [doc.doc_id for doc in step_docs]},
                )
            )
            for doc in step_docs:
                if doc.doc_id not in seen_doc_ids:
                    docs.append(doc)
                    seen_doc_ids.add(doc.doc_id)

        prompt = build_grounded_qa_prompt(question=task.question, docs=all_docs)
        response = client.responses.create(model=model, input=prompt)
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
            answer=answer,
            citations=[doc.doc_id for doc in docs],
            retrieved_doc_ids=[doc.doc_id for doc in docs],
            steps=steps,
        )

    def _build_plan(self, question: str) -> list[str]:
        return build_plan(question)
