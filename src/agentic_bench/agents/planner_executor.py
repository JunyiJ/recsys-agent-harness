from __future__ import annotations

from agentic_bench.agents.base import BaseAgent
from agentic_bench.schemas import AgentResult, Task, TraceStep


class PlannerExecutorAgent(BaseAgent):
    name = "planner_executor"

    def run(self, task: Task) -> AgentResult:
        plan = self._build_plan(task.question)
        steps = [TraceStep(kind="plan", content=step) for step in plan]

        docs = []
        seen_doc_ids: set[str] = set()
        for step in plan:
            step_docs = self.search_tool.search(step, top_k=1)
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

        answer = " ".join(f"{doc.title}: {doc.text}" for doc in docs)
        steps.append(TraceStep(kind="synthesize", content=answer))

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            answer=answer,
            citations=[doc.doc_id for doc in docs],
            retrieved_doc_ids=[doc.doc_id for doc in docs],
            steps=steps,
        )

    def _build_plan(self, question: str) -> list[str]:
        if "cold-start" in question:
            return [
                "find one approach for cold-start recommendation",
                "find a second approach for cold-start recommendation",
                "find a tradeoff for personalization or coverage",
            ]
        return [
            "find what retrieval does in recommendation pipelines",
            "find what reranking does in recommendation pipelines",
        ]
