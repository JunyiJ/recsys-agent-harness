from __future__ import annotations

from agentic_bench.agents.base import BaseAgent
from agentic_bench.schemas import AgentResult, Task, TraceStep


class BaselineRAGAgent(BaseAgent):
    name = "baseline_rag"

    def run(self, task: Task) -> AgentResult:
        docs = self.search_tool.search(task.question, top_k=2)
        citations = [doc.doc_id for doc in docs]
        answer_parts = [f"{doc.title} ({doc.source}, {doc.year}): {doc.text}" for doc in docs]
        answer = " ".join(answer_parts)
        steps = [
            TraceStep(kind="retrieve", content=task.question, metadata={"doc_ids": citations}),
            TraceStep(kind="answer", content=answer),
        ]
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            answer=answer,
            citations=citations,
            retrieved_doc_ids=citations,
            steps=steps,
        )
