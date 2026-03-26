from __future__ import annotations

from agentic_bench.agents.base import BaseAgent
from agentic_bench.schemas import AgentResult, Task, TraceStep


class ReActAgent(BaseAgent):
    name = "react_agent"

    def run(self, task: Task) -> AgentResult:
        steps: list[TraceStep] = [
            TraceStep(kind="thought", content="Identify the main information need in the question."),
        ]

        initial_docs = self.search_tool.search(task.question, top_k=1)
        citations = [doc.doc_id for doc in initial_docs]
        steps.append(
            TraceStep(kind="retrieve", content=task.question, metadata={"doc_ids": citations})
        )

        follow_up = "tradeoff personalization coverage" if "cold-start" in task.question else "retrieval reranking candidate set"
        extra_docs = self.search_tool.search(follow_up, top_k=1)
        for doc in extra_docs:
            if doc.doc_id not in citations:
                citations.append(doc.doc_id)

        steps.append(
            TraceStep(
                kind="thought",
                content="Retrieved one more document to cover a likely missing detail.",
            )
        )
        steps.append(
            TraceStep(kind="retrieve", content=follow_up, metadata={"doc_ids": [doc.doc_id for doc in extra_docs]})
        )

        all_docs = initial_docs + [doc for doc in extra_docs if doc.doc_id not in {d.doc_id for d in initial_docs}]
        answer = " ".join(f"{doc.title}: {doc.text}" for doc in all_docs)
        steps.append(TraceStep(kind="answer", content=answer))

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            answer=answer,
            citations=citations,
            retrieved_doc_ids=[doc.doc_id for doc in all_docs],
            steps=steps,
        )
