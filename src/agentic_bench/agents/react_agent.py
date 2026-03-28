from __future__ import annotations

import os

from agentic_bench.agents.base import BaseAgent
from agentic_bench.llm_utils import (
    build_grounded_qa_prompt,
    extract_first_json_object,
    get_openai_client,
    parse_grounded_qa_response,
)
from agentic_bench.schemas import AgentResult, Document, Task, TraceStep


class ReActAgent(BaseAgent):
    name = "react_agent"
    max_searches = 3
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

    def _build_decision_prompt(
        self,
        task: Task,
        docs: list[Document],
        seen_queries: list[str],
        searches_used: int,
    ) -> str:
        references = "\n\n".join(
            f"{doc.doc_id} {doc.title} ({doc.source}, {doc.year}): {doc.text}" for doc in docs
        )
        seen_queries_text = "\n".join(f"- {query}" for query in seen_queries) or "- none"
        references_text = references or "No references retrieved yet."
        return f"""You are deciding the next action for a ReAct-style research agent.
        Choose exactly one action: "search" or "answer".

        Return only valid JSON with this exact schema:
        {{
        "thought": "string",
        "action": "search" | "answer",
        "query": "string"
        }}

        Rules:
        - If the current evidence is not enough to answer well, choose "search".
        - If there is enough evidence, choose "answer".
        - If you choose "search", provide one concise search query in "query".
        - If you choose "answer", set "query" to an empty string.
        - Avoid repeating previous queries.

        User question:
        {task.question}

        Searches used:
        {searches_used} of {self.max_searches}

        Previous queries:
        {seen_queries_text}

        Retrieved references:
        {references_text}
        """

    def _normalize_decision(self, response_text: str, docs: list[Document]) -> tuple[str, str, str, str]:
        parsed, parse_mode = extract_first_json_object(response_text)
        if not isinstance(parsed, dict):
            if docs:
                return "answer", "", "Defaulting to answer because the decision output was not valid JSON.", parse_mode
            return "search", "", "Defaulting to search because no references have been retrieved yet.", parse_mode

        thought_value = parsed.get("thought", "")
        thought = thought_value.strip() if isinstance(thought_value, str) else str(thought_value).strip()

        action_value = parsed.get("action", "")
        action = action_value.strip().lower() if isinstance(action_value, str) else ""
        if action not in {"search", "answer"}:
            action = "answer" if docs else "search"

        query_value = parsed.get("query", "")
        query = query_value.strip() if isinstance(query_value, str) else ""

        if not thought:
            thought = "Choose the next action based on the retrieved evidence."
        return action, query, thought, parse_mode

    def run(self, task: Task) -> AgentResult:
        client = get_openai_client()
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        steps: list[TraceStep] = []
        all_docs: list[Document] = []
        seen_doc_ids: set[str] = set()
        seen_queries: list[str] = []

        for search_idx in range(self.max_searches):
            decision_prompt = self._build_decision_prompt(
                task=task,
                docs=all_docs,
                seen_queries=seen_queries,
                searches_used=search_idx,
            )
            decision_response = client.responses.create(model=model, input=decision_prompt)
            decision_text = decision_response.output_text.strip()
            action, query, thought, parse_mode = self._normalize_decision(decision_text, all_docs)

            steps.append(
                TraceStep(
                    kind="thought",
                    content=thought,
                    metadata={
                        "action": action,
                        "query": query,
                        "parse_mode": parse_mode,
                        "search_index": search_idx + 1,
                    },
                )
            )

            if action == "answer":
                break

            if not query:
                steps.append(
                    TraceStep(
                        kind="thought",
                        content="Stopping because the next search query was empty.",
                        metadata={"stop_reason": "empty_query"},
                    )
                )
                break

            if query in seen_queries:
                steps.append(
                    TraceStep(
                        kind="thought",
                        content="Stopping because the next search query duplicated a previous query.",
                        metadata={"stop_reason": "duplicate_query", "query": query},
                    )
                )
                break

            seen_queries.append(query)
            retrieved_docs = self.search_tool.search(query, top_k=self.search_top_k)
            retrieved_doc_ids = [doc.doc_id for doc in retrieved_docs]
            steps.append(
                TraceStep(
                    kind="retrieve",
                    content=query,
                    metadata={"doc_ids": retrieved_doc_ids},
                )
            )

            new_docs = [doc for doc in retrieved_docs if doc.doc_id not in seen_doc_ids]
            if not new_docs:
                steps.append(
                    TraceStep(
                        kind="thought",
                        content="Stopping because retrieval produced no new evidence.",
                        metadata={"stop_reason": "no_new_evidence", "query": query},
                    )
                )
                break

            for doc in new_docs:
                seen_doc_ids.add(doc.doc_id)
                all_docs.append(doc)

        if not all_docs:
            return self._build_no_references_result(task, steps)

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
            answer=parsed_response["answer"],
            citations=parsed_response["citations"],
            retrieved_doc_ids=[doc.doc_id for doc in all_docs],
            steps=steps,
        )
