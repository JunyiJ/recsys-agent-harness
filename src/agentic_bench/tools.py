from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_bench.schemas import Document


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(slots=True)
class LocalKeywordSearchTool:
    corpus: list[Document]

    def search(self, query: str, top_k: int = 2) -> list[Document]:
        query_tokens = _tokenize(query)
        scored: list[tuple[int, Document]] = []
        for doc in self.corpus:
            overlap = len(query_tokens & _tokenize(f"{doc.title} {doc.text}"))
            scored.append((overlap, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]
