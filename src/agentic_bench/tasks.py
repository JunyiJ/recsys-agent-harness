from __future__ import annotations

import json
from pathlib import Path

from agentic_bench.schemas import Document, Task


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_tasks(path: Path) -> list[Task]:
    rows = _load_jsonl(path)
    return [Task(**row) for row in rows]


def load_corpus(path: Path) -> list[Document]:
    rows = _load_jsonl(path)
    return [Document(**row) for row in rows]
