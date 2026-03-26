from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_bench.schemas import AgentResult, Task
from agentic_bench.tools import LocalKeywordSearchTool


class BaseAgent(ABC):
    name: str

    def __init__(self, search_tool: LocalKeywordSearchTool) -> None:
        self.search_tool = search_tool

    @abstractmethod
    def run(self, task: Task) -> AgentResult:
        raise NotImplementedError
