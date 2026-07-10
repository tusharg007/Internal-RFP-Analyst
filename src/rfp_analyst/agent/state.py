"""Agent state container."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    query: str
    chat_history: list | None = None
    intent: str = "search"
    tool_trace: list[dict] = field(default_factory=list)
    retrieved_documents: list = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    tool_outputs: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    prompt: str = ""
    final_answer: str = ""
    verification: dict = field(default_factory=dict)
