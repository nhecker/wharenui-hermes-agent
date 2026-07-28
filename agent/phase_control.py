"""Generic phase-control primitives for Hermes Agent."""
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

@dataclass
class ControlOutcome:
    action: str  # "enter" | "resume" | "close"
    handler: str
    tool_result: str
    payload: dict[str, Any] = field(default_factory=dict)

class PhaseHandler(Protocol):
    def run(self, agent: Any, messages: list, effective_task_id: str) -> Optional["ControlOutcome"]: ...
