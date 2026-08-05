"""Generic phase-control primitives for Hermes Agent."""
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

PHASE_CONTROL_API_VERSION = 1


@dataclass
class ControlOutcome:
    action: str  # "enter" | "resume" | "close"
    handler: str
    tool_result: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubturnResult:
    """Result of one bounded model turn."""
    content: Optional[str] = None
    tool_calls_used: bool = False
    finish_reason: str = "stop"


class PhaseHandler(Protocol):
    def run(self, agent: Any, messages: list, effective_task_id: str) -> Optional["ControlOutcome"]: ...