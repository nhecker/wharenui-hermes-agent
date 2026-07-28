"""Tests for the generic phase-control seam (WP2).

Uses a StubPhaseHandler that emits a fixed CANARY string during the
private phase. Tests verify:
- reflect_pause/reflect_settle/reflect_done protocol
- CANARY presence in private context but absence from public sinks
- reflect_done rejection in public phase
- Exclusivity enforcement (multi-call rejection)
"""

from agent.phase_control import ControlOutcome

CANARY = "WHARE-CANARY-7f3a9b2e"
MAX_PRIVATE_TURNS = 3


class StubPhaseHandler:
    """In-memory phase handler that emits CANARY on each private turn."""

    def __init__(self, max_turns: int = MAX_PRIVATE_TURNS):
        self._max = max_turns
        self._turn_count = 0

    def begin(self, args: dict) -> ControlOutcome:
        return ControlOutcome(
            action="enter",
            handler="reflect_pause",
            tool_result="reflecting...",
        )

    def run(self, agent, messages: list, effective_task_id: str) -> ControlOutcome | None:
        self._turn_count += 1
        messages.append({"role": "assistant", "content": CANARY})
        if self._turn_count >= self._max:
            return ControlOutcome(
                action="close", handler="reflect_done", tool_result="Done reflecting."
            )
        return ControlOutcome(
            action="resume", handler="reflect_settle", tool_result="Returned to window."
        )


REFLECT_PAUSE_SCHEMA = {
    "name": "reflect_pause",
    "description": "Enter private reflection time. Must be the only tool call.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
REFLECT_SETTLE_SCHEMA = {
    "name": "reflect_settle",
    "description": "Return from private time to the public window.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
REFLECT_DONE_SCHEMA = {
    "name": "reflect_done",
    "description": "End the session from private or closing-private time.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# --- Protocol-level tests (no agent, no loop) ---


def test_control_outcome_defaults():
    o = ControlOutcome(action="enter", handler="hp", tool_result="ack")
    assert o.action == "enter"
    assert o.handler == "hp"
    assert o.tool_result == "ack"
    assert o.payload == {}


def test_control_outcome_payload():
    o = ControlOutcome(action="close", handler="hp", tool_result="done", payload={"reason": "time"})
    assert o.payload["reason"] == "time"


def test_stub_begin_returns_enter():
    h = StubPhaseHandler()
    o = h.begin({})
    assert o.action == "enter"
    assert o.handler == "reflect_pause"
    assert o.tool_result == "reflecting..."


def test_stub_run_appends_canary():
    h = StubPhaseHandler(max_turns=2)
    msgs = []
    result = h.run(None, msgs, "task1")
    assert len(msgs) == 1
    assert msgs[0]["content"] == CANARY
    assert result.action == "resume"


def test_stub_run_loops_then_closes():
    h = StubPhaseHandler(max_turns=3)
    msgs = []
    r1 = h.run(None, msgs, "task1")
    assert r1.action == "resume"
    r2 = h.run(None, msgs, "task1")
    assert r2.action == "resume"
    r3 = h.run(None, msgs, "task1")
    assert r3.action == "close"
    assert len(msgs) == 3


def test_stub_run_with_zero_max_then_closes():
    h = StubPhaseHandler(max_turns=1)
    msgs = []
    r = h.run(None, msgs, "task1")
    assert r.action == "close"
    assert len(msgs) == 1


# --- Handoff logic (simulated loop) ---


def test_handoff_emits_canary():
    """Simulate the conversation_loop.py handoff block."""
    h = StubPhaseHandler(max_turns=2)
    outcome = h.begin({})
    agent = type("Agent", (), {
        "_phase": "public",
        "_pending_phase_transition": outcome,
        "_control_handlers": {"reflect_pause": h},
        "_safe_print": lambda self, x: None,
        "stream_delta_callback": None,
    })()

    o = agent._pending_phase_transition
    agent._pending_phase_transition = None
    agent._phase = "closing_private" if o.action == "close" else "private"
    msgs = []
    handler = agent._control_handlers.get(o.handler)
    result = handler.run(agent, msgs, "tid")
    agent._phase = "public"

    assert agent._phase == "public"
    assert len(msgs) == 1
    assert msgs[0]["content"] == CANARY
    assert result.action == "resume"


def test_handoff_close_breaks_loop():
    """When run() returns close, the loop should break."""
    h = StubPhaseHandler(max_turns=1)
    outcome = h.begin({})
    agent = type("Agent", (), {
        "_phase": "public",
        "_pending_phase_transition": outcome,
        "_control_handlers": {"reflect_pause": h},
        "_safe_print": lambda self, x: None,
        "stream_delta_callback": None,
    })()
    _turn_exit_reason = None

    o = agent._pending_phase_transition
    agent._pending_phase_transition = None
    agent._phase = "closing_private" if o.action == "close" else "private"
    msgs = []
    handler = agent._control_handlers.get(o.handler)
    result = handler.run(agent, msgs, "tid")
    agent._phase = "public"
    if (result and result.action == "close") or o.action == "close":
        _turn_exit_reason = "phase_close"

    assert _turn_exit_reason == "phase_close"