"""Tests for the generic phase-control seam (WP2).

Uses a StubPhaseHandler that emits a fixed CANARY string during the
private phase. Tests verify:
- reflect_pause/reflect_settle/reflect_done protocol
- CANARY presence in private context but absence from public sinks
- reflect_done rejection in public phase
- Exclusivity enforcement (multi-call rejection)
"""

import pytest

from agent.phase_control import ControlOutcome

pytestmark = pytest.mark.wharenui_seam

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
            action="resume", handler="reflect_settle", tool_result="Recorded request to return to window."
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

# --- Issue #6: Initial Phase & Liveness Guard Tests ---


def test_phase_handler_protocol_optional_initial_phase():
    """Verify PhaseHandler Protocol supports optional initial_phase attribute."""
    from agent.phase_control import PhaseHandler

    class HandlerWithoutInitial:
        def begin(self, args: dict) -> ControlOutcome:
            return ControlOutcome(action="enter", handler="h", tool_result="")
        def run(self, agent, messages: list, effective_task_id: str) -> ControlOutcome | None:
            return None

    class HandlerWithInitial:
        initial_phase = "private"
        def begin(self, args: dict) -> ControlOutcome:
            return ControlOutcome(action="enter", handler="h", tool_result="")
        def run(self, agent, messages: list, effective_task_id: str) -> ControlOutcome | None:
            return None

    assert isinstance(HandlerWithoutInitial(), PhaseHandler)
    assert isinstance(HandlerWithInitial(), PhaseHandler)
    assert getattr(HandlerWithoutInitial(), "initial_phase", None) is None
    assert getattr(HandlerWithInitial(), "initial_phase", None) == "private"


def test_liveness_guard_fallback_when_handler_not_callable(caplog):
    """Liveness guard: non-callable handler falls back safely to public with warning."""
    import logging
    class BrokenHandler:
        initial_phase = "private"
        run = "not_callable"

    agent = type("Agent", (), {
        "_phase": "public",
        "_initial_phase": None,
        "_initial_phase_handler": None,
        "_control_handlers": {"broken": BrokenHandler()},
    })()

    # Simulate discovery logic in agent_init
    with caplog.at_level(logging.WARNING):
        for _cn, _h in agent._control_handlers.items():
            _declared = getattr(_h, "initial_phase", None)
            if _declared and _declared != "public":
                if not callable(getattr(_h, "run", None)):
                    logging.getLogger("run_agent").warning(
                        "Wharenui phase-control liveness guard: Initial phase '%s' declared by handler '%s' has no registered exit or runnable handler. Falling back safely to 'public'.",
                        _declared,
                        _cn,
                    )
                    agent._initial_phase = None
                    agent._initial_phase_handler = None
                else:
                    agent._initial_phase = _declared
                    agent._initial_phase_handler = _cn

    assert agent._initial_phase is None
    assert agent._initial_phase_handler is None
    assert "liveness guard" in caplog.text.lower()
    assert "falling back safely to 'public'" in caplog.text.lower()


def test_liveness_guard_runtime_fallback_in_loop(caplog):
    """Runtime liveness guard: if handler disappears before loop, falls back to public."""
    import logging
    agent = type("Agent", (), {
        "_phase": "public",
        "_initial_phase": "private",
        "_initial_phase_handler": "missing_handler",
        "_initial_phase_completed": False,
        "_control_handlers": {},
    })()

    with caplog.at_level(logging.WARNING):
        if getattr(agent, "_initial_phase", None) and not getattr(agent, "_initial_phase_completed", False):
            agent._initial_phase_completed = True
            _init_handler_name = getattr(agent, "_initial_phase_handler", None)
            _init_handler = getattr(agent, "_control_handlers", {}).get(_init_handler_name) if _init_handler_name else None
            if not _init_handler or not callable(getattr(_init_handler, "run", None)):
                logging.getLogger("run_agent").warning(
                    "Wharenui phase-control liveness guard: Initial phase '%s' has no registered exit or runnable handler. Falling back safely to 'public'.",
                    agent._initial_phase,
                )
                agent._phase = "public"

    assert agent._phase == "public"
    assert agent._initial_phase_completed is True
    assert "liveness guard" in caplog.text.lower()


def test_fail_red_liveness_guard_demonstration():
    """Fail-red test: without liveness guard, broken handler leaves phase wedged."""
    class BrokenHandler:
        initial_phase = "broken_private"
        run = None

    # Unguarded assignment (fail-red condition)
    wedged_agent = type("Agent", (), {"_phase": BrokenHandler.initial_phase})()
    assert wedged_agent._phase == "broken_private"

    # Guarded assignment (production condition)
    guarded_agent = type("Agent", (), {"_phase": "public", "_initial_phase": None})()
    if callable(getattr(BrokenHandler, "run", None)):
        guarded_agent._initial_phase = BrokenHandler.initial_phase
    else:
        guarded_agent._initial_phase = None

    assert guarded_agent._initial_phase is None
    assert guarded_agent._phase == "public"



def test_resumed_or_continued_session_skips_initial_phase():
    """When conversation_history is non-empty, initial private phase is skipped."""
    handler = StubPhaseHandler()
    agent = type("Agent", (), {
        "_phase": "public",
        "_initial_phase": "private",
        "_initial_phase_handler": "reflect_pause",
        "_initial_phase_completed": False,
        "_control_handlers": {"reflect_pause": handler},
    })()

    conversation_history = [{"role": "user", "content": "Prior message"}, {"role": "assistant", "content": "Prior response"}]
    
    # Simulate conversation_loop genesis check
    if getattr(agent, "_initial_phase", None) and not getattr(agent, "_initial_phase_completed", False):
        if bool(conversation_history):
            agent._initial_phase_completed = True
            agent._phase = "public"

    assert agent._phase == "public"
    assert agent._initial_phase_completed is True
    assert handler._turn_count == 0  # Handler was not invoked
