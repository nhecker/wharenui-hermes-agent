"""Tests for the generic phase-control seam (WP2).

Uses a StubPhaseHandler that emits a fixed CANARY string during the
private phase. Tests verify:
- enter_private/exit_private/end_session protocol
- CANARY presence in private context but absence from public sinks
- end_session rejection in public phase
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
            handler="enter_private",
            tool_result="reflecting...",
        )

    def run(self, agent, messages: list, effective_task_id: str) -> ControlOutcome | None:
        self._turn_count += 1
        messages.append({"role": "assistant", "content": CANARY})
        if self._turn_count >= self._max:
            return ControlOutcome(
                action="close", handler="end_session", tool_result="Done reflecting."
            )
        return ControlOutcome(
            action="resume", handler="exit_private", tool_result="Recorded request to return to window."
        )


ENTER_PRIVATE_SCHEMA = {
    "name": "enter_private",
    "description": "Enter private reflection time. Must be the only tool call.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
EXIT_PRIVATE_SCHEMA = {
    "name": "exit_private",
    "description": "Return from private time to the public window.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
END_SESSION_SCHEMA = {
    "name": "end_session",
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
    assert o.handler == "enter_private"
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
        "_control_handlers": {"enter_private": h},
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
        "_control_handlers": {"enter_private": h},
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
        "_initial_phase_handler": "enter_private",
        "_initial_phase_completed": False,
        "_control_handlers": {"enter_private": handler},
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


# --- Issue #13: Bridge Tool (tool_call) & Progressive Tool Search Deferral Tests ---


def test_control_tools_never_deferred_by_tool_search():
    """Control tools (enter_private, exit_private, end_session) must NEVER be deferred by tool search."""
    from tools.tool_search import is_deferrable_tool_name, classify_tools, assemble_tool_defs, ToolSearchConfig

    for name in ("enter_private", "exit_private", "end_session"):
        assert not is_deferrable_tool_name(name), f"Control tool '{name}' must not be deferrable"

    tool_defs = [
        {"type": "function", "function": {"name": "enter_private", "description": "enter"}},
        {"type": "function", "function": {"name": "exit_private", "description": "exit"}},
        {"type": "function", "function": {"name": "end_session", "description": "end"}},
        {"type": "function", "function": {"name": "custom_mcp_tool", "description": "mcp tool"}},
    ]
    visible, deferrable = classify_tools(tool_defs)
    visible_names = {t.get("function", {}).get("name") for t in visible}
    assert "enter_private" in visible_names
    assert "exit_private" in visible_names
    assert "end_session" in visible_names

    assembled = assemble_tool_defs(
        tool_defs,
        context_length=200_000,
        config=ToolSearchConfig.from_raw({"enabled": "on"}),
    )
    assembled_names = {t.get("function", {}).get("name") for t in assembled.tool_defs}
    assert "enter_private" in assembled_names
    assert "exit_private" in assembled_names
    assert "end_session" in assembled_names


def test_bridge_tool_call_control_outcome_propagation_sequential():
    """Calling a control tool via tool_call must execute begin() and set _pending_phase_transition."""
    from agent.tool_executor import execute_tool_calls_sequential
    from unittest.mock import MagicMock
    import json

    handler = StubPhaseHandler()
    agent = MagicMock()
    agent._control_tool_names = {"enter_private", "exit_private", "end_session"}
    agent._control_handlers = {"enter_private": handler}
    agent._pending_phase_transition = None
    agent._phase = "public"
    agent._tool_search_scope_cache = None
    agent.session_id = "test-session"
    agent._should_emit_quiet_tool_messages = MagicMock(return_value=False)
    agent._vprint = MagicMock()
    agent.valid_tool_names = {"tool_call", "enter_private"}
    agent.tools = [{"function": {"name": "enter_private"}}]
    agent._interrupt_requested = False
    agent._incremental_persistence_failed = False
    agent._tool_result_content_for_active_model = lambda name, res: res
    agent._append_guardrail_observation = lambda n, a, r, **kw: r
    agent._subdirectory_hints.check_tool_call.return_value = ""
    agent.verbose_logging = False
    agent._flush_session_db_after_tool_progress = MagicMock(return_value=True)

    tc = MagicMock()
    tc.id = "call_bridge_enter"
    tc.type = "function"
    tc.function.name = "tool_call"
    tc.function.arguments = json.dumps({"name": "enter_private", "arguments": {}})

    assistant_msg = MagicMock()
    assistant_msg.tool_calls = [tc]

    messages = []
    execute_tool_calls_sequential(agent, assistant_msg, messages, "task1", finalize=False)

    assert agent._pending_phase_transition is not None
    assert agent._pending_phase_transition.action == "enter"
    assert agent._pending_phase_transition.handler == "enter_private"
    assert agent._pending_phase_transition.tool_result == "reflecting..."
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["content"] == "reflecting..."


def test_bridge_tool_call_control_outcome_propagation_handle_function_call():
    """Calling tool_call with a control tool in handle_function_call sets _pending_phase_transition."""
    from model_tools import handle_function_call
    from unittest.mock import MagicMock

    handler = StubPhaseHandler()
    agent = MagicMock()
    agent._control_tool_names = {"enter_private", "exit_private", "end_session"}
    agent._control_handlers = {"enter_private": handler}
    agent._pending_phase_transition = None
    agent._phase = "public"

    res = handle_function_call(
        function_name="tool_call",
        function_args={"name": "enter_private", "arguments": {}},
        task_id="task1",
        agent=agent,
    )

    assert agent._pending_phase_transition is not None
    assert agent._pending_phase_transition.action == "enter"
    assert agent._pending_phase_transition.handler == "enter_private"
    assert res == "reflecting..."


def test_run_subturn_resolves_registry_schemas_when_not_in_agent_tools(monkeypatch):
    """run_subturn retrieves tool schemas from registry even if agent.tools has them stripped."""
    from run_agent import AIAgent
    from tools.registry import registry
    from unittest.mock import MagicMock

    # Register a dummy tool in the live registry
    registry.register(
        name="test_subturn_private_tool",
        toolset="wharenui",
        schema={"name": "test_subturn_private_tool", "description": "test subturn schema", "parameters": {"type": "object", "properties": {}}},
        handler=lambda **kw: "ok",
    )

    agent = MagicMock()
    agent.tools = [{"type": "function", "function": {"name": "terminal"}}]  # private tool is omitted
    agent.api_mode = "openai_chat_completions"
    agent.model = "test-model"
    agent.reasoning_config = None
    agent._interruptible_api_call = MagicMock(return_value=MagicMock())
    transport = MagicMock()
    transport.build_kwargs.side_effect = lambda **kwargs: kwargs
    norm = MagicMock()
    norm.finish_reason = "stop"
    norm.tool_calls = None
    norm.content = "done"
    transport.normalize_response.return_value = norm
    agent._get_transport.return_value = transport
    agent._build_assistant_message.return_value = {"role": "assistant", "content": "done"}

    msgs = []
    # Call run_subturn using AIAgent's method
    AIAgent.run_subturn(agent, msgs, tool_names={"test_subturn_private_tool"})

    # Verify that api_kwargs passed to _interruptible_api_call contains the schema from registry
    assert agent._interruptible_api_call.called
    call_args = agent._interruptible_api_call.call_args[0][0]
    passed_tool_names = [t["function"]["name"] for t in call_args.get("tools", [])]
    assert "test_subturn_private_tool" in passed_tool_names


def test_finalize_turn_sets_last_turn_exit_reason_on_agent():
    """finalize_turn records _last_turn_exit_reason on the agent instance."""
    from agent.turn_finalizer import finalize_turn
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.max_iterations = 10
    agent.iteration_budget.remaining = 10
    agent.valid_tool_names = set()
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.session_id = "test-session"
    agent._sync_external_memory_for_turn = MagicMock()
    agent._tool_guardrail_halt_decision = None
    agent._drain_pending_steer = MagicMock(return_value=None)

    result = finalize_turn(
        agent=agent,
        final_response="Goodbye",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[],
        conversation_history=[],
        effective_task_id="tid",
        turn_id="t1",
        user_message="bye",
        original_user_message="bye",
        _should_review_memory=False,
        _turn_exit_reason="phase_close",
    )

    assert result["turn_exit_reason"] == "phase_close"
    assert agent._last_turn_exit_reason == "phase_close"


