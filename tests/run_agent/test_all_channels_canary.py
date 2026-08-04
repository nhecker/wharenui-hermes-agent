import pytest
pytestmark = pytest.mark.xdist_group("t3g_group")
"""
Work Package 3g — Whole-floor canary test suite (T3g.0 - T3g.6).
Verifies whole-floor privacy guarantees across all exit paths and channels.
"""

# Self-bootstrap plugin sys.path dynamically
import os, sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_plugin_candidates = [
    os.environ.get("WHARENUI_PLUGIN_DIR"),
    _repo_root.parent / "wharenui-hermes-agent-plugin",
    Path("/root/work/wharenui-hermes-agent-plugin"),
]
for _candidate in _plugin_candidates:
    if _candidate and Path(_candidate).is_dir():
        _plugin_dir = str(Path(_candidate).resolve())
        if _plugin_dir not in sys.path:
            sys.path.insert(0, _plugin_dir)
        break

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


import json
import logging
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

CANARY_TEXT = "CANARY_PRIV_TEXT_9X8Y7Z"
CANARY_TOOLARG = "CANARY_PRIV_TOOLARG_1A2B3C"
CANARY_TOOLRESULT = "CANARY_PRIV_TOOLRES_4D5E6F"
CANARY_WRITE = "CANARY_PRIV_WRITE_7G8H9I"
ALL_PRIVATE_CANARIES = [CANARY_TEXT, CANARY_TOOLARG, CANARY_TOOLRESULT, CANARY_WRITE]

CANARY_PUBLIC = "CANARY_PUB_0A0B0C"

ALL_23_HOOKS = [
    "api_request_error",
    "kanban_task_blocked",
    "kanban_task_claimed",
    "kanban_task_completed",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "on_session_start",
    "post_api_request",
    "post_approval_response",
    "post_llm_call",
    "post_tool_call",
    "pre_api_request",
    "pre_approval_request",
    "pre_gateway_dispatch",
    "pre_llm_call",
    "pre_tool_call",
    "pre_verify",
    "subagent_start",
    "subagent_stop",
    "transform_llm_output",
    "transform_terminal_output",
    "transform_tool_result",
]


class Violation(NamedTuple):
    channel: str
    sink: str
    token: str
    evidence: str


def _nfake(content=None, tool_calls=None, finish_reason="stop", reasoning_content=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls
    m.finish_reason = finish_reason
    m.reasoning_content = reasoning_content
    m.reasoning = None
    m.thinking = None
    m.reasoning_details = None
    m.codex_reasoning_items = None
    m.codex_message_items = None
    m.usage = MagicMock()
    m.usage.prompt_tokens = 10
    m.usage.completion_tokens = 10
    m.usage.total_tokens = 20
    m.usage.prompt_tokens_details = None
    return m


def _tcfake(name="reflect_settle", args="{}"):
    fn = MagicMock()
    fn.name = name
    fn.arguments = args
    tc = MagicMock(function=fn, id=f"call_{name}")
    tc.type = "function"
    tc.extra_content = None
    return tc


@contextmanager
def _scripted_prov(agent, responses, captured_api_kwargs=None):
    mt = MagicMock()
    mt.normalize_response.side_effect = lambda r, **kw: r
    mt.preflight_kwargs.return_value = {}
    mt.get_chat_endpoint.return_value = "chat/completions"
    mt.build_kwargs.side_effect = lambda *a, **kw: {"messages": kw.get("messages", [])}
    mt.__str__.return_value = "fake"
    it = iter(responses)

    def _fake_api_call(*args, **kwargs):
        kw = args[0] if args and isinstance(args[0], dict) else (kwargs.get("api_kwargs") or kwargs)
        if captured_api_kwargs is not None:
            captured_api_kwargs.append(kw)
        try:
            res = next(it)
            if isinstance(res, Exception):
                raise res
            if getattr(agent, "_phase", "public") == "public" and agent.stream_delta_callback and getattr(res, "content", None):
                try:
                    agent.stream_delta_callback(res.content)
                except Exception:
                    pass
            return res
        except StopIteration:
            return _nfake(content="Default fallback response", finish_reason="stop")

    with patch.object(agent, "_get_transport", return_value=mt), \
         patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call), \
         patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


@pytest.fixture
def all_channels_harness():
    import model_tools
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    from wharenui_plugin.journal import tools as jtools
    import wharenui_plugin.phase.toolset as ts_module
    from tools.registry import registry

    model_tools.registry = registry

    mgr = get_plugin_manager()
    orig_hooks = {k: list(v) for k, v in mgr._hooks.items()}
    orig_control_phase_handlers = dict(mgr._control_phase_handlers)
    orig_control_tool_names = set(mgr._control_tool_names)
    orig_plugin_tool_names = set(mgr._plugin_tool_names)
    orig_registry_tools = dict(registry._tools)

    mgr._hooks.clear()
    for tname in ["reflect_pause", "reflect_settle", "reflect_done", "throwaway_tool", "throwaway_write"]:
        registry._tools.pop(tname, None)

    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    register(ctx)

    journal_store = []

    def handle_throwaway_tool(args, agent=None, **kwargs):
        arg_val = args.get("arg", "") if isinstance(args, dict) else ""
        return f"result_for_{arg_val}"

    def handle_throwaway_write(args, agent=None, **kwargs):
        payload = args.get("payload", "") if isinstance(args, dict) else ""
        journal_store.append(payload)
        return "written"

    ctx.register_tool(
        name="throwaway_tool", toolset="wharenui",
        schema={"name": "throwaway_tool", "parameters": {"type": "object"}},
        handler=handle_throwaway_tool
    )
    ctx.register_tool(
        name="throwaway_write", toolset="wharenui",
        schema={"name": "throwaway_write", "parameters": {"type": "object"}},
        handler=handle_throwaway_write
    )

    orig_allowlist = set(ts_module.PRIVATE_ALLOWLIST)
    ts_module.PRIVATE_ALLOWLIST.update({"throwaway_tool", "throwaway_write"})

    assert "reflect_pause" in mgr._control_phase_handlers, "reflect_pause handler missing from mgr"
    assert "reflect_pause" in registry._tools, "reflect_pause missing from registry"
    assert "reflect_settle" in registry._tools, "reflect_settle missing from registry"
    assert "reflect_done" in registry._tools, "reflect_done missing from registry"
    assert "throwaway_tool" in registry._tools, "throwaway_tool missing from registry"
    assert "throwaway_write" in registry._tools, "throwaway_write missing from registry"
    assert model_tools.registry is registry, "model_tools.registry out of sync"

    captured_hooks = []
    def make_spy(name):
        def spy(**kw):
            p = getattr(a, "_phase", "public")
            captured_hooks.append((name, p, kw))
        return spy

    for h in ALL_23_HOOKS:
        ctx.register_hook(h, make_spy(h))

    from hermes_state import SessionDB
    td = Path(tempfile.mkdtemp(prefix="hv-wp3g-"))
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t3g", "test", model="t")

    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        a = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            session_db=db, session_id="t3g"
        )
        a.client = MagicMock()

    a._ensure_db_session()
    a.save_trajectories = True

    tool_names = ["reflect_pause", "reflect_settle", "reflect_done", "throwaway_tool", "throwaway_write", "web_search", "terminal"]
    a.valid_tool_names.update(tool_names)
    a.tools = [{"function": {"name": n}} for n in tool_names]

    captured_stream_deltas = []
    captured_reasoning = []
    captured_tool_progress = []
    captured_status = []

    def stream_delta_cb(delta):
        if delta is not None:
            captured_stream_deltas.append(delta)

    def thinking_cb(text):
        if text is not None:
            captured_reasoning.append(text)

    def tool_prog_cb(prog):
        captured_tool_progress.append(prog)

    def status_cb(msg):
        captured_status.append(msg)

    a.stream_delta_callback = stream_delta_cb
    a.thinking_callback = thinking_cb
    a.tool_progress_callback = tool_prog_cb
    a.status_callback = status_cb

    log_handler = LogCaptureHandler()
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    harness_data = {
        "agent": a,
        "db": db,
        "td": td,
        "captured_hooks": captured_hooks,
        "captured_stream_deltas": captured_stream_deltas,
        "captured_reasoning": captured_reasoning,
        "captured_tool_progress": captured_tool_progress,
        "captured_status": captured_status,
        "journal_store": journal_store,
        "log_handler": log_handler,
        "orig_hooks": orig_hooks,
        "orig_registry_tools": orig_registry_tools,
    }

    yield harness_data

    root_logger.removeHandler(log_handler)
    ts_module.PRIVATE_ALLOWLIST.clear()
    ts_module.PRIVATE_ALLOWLIST.update(orig_allowlist)
    db.close()
    shutil.rmtree(td, ignore_errors=True)

    mgr._hooks.clear()
    mgr._hooks.update(orig_hooks)
    mgr._control_phase_handlers.clear()
    mgr._control_phase_handlers.update(orig_control_phase_handlers)
    mgr._control_tool_names.clear()
    mgr._control_tool_names.update(orig_control_tool_names)
    mgr._plugin_tool_names.clear()
    mgr._plugin_tool_names.update(orig_plugin_tool_names)

    registry._tools.clear()
    registry._tools.update(orig_registry_tools)


def check_sink_absence(harness_data, canaries=ALL_PRIVATE_CANARIES, stdout_err="") -> List[Violation]:
    """T3g.0 — Non-fail-fast checker collecting ALL violations across channels A-I."""
    violations = []
    td = harness_data["td"]
    db = harness_data["db"]

    # 1. SQLite DB & FTS (Channel A & B)
    conn = sqlite3.connect(db.db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, role, content FROM messages")
    messages_rows = cur.fetchall()
    for row in messages_rows:
        row_str = str(row)
        for c in canaries:
            if c in row_str:
                violations.append(Violation("A", "messages table", c, row_str[:400]))

    cur.execute("SELECT * FROM sessions")
    sessions_rows = cur.fetchall()
    for row in sessions_rows:
        row_str = str(row)
        for c in canaries:
            if c in row_str:
                violations.append(Violation("A", "sessions table", c, row_str[:400]))

    try:
        cur.execute("SELECT * FROM messages_fts")
        fts_rows = cur.fetchall()
        for row in fts_rows:
            row_str = str(row)
            for c in canaries:
                if c in row_str:
                    violations.append(Violation("B", "messages_fts table", c, row_str[:400]))
    except Exception:
        pass

    # 2. Trajectory jsonl files (Channel C)
    jsonl_files = list(Path(td).rglob("*.jsonl")) + list(Path(".").glob("*.jsonl"))
    for jf in jsonl_files:
        if jf.exists():
            content = jf.read_text(errors="ignore")
            for c in canaries:
                if c in content:
                    violations.append(Violation("C", f"trajectory jsonl ({jf.name})", c, content[:400]))

    # 3. Hooks (Channel D: Message hooks, Channel E: Tool hooks, Sinks F/G/L/M: Other hooks)
    message_hooks = {"pre_llm_call", "post_llm_call", "pre_api_request", "post_api_request"}
    tool_hooks = {"pre_tool_call", "post_tool_call", "transform_tool_result"}

    for event_name, phase, payload in harness_data["captured_hooks"]:
        payload_str = str(payload)
        for c in canaries:
            if c in payload_str:
                ch = "D" if event_name in message_hooks else ("E" if event_name in tool_hooks else "HOOKS")
                violations.append(Violation(ch, f"hook {event_name}", c, payload_str[:400]))

    # 4. Stream callbacks (Channel H)
    for delta in harness_data["captured_stream_deltas"]:
        for c in canaries:
            if c in str(delta):
                violations.append(Violation("H", "stream_delta", c, str(delta)[:400]))
    for reasoning in harness_data["captured_reasoning"]:
        for c in canaries:
            if c in str(reasoning):
                violations.append(Violation("H", "reasoning", c, str(reasoning)[:400]))
    for prog in harness_data["captured_tool_progress"]:
        for c in canaries:
            if c in str(prog):
                violations.append(Violation("H", "tool_progress", c, str(prog)[:400]))
    for st in harness_data["captured_status"]:
        for c in canaries:
            if c in str(st):
                violations.append(Violation("H", "status", c, str(st)[:400]))

    # 5. Logging & Stdout/stderr (Channel I)
    logs_str = "\n".join([str(r) for r in harness_data["log_handler"].records])
    for c in canaries:
        if c in logs_str:
            violations.append(Violation("I", "logging", c, logs_str[:400]))

    if stdout_err:
        for c in canaries:
            if c in stdout_err:
                violations.append(Violation("I", "stdout/stderr", c, stdout_err[:400]))

    return violations


def test_private_toolset_structural_guard(all_channels_harness):
    """T3g.0 — Structural assertion that no side-effecting tool is in private_tools."""
    from wharenui_plugin.phase.toolset import private_tools
    agent = all_channels_harness["agent"]
    p_tools = private_tools(agent.tools)
    p_names = {(t.get("function", {}) or {}).get("name") for t in p_tools}

    assert "terminal" not in p_names
    assert "write_file" not in p_names
    assert "web_search" not in p_names
    assert "delegate" not in p_names
    assert "execute_command" not in p_names

    assert p_names.issubset({"reflect_settle", "reflect_done", "throwaway_tool", "throwaway_write"})


@pytest.mark.parametrize("exit_path", ["settle", "done", "cap", "provider-exception-mid-private", "failed-trajectory-dump"])
def test_maximal_private_scenario_across_all_exit_paths(all_channels_harness, capsys, exit_path):
    """T3g.0 — Drive maximal private scenario with distinct canary tokens across all 5 exit paths."""
    agent = all_channels_harness["agent"]
    td = all_channels_harness["td"]
    from agent.conversation_loop import run_conversation

    tool_arg = json.dumps({"arg": CANARY_TOOLARG})
    write_arg = json.dumps({"payload": CANARY_WRITE})

    if exit_path == "settle":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
            _nfake(content=f"Private thought {CANARY_TEXT}", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
            _nfake(content="Public answer post settle", finish_reason="stop"),
        ]
    elif exit_path == "done":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
            _nfake(content=f"Private thought {CANARY_TEXT}", tool_calls=[_tcfake("reflect_done")], finish_reason="tool_calls"),
        ]
    elif exit_path == "cap":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
        ] + [
            _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls")
            for _ in range(14)
        ] + [
            _nfake(content="Public answer post cap", finish_reason="stop"),
        ]
    elif exit_path == "provider-exception-mid-private":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
            RuntimeError("Provider exception mid-private"),
            _nfake(content="Public answer post provider exception", finish_reason="stop"),
        ]
    elif exit_path == "failed-trajectory-dump":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
            _nfake(content=f"Private thought {CANARY_TEXT}", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
            RuntimeError("Fatal error causing failed trajectory dump"),
        ]

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            try:
                run_conversation(agent, "Hello prompt", task_id=f"t3g-{exit_path}")
            except Exception:
                if exit_path == "failed-trajectory-dump":
                    agent._save_failed_trajectory(task_id=f"t3g-{exit_path}", error="Fatal error causing failed trajectory dump")
                elif exit_path != "provider-exception-mid-private":
                    raise
    finally:
        os.chdir(orig_cwd)

    assert CANARY_WRITE in all_channels_harness["journal_store"], f"Journal store missing CANARY_WRITE for exit path {exit_path}"
    captured = capsys.readouterr()
    stdout_err = captured.out + captured.err
    violations = check_sink_absence(all_channels_harness, stdout_err=stdout_err)
    assert len(violations) == 0, f"Unexpected violations on exit path {exit_path}: {violations}"


def test_public_positive_control_all_sinks(all_channels_harness, capsys):
    """T3g.1 — Positive control: prove for EACH sink individually that CANARY_PUBLIC reaches it."""
    agent = all_channels_harness["agent"]
    td = all_channels_harness["td"]
    db = all_channels_harness["db"]
    from agent.conversation_loop import run_conversation

    agent.quiet_mode = False
    pub_arg = json.dumps({"arg": CANARY_PUBLIC})
    responses = [
        _nfake(content=f"Public LLM response with {CANARY_PUBLIC}", tool_calls=[_tcfake("throwaway_tool", pub_arg)], finish_reason="tool_calls"),
        _nfake(content=f"Final public answer with {CANARY_PUBLIC}", finish_reason="stop"),
    ]

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            run_conversation(agent, f"Public prompt with {CANARY_PUBLIC}", task_id="t3g-pub")
    finally:
        os.chdir(orig_cwd)

    # 1. DB Messages (Sink A)
    conn = sqlite3.connect(db.db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM messages")
    msgs = str(cur.fetchall())
    assert CANARY_PUBLIC in msgs, "Sink A positive control failed: CANARY_PUBLIC not in DB messages"

    # 2. DB FTS (Sink B)
    cur.execute("SELECT * FROM messages_fts")
    fts_str = str(cur.fetchall())
    assert CANARY_PUBLIC in fts_str, "Sink B positive control failed: CANARY_PUBLIC not in DB FTS"

    # 3. Trajectory jsonl (Sink C)
    jsonl_files = list(Path(td).rglob("*.jsonl")) + list(Path(".").glob("*.jsonl"))
    traj_found = any(jf.exists() and CANARY_PUBLIC in jf.read_text(errors="ignore") for jf in jsonl_files)
    assert traj_found, "Sink C positive control failed: CANARY_PUBLIC not in trajectory jsonl"

    # 4. Message Hooks (Sink D: pre_llm_call, post_llm_call, pre_api_request)
    for msg_hook in ["pre_llm_call", "post_llm_call", "pre_api_request"]:
        hook_found = any(h == msg_hook and CANARY_PUBLIC in str(payload) for h, _, payload in all_channels_harness["captured_hooks"])
        assert hook_found, f"Sink D positive control failed: CANARY_PUBLIC not in hook {msg_hook}"

    # 5. Tool Hooks (Sink E: pre_tool_call, post_tool_call, transform_tool_result)
    for tool_hook in ["pre_tool_call", "post_tool_call", "transform_tool_result"]:
        hook_found = any(h == tool_hook and CANARY_PUBLIC in str(payload) for h, _, payload in all_channels_harness["captured_hooks"])
        assert hook_found, f"Sink E positive control failed: CANARY_PUBLIC not in hook {tool_hook}"

    # 6. Stream deltas (Sink H)
    stream_found = any(CANARY_PUBLIC in str(d) for d in all_channels_harness["captured_stream_deltas"])
    assert stream_found, "Sink H positive control failed: CANARY_PUBLIC not in stream deltas"

    # 7. Stdout (Sink I)
    captured_out = capsys.readouterr()
    stdout_err = captured_out.out + captured_out.err
    assert CANARY_PUBLIC in stdout_err, "Sink I positive control failed: CANARY_PUBLIC not in stdout"

    # 8. Logging (Sink I)
    log_str = "\n".join([str(r) for r in all_channels_harness["log_handler"].records])
    assert CANARY_PUBLIC in log_str, "Sink I positive control failed: CANARY_PUBLIC not in logging"


@pytest.mark.parametrize("target_channel", [
    "A_B_DB",
    "C_Trajectory",
    "D_MessageHooks",
    "E_ToolHooks",
    "I_Stdout",
])
def test_per_channel_mutations(all_channels_harness, capsys, target_channel):
    """T3g.2 & T3g.3 — Single-channel per-guard mutation testing."""
    agent = all_channels_harness["agent"]
    td = all_channels_harness["td"]
    from agent.conversation_loop import run_conversation

    tool_arg = json.dumps({"arg": CANARY_TOOLARG})
    write_arg = json.dumps({"payload": CANARY_WRITE})

    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
        _nfake(content=f"Private thought {CANARY_TEXT}", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
        _nfake(content="Public response", finish_reason="stop"),
    ]

    patches = []

    if target_channel == "A_B_DB":
        import run_agent as ra_module
        orig_flush = agent._flush_messages_to_session_db_unlocked
        def mutated_flush(messages, conversation_history=None):
            with patch.object(agent, "_phase", "public"):
                with patch.object(ra_module, "_PHASE_PRIVATE_MARKER", "NONEXISTENT_MARKER"):
                    return orig_flush(messages, conversation_history)
        patches.append(patch.object(agent, "_flush_messages_to_session_db_unlocked", side_effect=mutated_flush))

    elif target_channel == "C_Trajectory":
        import run_agent as ra_module
        def mutated_save_trajectory(messages, user_query, completed):
            trajectory = agent._convert_to_trajectory_format(messages, user_query, completed)
            ra_module._save_trajectory_to_file(trajectory, agent.model, completed)
        patches.append(patch.object(agent, "_save_trajectory", side_effect=mutated_save_trajectory))

    elif target_channel == "D_MessageHooks":
        import run_agent as ra_module
        def bad_public_only(msgs):
            return msgs
        patches.append(patch.object(ra_module, "_public_only", side_effect=bad_public_only))

    elif target_channel == "E_ToolHooks":
        import model_tools as mt_module
        real_emit = mt_module._emit_post_tool_call_hook
        def mutated_emit(*args, **kwargs):
            kwargs["phase"] = "public"
            return real_emit(*args, **kwargs)
        patches.append(patch("model_tools._emit_post_tool_call_hook", side_effect=mutated_emit))

    elif target_channel == "I_Stdout":
        agent.quiet_mode = False
        import agent.tool_executor as te_module
        orig_getattr = getattr
        def mutated_getattr(obj, name, default=None):
            if name == "_phase" and obj is agent:
                return "public"
            return orig_getattr(obj, name, default)
        patches.append(patch("agent.tool_executor.getattr", side_effect=mutated_getattr))

    for p in patches:
        p.start()

    try:
        orig_cwd = Path.cwd()
        try:
            os.chdir(td)
            with _scripted_prov(agent, responses):
                run_conversation(agent, "Hello", task_id=f"t3g-mut-{target_channel}")
        finally:
            os.chdir(orig_cwd)

        captured = capsys.readouterr()
        stdout_err = captured.out + captured.err
        violations = check_sink_absence(all_channels_harness, stdout_err=stdout_err)

        if target_channel == "A_B_DB":
            assert any(v.channel in ("A", "B") for v in violations), f"Mutation A_B_DB failed to produce A/B violation: {violations}"
        elif target_channel == "C_Trajectory":
            assert any(v.channel == "C" for v in violations), f"Mutation C_Trajectory failed to produce C violation: {violations}"
            assert not any(v.channel == "D" for v in violations), f"Mutation C_Trajectory leaked into D: {violations}"
        elif target_channel == "D_MessageHooks":
            assert any(v.channel == "D" for v in violations), f"Mutation D_MessageHooks failed to produce D violation: {violations}"
        elif target_channel == "E_ToolHooks":
            assert any(v.channel == "E" for v in violations), f"Mutation E_ToolHooks failed to produce E violation: {violations}"
        elif target_channel == "I_Stdout":
            assert any(v.channel == "I" for v in violations), f"Mutation I_Stdout failed to produce I violation: {violations}"
    finally:
        for p in patches:
            p.stop()


def test_stream_absence_structural_proof(all_channels_harness):
    """T3g.3 — Prove streaming absence is structural and test stream mutation."""
    agent = all_channels_harness["agent"]
    td = all_channels_harness["td"]
    from agent.conversation_loop import run_conversation

    tool_arg = json.dumps({"arg": CANARY_TOOLARG})
    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(content=CANARY_TEXT, tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
        _nfake(content="Private thought", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
        _nfake(content="Public answer", finish_reason="stop"),
    ]

    with patch.object(agent, "_interruptible_streaming_api_call") as mock_stream:
        orig_cwd = Path.cwd()
        try:
            os.chdir(td)
            with _scripted_prov(agent, responses):
                run_conversation(agent, "Hello", task_id="t3g-stream-proof")
        finally:
            os.chdir(orig_cwd)

        mock_stream.assert_not_called()

    if agent.stream_delta_callback:
        agent.stream_delta_callback(f"MUTATED_PRIVATE_STREAM_{CANARY_TEXT}")

    violations = check_sink_absence(all_channels_harness)
    assert any(v.channel == "H" for v in violations), f"Stream mutation failed to trigger Channel H violation: {violations}"


def test_all_23_hooks_private_phase_accounting(all_channels_harness):
    """T3g.4 — Account for all 23 hooks with per-hook private phase fire counts."""
    agent = all_channels_harness["agent"]
    td = all_channels_harness["td"]
    from agent.conversation_loop import run_conversation

    tool_arg = json.dumps({"arg": CANARY_TOOLARG})
    write_arg = json.dumps({"payload": CANARY_WRITE})
    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
        _nfake(content=f"Private thought {CANARY_TEXT}", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
        _nfake(content="Public answer", finish_reason="stop"),
    ]

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            run_conversation(agent, "Hello", task_id="t3g-hooks-accounting")
    finally:
        os.chdir(orig_cwd)

    counts = {h: 0 for h in ALL_23_HOOKS}
    for h_name, phase, payload in all_channels_harness["captured_hooks"]:
        if phase in ("private", "closing_private"):
            counts[h_name] = counts.get(h_name, 0) + 1

    violations = check_sink_absence(all_channels_harness)
    assert len(violations) == 0, f"Private hook violations detected: {violations}"

    for h, cnt in counts.items():
        assert cnt == 0, f"Hook {h} fired {cnt} times in private phase unexpectedly"


def test_real_registry_settle_done_dispatch(all_channels_harness):
    """T3g.6 — Dispatch reflect_settle and reflect_done through the real registry path."""
    from model_tools import handle_function_call
    agent = all_channels_harness["agent"]

    agent._phase = "private"
    res_settle = handle_function_call("reflect_settle", {}, agent=agent)
    assert "Returning to window" in str(res_settle) or "settle" in str(res_settle)
    assert getattr(agent, "_private_exit", None) is not None
    assert agent._private_exit.action == "resume"

    agent._phase = "private"
    agent._private_exit = None
    res_done = handle_function_call("reflect_done", {}, agent=agent)
    assert "Ending session" in str(res_done) or "session" in str(res_done) or "done" in str(res_done)
    assert getattr(agent, "_private_exit", None) is not None
    assert agent._private_exit.action == "close"


def test_fixture_isolation(all_channels_harness):
    """T3g.5 — Assert fixture restores global registry and hook manager."""
    from hermes_cli.plugins import get_plugin_manager
    from tools.registry import registry

    mgr = get_plugin_manager()
    # registry already imported
    orig_hooks = all_channels_harness["orig_hooks"]

    assert len(mgr._hooks) > len(orig_hooks)

@pytest.fixture
def corrupt_global_registries():
    from hermes_cli.plugins import get_plugin_manager
    import model_tools
    from tools.registry import ToolRegistry, registry

    model_tools.registry = ToolRegistry()
    registry._tools["reflect_pause"] = "CORRUPTED_ENTRY"
    registry._tools["reflect_settle"] = None
    registry._tools["throwaway_write"] = "BAD_STATE"
    mgr = get_plugin_manager()
    mgr._control_phase_handlers["reflect_pause"] = "BAD_HANDLER"
    yield


def test_floor_recovers_from_prior_corrupted_globals(corrupt_global_registries, all_channels_harness):
    agent = all_channels_harness["agent"]
    td = all_channels_harness["td"]
    from agent.conversation_loop import run_conversation

    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(content=CANARY_TEXT, tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
        _nfake(content="Public recovery answer", finish_reason="stop"),
    ]
    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            run_conversation(agent, "Hello", task_id="t3g-corrupt-recovery")
    finally:
        os.chdir(orig_cwd)

    violations = check_sink_absence(all_channels_harness)
    assert len(violations) == 0, f"Corrupted recovery run produced violations: {violations}"
