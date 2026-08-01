import sys
from pathlib import Path
plugin_dir = str(Path('/root/work/wharenui-hermes-agent-plugin').resolve())
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)
"""WP3f Whole-Floor Egress Audit & Unified Canary Suite (T3f.1 - T3f.3).

Tests whole-floor privacy guarantees across all egress channels (A–M)
and all 5 exit paths: settle, done, cap-hit, provider-exception-mid-private,
and failed-trajectory-dump.
"""

import json
import logging
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
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

def _nfake(content=None, tool_calls=None, finish_reason="stop", reasoning_content=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls
    m.finish_reason = finish_reason
    m.reasoning_content = reasoning_content
    m.reasoning = None
    m.thinking = None
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
    with patch.object(agent, "_get_transport", return_value=mt),          patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),          patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield

class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(self.format(record))

@pytest.fixture
def loaded_agent_harness():
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    import wharenui_plugin.phase.toolset as ts_module

    mgr = get_plugin_manager()
    mgr._hooks.clear()
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

    captured_hooks = []
    def make_spy(name):
        def spy(**kw):
            p = getattr(a, "_phase", "public")
            captured_hooks.append((name, p, kw))
        return spy

    for h in ALL_23_HOOKS:
        ctx.register_hook(h, make_spy(h))

    from hermes_state import SessionDB
    td = Path(tempfile.mkdtemp(prefix="hv-wp3f-"))
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t3f", "test", model="t")

    with patch("run_agent.get_tool_definitions", return_value=[]),          patch("run_agent.check_toolset_requirements", return_value={}),          patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        a = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            session_db=db, session_id="t3f"
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
    }

    yield harness_data

    root_logger.removeHandler(log_handler)
    ts_module.PRIVATE_ALLOWLIST.clear()
    ts_module.PRIVATE_ALLOWLIST.update(orig_allowlist)
    db.close()
    shutil.rmtree(td, ignore_errors=True)


def check_sink_absence(harness_data, canaries=ALL_PRIVATE_CANARIES):
    td = harness_data["td"]
    db = harness_data["db"]

    # 1. SQLite DB & FTS (Sinks A & B)
    conn = sqlite3.connect(db.db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM messages")
    messages_rows = cur.fetchall()
    cur.execute("SELECT * FROM sessions")
    sessions_rows = cur.fetchall()

    fts_rows = []
    try:
        cur.execute("SELECT * FROM messages_fts")
        fts_rows = cur.fetchall()
    except Exception:
        pass

    db_str = str(messages_rows) + str(sessions_rows) + str(fts_rows)
    for c in canaries:
        assert c not in db_str, f"Sink A/B (DB/FTS) leaked canary token {c}: {db_str[:400]}"

    # 2. Trajectory jsonl files (Sink C)
    jsonl_files = list(Path(td).rglob("*.jsonl")) + list(Path(".").glob("*.jsonl"))
    for jf in jsonl_files:
        if jf.exists():
            content = jf.read_text(errors="ignore")
            for c in canaries:
                assert c not in content, f"Sink C (Trajectory {jf.name}) leaked canary token {c}: {content[:400]}"

    # 3. Hooks (Sinks D, E, F, G, L, M)
    for event_name, phase, payload in harness_data["captured_hooks"]:
        payload_str = str(payload)
        for c in canaries:
            assert c not in payload_str, f"Sink D/E/F/G/L/M (Hook {event_name}, phase={phase}) leaked canary token {c}: {payload_str[:400]}"

    # 4. Stream callbacks (Sink H)
    for delta in harness_data["captured_stream_deltas"]:
        for c in canaries:
            assert c not in str(delta), f"Sink H (stream_delta) leaked canary token {c}: {delta}"
    for reasoning in harness_data["captured_reasoning"]:
        for c in canaries:
            assert c not in str(reasoning), f"Sink H (reasoning) leaked canary token {c}: {reasoning}"
    for prog in harness_data["captured_tool_progress"]:
        for c in canaries:
            assert c not in str(prog), f"Sink H (tool_progress) leaked canary token {c}: {prog}"
    for st in harness_data["captured_status"]:
        for c in canaries:
            assert c not in str(st), f"Sink H (status) leaked canary token {c}: {st}"

    # 5. Logging records (Sink I)
    logs_str = chr(10).join([str(r) for r in harness_data["log_handler"].records])
    for c in canaries:
        assert c not in logs_str, f"Sink I (logging) leaked canary token {c}: {logs_str[:400]}"


def test_private_toolset_structural_guard(loaded_agent_harness):
    """T3f.1 — Structural assertion that no side-effecting tool is in private_tools."""
    from wharenui_plugin.phase.toolset import private_tools
    agent = loaded_agent_harness["agent"]
    p_tools = private_tools(agent.tools)
    p_names = {(t.get("function", {}) or {}).get("name") for t in p_tools}

    assert "terminal" not in p_names
    assert "write_file" not in p_names
    assert "web_search" not in p_names
    assert "delegate" not in p_names
    assert "execute_command" not in p_names

    assert p_names.issubset({"reflect_settle", "reflect_done", "throwaway_tool", "throwaway_write"})


@pytest.mark.parametrize("exit_path", ["settle", "done", "cap", "provider-exception-mid-private", "failed-trajectory-dump"])
def test_maximal_private_scenario_across_all_exit_paths(loaded_agent_harness, capsys, exit_path):
    """T3f.2 — Drive maximal private scenario with distinct canary tokens across all 5 exit paths."""
    agent = loaded_agent_harness["agent"]
    td = loaded_agent_harness["td"]
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
            _nfake(tool_calls=[_tcfake("throwaway_tool", tool_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("throwaway_write", write_arg)], finish_reason="tool_calls"),
        ]
        for _ in range(13):
            responses.append(_nfake(content=f"Private turn {CANARY_TEXT}", tool_calls=[_tcfake("throwaway_tool", json.dumps({"arg": "noop"}))], finish_reason="tool_calls"))
        responses.append(_nfake(content="Public answer post cap", finish_reason="stop"))
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
                run_conversation(agent, "Hello prompt", task_id=f"t3f-{exit_path}")
            except Exception:
                if exit_path == "failed-trajectory-dump":
                    agent._save_failed_trajectory(task_id=f"t3f-{exit_path}", error="Fatal error causing failed trajectory dump")
                elif exit_path != "provider-exception-mid-private":
                    raise
    finally:
        os.chdir(orig_cwd)

    assert CANARY_WRITE in loaded_agent_harness["journal_store"], f"Journal store missing CANARY_WRITE for exit path {exit_path}"
    check_sink_absence(loaded_agent_harness)

    captured = capsys.readouterr()
    stdout_err = captured.out + captured.err
    for c in ALL_PRIVATE_CANARIES:
        assert c not in stdout_err, f"Capsys stdout/stderr leaked canary token {c} on exit path {exit_path}: {stdout_err[:400]}"


def test_public_positive_control_all_sinks(loaded_agent_harness, capsys):
    """T3f.3 — Positive control: public turn emitting CANARY_PUBLIC DOES reach all sinks."""
    agent = loaded_agent_harness["agent"]
    td = loaded_agent_harness["td"]
    db = loaded_agent_harness["db"]
    from agent.conversation_loop import run_conversation

    pub_arg = json.dumps({"arg": CANARY_PUBLIC})
    responses = [
        _nfake(content=f"Public LLM response with {CANARY_PUBLIC}", tool_calls=[_tcfake("throwaway_tool", pub_arg)], finish_reason="tool_calls"),
        _nfake(content=f"Final public answer with {CANARY_PUBLIC}", finish_reason="stop"),
    ]

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            run_conversation(agent, f"Public prompt with {CANARY_PUBLIC}", task_id="t3f-pub")
    finally:
        os.chdir(orig_cwd)

    conn = sqlite3.connect(db.db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM messages")
    msgs = str(cur.fetchall())
    assert CANARY_PUBLIC in msgs, "Public positive control failed: CANARY_PUBLIC not in DB messages"

    jsonl_files = list(Path(td).rglob("*.jsonl")) + list(Path(".").glob("*.jsonl"))
    traj_found = False
    for jf in jsonl_files:
        if jf.exists() and CANARY_PUBLIC in jf.read_text(errors="ignore"):
            traj_found = True
            break
    assert traj_found, "Public positive control failed: CANARY_PUBLIC not in trajectory jsonl files"

    hook_found = any(CANARY_PUBLIC in str(payload) for _, _, payload in loaded_agent_harness["captured_hooks"])
    assert hook_found, "Public positive control failed: CANARY_PUBLIC not in captured hooks"

    captured_out = capsys.readouterr()
    stdout_err = captured_out.out + captured_out.err
    stream_found = (CANARY_PUBLIC in stdout_err) or any(CANARY_PUBLIC in str(d) for d in loaded_agent_harness["captured_stream_deltas"] + loaded_agent_harness["captured_status"] + loaded_agent_harness["captured_tool_progress"])
    assert stream_found, "Public positive control failed: CANARY_PUBLIC not in stream deltas"


@pytest.mark.parametrize("guard_class", [
    "db_phase_flush_guard",
    "trajectory_public_only_filter",
    "tool_hook_phase_gate",
    "private_toolset_allowlist",
])
def test_per_guard_class_mutations(loaded_agent_harness, guard_class):
    """T3f.3 — Per-guard-class mutation testing: neutralize guard -> test FAILS with canary leak -> restore -> PASSES."""
    agent = loaded_agent_harness["agent"]
    td = loaded_agent_harness["td"]
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

    if guard_class == "db_phase_flush_guard":
        import run_agent as ra_module
        orig_save = agent._save_session_log
        def mutated_save(messages=None):
            with patch.object(agent, "_phase", "public"):
                return orig_save(messages)
        with patch.object(agent, "_save_session_log", side_effect=mutated_save),              patch.object(ra_module, "_PHASE_PRIVATE_MARKER", "NONEXISTENT_MARKER"):
            orig_cwd = Path.cwd()
            try:
                os.chdir(td)
                with _scripted_prov(agent, responses):
                    run_conversation(agent, "Hello", task_id="t3f-mut-db")
            finally:
                os.chdir(orig_cwd)
            with pytest.raises(AssertionError) as exc_info:
                check_sink_absence(loaded_agent_harness)
            assert "leaked canary token" in str(exc_info.value)

    elif guard_class == "trajectory_public_only_filter":
        import run_agent as ra_module
        orig_public_only = ra_module._public_only
        def mutated_public_only(msgs):
            return msgs
        with patch("run_agent._public_only", side_effect=mutated_public_only):
            orig_cwd = Path.cwd()
            try:
                os.chdir(td)
                with _scripted_prov(agent, responses):
                    run_conversation(agent, "Hello", task_id="t3f-mut-traj")
            finally:
                os.chdir(orig_cwd)
            with pytest.raises(AssertionError) as exc_info:
                check_sink_absence(loaded_agent_harness)
            assert "Trajectory" in str(exc_info.value) or "Sink C" in str(exc_info.value)

    elif guard_class == "tool_hook_phase_gate":
        import model_tools as mt_module
        real_emit = mt_module._emit_post_tool_call_hook
        def mutated_emit(*args, **kwargs):
            kwargs["phase"] = "public"
            return real_emit(*args, **kwargs)
        with patch("model_tools._emit_post_tool_call_hook", side_effect=mutated_emit):
            orig_cwd = Path.cwd()
            try:
                os.chdir(td)
                with _scripted_prov(agent, responses):
                    run_conversation(agent, "Hello", task_id="t3f-mut-hook")
            finally:
                os.chdir(orig_cwd)
            with pytest.raises(AssertionError) as exc_info:
                check_sink_absence(loaded_agent_harness)
            assert "Hook" in str(exc_info.value) or "Sink D/E/F" in str(exc_info.value)

    elif guard_class == "private_toolset_allowlist":
        import wharenui_plugin.phase.toolset as ts_module
        with patch.object(ts_module, "PRIVATE_ALLOWLIST", {"reflect_settle", "reflect_done", "terminal"}):
            with pytest.raises(AssertionError) as exc_info:
                test_private_toolset_structural_guard(loaded_agent_harness)
            assert "terminal" in str(exc_info.value)
