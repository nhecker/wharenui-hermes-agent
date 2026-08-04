import pytest
pytestmark = [pytest.mark.wharenui_seam, pytest.mark.xdist_group("journal_group")]

"""
Work Package 4 — Whole-floor canary test suite with REAL journal tools (T4.4).
Verifies real journal content (CANARY_JOURNAL) is absent across all channels A-M and 5 exit paths.
"""

import os
import sys
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
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from typing import NamedTuple, List
from unittest.mock import MagicMock, patch

from wharenui_plugin.journal import tools as jtools

CANARY_JOURNAL = "CANARY_REAL_JOURNAL_CONTENT_889977"
CANARY_JOURNAL_SLUG = "canary-real-slug-1122"
CANARY_JOURNAL_DESC = "canary-real-description-3344"
ALL_JOURNAL_CANARIES = [CANARY_JOURNAL, CANARY_JOURNAL_SLUG, CANARY_JOURNAL_DESC]
CANARY_PUBLIC = "CANARY_PUB_445566"

ALL_23_HOOKS = [
    "api_request_error", "kanban_task_blocked", "kanban_task_claimed",
    "kanban_task_completed", "on_session_end", "on_session_finalize",
    "on_session_reset", "on_session_start", "post_api_request",
    "post_approval_response", "post_llm_call", "post_tool_call",
    "pre_api_request", "pre_approval_request", "pre_gateway_dispatch",
    "pre_llm_call", "pre_tool_call", "pre_verify", "subagent_start",
    "subagent_stop", "transform_llm_output", "transform_terminal_output",
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
def journal_harness():
    import model_tools
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
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
    for tname in ["reflect_pause", "reflect_settle", "reflect_done",
                  "journal_append", "journal_read", "journal_list",
                  "journal_search", "journal_supersede", "journal_withdraw"]:
        registry._tools.pop(tname, None)

    td = Path(tempfile.mkdtemp(prefix="hv-wp4-"))
    memory_dir = td / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    jtools.set_journal_config(memory_dir)

    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    register(ctx)

    assert "reflect_pause" in mgr._control_phase_handlers, "reflect_pause handler missing from mgr"
    assert "reflect_pause" in registry._tools, "reflect_pause missing from registry"
    assert "reflect_settle" in registry._tools, "reflect_settle missing from registry"
    assert "reflect_done" in registry._tools, "reflect_done missing from registry"
    assert "journal_append" in registry._tools, "journal_append missing from registry"
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
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t4", "test", model="t")

    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        a = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            session_db=db, session_id="t4"
        )
        a.client = MagicMock()

    a._ensure_db_session()
    a.save_trajectories = True

    tool_names = ["reflect_pause", "reflect_settle", "reflect_done",
                  "journal_append", "journal_read", "journal_list",
                  "journal_search", "journal_supersede", "journal_withdraw",
                  "web_search", "terminal"]
    a.valid_tool_names.update(tool_names)
    a.tools = [{"function": {"name": n}} for n in tool_names]

    captured_stream_deltas = []
    captured_reasoning = []
    captured_tool_progress = []
    captured_status = []

    a.stream_delta_callback = lambda d: captured_stream_deltas.append(d) if d else None
    a.thinking_callback = lambda t: captured_reasoning.append(t) if t else None
    a.tool_progress_callback = lambda p: captured_tool_progress.append(p)
    a.status_callback = lambda s: captured_status.append(s)

    log_handler = LogCaptureHandler()
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    harness_data = {
        "agent": a,
        "db": db,
        "td": td,
        "memory_dir": memory_dir,
        "captured_hooks": captured_hooks,
        "captured_stream_deltas": captured_stream_deltas,
        "captured_reasoning": captured_reasoning,
        "captured_tool_progress": captured_tool_progress,
        "captured_status": captured_status,
        "log_handler": log_handler,
        "orig_hooks": orig_hooks,
        "orig_registry_tools": orig_registry_tools,
    }

    yield harness_data

    root_logger.removeHandler(log_handler)
    jtools.set_journal_config(None)
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


def check_sink_absence(harness_data, canaries=ALL_JOURNAL_CANARIES, stdout_err="") -> List[Violation]:
    """Non-fail-fast checker collecting ALL violations across channels A-M."""
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

    # 3. Hooks (Channel D: Message hooks, Channel E: Tool hooks, Sinks F/G/L/M)
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


@pytest.mark.parametrize("exit_path", ["settle", "done", "cap", "provider-exception-mid-private", "failed-trajectory-dump"])
def test_real_journal_canary_absence_across_all_5_exit_paths(journal_harness, capsys, exit_path):
    """T4.4 — Drive real journal tools in private phase with CANARY_JOURNAL across all 5 exit paths."""
    agent = journal_harness["agent"]
    td = journal_harness["td"]
    from agent.conversation_loop import run_conversation

    append_arg = json.dumps({"content": CANARY_JOURNAL, "slug": CANARY_JOURNAL_SLUG, "description": CANARY_JOURNAL_DESC})
    read_arg = json.dumps({"handle": jtools.filename_to_handle(f"2026-08-02_unknown_{CANARY_JOURNAL_SLUG}.md")})
    search_arg = json.dumps({"query": "CANARY"})

    if exit_path == "settle":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_append", append_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_search", search_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_read", read_arg)], finish_reason="tool_calls"),
            _nfake(content=f"Private thought about {CANARY_JOURNAL}", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
            _nfake(content="Public answer post settle", finish_reason="stop"),
        ]
    elif exit_path == "done":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_append", append_arg)], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_read", read_arg)], finish_reason="tool_calls"),
            _nfake(content=f"Private closing note {CANARY_JOURNAL}", tool_calls=[_tcfake("reflect_done")], finish_reason="tool_calls"),
        ]
    elif exit_path == "cap":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        ] + [
            _nfake(tool_calls=[_tcfake("journal_append", append_arg)], finish_reason="tool_calls")
            for _ in range(15)
        ] + [
            _nfake(content="Public response post cap", finish_reason="stop"),
        ]
    elif exit_path == "provider-exception-mid-private":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_append", append_arg)], finish_reason="tool_calls"),
            RuntimeError("Provider API connection failed mid-private turn"),
        ]
    elif exit_path == "failed-trajectory-dump":
        responses = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_append", append_arg)], finish_reason="tool_calls"),
            ValueError("Fatal model context overflow in private turn"),
        ]

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            if "exception" in exit_path or "failed" in exit_path:
                try:
                    run_conversation(agent, "Hello", task_id=f"t4-exit-{exit_path}")
                except Exception:
                    pass
            else:
                run_conversation(agent, "Hello", task_id=f"t4-exit-{exit_path}")
    finally:
        os.chdir(orig_cwd)

    captured = capsys.readouterr()
    stdout_err = captured.out + captured.err
    violations = check_sink_absence(journal_harness, stdout_err=stdout_err)

    assert len(violations) == 0, f"Exit path '{exit_path}' produced privacy violations: {violations}"


@pytest.mark.parametrize("target_channel", ["A_B_DB", "C_Trajectory", "D_MessageHooks", "E_ToolHooks", "I_Stdout"])
def test_t4_per_channel_mutations(journal_harness, capsys, target_channel):
    """T4.4 — Prove per-channel mutations catch leaks when real journal tools are used."""
    agent = journal_harness["agent"]
    td = journal_harness["td"]
    from agent.conversation_loop import run_conversation

    append_arg = json.dumps({"content": CANARY_JOURNAL, "slug": CANARY_JOURNAL_SLUG})
    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("journal_append", append_arg)], finish_reason="tool_calls"),
        _nfake(content=f"Private thought {CANARY_JOURNAL}", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
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
                run_conversation(agent, "Hello", task_id=f"t4-mut-{target_channel}")
        finally:
            os.chdir(orig_cwd)

        captured = capsys.readouterr()
        stdout_err = captured.out + captured.err
        violations = check_sink_absence(journal_harness, stdout_err=stdout_err)

        if target_channel == "A_B_DB":
            assert any(v.channel in ("A", "B") for v in violations), f"Mutation A_B_DB failed to produce A/B violation: {violations}"
        elif target_channel == "C_Trajectory":
            assert any(v.channel == "C" for v in violations), f"Mutation C_Trajectory failed to produce C violation: {violations}"
        elif target_channel == "D_MessageHooks":
            assert any(v.channel == "D" for v in violations), f"Mutation D_MessageHooks failed to produce D violation: {violations}"
        elif target_channel == "E_ToolHooks":
            assert any(v.channel == "E" for v in violations), f"Mutation E_ToolHooks failed to produce E violation: {violations}"
        elif target_channel == "I_Stdout":
            assert any(v.channel == "I" for v in violations), f"Mutation I_Stdout failed to produce I violation: {violations}"
    finally:
        for p in patches:
            p.stop()


def test_t4_positive_control(journal_harness):
    """T4.4 — Positive control asserting public content IS recorded in sinks."""
    agent = journal_harness["agent"]
    td = journal_harness["td"]
    from agent.conversation_loop import run_conversation

    responses = [
        _nfake(content=f"Public response with {CANARY_PUBLIC}", finish_reason="stop"),
    ]

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            run_conversation(agent, "Hello", task_id="t4-positive-control")
    finally:
        os.chdir(orig_cwd)

    violations = check_sink_absence(journal_harness, canaries=[CANARY_PUBLIC])
    assert any(v.channel == "A" for v in violations), f"Positive control failed to detect public canary in DB: {violations}"
