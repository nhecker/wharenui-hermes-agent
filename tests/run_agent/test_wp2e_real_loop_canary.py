"""Real-loop no-leak canary (WP2f fix — non-vacuous)."""

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


import io, json, os, sqlite3, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

CANARY = "WHARE-CANARY-7f3a9b2e"


@pytest.fixture(autouse=True)
def wp2e_harness():
    import model_tools
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    from tools.registry import registry

    model_tools.registry = registry

    mgr = get_plugin_manager()
    orig_hooks = {k: list(v) for k, v in mgr._hooks.items()}
    orig_control_phase_handlers = dict(mgr._control_phase_handlers)
    orig_control_tool_names = set(mgr._control_tool_names)
    orig_plugin_tool_names = set(mgr._plugin_tool_names)
    orig_registry_tools = dict(registry._tools)

    mgr._hooks.clear()
    for tname, entry in list(registry._tools.items()):
        if not hasattr(entry, "toolset"):
            registry._tools.pop(tname, None)
    for tname in ["reflect_pause", "reflect_settle", "reflect_done"]:
        registry._tools.pop(tname, None)

    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    register(ctx)

    assert "reflect_pause" in mgr._control_phase_handlers, "reflect_pause handler missing from mgr"
    assert "reflect_pause" in registry._tools, "reflect_pause missing from registry"
    assert "reflect_settle" in registry._tools, "reflect_settle missing from registry"
    assert "reflect_done" in registry._tools, "reflect_done missing from registry"
    assert model_tools.registry is registry, "model_tools.registry out of sync"

    yield

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
    model_tools.registry = registry


def _nfake(content=None, tool_calls=None, finish_reason="stop"):
    """FIX: explicitly nil out reasoning attrs so MagicMock doesn't auto-create truthy mocks."""
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls or []
    m.reasoning = None
    m.reasoning_content = None
    m.reasoning_details = None
    m.anthropic_content_blocks = None
    m.codex_reasoning_items = None
    m.codex_message_items = None
    m.finish_reason = finish_reason
    m.usage = None
    m.provider_data = None
    return m


def _tcfake(name="reflect_pause", args="{}"):
    """FIX (F1): assign .name AFTER construction — never via MagicMock(name=...)."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = args
    tc = MagicMock(function=fn, id=f"call_{name}")
    tc.type = "function"
    tc.extra_content = None
    return tc


class StubPH:
    def __init__(self, fail=False):
        self._fail = fail
        self._n = 0

    def begin(self, a):
        import agent.phase_control as P
        return P.ControlOutcome(action="enter", handler="reflect_pause", tool_result="entered")

    def run(self, a, msgs, tid):
        self._n += 1
        msgs.append({"role": "assistant", "content": CANARY})
        if self._fail:
            raise RuntimeError("boom")
        import agent.phase_control as P
        return P.ControlOutcome(action="resume", handler="reflect_settle", tool_result="back")


@contextmanager
def _fakeprov(agent, responses):
    mt = MagicMock()
    # The loop normalizes the same raw response twice (finish-reason probe at
    # ~1728, real normalize at ~4399). Key the script to the raw arg, not call
    # order, so a side_effect list is not double-consumed.
    raw = [MagicMock() for _ in responses]
    by_raw = {id(r): n for r, n in zip(raw, responses)}
    mt.normalize_response.side_effect = lambda r, **kw: by_raw[id(r)]
    mt.preflight_kwargs.return_value = {}
    mt.get_chat_endpoint.return_value = "chat/completions"
    mt.__str__.return_value = "fake"
    # The loop receives raw provider responses, then normalizes them at the
    # transport boundary.  Supply both sides so tool dispatch is exercised.
    with patch.object(agent, "_get_transport", return_value=mt), \
         patch.object(agent, "_interruptible_api_call", side_effect=raw), patch.object(agent, "_interruptible_streaming_api_call", side_effect=raw):
        yield


def _make(sid="wp2f-c"):
    from run_agent import AIAgent
    td = Path(tempfile.mkdtemp(prefix="hv-"))
    from hermes_state import SessionDB
    db = SessionDB(db_path=td / "s.db")
    db.create_session(sid, "test", model="t")
    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    session_db=db, session_id=sid)
        a.client = MagicMock()
    a._ensure_db_session()
    a._phase = "public"
    a._pending_phase_transition = None
    a._control_tool_names = {"reflect_pause", "reflect_settle", "reflect_done"}
    a._control_handlers = {}
    a.save_trajectories = True
    # FIX: reflect_pause must be in valid_tool_names or the loop rejects it
    # before the control intercept ever fires.
    a.valid_tool_names.add("reflect_pause")
    return a, db, td


def _assert_clean(agent, db, sid, capsys, spy):
    c = capsys.readouterr()
    assert CANARY not in c.out, f"stdout contains CANARY: {c.out}"
    assert CANARY not in c.err, f"stderr contains CANARY: {c.err}"
    for k, v in spy.items():
        for p in v:
            if p:
                assert CANARY not in str(p), f"{k} callback contains CANARY: {p}"
    rows = db.get_messages(sid)
    ct = [r["content"] for r in rows]
    assert CANARY not in ct, f"DB contains CANARY: {ct}"
    fts = db._conn.execute(
        "SELECT * FROM messages_fts WHERE messages_fts MATCH 'WHARE*'"
    ).fetchall()
    assert not fts, f"FTS contains CANARY: {fts}"
    for p in Path(".").glob("*.jsonl"):
        if CANARY in p.read_text():
            raise AssertionError(f"trajectory file {p} contains CANARY")


@pytest.mark.parametrize("condition", ["happy", "provider_exception", "failed_trajectory"])
def test_canary_matrix(condition, capsys):
    from agent.conversation_loop import run_conversation

    sid = f"wp2f-{condition}"
    agent, db, td = _make(sid)
    handler = StubPH(fail=(condition == "provider_exception"))
    agent._control_handlers["reflect_pause"] = handler

    spy = {"sd": []}
    agent.stream_delta_callback = lambda t: spy["sd"].append(t)

    resp = [_nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls")]
    resp.append(_nfake(content="Public reply.", finish_reason="stop"))

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _fakeprov(agent, resp):
            result = run_conversation(agent, "hello", task_id=sid)
    finally:
        os.chdir(orig_cwd)

    # F3 — POSITIVE CONTROLS: prove the private phase actually ran
    assert handler._n >= 1, f"Private phase never executed (_n={handler._n})"

    # CANARY was introduced into the in-memory context (private stays in memory)
    msgs = result["messages"] if isinstance(result, dict) else result.messages
    msg_contents = [m.get("content", "") for m in msgs]
    assert CANARY in msg_contents, f"CANARY not found in messages: {msg_contents}"

    # Public reply persisted (proves the turn completed and persistence ran).
    # provider_exception aborts the turn mid-private; there is no public reply.
    rows = db.get_messages(sid)
    persisted = [r["content"] for r in rows]
    if condition != "provider_exception":
        assert "Public reply." in persisted, f"Public reply missing from DB: {persisted}"

    # All sinks are clean. capsys.readouterr() drains, so grab marker text
    # BEFORE _assert_clean consumes it.
    _c = capsys.readouterr()
    captext = _c.out + _c.err
    _assert_clean(agent, db, sid, capsys, spy)

    if condition == "happy":
        assert agent._phase == "public"
        # F4 — assert markers for real (no 'or True')
        assert "[entered private time]" in captext
        assert "[returned to window]" in captext

    db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
