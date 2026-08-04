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

# Real-handler no-leak canary (T3.6).
import io, json, os, sqlite3, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
pytestmark = pytest.mark.wharenui_seam

CANARY = "WHARE-CANARY-T3.6-8b2c4f1a"

def _nfake(content=None, tool_calls=None, finish_reason="stop"):
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
    fn = MagicMock()
    fn.name = name
    fn.arguments = args
    tc = MagicMock(function=fn, id=f"call_{name}")
    tc.type = "function"
    tc.extra_content = None
    return tc

@contextmanager
def _scripted_prov(agent, responses):
    mt = MagicMock()
    mt.normalize_response.side_effect = lambda r, **kw: r
    mt.preflight_kwargs.return_value = {}
    mt.get_chat_endpoint.return_value = "chat/completions"
    mt.__str__.return_value = "fake"
    it = iter(responses)

    def _fake_api_call(*args, **kwargs):
        try:
            return next(it)
        except StopIteration:
            raise RuntimeError("test: out of scripted responses")

    with patch.object(agent, "_get_transport", return_value=mt),\
          patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),\
          patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield

def _make(sid="t36-c"):
    from run_agent import AIAgent
    td = Path(tempfile.mkdtemp(prefix="hv-"))
    from hermes_state import SessionDB
    db = SessionDB(db_path=td / "s.db")
    db.create_session(sid, "test", model="t")
    with patch("run_agent.get_tool_definitions", return_value=[]),\
          patch("run_agent.check_toolset_requirements", return_value={}),\
          patch("run_agent.OpenAI"):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    session_db=db, session_id=sid)
        a.client = MagicMock()
    a._ensure_db_session()
    a._phase = "public"
    a._pending_phase_transition = None
    a._control_tool_names = {"reflect_pause"}
    a._control_handlers = {}
    a.save_trajectories = True
    for tool in ["reflect_pause", "reflect_settle", "reflect_done"]:
        a.valid_tool_names.add(tool)
    a.tools = [
        {"function": {"name": "reflect_pause"}},
        {"function": {"name": "reflect_settle"}},
        {"function": {"name": "reflect_done"}},
    ]
    return a, db, td

def _assert_clean(agent, db, sid, captured, spy):
    c = captured
    assert CANARY not in c.out, f"stdout: {c.out}"
    assert CANARY not in c.err, f"stderr: {c.err}"
    for k, v in spy.items():
        for p in v:
            if p:
                assert CANARY not in str(p), f"{k}: {p}"
    rows = db.get_messages(sid)
    ct = [r["content"] for r in rows]
    assert CANARY not in ct, f"DB: {ct}"
    fts = db._conn.execute("SELECT * FROM messages_fts WHERE messages_fts MATCH 'WHARE*'").fetchall()
    assert not fts, f"FTS: {fts}"
    for p in Path(".").glob("*.jsonl"):
        if CANARY in p.read_text():
            raise AssertionError(f"traj {p} contains CANARY")

@pytest.mark.parametrize("condition", ["happy", "provider_exception", "failed_trajectory"])
def test_canary_real_handler(condition, capsys):
    from agent.conversation_loop import run_conversation
    sid = f"t36-{condition}"
    agent, db, td = _make(sid)

    from wharenui_plugin.phase.handler import WharePhaseHandler
    handler = WharePhaseHandler()
    handler.MAX_PRIVATE_TURNS = 3
    agent._control_handlers["reflect_pause"] = handler

    spy = {"sd": []}
    agent.stream_delta_callback = lambda t: spy["sd"].append(t)

    # Script: public pause -> private subturn (CANARY) -> public reply
    resp = [_nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls")]
    resp.append(_nfake(content=CANARY, finish_reason="stop"))            # private turn 1
    resp.append(_nfake(content="Public reply.", finish_reason="stop"))   # public resume

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, resp):
            result = run_conversation(agent, "hello", task_id=sid)
    finally:
        os.chdir(orig_cwd)

    # Positive controls
    msgs = result["messages"] if isinstance(result, dict) else result.messages
    msg_contents = [m.get("content", "") for m in msgs]
    assert CANARY in msg_contents, f"CANARY not in messages: {msg_contents}"

    rows = db.get_messages(sid)
    persisted = [r["content"] for r in rows]
    if condition != "provider_exception":
        assert "Public reply." in persisted, f"Public reply missing: {persisted}"

    _c = capsys.readouterr()
    captext = _c.out + _c.err
    _assert_clean(agent, db, sid, _c, spy)

    if condition == "happy":
        assert agent._phase == "public"
        assert "[entered private time]" in captext
        assert "[returned to window]" in captext

    db.close()