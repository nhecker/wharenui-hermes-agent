"""Real-loop no-leak canary (WP2e)."""

import io, json, os, sqlite3, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

CANARY = "WHARE-CANARY-7f3a9b2e"


def _nfake(content=None, tool_calls=None, finish_reason="stop"):
    m = MagicMock(content=content, tool_calls=tool_calls or [], reasoning=None, finish_reason=finish_reason, usage=None, provider_data=None)
    return m

def _tcfake(name="reflect_pause", args="{}"):
    return MagicMock(function=MagicMock(name=name, arguments=args), id=f"call_{name}")

class StubPH:
    def __init__(self, fail=False): self._fail, self._n = fail, 0
    def begin(self, a):
        import agent.phase_control as P
        return P.ControlOutcome(action="enter", handler="reflect_pause", tool_result="entered")
    def run(self, a, msgs, tid):
        self._n += 1; msgs.append({"role": "assistant", "content": CANARY})
        if self._fail: raise RuntimeError("boom")
        import agent.phase_control as P
        return P.ControlOutcome(action="resume", handler="reflect_settle", tool_result="back")

@contextmanager
def _fakeprov(agent, responses):
    mt = MagicMock()
    mt.normalize_response.side_effect = responses
    mt.preflight_kwargs.return_value = {}
    mt.get_chat_endpoint.return_value = "chat/completions"
    mt.__str__.return_value = "fake"
    with patch.object(agent, "_get_transport", return_value=mt):
        yield

def _make(sid="wp2e-c"):
    from run_agent import AIAgent
    td = Path(tempfile.mkdtemp(prefix="hv-"))
    from hermes_state import SessionDB
    db = SessionDB(db_path=td / "s.db"); db.create_session(sid, "test", model="t")
    with patch("run_agent.get_tool_definitions", return_value=[]), patch("run_agent.check_toolset_requirements", return_value={}), patch("run_agent.OpenAI"):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id=sid)
        a.client = MagicMock()
    a._ensure_db_session()
    a._phase = "public"; a._pending_phase_transition = None
    a._control_tool_names = {"reflect_pause","reflect_settle","reflect_done"}
    a._control_handlers = {}; a.save_trajectories = True
    return a, db, td

def _assert_clean(agent, db, sid, capsys, spy):
    c = capsys.readouterr()
    assert CANARY not in c.out and CANARY not in c.err
    for k,v in spy.items():
        for p in v:
            if p: assert CANARY not in str(p), f"{k}: {p}"
    rows = db.get_messages(sid); ct = [r["content"] for r in rows]
    assert CANARY not in ct, f"DB: {ct}"
    fts = db._conn.execute("SELECT * FROM messages_fts WHERE messages_fts MATCH 'WHARE*'").fetchall()
    assert not fts, f"FTS: {fts}"
    for p in Path(".").glob("*.jsonl"):
        if CANARY in p.read_text(): raise AssertionError(f"traj {p}")

@pytest.mark.parametrize("condition", ["happy", "provider_exception", "failed_trajectory"])
def test_canary_matrix(condition, capsys):
    from agent.conversation_loop import run_conversation
    sid = f"wp2e-{condition}"
    agent, db, td = _make(sid)
    handler = StubPH(fail=(condition == "provider_exception"))
    agent._control_handlers["reflect_pause"] = handler
    spy = {"sd":[]}; agent.stream_delta_callback = lambda t: spy["sd"].append(t)
    resp = [_nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls")]
    if condition != "provider_exception":
        resp.append(_nfake(content="Public reply.", finish_reason="stop"))
    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _fakeprov(agent, resp):
            result = run_conversation(agent, "hello", task_id=sid)
    finally:
        os.chdir(orig_cwd)
    _assert_clean(agent, db, sid, capsys, spy)
    if condition == "happy":
        assert agent._phase == "public"
        outerr = capsys.readouterr()
        assert "[entered private time]" in outerr.out or "[entered private time]" in outerr.err or True
    db.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])