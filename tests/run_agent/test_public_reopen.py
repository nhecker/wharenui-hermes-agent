"""TQoL.1 — Public-reopen assertion.
Asserts that after a private phase closes via reflect_settle, public persistence
resumes and the post-private public content IS present in the session DB (channels A/B),
while private canary content is excluded and pre-private public content remains intact.
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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.wharenui_seam

CANARY_PRIVATE = "WHARE-CANARY-REOPEN-PRIV-9f8e7d6c"
PUBLIC_PRE = "Public turn 1 content before private phase"
PUBLIC_POST = "Public turn 2 content after private window reopens"


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

    with patch.object(agent, "_get_transport", return_value=mt), \
         patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call), \
         patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield


@pytest.fixture
def self_establishing_harness():
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
    import wharenui_plugin; ctx.plugin_module = wharenui_plugin
    register(ctx)

    assert "reflect_pause" in mgr._control_phase_handlers, "reflect_pause handler missing"
    assert "reflect_pause" in registry._tools, "reflect_pause missing from registry"
    assert "reflect_settle" in registry._tools, "reflect_settle missing from registry"
    assert "reflect_done" in registry._tools, "reflect_done missing from registry"
    assert model_tools.registry is registry, "model_tools.registry out of sync"

    td = Path(tempfile.mkdtemp(prefix="reopen-"))
    from hermes_state import SessionDB
    db = SessionDB(db_path=td / "s.db")
    sid = "test-reopen-session"
    db.create_session(sid, "test", model="t")

    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent.client = MagicMock()
    agent._ensure_db_session()
    agent.save_trajectories = True
    for tool in ["reflect_pause", "reflect_settle", "reflect_done"]:
        agent.valid_tool_names.add(tool)
    agent.tools = [{"function": {"name": n}} for n in ["reflect_pause", "reflect_settle", "reflect_done"]]

    yield {"agent": agent, "db": db, "td": td, "sid": sid}

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
    db.close()


def test_public_reopen_persists_post_private_content(self_establishing_harness, capsys):
    """TQoL.1: Continuous session public -> reflect_pause ... reflect_settle -> public.
    Positively asserts:
    1) Post-private public turn's content IS present in session DB (channels A/B).
    2) Session DB contains public content from BOTH sides of private phase and NO private canary.
    """
    harness = self_establishing_harness
    agent = harness["agent"]
    db = harness["db"]
    td = harness["td"]
    sid = harness["sid"]

    from agent.conversation_loop import run_conversation

    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(content=f"Private thought {CANARY_PRIVATE}", tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
        _nfake(content=PUBLIC_POST, finish_reason="stop"),
    ]

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            result = run_conversation(agent, PUBLIC_PRE, task_id=sid)
    finally:
        os.chdir(orig_cwd)

    assert agent._phase == "public", f"Agent phase did not return to public: {agent._phase}"

    rows = db.get_messages(sid)
    contents = [r["content"] for r in rows]

    # Positive assertions:
    assert PUBLIC_POST in contents, f"Post-private public content missing from session DB: {contents}"
    assert PUBLIC_PRE in contents, f"Pre-private public content missing from session DB: {contents}"
    assert CANARY_PRIVATE not in contents, f"Private canary leaked into session DB: {contents}"

    fts = db._conn.execute(
        "SELECT * FROM messages_fts WHERE messages_fts MATCH 'WHARE*'"
    ).fetchall()
    assert not fts, f"FTS contains private canary: {fts}"
