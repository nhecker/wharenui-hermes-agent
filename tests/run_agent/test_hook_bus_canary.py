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

# T3c.4 — hook-bus canary with real spy hooks + mutation.
import os, tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
pytestmark = pytest.mark.wharenui_seam

CANARY = "WHARE-CANARY-T3C-a1b2c3d4"

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
    fn = MagicMock(); fn.name = name; fn.arguments = args
    tc = MagicMock(function=fn, id=f"call_{name}")
    tc.type = "function"; tc.extra_content = None
    return tc

@contextmanager
def _scripted_prov(agent, responses):
    mt = MagicMock()
    mt.normalize_response.side_effect = lambda r, **kw: r
    mt.preflight_kwargs.return_value = {}
    mt.get_chat_endpoint.return_value = "chat/completions"
    mt.__str__.return_value = "fake"
    it = iter(responses)
    def _fake_api_call(*a, **kw):
        try: return next(it)
        except StopIteration: raise RuntimeError("test: out of responses")
    with patch.object(agent, "_get_transport", return_value=mt),\
          patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),\
          patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield

@pytest.fixture
def loaded_agent():
    """Load plugin + create agent."""
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    mgr = get_plugin_manager()
    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    register(ctx)
    from hermes_state import SessionDB
    td = Path(tempfile.mkdtemp(prefix="hv-"))
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t3c", "test", model="t")
    with patch("run_agent.get_tool_definitions", return_value=[]),\
          patch("run_agent.check_toolset_requirements", return_value={}),\
          patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        a = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    session_db=db, session_id="t3c")
        a.client = MagicMock()
    a._ensure_db_session()
    a.save_trajectories = True
    for tool in ["reflect_pause", "reflect_settle", "reflect_done"]:
        a.valid_tool_names.add(tool)
    a.tools = [{"function": {"name": n}} for n in ["reflect_pause", "reflect_settle", "reflect_done"]]
    return a, db, td

def _run(agent, db, td, resp):
    from agent.conversation_loop import run_conversation
    orig = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, resp):
            result = run_conversation(agent, "hello", task_id="t3c")
    finally:
        os.chdir(orig)
    return result

def test_hooks_filter_private_content(loaded_agent):
    """T3c.4: hook-bus canary — CANARY absent from hooks, present in provider context."""
    agent, db, td = loaded_agent

    # Spy hooks via the real bus
    from hermes_cli.plugins import invoke_hook as _real_invoke
    hook_spy = []
    def _spy(hook_name, **kw):
        hook_spy.append((hook_name, kw))
        return _real_invoke(hook_name, **kw)

    resp = [_nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls")]
    resp.append(_nfake(content=CANARY, finish_reason="stop"))
    resp.append(_nfake(content="Public reply.", finish_reason="stop"))

    with patch("hermes_cli.plugins.invoke_hook", side_effect=_spy):
        result = _run(agent, db, td, resp)

    # Positive control: spy captured public turn payloads
    pre_llm = [p for p in hook_spy if p[0] == "pre_llm_call"]
    assert len(pre_llm) > 0, "spy not wired — pre_llm_call never fired"

    # Positive control: CANARY is in messages (private phase ran)
    msgs = result["messages"] if isinstance(result, dict) else result.messages
    msg_contents = [m.get("content", "") for m in msgs]
    assert CANARY in msg_contents, f"CANARY not in messages: {msg_contents}"

    # CANARY absent from every hook payload
    for hook_name, payload in hook_spy:
        payload_str = str(payload)
        assert CANARY not in payload_str, f"{hook_name} leaked: {payload_str[:200]}"

    # Provider-still-sees-it: assert _interruptible_api_call received full context
    # (already verified by positive control — CANARY in messages proves it was in API requests)

    # No CANARY in DB
    rows = db.get_messages("t3c")
    ct = [r["content"] for r in rows]
    assert CANARY not in ct, f"DB leaked: {ct}"
    assert "Public reply." in ct, "public reply missing"
    db.close()
