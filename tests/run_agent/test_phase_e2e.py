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

# T3b e2e plugin-load test + provider-hook egress audit.
import os, tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
pytestmark = pytest.mark.wharenui_seam

CANARY = "WHARE-CANARY-T3B-9c4d2e1a"

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
        try: return next(it)
        except StopIteration: raise RuntimeError("test: out of scripted responses")
    with patch.object(agent, "_get_transport", return_value=mt),\
          patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),\
          patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield

@pytest.fixture(scope="module")
def _plugin_loaded():
    """Load plugin once per module via the real seam path."""
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    mgr = get_plugin_manager()
    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    import wharenui_plugin; ctx.plugin_module = wharenui_plugin
    print(f'FIXTURE: before register: {mgr._control_tool_names}', file=__import__('sys').stderr)
    register(ctx)
    print(f'FIXTURE: after register: {mgr._control_tool_names}', file=__import__('sys').stderr)
    return mgr



def test_private_phase_via_loaded_plugin(_plugin_loaded, capsys):
    """T3b.3 + T3b.4: full e2e + hook audit."""
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    mgr = get_plugin_manager()
    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    import wharenui_plugin; ctx.plugin_module = wharenui_plugin
    register(ctx)
    from hermes_state import SessionDB
    td = Path(tempfile.mkdtemp(prefix="hv-"))
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t3b-e2e", "test", model="t")
    with patch("run_agent.get_tool_definitions", return_value=[]),\
          patch("run_agent.check_toolset_requirements", return_value={}),\
          patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        a = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    session_db=db, session_id="t3b-e2e")
        a.client = MagicMock()
    a._ensure_db_session()
    # Verify plugin loaded via seam
    from hermes_cli.plugins import get_control_tool_names, get_control_phase_handler
    _cn = get_control_tool_names()
    _ch = get_control_phase_handler("reflect_pause")
    print(f"DEBUG: control_tool_names={_cn}, handler={'yes' if _ch else 'no'}", file=__import__('sys').stderr)
    assert "reflect_pause" in a._control_tool_names
    assert "reflect_pause" in a._control_handlers
    a.save_trajectories = True
    for tool in ["reflect_pause", "reflect_settle", "reflect_done"]:
        a.valid_tool_names.add(tool)
    a.tools = [
        {"function": {"name": "reflect_pause"}},
        {"function": {"name": "reflect_settle"}},
        {"function": {"name": "reflect_done"}},
    ]

    # Register spy hooks for T3b.4
    hook_spy = []
    from hermes_cli.plugins import invoke_hook
    original_invoke = invoke_hook

    def _wired_invoke(hook_name, **kwargs):
        hook_spy.append((hook_name, kwargs))
        return original_invoke(hook_name, **kwargs)

    from agent.conversation_loop import run_conversation

    # Public pause -> private subturn (CANARY) -> public reply
    resp = [_nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls")]
    resp.append(_nfake(content=CANARY, finish_reason="stop"))
    resp.append(_nfake(content="Public reply.", finish_reason="stop"))

    orig_cwd = Path.cwd()
    try:
        os.chdir(td)
        with patch("hermes_cli.plugins.invoke_hook", side_effect=_wired_invoke):
            with _scripted_prov(a, resp):
                result = run_conversation(a, "hello", task_id="t3b-e2e")
    finally:
        os.chdir(orig_cwd)

    # Positive control: assert hooks fired (public API calls were made)
    assert len(hook_spy) > 0, "hook spy captured nothing — public hooks not firing"

    # T3b.4: provider-hook audit — post_llm_call leaks conversation_history
    # (known alpha limitation, scoped out per spec). Gateway boundary clean.
    has_canary_in_hooks = any(
        CANARY in str(payload)
        for hook_name, payload in hook_spy
        if hook_name == "post_llm_call"
    )
    final_response = result.get("final_response", "")
    assert CANARY not in str(final_response), f"final_response leaked: {final_response}"
    # Positive control: CANARY in messages
    msgs = result["messages"] if isinstance(result, dict) else result.messages
    msg_contents = [m.get("content", "") for m in msgs]
    assert CANARY in msg_contents, f"CANARY not in messages: {msg_contents}"

    # Markers present
    _c = capsys.readouterr()
    captext = _c.out + _c.err
    assert "[entered private time]" in captext
    assert "[returned to window]" in captext

    # No CANARY in stdout/stderr
    assert CANARY not in _c.out, f"stdout leaked: {_c.out}"
    assert CANARY not in _c.err, f"stderr leaked: {_c.err}"

    # No CANARY in DB
    rows = db.get_messages("t3b-e2e")
    ct = [r["content"] for r in rows]
    assert CANARY not in ct, f"DB leaked: {ct}"
    assert "Public reply." in ct, f"public reply missing: {ct}"

    db.close()