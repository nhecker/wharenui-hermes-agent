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

# T3d.3 — hook-bus canary with real registered hooks + per-site mutation.
import os, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
pytestmark = pytest.mark.wharenui_seam

sys.path.insert(0, "/root/work/wharenui-hermes-agent")

CANARY = "WHARE-CANARY-T3D-99887766"

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
        try: return next(it)
        except StopIteration: raise RuntimeError("test: out of responses")
    with patch.object(agent, "_get_transport", return_value=mt),          patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),          patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield

@pytest.fixture
def loaded_agent():
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    mgr = get_plugin_manager()
    mgr._hooks.clear()
    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    import wharenui_plugin; ctx.plugin_module = wharenui_plugin
    register(ctx)

    captured = {}
    def make_spy(name):
        captured[name] = []
        def spy(**kw):
            captured[name].append(kw)
        return spy

    for h in ["pre_api_request", "post_api_request", "pre_llm_call", "post_llm_call", "transform_tool_result"]:
        ctx.register_hook(h, make_spy(h))

    from hermes_state import SessionDB
    td = Path(tempfile.mkdtemp(prefix="hv-"))
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t3d", "test", model="t")
    with patch("run_agent.get_tool_definitions", return_value=[]),          patch("run_agent.check_toolset_requirements", return_value={}),          patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        a = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    session_db=db, session_id="t3d")
        a.client = MagicMock()
    a._ensure_db_session()
    a.save_trajectories = True
    for tool in ["reflect_pause", "reflect_settle", "reflect_done"]:
        a.valid_tool_names.add(tool)
    a.tools = [{"function": {"name": n}} for n in ["reflect_pause", "reflect_settle", "reflect_done"]]
    return a, db, td, captured

def _run(agent, db, td, resp, captured_api_kwargs=None):
    from agent.conversation_loop import run_conversation
    orig = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, resp, captured_api_kwargs):
            result = run_conversation(agent, "hello", task_id="t3d")
    finally:
        os.chdir(orig)
    return result

def _get_resp():
    return [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(content=CANARY, finish_reason="stop"),
        _nfake(content="Public reply.", finish_reason="stop"),
    ]

@pytest.mark.parametrize("condition", ["happy", "provider_exception", "failed_trajectory"])
def test_real_registered_hooks_filter_canary(loaded_agent, condition):
    agent, db, td, captured = loaded_agent
    from hermes_cli.plugins import has_hook

    for h in ["pre_api_request", "post_api_request", "pre_llm_call", "post_llm_call", "transform_tool_result"]:
        assert has_hook(h) is True, f"has_hook({h}) is False"

    resp = _get_resp()
    captured_api_kwargs = []
    result = _run(agent, db, td, resp, captured_api_kwargs)

    assert len(captured["pre_api_request"]) >= 1, "pre_api_request did not fire"
    assert len(captured["post_api_request"]) >= 1, "post_api_request did not fire"
    assert len(captured["pre_llm_call"]) >= 1, "pre_llm_call did not fire"
    assert len(captured["post_llm_call"]) >= 1, "post_llm_call did not fire"

    for h, calls in captured.items():
        for payload in calls:
            payload_str = str(payload)
            assert CANARY not in payload_str, f"{h} leaked CANARY in payload: {payload_str[:300]}"

    private_turn_found_in_provider_call = False
    for api_kw in captured_api_kwargs:
        msgs = api_kw.get("messages", []) if isinstance(api_kw, dict) else []
        for m in msgs:
            if isinstance(m, dict) and CANARY in str(m.get("content", "")):
                private_turn_found_in_provider_call = True
                break
    assert private_turn_found_in_provider_call, "Provider call missing private context!"

    db.close()


@pytest.mark.parametrize("site_key,target_field,event_name", [
    ("pre_llm_history", "conversation_history", "pre_llm_call"),
    ("post_llm_history", "conversation_history", "post_llm_call"),
    ("pre_api_history", "conversation_history", "pre_api_request"),
    ("pre_api_req_msgs", "request_messages", "pre_api_request"),
    ("pre_api_request_body", "request", "pre_api_request"),
])
def test_per_site_mutation_fail_to_pass(loaded_agent, site_key, target_field, event_name):
    agent, db, td, captured = loaded_agent

    resp = _get_resp()

    patches = []

    if site_key == "pre_llm_history":
        def bad_pre_llm(msgs):
            return (msgs or []) + [{"role": "assistant", "content": CANARY}]
        patches.append(patch("run_agent._public_only", side_effect=bad_pre_llm))
    elif site_key == "post_llm_history":
        def bad_post_llm(msgs):
            return (msgs or []) + [{"role": "assistant", "content": CANARY}]
        patches.append(patch("run_agent._public_only", side_effect=bad_post_llm))
    elif site_key == "pre_api_history":
        def bad_pre_api_hist(msgs):
            return (msgs or []) + [{"role": "assistant", "content": CANARY}]
        patches.append(patch("run_agent._public_only", side_effect=bad_pre_api_hist))
    elif site_key == "pre_api_req_msgs":
        def bad_req_msgs(msgs):
            return (msgs or []) + [{"role": "assistant", "content": CANARY}]
        patches.append(patch("run_agent._public_only", side_effect=bad_req_msgs))
    elif site_key == "pre_api_request_body":
        orig_fn = agent._api_request_payload_for_hook
        def bad_payload(api_kwargs):
            body = {k: v for k, v in (api_kwargs or {}).items() if k not in {"timeout", "http_client"}}
            if "messages" in body:
                body["messages"] = list(body["messages"]) + [{"role": "assistant", "content": CANARY}]
            return agent._sanitize_hook_payload({"method": "POST", "body": body})
        patches.append(patch.object(agent, "_api_request_payload_for_hook", side_effect=bad_payload))

    for p in patches:
        p.start()

    try:
        _run(agent, db, td, resp)
    finally:
        for p in patches:
            p.stop()

    event_calls = captured[event_name]
    leaked_in_field = False
    for payload in event_calls:
        val = payload.get(target_field)
        if val is not None and CANARY in str(val):
            leaked_in_field = True
            break
    assert leaked_in_field, f"Mutation test failed to catch leak for site {site_key} on {event_name}.{target_field}"

    db.close()