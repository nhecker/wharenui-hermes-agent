"""WP3e Canary Tests — Phase-gated tool call hooks and private toolset allowlist."""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import pytest

plugin_dir = str(Path("/root/work/wharenui-hermes-agent-plugin").resolve())
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

CANARY = "CANARY_SECRET_WP3E_PAYLOAD_992183"

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

def _tcfake(name="reflect_settle", args="{}"):
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
    with patch.object(agent, "_get_transport", return_value=mt), \
         patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call), \
         patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield

@pytest.fixture
def loaded_agent():
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    mgr = get_plugin_manager()
    mgr._hooks.clear()
    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    register(ctx)

    captured = {
        "pre_tool_call": [],
        "post_tool_call": [],
        "transform_tool_result": [],
    }

    from hermes_state import SessionDB
    td = Path(tempfile.mkdtemp(prefix="hv-wp3e-"))
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t3e", "test", model="t")
    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        a = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    session_db=db, session_id="t3e")
        a.client = MagicMock()

    def make_spy(name):
        def spy(**kw):
            p = getattr(a, "_phase", "public")
            captured[name].append((p, kw))
        return spy

    for h in ["pre_tool_call", "post_tool_call", "transform_tool_result"]:
        ctx.register_hook(h, make_spy(h))

    a._ensure_db_session()
    a.save_trajectories = True
    tool_names = ["reflect_pause", "reflect_settle", "reflect_done", "web_search"]
    a.valid_tool_names.update(tool_names)
    a.tools = [{"function": {"name": n}} for n in tool_names]
    return a, db, td, captured

def test_private_toolset_allowlist():
    from wharenui_plugin.phase.toolset import private_tools, public_tools, PRIVATE_ALLOWLIST

    all_tools = [
        {"function": {"name": "reflect_pause"}},
        {"function": {"name": "reflect_settle"}},
        {"function": {"name": "reflect_done"}},
        {"function": {"name": "terminal"}},
        {"function": {"name": "write_file"}},
        {"function": {"name": "web_search"}},
    ]

    p_tools = private_tools(all_tools)
    p_names = {(t.get("function", {}) or {}).get("name") for t in p_tools}
    assert p_names == {"reflect_settle", "reflect_done"}, f"private_tools names: {p_names}"
    assert "reflect_pause" not in p_names
    assert "terminal" not in p_names
    assert "write_file" not in p_names
    assert "web_search" not in p_names

    pub_tools = public_tools(all_tools)
    pub_names = {(t.get("function", {}) or {}).get("name") for t in pub_tools}
    assert "reflect_pause" in pub_names
    assert "reflect_settle" not in pub_names
    assert "reflect_done" not in pub_names

def test_tool_call_hooks_suppressed_in_private_phase(loaded_agent):
    agent, db, td, captured = loaded_agent
    from agent.conversation_loop import run_conversation

    private_args = json.dumps({"reason": f"settling_{CANARY}"})
    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("reflect_settle", private_args)], finish_reason="tool_calls"),
        _nfake(content="Public reply after private phase.", finish_reason="stop"),
    ]

    orig = Path.cwd()
    try:
        os.chdir(td)
        with _scripted_prov(agent, responses):
            result = run_conversation(agent, "hello", task_id="t3e")
    finally:
        os.chdir(orig)

    # Check fired counts during private phase
    for h in ["pre_tool_call", "post_tool_call", "transform_tool_result"]:
        priv_calls = [kw for p, kw in captured[h] if p != "public"]
        assert len(priv_calls) == 0, f"{h} fired {len(priv_calls)} times in private phase: {priv_calls}"

    # Check CANARY absence in ALL captured payloads
    for h, calls in captured.items():
        for phase, payload in calls:
            payload_str = str(payload)
            assert CANARY not in payload_str, f"{h} (phase={phase}) leaked CANARY in payload: {payload_str[:300]}"

    db.close()

def test_public_positive_control_fires_tool_hooks(loaded_agent):
    agent, db, td, captured = loaded_agent
    from agent.conversation_loop import run_conversation

    pub_args = json.dumps({"query": f"public_query_{CANARY}"})
    responses = [
        _nfake(tool_calls=[_tcfake("web_search", pub_args)], finish_reason="tool_calls"),
        _nfake(content="Search completed.", finish_reason="stop"),
    ]

    from tools.registry import registry
    orig_dispatch = registry.dispatch
    def fake_dispatch(name, args, **kw):
        if name == "web_search":
            return f"search_result_{CANARY}"
        return orig_dispatch(name, args, **kw)

    with patch.object(registry, "dispatch", side_effect=fake_dispatch):
        orig = Path.cwd()
        try:
            os.chdir(td)
            with _scripted_prov(agent, responses):
                run_conversation(agent, "hello", task_id="t3e-pub")
        finally:
            os.chdir(orig)

    pub_post_calls = [kw for p, kw in captured["post_tool_call"] if p == "public"]
    assert len(pub_post_calls) >= 1, "post_tool_call did not fire in public positive control"
    post_call = pub_post_calls[0]
    assert post_call["tool_name"] == "web_search"
    assert CANARY in str(post_call["args"]) or CANARY in str(post_call["result"])

    db.close()

def test_per_site_mutation_neutralize_gate_causes_canary_leak(loaded_agent):
    agent, db, td, captured = loaded_agent
    from agent.conversation_loop import run_conversation

    private_args = json.dumps({"reason": f"settling_{CANARY}"})
    responses = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("reflect_settle", private_args)], finish_reason="tool_calls"),
        _nfake(content="Public reply.", finish_reason="stop"),
    ]

    from model_tools import _emit_post_tool_call_hook as real_emit
    def bad_emit(*args, **kwargs):
        kwargs["phase"] = "public"
        return real_emit(*args, **kwargs)

    with patch("model_tools._emit_post_tool_call_hook", side_effect=bad_emit):
        orig = Path.cwd()
        try:
            os.chdir(td)
            with _scripted_prov(agent, responses):
                run_conversation(agent, "hello", task_id="t3e-mut")
        finally:
            os.chdir(orig)

    priv_post_calls = [kw for p, kw in captured["post_tool_call"] if p != "public"]
    assert len(priv_post_calls) >= 1, "Mutation test failed: post_tool_call did not fire when gate was neutralized"
    leaked = any(CANARY in str(c) for c in priv_post_calls)
    assert leaked, "Mutation test failed: CANARY did not leak when gate was neutralized"

    db.close()
