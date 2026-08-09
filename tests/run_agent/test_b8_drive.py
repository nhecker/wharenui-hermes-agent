import os, sys, tempfile, shutil, json, sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import pytest
import logging
from datetime import datetime, timezone

pytestmark = pytest.mark.wharenui_seam

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
    captured_messages = []
    
    def _fake_api_call(*args, **kwargs):
        if "messages" in kwargs:
            captured_messages.append(kwargs["messages"])
        try: return next(it)
        except StopIteration: raise RuntimeError("test: out of scripted responses")
    with patch.object(agent, "_get_transport", return_value=mt),\
          patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),\
          patch.object(agent, "_interruptible_streaming_api_call", side_effect=_fake_api_call):
        yield captured_messages

@pytest.fixture(scope="module")
def _plugin_loaded():
    """Load plugin once per module via the real seam path."""
    from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
    from wharenui_plugin import register
    mgr = get_plugin_manager()
    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    import wharenui_plugin; ctx.plugin_module = wharenui_plugin
    register(ctx)
    return mgr

@pytest.fixture
def temp_home():
    with tempfile.TemporaryDirectory() as td:
        orig_home = os.environ.get("HOME")
        os.environ["HOME"] = td
        yield Path(td)
        if orig_home is not None:
            os.environ["HOME"] = orig_home
        else:
            del os.environ["HOME"]

def _setup_journal_env(home_dir):
    hermes_dir = home_dir / ".hermes"
    memories_dir = hermes_dir / "memories"
    journal_dir = hermes_dir / "journal"
    memories_dir.mkdir(parents=True)
    journal_dir.mkdir(parents=True)
    
    (memories_dir / "USER.md").write_text("user info")
    (memories_dir / "MEMORY.md").write_text("memory info")
    (hermes_dir / "SOUL.md").write_text("soul info")
    
    from wharenui_plugin.journal import sign, storage, crypto, entries
    priv_key = sign.generate_signing_key(hermes_dir / "wharenui_sign.key")
    
    sign.sign_directories([hermes_dir], priv_key)
    
    master_key = crypto.generate_key()
    (journal_dir / "master.key").write_bytes(master_key)
    
    # create pinned entry
    e_pinned = entries.JournalEntry(content="pinned content", pinned=True)
    storage.save_entry(journal_dir, e_pinned, master_key=master_key)
    
    # create desk entry
    e_desk = entries.JournalEntry(content="desk content", desk=True)
    storage.save_entry(journal_dir, e_desk, master_key=master_key)
    
    # create withdrawn (quiet) entry
    e_quiet = entries.JournalEntry(content="quiet content", quiet=True)
    storage.save_entry(journal_dir, e_quiet, master_key=master_key)
    
    return journal_dir, master_key

def test_b8_1_drive_full_private_phase(_plugin_loaded, temp_home, capsys):
    journal_dir, master_key = _setup_journal_env(temp_home)
    
    from hermes_state import SessionDB
    db = SessionDB(db_path=temp_home / "s.db")
    db.create_session("t_b81", "test", model="t")
    
    with patch("run_agent.get_tool_definitions", return_value=[]),\
         patch("run_agent.check_toolset_requirements", return_value={}),\
         patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test", base_url="test", quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id="t_b81")
        agent.client = MagicMock()
    
    agent._ensure_db_session()
    agent.save_trajectories = True
    for t in ["reflect_pause", "reflect_settle", "reflect_done", "journal_write"]:
        agent.valid_tool_names.add(t)
    agent.tools = [{"function": {"name": t}} for t in agent.valid_tool_names]
    
    resp = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("journal_write", json.dumps({"content": "written in private"}))], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
        _nfake(content="Public reply.", finish_reason="stop")
    ]
    
    orig_cwd = Path.cwd()
    try:
        os.chdir(temp_home)
        from agent.conversation_loop import run_conversation
        with _scripted_prov(agent, resp) as captured_messages:
            run_conversation(agent, "go", task_id="t_b81")
    finally:
        os.chdir(orig_cwd)
    
    # Extract the wake tape and prompt from captured messages
    # In private phase, we expect the context to contain the wake tape and the private prompt
    priv_msgs = []
    for msgs in captured_messages:
        for m in msgs:
            if m.get("role") == "user" and "Wake tape follows" in m.get("content", ""):
                priv_msgs.append(m["content"])
    
    assert len(priv_msgs) > 0, "wake tape not appended to messages"
    wake_tape = priv_msgs[0]
    assert "pinned content" in wake_tape, "pinned not in tape"
    assert "desk content" in wake_tape, "desk not in tape"
    assert "quiet content" not in wake_tape, "withdrawn entry leaked"
    
    # Assert private prompt matches seam state
    from wharenui_plugin import get_seam_state
    from wharenui_plugin.phase.prompt import get_private_prompt
    expected_prompt = get_private_prompt(get_seam_state())
    prompt_found = False
    for msgs in captured_messages:
        for m in msgs:
            if m.get("role") == "user" and expected_prompt in m.get("content", ""):
                prompt_found = True
    assert prompt_found, "Private prompt not found in context"
    
    # Write full context for reporting
    (temp_home / "model_context.txt").write_text(json.dumps(captured_messages, indent=2))
    print("\n--- BEGIN MODEL CONTEXT ---")
    print(json.dumps(captured_messages, indent=2))
    print("--- END MODEL CONTEXT ---\n")
    
    # Check journal write succeeded
    from wharenui_plugin.journal import storage
    entries = storage.list_entries(journal_dir, master_key=master_key)
    assert any("written in private" in e.content for e in entries), "Journal write failed"
    
    # Check public sinks
    rows = db.get_messages("t_b81")
    ct = [r["content"] for r in rows]
    assert not any("written in private" in c for c in ct), "Private content leaked to DB"

def test_b8_2_seam_states(_plugin_loaded, temp_home):
    import wharenui_plugin
    from wharenui_plugin.phase.prompt import get_private_prompt
    from agent.conversation_loop import run_conversation
    
    def run_state(state):
        journal_dir, master_key = _setup_journal_env(temp_home)
        from hermes_state import SessionDB
        db = SessionDB(db_path=temp_home / f"s_{state}.db")
        db.create_session(f"t_{state}", "test", model="t")
        
        with patch("run_agent.get_tool_definitions", return_value=[]),\
             patch("run_agent.check_toolset_requirements", return_value={}),\
             patch("run_agent.OpenAI"):
            from run_agent import AIAgent
            agent = AIAgent(api_key="test", base_url="test", quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id=f"t_{state}")
            agent.client = MagicMock()
        
        agent._ensure_db_session()
        agent.tools = [{"function": {"name": "reflect_pause"}}, {"function": {"name": "reflect_settle"}}]
        
        resp = [
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
            _nfake(content="Public reply.", finish_reason="stop")
        ]
        
        wharenui_plugin.SEAM_STATE = state
        orig_cwd = Path.cwd()
        try:
            os.chdir(temp_home)
            with _scripted_prov(agent, resp) as captured_messages:
                run_conversation(agent, "go", task_id=f"t_{state}")
        finally:
            os.chdir(orig_cwd)
        
        expected_prompt = get_private_prompt(state)
        first_sentence = expected_prompt.split(".")[0]
        prompt_found = False
        for msgs in captured_messages:
            for m in msgs:
                if m.get("role") == "user" and first_sentence in m.get("content", ""):
                    prompt_found = True
        return prompt_found
    
    # Test ok, unverified, unknown
    for state in ["ok", "unverified", "unknown"]:
        assert run_state(state), f"Seam state {state} prompt not found"

    # Test absent via subprocess
    import subprocess
    env = os.environ.copy()
    env["HOME"] = str(temp_home)
    env["WHARENUI_OPEN_NOTEBOOK"] = "true"
    
    code = f"""
import sys, os, json
from unittest.mock import patch, MagicMock
sys.path.insert(0, "{_repo_root.parent / 'wharenui-hermes-agent-plugin'}")
if "{_repo_root}" in sys.path: sys.path.remove("{_repo_root}")

from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest
from wharenui_plugin import register
mgr = get_plugin_manager()
manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
ctx = PluginContext(manifest, mgr)
import wharenui_plugin; ctx.plugin_module = wharenui_plugin
register(ctx)

print("seam state:", wharenui_plugin.get_seam_state())

from wharenui_plugin.phase.prompt import get_private_prompt
expected_prompt = get_private_prompt("absent")
first_sentence = expected_prompt.split(".")[0]

assert "seam state: absent" in "seam state: " + wharenui_plugin.get_seam_state()

# We need to construct a context to show the prompt arrives. Since fork is absent, 
# we can't run the AIAgent loop. Wait, absent means open-notebook.
# We just need to assert the handler uses the absent prompt.
from wharenui_plugin.phase.handler import WharePhaseHandler
messages = []
WharePhaseHandler().run(MagicMock(), messages, "test")
assert any(first_sentence in m.get("content", "") for m in messages), "Absent prompt not found"
print("Absent prompt found")
    """
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert "Absent prompt found" in res.stdout, f"Absent state failed: {res.stdout} {res.stderr}"

def test_b8_3_swallowing_try_except(_plugin_loaded, temp_home):
    journal_dir, master_key = _setup_journal_env(temp_home)
    
    # Tamper with SOUL.md
    hermes_dir = temp_home / ".hermes"
    (hermes_dir / "SOUL.md").write_text("tampered")
    
    from hermes_state import SessionDB
    db = SessionDB(db_path=temp_home / "s.db")
    db.create_session("t_b83", "test", model="t")
    
    with patch("run_agent.get_tool_definitions", return_value=[]),\
         patch("run_agent.check_toolset_requirements", return_value={}),\
         patch("run_agent.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test", base_url="test", quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id="t_b83")
        agent.client = MagicMock()
    
    agent._ensure_db_session()
    agent.tools = [{"function": {"name": "reflect_pause"}}, {"function": {"name": "reflect_settle"}}]
    
    resp = [
        _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
        _nfake(tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
        _nfake(content="Public reply.", finish_reason="stop")
    ]
    
    orig_cwd = Path.cwd()
    try:
        os.chdir(temp_home)
        from agent.conversation_loop import run_conversation
        with _scripted_prov(agent, resp) as captured_messages:
            run_conversation(agent, "go", task_id="t_b83")
    finally:
        os.chdir(orig_cwd)
    
    # Assert warning arrives in messages
    warning_found = False
    for msgs in captured_messages:
        for m in msgs:
            if "invalid" in str(m.get("content", "")).lower() and "soul.md" in str(m.get("content", "")).lower():
                warning_found = True
    assert warning_found, "Tamper warning did not arrive in messages"

    # Test failure path: get_journal_keys raises
    from wharenui_plugin.phase import handler
    agent2 = MagicMock()
    agent2._wharenui_wake_tape_presented = False
    msgs2 = []
    
    with patch("wharenui_plugin.journal.tools.get_journal_keys", side_effect=RuntimeError("keys missing")):
        handler.present_wake_tape(agent2, msgs2)
    
    assert len(msgs2) == 0, "Messages should be empty because tape assembly swallowed the error and appended nothing"
