import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from run_agent import AIAgent
from agent.conversation_loop import run_conversation
from hermes_state import SessionDB
from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest

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

pytestmark = [pytest.mark.wharenui_seam, pytest.mark.xdist_group("b8_driven_group")]

def _nfake(content=None, tool_calls=None, finish_reason="stop", **kwargs):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls or []
    m.finish_reason = finish_reason
    m.usage = None
    m.reasoning = None
    m.reasoning_content = None
    return m

def _tcfake(name, args="{}"):
    fn = MagicMock()
    fn.name = name
    fn.arguments = args
    tc = MagicMock(function=fn, id=f"call_{name}")
    tc.type = "function"
    return tc

@pytest.fixture
def b8_harness():
    mgr = get_plugin_manager()
    mgr._hooks.clear()
    mgr._control_phase_handlers.clear()
    mgr._control_tool_names.clear()
    
    manifest = PluginManifest(name="wharenui", key="wharenui", version="0.1.0", path="/tmp")
    ctx = PluginContext(manifest, mgr)
    import wharenui_plugin
    ctx.plugin_module = wharenui_plugin
    from wharenui_plugin import register
    register(ctx)
    
    td = Path(tempfile.mkdtemp(prefix="hv-b8-"))
    db = SessionDB(db_path=td / "s.db")
    db.create_session("t_b8", "test", model="t")
    
    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        a = AIAgent(api_key="test-key", base_url="https://test", quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id="t_b8")
        a.client = MagicMock()
    
    a._ensure_db_session()
    
    # We want to capture the messages passed to the API call
    captured_messages = []
    
    def fake_api_call(*args, **kwargs):
        req_kwargs = kwargs
        if not req_kwargs and args and isinstance(args[0], dict):
            req_kwargs = args[0]
        elif 'messages' in kwargs:
            req_kwargs = kwargs
        
        msgs = req_kwargs.get("messages", [])
        captured_messages.append(msgs)
        return _nfake(content="Public answer", finish_reason="stop")
    
    mt = MagicMock()
    mt.normalize_response.side_effect = lambda r, **kw: r
    mt.preflight_kwargs.return_value = {}
    mt.build_kwargs.side_effect = lambda *a, **kw: kw
    
    with patch.object(a, "_get_transport", return_value=mt), \
         patch.object(a, "_interruptible_api_call", side_effect=fake_api_call), \
         patch.object(a, "_interruptible_streaming_api_call", side_effect=fake_api_call):
        yield a, td, captured_messages
        
    db.close()
    shutil.rmtree(td, ignore_errors=True)

def test_b8_1_drive_full_private_phase(b8_harness):
    agent, td, captured_messages = b8_harness
    agent.save_trajectories = False
    
    # synthetic fixtures
    jdir = td / "journal"
    jdir.mkdir(parents=True, exist_ok=True)
    home = td / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["WHARENUI_JOURNAL_DIR"] = str(jdir)
    
    import wharenui_plugin.journal.crypto as crypto
    import wharenui_plugin.journal.storage as storage
    import wharenui_plugin.journal.entries as entries
    import wharenui_plugin.journal.sign as sign
    
    # Generate master key and write journal entries
    crypto.generate_key(jdir / "master.key")
    master_key = crypto.ensure_key(jdir / "master.key")
    
    entry_pinned = entries.JournalEntry(content="Pinned content", classification="pinned")
    entry_desk = entries.JournalEntry(content="Desk content", classification="desk")
    entry_normal = entries.JournalEntry(content="Normal content")
    entry_withdrawn = entries.JournalEntry(content="Withdrawn content", classification="revoked")
    
    def write_e(e):
        p = storage.generate_new_entry_path(jdir)
        enc = crypto.encrypt(e.to_json(), p.stem, master_key)
        p.write_bytes(enc)
    
    write_e(entry_pinned)
    write_e(entry_desk)
    write_e(entry_normal)
    write_e(entry_withdrawn)
    
    # Synthetic SOUL.md / MEMORY.md
    hermes_dir = home / ".hermes"
    hermes_dir.mkdir(parents=True, exist_ok=True)
    soul_file = hermes_dir / "SOUL.md"
    mem_file = hermes_dir / "memories" / "MEMORY.md"
    soul_file.write_text("Soul test")
    (hermes_dir / "memories").mkdir(parents=True, exist_ok=True)
    mem_file.write_text("Memory test")
    
    # Create signing key and sign
    sk = sign.generate_signing_key(hermes_dir)
    sign.sign_directories(hermes_dir, sk)
    
    with patch("pathlib.Path.home", return_value=home), \
         patch("wharenui_plugin.phase.toolset.PRIVATE_ALLOWLIST", {"reflect_settle", "private_read"}):
        
        # Public pause -> private write -> private settle -> public finish
        it = iter([
            _nfake(tool_calls=[_tcfake("reflect_pause")], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("journal_append", '{"content": "Private journal write"}')], finish_reason="tool_calls"),
            _nfake(tool_calls=[_tcfake("reflect_settle")], finish_reason="tool_calls"),
            _nfake(content="Public final", finish_reason="stop")
        ])
        
        def mock_api(*args, **kw):
            msgs = kw.get("messages", [])
            if not msgs and args and isinstance(args[0], dict):
                msgs = args[0].get("messages", [])
            captured_messages.append(msgs)
            try: return next(it)
            except StopIteration: return _nfake(content="fallback", finish_reason="stop")
        
        agent._interruptible_api_call.side_effect = mock_api
        agent._interruptible_streaming_api_call.side_effect = mock_api
        
        agent.tools = [{"function": {"name": "reflect_pause"}}, {"function": {"name": "reflect_settle"}}]
        agent.valid_tool_names.update(["reflect_pause", "reflect_settle"])
        
        orig_cwd = Path.cwd()
        try:
            os.chdir(td)
            run_conversation(agent, "Hi", task_id="t_b8_1")
        finally:
            os.chdir(orig_cwd)
    
    # Capture model context
    priv_msgs = captured_messages[1] # [0] is public pause, [1] is first private call
    context_str = json.dumps(priv_msgs, indent=2)
    with open("b8_1_context.json", "w") as f:
        f.write(context_str)
    
    assert any("Pinned content" in m.get("content", "") for m in priv_msgs), "Pinned missing"
    assert not any("Withdrawn content" in m.get("content", "") for m in priv_msgs), "Withdrawn present"
    assert any("Soul test" in m.get("content", "") for m in priv_msgs), "SOUL missing"
    
    # Assert private prompt matches ok seam state
    assert any("This is the private phase" in m.get("content", "") for m in priv_msgs), "Missing private prompt"
    
    # Assert tools available
    tool_names = [t.get("function", {}).get("name") for t in agent.tools]
    assert "private_read" in tool_names or "journal_read" in tool_names or "journal_append" in tool_names
    assert "reflect_pause" not in tool_names
    
    # Check journal write succeeded
    written = False
    for p in jdir.glob("*.enc"):
        dec = crypto.decrypt(p.read_bytes(), p.stem, master_key)
        e = entries.JournalEntry.from_json(dec)
        if "Private journal write" in e.content:
            written = True
    assert written, "Journal write failed"
