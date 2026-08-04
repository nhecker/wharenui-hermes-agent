"""WP2c regression tests: trajectory and real SessionDB isolation."""

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
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def session_db_harness():
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


@pytest.mark.parametrize("completed", [True, False])
def test_trajectory_excludes_private_messages(tmp_path, completed):
    from agent.trajectory import save_trajectory
    from run_agent import AIAgent, _PHASE_PRIVATE_MARKER

    path = tmp_path / ("success.jsonl" if completed else "failed.jsonl")
    agent = AIAgent.__new__(AIAgent)
    agent.save_trajectories = True
    agent.model = "test/model"
    agent._convert_to_trajectory_format = lambda messages, query, done: messages
    messages = [
        {"role": "user", "content": "public"},
        {"role": "assistant", "content": "private-canary", _PHASE_PRIVATE_MARKER: True},
    ]
    with patch(
        "run_agent._save_trajectory_to_file",
        side_effect=lambda t, m, c: save_trajectory(t, m, c, str(path)),
    ):
        agent._save_trajectory(messages, "q", completed)

    record = json.loads(path.read_text())
    assert record["conversations"] == [{"role": "user", "content": "public"}]
    assert "private-canary" not in path.read_text()


def _make_real_agent(db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent


def test_real_session_db_skips_private_marker(tmp_path):
    from hermes_state import SessionDB
    from run_agent import _PHASE_PRIVATE_MARKER

    session_id = "wp2c-layer2"
    db = SessionDB(db_path=Path(tmp_path) / "session.db")
    db.create_session(session_id, "test", model="test/model")
    try:
        agent = _make_real_agent(db, session_id)
        public = {"role": "user", "content": "public tail"}
        private = {
            "role": "assistant",
            "content": "WHARE-CANARY-7f3a9b2e",
            _PHASE_PRIVATE_MARKER: True,
        }
        agent._flush_messages_to_session_db([public], [])
        agent._phase = "private"
        agent._flush_messages_to_session_db([public, private], [])
        agent._phase = "public"
        agent._flush_messages_to_session_db(
            [public, private, {"role": "assistant", "content": "public reply"}], []
        )

        contents = [row["content"] for row in db.get_messages(session_id)]
        assert "public tail" in contents
        assert "public reply" in contents
        assert "WHARE-CANARY-7f3a9b2e" not in contents
        fts = db._conn.execute(
            "SELECT * FROM messages_fts WHERE messages_fts MATCH 'WHARE*'"
        ).fetchall()
        assert not fts
    finally:
        db.close()


def test_private_and_closing_private_flush_guards(tmp_path):
    from run_agent import AIAgent
    from unittest.mock import MagicMock

    for phase in ("private", "closing_private"):
        agent = AIAgent.__new__(AIAgent)
        agent._persist_disabled = False
        agent._phase = phase
        agent._session_db = MagicMock()
        agent._session_persist_lock = None
        agent._flush_messages_to_session_db([{"role": "assistant", "content": "x"}])
        assert agent._session_db.insert_message.call_count == 0


def test_trajectory_marker_is_generic():
    from run_agent import _PHASE_PRIVATE_MARKER
    assert _PHASE_PRIVATE_MARKER == "_phase_private"
    assert "wharenui" not in _PHASE_PRIVATE_MARKER
    assert "journal" not in _PHASE_PRIVATE_MARKER


if __name__ == "__main__":
    print("use pytest")
    raise SystemExit(0)
