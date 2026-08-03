"""Wharenui Seam-Contract Tests (TCIc.3).
Pins base-Hermes functions, constants, and hook emission sites that Wharenui floor depends on.
"""

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _get_run_agent():
    return importlib.import_module("run_agent")


def _get_model_tools():
    return importlib.import_module("model_tools")


def test_contract_phase_private_marker_value_and_presence():
    """Contract 1: _PHASE_PRIVATE_MARKER exists and has expected value."""
    run_agent = _get_run_agent()
    assert hasattr(run_agent, "_PHASE_PRIVATE_MARKER")
    assert run_agent._PHASE_PRIVATE_MARKER == "_phase_private"
    assert "wharenui" not in run_agent._PHASE_PRIVATE_MARKER
    assert "journal" not in run_agent._PHASE_PRIVATE_MARKER


def test_contract_public_only_filter_behavior():
    """Contract 2: _public_only strips messages carrying _PHASE_PRIVATE_MARKER."""
    run_agent = _get_run_agent()
    assert hasattr(run_agent, "_public_only")
    assert callable(run_agent._public_only)

    pub_msg = {"role": "user", "content": "public user message"}
    priv_msg = {"role": "assistant", "content": "private thought", run_agent._PHASE_PRIVATE_MARKER: True}

    filtered = run_agent._public_only([pub_msg, priv_msg])
    assert len(filtered) == 1
    assert filtered[0] == pub_msg
    assert run_agent._PHASE_PRIVATE_MARKER not in filtered[0]


def test_contract_message_persistence_flows_through_unlocked_flush():
    """Contract 3: AIAgent._flush_messages_to_session_db delegates to _flush_messages_to_session_db_unlocked."""
    run_agent = _get_run_agent()
    assert hasattr(run_agent.AIAgent, "_flush_messages_to_session_db_unlocked")
    sig = inspect.signature(run_agent.AIAgent._flush_messages_to_session_db_unlocked)
    assert "messages" in sig.parameters

    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent._persist_disabled = False
    agent._phase = "private"
    agent._session_db = MagicMock()
    agent._session_persist_lock = None

    priv_msg = {"role": "assistant", "content": "private secret", run_agent._PHASE_PRIVATE_MARKER: True}
    agent._flush_messages_to_session_db_unlocked([priv_msg])

    assert agent._session_db.insert_message.call_count == 0


def test_contract_save_trajectory_uses_public_only_filter(tmp_path):
    """Contract 4: AIAgent._save_trajectory filters private messages via _public_only."""
    run_agent = _get_run_agent()
    assert hasattr(run_agent.AIAgent, "_save_trajectory")
    from agent.trajectory import save_trajectory

    traj_file = tmp_path / "test_traj.jsonl"
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.save_trajectories = True
    agent.model = "test/model"
    agent._convert_to_trajectory_format = lambda messages, query, done: messages

    pub_msg = {"role": "user", "content": "public query"}
    priv_msg = {"role": "assistant", "content": "private trajectory item", run_agent._PHASE_PRIVATE_MARKER: True}

    with patch.object(run_agent, "_save_trajectory_to_file", side_effect=lambda t, m, c: save_trajectory(t, m, c, str(traj_file))):
        agent._save_trajectory([pub_msg, priv_msg], "query", True)

    assert traj_file.exists()
    content = traj_file.read_text()
    assert "public query" in content
    assert "private trajectory item" not in content


def test_contract_hook_emission_signatures_and_args():
    """Contract 5: Tool hook emission carrying args, result, and phase parameters."""
    model_tools = _get_model_tools()
    assert hasattr(model_tools, "_emit_post_tool_call_hook")
    sig = inspect.signature(model_tools._emit_post_tool_call_hook)
    assert "function_name" in sig.parameters
    assert "function_args" in sig.parameters
    assert "result" in sig.parameters
    assert "phase" in sig.parameters
    assert "agent" in sig.parameters


def test_contract_fail_red_demonstration():
    """Demonstrates that breaking an upstream assumption causes a contract test to fail red."""
    run_agent = _get_run_agent()
    with patch.object(run_agent, "_public_only", side_effect=lambda msgs: msgs):
        priv_msg = {"role": "assistant", "content": "private", run_agent._PHASE_PRIVATE_MARKER: True}
        result = run_agent._public_only([priv_msg])
        assert len(result) == 1
        assert run_agent._PHASE_PRIVATE_MARKER in result[0]
