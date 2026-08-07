import pytest
pytestmark = [pytest.mark.wharenui_seam, pytest.mark.xdist_group("journal_group")]

"""
Work Package 4 — Cross-session persistence & encryption/signature verification (T4.5).
Session A writes private journal entry -> Session B (fresh process) reads, verifies encryption & signature, and tests tombstone withdrawal.
"""

import os
import sys
import json
from pathlib import Path

# Self-bootstrap sys.path
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

from wharenui_plugin.journal import tools as jtools, crypto, sign

CANARY_PERSIST_JOURNAL = "CANARY_JOURNAL_PERSISTENCE_TEXT_554433"


def _parse_res(res):
    if isinstance(res, str):
        return json.loads(res)
    return res


class SessionAgent:
    def __init__(self, session_id="session-a", _phase="private"):
        self._phase = _phase
        self.model = "gpt-4o"
        self.provider = "openai"
        self.session_id = session_id
        self.runtime_id = "rt-999"


def test_cross_session_persistence_encrypted_signed(tmp_path):
    """T4.5 — Private write in Session A, read in fresh Session B.

    Asserts:
    1. Content survives A -> B.
    2. Raw on-disk bytes do NOT contain CANARY_PERSIST_JOURNAL plaintext (encrypted).
    3. Ed25519 signature verifies.
    4. Tombstone/withdraw hides from read while signed record remains on disk.
    """
    store_dir = tmp_path / "journal_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic key & signing key in throwaway store
    master_key = crypto.generate_key(store_dir / "journal.key")
    signing_key = sign.generate_signing_key(store_dir / "signing.key")
    verifying_key = signing_key.public_key()

    # --- SESSION A ---
    jtools.set_journal_config(store_dir, master_key=master_key)
    session_a_agent = SessionAgent(session_id="session-A")

    append_res = _parse_res(jtools.handle_journal_append({
        "content": CANARY_PERSIST_JOURNAL,
        "slug": "persistence-canary",
        "description": "Cross-session persistence test entry",
        "tags": ["persistence", "test"],
    }, agent=session_a_agent))

    handle = append_res["handle"]
    written_filename = append_res["filename"]
    assert handle.startswith("h_")

    # Verify on-disk file bytes are ENCRYPTED (no plaintext canary in raw file)
    on_disk_path = store_dir / written_filename
    assert on_disk_path.exists(), f"Written entry file {written_filename} does not exist!"
    raw_file_bytes = on_disk_path.read_bytes()

    assert CANARY_PERSIST_JOURNAL.encode("utf-8") not in raw_file_bytes, (
        "CRITICAL PRIVACY VIOLATION: Plaintext canary found in raw on-disk file bytes!"
    )
    assert crypto.is_encrypted(raw_file_bytes), "On-disk file bytes are not Fernet encrypted!"

    # Verify signature file exists and verifies
    sig_path = store_dir / f"{written_filename}.sig"
    assert sig_path.exists(), f"Signature file {sig_path.name} missing!"
    assert sign.verify_entry(on_disk_path, verifying_key) is True, "Ed25519 signature verification failed!"

    # Clean up Session A config in-memory state
    jtools.set_journal_config(None, None)

    # --- SESSION B (Fresh Agent / Process Simulation) ---
    jtools.set_journal_config(store_dir, master_key=master_key)
    session_b_agent = SessionAgent(session_id="session-B")

    # Session B reads entry written by Session A
    read_data = _parse_res(jtools.handle_journal_read({"handle": handle}, agent=session_b_agent))

    # Assert (a) Content survives A -> B
    assert read_data["content"] == CANARY_PERSIST_JOURNAL
    assert read_data["description"] == "Cross-session persistence test entry"
    # Assert (c) Signature verifies in Session B
    assert read_data["signature_valid"] is True

    # Assert Session B search finds handle
    search_hits = _parse_res(jtools.handle_journal_search({"query": "persistence"}, agent=session_b_agent))
    assert any(h["handle"] == handle for h in search_hits)

    # Assert (d) Tombstone/withdraw in Session B hides from read while signed record remains
    withdraw_res = _parse_res(jtools.handle_journal_withdraw({"handle": handle, "reason": "withdrawing canary"}, agent=session_b_agent))
    assert withdraw_res["status"] == "success"

    # Subsequent read in Session B raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        jtools.handle_journal_read({"handle": handle}, agent=session_b_agent)

    # After T1 (WP-TIDY), withdrawn entries are renamed to <token>.tomb.md.
    # The sig file stays at <token>.md.sig (canonical — signature_path_for uses the leading token).
    # Check both the renamed file and the unmoved original (legacy tolerance).
    token = on_disk_path.stem  # <token> from <token>.md
    tomb_path = on_disk_path.parent / f"{token}.tomb.md"
    renamed = tomb_path.exists() and not on_disk_path.exists()
    stayed = on_disk_path.exists()
    assert renamed or stayed, (
        f"Original entry file was unlinked instead of tombstoned!\n"
        f"  original: {on_disk_path} exists={on_disk_path.exists()}\n"
        f"  renamed:  {tomb_path} exists={tomb_path.exists()}"
    )
    assert sig_path.exists(), "Original signature file was unlinked!"
    verify_target = tomb_path if renamed else on_disk_path
    assert sign.verify_entry(verify_target, verifying_key) is True, "Original signature compromised after withdrawal!"
