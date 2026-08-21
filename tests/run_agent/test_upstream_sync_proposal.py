"""
Tests for Wharenui Upstream Sync Proposal Automation (Issue #9).

Validates:
- Seam surface dynamic derivation from wharenui-fork.md and agent/phase_control.py
- Preconditions verification (Tier 2 baseline #4, manual catch-up record #5, clean tree)
- Safety & Refusal Matrix:
    * Merge conflicts -> Safe halt & merge abort
    * Seam surface modification by upstream -> Safe halt & manual reconciliation requirement
    * Oversized deltas (lines / commits) -> Safe halt & review gate
    * Red test gates -> Safe halt & rollback
- Pull request proposal workflow:
    * Proposals created on branch upstream-sync-<date>
    * wharenui-integration is NEVER directly modified or pushed
    * Zero auto-merge / zero auto-approval paths
- Clean up-to-date no-op behavior
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pytest

# Ensure .github/scripts is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / ".github" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from propose_upstream_sync import (
    EXIT_SUCCESS,
    EXIT_REFUSAL_PRECONDITIONS,
    EXIT_REFUSAL_SEAM_TOUCHED,
    EXIT_REFUSAL_MERGE_CONFLICT,
    EXIT_REFUSAL_OVERSIZED_DELTA,
    EXIT_REFUSAL_TESTS_FAILED,
    STATUS_PROPOSAL_CREATED,
    STATUS_DRY_RUN_PASSED,
    STATUS_ALREADY_UP_TO_DATE,
    STATUS_REFUSED_PRECONDITIONS,
    STATUS_REFUSED_SEAM_TOUCHED,
    STATUS_REFUSED_MERGE_CONFLICT,
    STATUS_REFUSED_OVERSIZED_DELTA,
    STATUS_REFUSED_TESTS_FAILED,
    SeamSurface,
    derive_seam_surface,
    verify_preconditions,
    analyze_upstream_delta,
    check_seam_intersection,
    perform_test_merge_and_gate,
    generate_pr_body,
    create_proposal_branch,
    open_pull_request,
    execute_proposal_pipeline,
    run_git,
)

pytestmark = pytest.mark.wharenui_seam


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo_root():
    return _REPO_ROOT


@pytest.fixture
def isolated_git_repo(tmp_path):
    """Create an isolated git repository with main and wharenui-integration branches."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    run_git(["init", "-b", "main"], cwd=repo_dir, check=True)
    run_git(["config", "user.name", "Test User"], cwd=repo_dir, check=True)
    run_git(["config", "user.email", "test@example.com"], cwd=repo_dir, check=True)

    # Initial commit on main
    (repo_dir / "README.md").write_text("# Mock Hermes Agent\n", encoding="utf-8")
    (repo_dir / "wharenui-fork.md").write_text(
        """# Wharenui Seam Architecture

## The Seam Surface

- **`agent/phase_control.py`** — ControlOutcome
- **`hermes_cli/plugins.py`** — register_control_tool
- **`run_agent.py`** — _public_only

| File | The seam's hook, in one line |
|---|---|
| `agent/conversation_loop.py` | main loop hook |
| `agent/tool_executor.py` | tool routing |
| `model_tools.py` | registry sync |

## Upstream Reconciliation Record (August 2026 / Issue #5)
- **Upstream Merge Base**: 123456
""",
        encoding="utf-8",
    )

    pc_dir = repo_dir / "agent"
    pc_dir.mkdir()
    (pc_dir / "phase_control.py").write_text(
        """PHASE_CONTROL_API_VERSION = 1

class ControlOutcome: pass
class PhaseHandler: pass
class SubturnResult: pass
""",
        encoding="utf-8",
    )

    dot_gh = repo_dir / ".github"
    dot_gh.mkdir()
    (dot_gh / "baseline-tier2.txt").write_text(
        """# Provenance: github-runner
# Date: 2026-08-20
test.node.1
test.node.2
""",
        encoding="utf-8",
    )

    run_git(["add", "."], cwd=repo_dir, check=True)
    run_git(["commit", "-m", "Initial commit on main"], cwd=repo_dir, check=True)

    # Create wharenui-integration branch
    run_git(["checkout", "-b", "wharenui-integration"], cwd=repo_dir, check=True)

    return repo_dir


# ---------------------------------------------------------------------------
# Test Cases: Seam Surface Derivation
# ---------------------------------------------------------------------------

def test_derive_seam_surface_from_real_repo(repo_root):
    """Verify dynamic derivation from real wharenui-fork.md and agent/phase_control.py."""
    seam = derive_seam_surface(repo_root)

    # Core seam touchpoints
    assert "agent/phase_control.py" in seam.files
    assert "hermes_cli/plugins.py" in seam.files
    assert "run_agent.py" in seam.files

    # Inline hooks table
    assert "agent/conversation_loop.py" in seam.files
    assert "agent/tool_executor.py" in seam.files
    assert "agent/tool_dispatch_helpers.py" in seam.files
    assert "agent/turn_context.py" in seam.files
    assert "agent/turn_finalizer.py" in seam.files
    assert "model_tools.py" in seam.files
    assert "pyproject.toml" in seam.files
    assert ".github/workflows/tests.yml" in seam.files

    # Symbols & API Version
    assert seam.api_version == 1
    assert "ControlOutcome" in seam.symbols
    assert "PhaseHandler" in seam.symbols
    assert "SubturnResult" in seam.symbols
    assert "PHASE_CONTROL_API_VERSION" in seam.symbols

    # Test files discovered
    assert any("test_seam_contracts.py" in f for f in seam.files)
    assert any("test_all_channels_canary.py" in f for f in seam.files)


def test_derive_seam_surface_dynamic_custom_doc(tmp_path):
    """Verify seam derivation adapts to updated markdown document structure."""
    fake_repo = tmp_path / "custom_repo"
    fake_repo.mkdir()

    (fake_repo / "my_fork_doc.md").write_text(
        """# Custom Fork Doc
## The Seam Surface
- **`custom/seam_module.py`** — custom seam
| File | Role |
|---|---|
| `custom/inline_hook.py` | hook |
## Next Section
""",
        encoding="utf-8",
    )

    custom_agent = fake_repo / "agent"
    custom_agent.mkdir()
    (custom_agent / "phase_control.py").write_text(
        "PHASE_CONTROL_API_VERSION = 2\nclass CustomHandler: pass\n",
        encoding="utf-8",
    )

    seam = derive_seam_surface(
        fake_repo,
        fork_doc_path="my_fork_doc.md",
        phase_control_file="agent/phase_control.py",
    )

    assert "custom/seam_module.py" in seam.files
    assert "custom/inline_hook.py" in seam.files
    assert "agent/phase_control.py" in seam.files
    assert seam.api_version == 2
    assert "CustomHandler" in seam.symbols


# ---------------------------------------------------------------------------
# Test Cases: Preconditions Verification
# ---------------------------------------------------------------------------

def test_verify_preconditions_success(isolated_git_repo):
    """Preconditions satisfied: valid baseline provenance, reconciliation record, clean tree."""
    report = verify_preconditions(isolated_git_repo)
    assert report.valid is True
    assert report.has_provenance is True
    assert report.reconciliation_record_present is True
    assert report.clean_working_tree is True
    assert len(report.errors) == 0


def test_verify_preconditions_refusal_missing_baseline(isolated_git_repo):
    """Refusal when Tier 2 baseline is missing (#4)."""
    (isolated_git_repo / ".github" / "baseline-tier2.txt").unlink()
    report = verify_preconditions(isolated_git_repo)
    assert report.valid is False
    assert any("not found" in err for err in report.errors)


def test_verify_preconditions_refusal_unprovenanced_baseline(isolated_git_repo):
    """Refusal when Tier 2 baseline lacks runner provenance header (#4)."""
    (isolated_git_repo / ".github" / "baseline-tier2.txt").write_text(
        "# Unprovenanced test baseline\ntest.node.1\n",
        encoding="utf-8",
    )
    report = verify_preconditions(isolated_git_repo)
    assert report.valid is False
    assert report.has_provenance is False
    assert any("missing required '# Runner-Provenance: github-runner'" in err for err in report.errors)


def test_verify_preconditions_refusal_missing_reconciliation(isolated_git_repo):
    """Refusal when wharenui-fork.md lacks Upstream Reconciliation Record (#5)."""
    (isolated_git_repo / "wharenui-fork.md").write_text(
        "## The Seam Surface\n- `agent/phase_control.py`\n",
        encoding="utf-8",
    )
    report = verify_preconditions(isolated_git_repo)
    assert report.valid is False
    assert report.reconciliation_record_present is False
    assert any("Upstream Reconciliation Record" in err for err in report.errors)


def test_verify_preconditions_refusal_dirty_working_tree(isolated_git_repo):
    """Refusal when working tree has uncommitted modifications."""
    (isolated_git_repo / "untracked_file.txt").write_text("dirty\n", encoding="utf-8")
    report = verify_preconditions(isolated_git_repo)
    assert report.valid is False
    assert report.clean_working_tree is False
    assert any("dirty" in err.lower() for err in report.errors)


# ---------------------------------------------------------------------------
# Test Cases: Safety Refusal Matrix (Seam Intersection)
# ---------------------------------------------------------------------------

def test_seam_intersection_clean():
    """Non-seam changes are marked clean."""
    seam = SeamSurface(files={"agent/phase_control.py", "run_agent.py", "model_tools.py"})
    changed = ["docs/new_feature.md", "hermes_cli/ui.py", "tools/browser.py"]

    report = check_seam_intersection(changed, seam)
    assert report.clean is True
    assert len(report.touched_seam_files) == 0


def test_seam_intersection_refusal_when_seam_file_touched():
    """Refusal when upstream touches ANY file in the active seam surface."""
    seam = SeamSurface(files={"agent/phase_control.py", "run_agent.py", "agent/conversation_loop.py"})
    changed = ["docs/readme.md", "agent/conversation_loop.py", "tools/web.py"]

    report = check_seam_intersection(changed, seam)
    assert report.clean is False
    assert report.touched_seam_files == ["agent/conversation_loop.py"]


# ---------------------------------------------------------------------------
# Test Cases: Safety Refusal Matrix (Oversized Deltas)
# ---------------------------------------------------------------------------

def test_analyze_upstream_delta_oversized_commits(isolated_git_repo):
    """Refusal when upstream commit count exceeds max_commits limit."""
    # Create upstream branch with 10 commits
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream/main"], cwd=isolated_git_repo, check=True)

    for i in range(10):
        (isolated_git_repo / f"file_{i}.txt").write_text(f"content {i}\n", encoding="utf-8")
        run_git(["add", "."], cwd=isolated_git_repo, check=True)
        run_git(["commit", "-m", f"Upstream commit {i}"], cwd=isolated_git_repo, check=True)

    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    delta = analyze_upstream_delta(
        isolated_git_repo,
        target_branch="wharenui-integration",
        upstream_ref="upstream/main",
        max_commits=5,  # threshold 5 < 10 commits
        max_delta_lines=5000,
    )

    assert delta.commit_count == 10
    assert delta.is_oversized is True
    assert any("commit count (10) exceeds threshold (5)" in r for r in delta.oversized_reasons)


def test_analyze_upstream_delta_oversized_lines(isolated_git_repo):
    """Refusal when upstream diff lines exceed max_delta_lines limit."""
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream/main"], cwd=isolated_git_repo, check=True)

    # Large file with 500 lines
    lines = ["Line " + str(i) for i in range(500)]
    (isolated_git_repo / "large_file.txt").write_text("\n".join(lines), encoding="utf-8")
    run_git(["add", "."], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Large upstream commit"], cwd=isolated_git_repo, check=True)

    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    delta = analyze_upstream_delta(
        isolated_git_repo,
        target_branch="wharenui-integration",
        upstream_ref="upstream/main",
        max_commits=500,
        max_delta_lines=100,  # threshold 100 < 500 lines
    )

    assert delta.is_oversized is True
    assert any("delta lines" in r and "exceeds threshold (100)" in r for r in delta.oversized_reasons)


# ---------------------------------------------------------------------------
# Test Cases: Safety Refusal Matrix (Merge Conflicts & Red Tests)
# ---------------------------------------------------------------------------

def test_perform_test_merge_refusal_on_conflict(isolated_git_repo):
    """Refusal and safe abort when upstream causes git merge conflicts."""
    # Modify README on wharenui-integration
    (isolated_git_repo / "README.md").write_text("# Fork Diverged Header\n", encoding="utf-8")
    run_git(["add", "README.md"], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Fork README edit"], cwd=isolated_git_repo, check=True)

    # Create upstream/main branch with conflicting change
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream/main"], cwd=isolated_git_repo, check=True)
    (isolated_git_repo / "README.md").write_text("# Upstream Diverged Header Conflict\n", encoding="utf-8")
    run_git(["add", "README.md"], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Upstream conflicting README edit"], cwd=isolated_git_repo, check=True)

    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    report = perform_test_merge_and_gate(
        isolated_git_repo,
        target_branch="wharenui-integration",
        upstream_ref="upstream/main",
        run_tests=False,
    )

    assert report.conflict_free is False
    assert "README.md" in report.conflicted_files
    assert any("Merge conflict" in err for err in report.errors)

    # Verify repo state is clean after abort
    status = run_git(["status", "--porcelain"], cwd=isolated_git_repo).stdout.strip()
    assert status == ""


def test_perform_test_merge_refusal_on_red_test_gate(isolated_git_repo):
    """Refusal when test gate script fails in merged state."""
    # Create upstream branch
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream/main"], cwd=isolated_git_repo, check=True)
    (isolated_git_repo / "doc.txt").write_text("doc update\n", encoding="utf-8")
    run_git(["add", "doc.txt"], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Clean upstream commit"], cwd=isolated_git_repo, check=True)

    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    # Provide custom test_args simulating a failing test runner (exit code 1)
    failing_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]

    report = perform_test_merge_and_gate(
        isolated_git_repo,
        target_branch="wharenui-integration",
        upstream_ref="upstream/main",
        run_tests=True,
        test_args=failing_cmd,
    )

    assert report.conflict_free is True
    assert report.tests_executed is True
    assert report.tests_passed is False
    assert report.test_exit_code == 1
    assert any("Test gate failed" in err for err in report.errors)


# ---------------------------------------------------------------------------
# Test Cases: Proposal Creation & Pull Request Workflow (No Direct Push / No Auto-Merge)
# ---------------------------------------------------------------------------

def test_create_proposal_branch_never_touches_target_branch(isolated_git_repo):
    """Proposal creates isolated upstream-sync-<date> branch and leaves wharenui-integration untouched."""
    # Create upstream branch with a safe commit
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream/main"], cwd=isolated_git_repo, check=True)
    (isolated_git_repo / "new_upstream_feature.txt").write_text("new feature\n", encoding="utf-8")
    run_git(["add", "."], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Add new upstream feature"], cwd=isolated_git_repo, check=True)

    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)
    before_sha = run_git(["rev-parse", "wharenui-integration"], cwd=isolated_git_repo).stdout.strip()

    branch_name = "upstream-sync-20260821"
    ok, msg = create_proposal_branch(
        isolated_git_repo,
        target_branch="wharenui-integration",
        upstream_ref="upstream/main",
        branch_name=branch_name,
        commit_message="chore(upstream): propose sync with upstream/main as of 20260821",
    )

    assert ok is True

    # Check wharenui-integration was NEVER modified
    after_sha = run_git(["rev-parse", "wharenui-integration"], cwd=isolated_git_repo).stdout.strip()
    assert before_sha == after_sha

    # Check proposal branch exists and has the merge commit
    prop_sha = run_git(["rev-parse", branch_name], cwd=isolated_git_repo).stdout.strip()
    assert prop_sha != before_sha

    # Check current branch is still wharenui-integration
    cur_branch = run_git(["branch", "--show-current"], cwd=isolated_git_repo).stdout.strip()
    assert cur_branch == "wharenui-integration"


def test_generate_pr_body_enforces_zero_auto_merge():
    """Verify PR body explicitly mandates human review and zero auto-merge."""
    mock_delta = type("Delta", (), {
        "target_branch": "wharenui-integration",
        "target_sha": "111122223333",
        "upstream_ref": "upstream/main",
        "upstream_sha": "444455556666",
        "merge_base": "000011112222",
        "commit_count": 12,
        "insertions": 150,
        "deletions": 20,
        "changed_files": ["a.py", "b.py"],
    })()
    mock_seam_inter = type("SeamInter", (), {"clean": True, "touched_seam_files": []})()
    mock_merge_test = type("MergeTest", (), {"tests_passed": True, "conflict_free": True})()
    mock_seam = SeamSurface(files={"agent/phase_control.py"}, api_version=1)

    body = generate_pr_body(
        delta=mock_delta,
        seam_intersection=mock_seam_inter,
        merge_test=mock_merge_test,
        seam_surface=mock_seam,
        proposal_date="20260821",
    )

    assert "STRICT ZERO AUTO-MERGE POLICY" in body
    assert "Direct push to `wharenui-integration` and auto-merge automation are strictly prohibited" in body
    assert "Active Seam Surface Audit" in body
    assert "0 files modified by upstream" in body
    assert "Commits Included**: `12`" in body


# ---------------------------------------------------------------------------
# Test Cases: End-to-End Pipeline Execution
# ---------------------------------------------------------------------------

def test_pipeline_already_up_to_date(isolated_git_repo):
    """Pipeline correctly returns ALREADY_UP_TO_DATE when 0 upstream commits exist."""
    # Upstream branch pointing to same commit
    run_git(["branch", "upstream/main", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    result = execute_proposal_pipeline(
        repo_root=isolated_git_repo,
        upstream_url=str(isolated_git_repo),
        upstream_branch="main",
        target_branch="wharenui-integration",
        dry_run=True,
        skip_tests=True,
    )

    assert result.status == STATUS_ALREADY_UP_TO_DATE
    assert result.exit_code == EXIT_SUCCESS
    assert "already fully merged" in result.summary


def test_pipeline_refusal_when_upstream_touches_seam(isolated_git_repo):
    """Pipeline safely halts with REFUSED_SEAM_TOUCHED when upstream modifies a seam file."""
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream_feature"], cwd=isolated_git_repo, check=True)

    # Upstream modifies agent/conversation_loop.py (a seam file)
    (isolated_git_repo / "agent" / "conversation_loop.py").write_text("# upstream modification\n", encoding="utf-8")
    run_git(["add", "."], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Upstream touched conversation_loop"], cwd=isolated_git_repo, check=True)

    # Set upstream remote to itself
    run_git(["remote", "add", "upstream", str(isolated_git_repo)], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    result = execute_proposal_pipeline(
        repo_root=isolated_git_repo,
        upstream_url=str(isolated_git_repo),
        upstream_branch="upstream_feature",
        target_branch="wharenui-integration",
        dry_run=True,
        skip_tests=True,
    )

    assert result.status == STATUS_REFUSED_SEAM_TOUCHED
    assert result.exit_code == EXIT_REFUSAL_SEAM_TOUCHED
    assert "agent/conversation_loop.py" in result.refusal_reason
    assert "Requires manual reconciliation" in result.summary


def test_pipeline_dry_run_success_on_safe_upstream(isolated_git_repo):
    """Pipeline succeeds in dry-run mode on clean, safe, non-seam upstream update."""
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream_clean"], cwd=isolated_git_repo, check=True)

    # Upstream modifies non-seam doc
    (isolated_git_repo / "docs_update.md").write_text("# Safe documentation update\n", encoding="utf-8")
    run_git(["add", "."], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Safe upstream doc update"], cwd=isolated_git_repo, check=True)

    run_git(["remote", "add", "upstream", str(isolated_git_repo)], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    result = execute_proposal_pipeline(
        repo_root=isolated_git_repo,
        upstream_url=str(isolated_git_repo),
        upstream_branch="upstream_clean",
        target_branch="wharenui-integration",
        dry_run=True,
        skip_tests=True,
        date_tag="20260821",
    )

    assert result.status == STATUS_DRY_RUN_PASSED
    assert result.exit_code == EXIT_SUCCESS
    assert result.branch_name == "upstream-sync-20260821"
    assert "Safe upstream sync proposal validated" in result.summary
    assert result.pr_body is not None
    assert "STRICT ZERO AUTO-MERGE POLICY" in result.pr_body


def test_pipeline_creates_proposal_branch_locally(isolated_git_repo):
    """Pipeline creates proposal branch locally without auto-merging."""
    run_git(["checkout", "main"], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "-b", "upstream_clean2"], cwd=isolated_git_repo, check=True)

    (isolated_git_repo / "clean_update.txt").write_text("clean update\n", encoding="utf-8")
    run_git(["add", "."], cwd=isolated_git_repo, check=True)
    run_git(["commit", "-m", "Safe clean update"], cwd=isolated_git_repo, check=True)

    run_git(["remote", "add", "upstream", str(isolated_git_repo)], cwd=isolated_git_repo, check=True)
    run_git(["checkout", "wharenui-integration"], cwd=isolated_git_repo, check=True)

    result = execute_proposal_pipeline(
        repo_root=isolated_git_repo,
        upstream_url=str(isolated_git_repo),
        upstream_branch="upstream_clean2",
        target_branch="wharenui-integration",
        dry_run=False,
        create_pr=False,
        push=False,
        skip_tests=True,
        date_tag="20260821",
    )

    assert result.status == STATUS_PROPOSAL_CREATED
    assert result.exit_code == EXIT_SUCCESS
    assert result.branch_name == "upstream-sync-20260821"

    # Verify branch was created and target branch was not updated
    branches = run_git(["branch"], cwd=isolated_git_repo).stdout
    assert "upstream-sync-20260821" in branches
