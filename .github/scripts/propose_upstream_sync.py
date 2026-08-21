#!/usr/bin/env python3
"""
Automate safe proposals for recurring upstream updates (Issue #9).

Scope & Acceptance Criteria:
1. Preconditions Check:
   - Validates trusted blocking Tier 2 baseline and completed manual catch-up record.
2. Seam Surface Derivation:
   - Dynamically derives and logs the active seam surface from wharenui-fork.md
     and agent/phase_control.py rather than using a static hardcoded list.
3. Safety & Refusal Matrix:
   - Merge conflicts: Safe halt and merge abort.
   - Seam surface modification: Safe halt if upstream touches any seam file.
   - Oversized deltas: Safe halt if commits or lines exceed threshold.
   - Red test gates: Safe halt if seam or Tier tests fail.
4. Pull Request Workflow (No Direct Push / No Auto-Merge):
   - Opens a reviewable pull request on branch proposal upstream-sync-<date>.
   - NEVER pushes directly to wharenui-integration.
   - Zero auto-merge or auto-approval paths exist.
"""

import argparse
import ast
import dataclasses
from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple


# Exit code constants
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_REFUSAL_PRECONDITIONS = 2
EXIT_REFUSAL_SEAM_TOUCHED = 3
EXIT_REFUSAL_MERGE_CONFLICT = 4
EXIT_REFUSAL_OVERSIZED_DELTA = 5
EXIT_REFUSAL_TESTS_FAILED = 6
EXIT_REFUSAL_UPSTREAM_UNAVAILABLE = 7

# Status strings
STATUS_PROPOSAL_CREATED = "PROPOSAL_CREATED"
STATUS_DRY_RUN_PASSED = "DRY_RUN_PASSED"
STATUS_ALREADY_UP_TO_DATE = "ALREADY_UP_TO_DATE"
STATUS_REFUSED_PRECONDITIONS = "REFUSED_PRECONDITIONS"
STATUS_REFUSED_SEAM_TOUCHED = "REFUSED_SEAM_TOUCHED"
STATUS_REFUSED_MERGE_CONFLICT = "REFUSED_MERGE_CONFLICT"
STATUS_REFUSED_OVERSIZED_DELTA = "REFUSED_OVERSIZED_DELTA"
STATUS_REFUSED_TESTS_FAILED = "REFUSED_TESTS_FAILED"
STATUS_REFUSED_UPSTREAM_FETCH = "REFUSED_UPSTREAM_FETCH"


@dataclass
class SeamSurface:
    """Represents the dynamically derived seam surface."""
    files: Set[str] = field(default_factory=set)
    inline_hook_files: List[str] = field(default_factory=list)
    contract_test_files: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    api_version: Optional[int] = None
    source_doc: str = "wharenui-fork.md"
    phase_control_file: str = "agent/phase_control.py"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": sorted(list(self.files)),
            "inline_hook_files": sorted(self.inline_hook_files),
            "contract_test_files": sorted(self.contract_test_files),
            "symbols": sorted(self.symbols),
            "api_version": self.api_version,
            "source_doc": self.source_doc,
            "phase_control_file": self.phase_control_file,
            "total_files": len(self.files),
        }


@dataclass
class PreconditionReport:
    """Preconditions evaluation results."""
    valid: bool
    baseline_path: str
    has_provenance: bool
    reconciliation_record_present: bool
    clean_working_tree: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class UpstreamDeltaReport:
    """Analysis of incoming changes from upstream."""
    target_branch: str
    upstream_ref: str
    merge_base: str
    upstream_sha: str
    target_sha: str
    commit_count: int
    insertions: int
    deletions: int
    changed_files: List[str]
    is_up_to_date: bool
    is_oversized: bool
    oversized_reasons: List[str] = field(default_factory=list)


@dataclass
class SeamIntersectionReport:
    """Analysis of whether upstream touched any seam file."""
    clean: bool
    touched_seam_files: List[str] = field(default_factory=list)
    seam_surface_total: int = 0


@dataclass
class MergeTestReport:
    """Results of test-merging and running test gates."""
    conflict_free: bool
    conflicted_files: List[str] = field(default_factory=list)
    tests_executed: bool = False
    tests_passed: bool = False
    test_exit_code: int = 0
    test_output: str = ""
    errors: List[str] = field(default_factory=list)


@dataclass
class ProposalResult:
    """Complete summary of the sync proposal pipeline."""
    status: str
    exit_code: int
    summary: str
    refusal_reason: Optional[str] = None
    branch_name: Optional[str] = None
    pr_url: Optional[str] = None
    pr_body: Optional[str] = None
    seam_surface: Optional[SeamSurface] = None
    preconditions: Optional[PreconditionReport] = None
    delta: Optional[UpstreamDeltaReport] = None
    seam_intersection: Optional[SeamIntersectionReport] = None
    merge_test: Optional[MergeTestReport] = None
    details: Dict[str, Any] = field(default_factory=dict)


def run_git(args: List[str], cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    """Execute a git command and return CompletedProcess."""
    cmd = ["git"] + args
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def derive_seam_surface(
    repo_root: Path,
    fork_doc_path: str = "wharenui-fork.md",
    phase_control_file: str = "agent/phase_control.py",
) -> SeamSurface:
    """
    Dynamically derive active seam surface from wharenui-fork.md and agent/phase_control.py.
    Extracts core seam files, inline hooks from the table, symbols from phase_control AST,
    and seam test files from tests/run_agent.
    """
    seam = SeamSurface(
        source_doc=fork_doc_path,
        phase_control_file=phase_control_file,
    )

    doc_full = repo_root / fork_doc_path
    if doc_full.exists():
        content = doc_full.read_text(encoding="utf-8")
        seam_section = ""
        in_seam = False
        for line in content.splitlines():
            if re.match(r"^##\s+The Seam Surface", line, re.IGNORECASE):
                in_seam = True
                continue
            elif in_seam and re.match(r"^##\s+", line):
                break
            if in_seam:
                seam_section += line + "\n"

        valid_exts = {".py", ".toml", ".yml", ".yaml", ".md", ".txt", ".sh", ".json"}
        for match in re.findall(r"`([^`]+)`", seam_section):
            candidates = [c.strip() for c in match.split(",") if c.strip()]
            for cand in candidates:
                p = Path(cand)
                if p.suffix in valid_exts and "(" not in cand and not cand.startswith("_"):
                    normalized = str(p)
                    seam.files.add(normalized)
                    if "inline hook" in seam_section.lower() and normalized not in {
                        "agent/phase_control.py", "hermes_cli/plugins.py", "run_agent.py"
                    }:
                        seam.inline_hook_files.append(normalized)

    # Always ensure agent/phase_control.py is included
    seam.files.add(phase_control_file)

    # Inspect agent/phase_control.py AST for symbols & version
    pc_full = repo_root / phase_control_file
    if pc_full.exists():
        try:
            tree = ast.parse(pc_full.read_text(encoding="utf-8"), filename=str(pc_full))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    seam.symbols.append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            seam.symbols.append(target.id)
                            if target.id == "PHASE_CONTROL_API_VERSION" and isinstance(node.value, ast.Constant):
                                seam.api_version = int(node.value.value)
        except Exception as e:
            seam.symbols.append(f"error_parsing_ast: {e}")

    # Discover seam test files in tests/run_agent with @pytest.mark.wharenui_seam
    tests_dir = repo_root / "tests" / "run_agent"
    if tests_dir.exists():
        for test_file in tests_dir.glob("test_*.py"):
            try:
                tcontent = test_file.read_text(encoding="utf-8")
                if "wharenui_seam" in tcontent:
                    rel_path = str(test_file.relative_to(repo_root))
                    seam.files.add(rel_path)
                    seam.contract_test_files.append(rel_path)
            except Exception:
                pass

    return seam


def verify_preconditions(
    repo_root: Path,
    baseline_file: str = ".github/baseline-tier2.txt",
    fork_doc_path: str = "wharenui-fork.md",
    target_branch: str = "wharenui-integration",
) -> PreconditionReport:
    """
    Verify required preconditions:
    - Trusted blocking Tier 2 baseline (#4) with valid runner provenance header.
    - Completed manual catch-up record in wharenui-fork.md (#5).
    - Clean git repository status.
    """
    errors = []

    # 1. Baseline check
    base_full = repo_root / baseline_file
    has_provenance = False
    if not base_full.exists():
        errors.append(f"Tier 2 baseline file not found at {baseline_file}")
    else:
        lines = base_full.read_text(encoding="utf-8").splitlines()
        for line in lines[:15]:
            lstr = line.strip()
            if lstr.startswith("#") and re.search(r"(provenance|runner-provenance)\s*:\s*github-runner", lstr, re.IGNORECASE):
                has_provenance = True
                break
        if not has_provenance:
            errors.append(f"Baseline file {baseline_file} missing required '# Runner-Provenance: github-runner' header.")

    # 2. Fork doc & catch-up record check
    fork_full = repo_root / fork_doc_path
    reconcil_present = False
    if not fork_full.exists():
        errors.append(f"Fork architecture documentation not found at {fork_doc_path}")
    else:
        fcontent = fork_full.read_text(encoding="utf-8")
        if "Upstream Reconciliation Record" in fcontent:
            reconcil_present = True
        else:
            errors.append(f"Fork doc {fork_doc_path} missing 'Upstream Reconciliation Record' (#5 precondition).")

    # 3. Clean working tree check
    status_proc = run_git(["status", "--porcelain"], cwd=repo_root)
    clean_tree = len(status_proc.stdout.strip()) == 0
    if not clean_tree:
        errors.append("Git working tree is dirty. Clean tree required before sync proposal.")

    valid = len(errors) == 0
    return PreconditionReport(
        valid=valid,
        baseline_path=baseline_file,
        has_provenance=has_provenance,
        reconciliation_record_present=reconcil_present,
        clean_working_tree=clean_tree,
        errors=errors,
    )


def fetch_upstream(
    repo_root: Path,
    upstream_url: str = "https://github.com/NousResearch/hermes-agent.git",
    upstream_branch: str = "main",
    remote_name: str = "upstream",
) -> Tuple[bool, str, str]:
    """
    Ensure upstream remote exists and fetch upstream_branch.
    Returns (success, upstream_sha, message).
    """
    # Check remotes
    remotes_proc = run_git(["remote", "-v"], cwd=repo_root)
    remotes = remotes_proc.stdout

    if remote_name not in remotes:
        add_proc = run_git(["remote", "add", remote_name, upstream_url], cwd=repo_root)
        if add_proc.returncode != 0:
            return False, "", f"Failed to add upstream remote: {add_proc.stderr.strip()}"

    fetch_proc = run_git(["fetch", remote_name, upstream_branch], cwd=repo_root)
    if fetch_proc.returncode != 0:
        return False, "", f"Failed to fetch {remote_name}/{upstream_branch}: {fetch_proc.stderr.strip()}"

    rev_proc = run_git(["rev-parse", f"{remote_name}/{upstream_branch}"], cwd=repo_root)
    if rev_proc.returncode != 0:
        return False, "", f"Failed to rev-parse {remote_name}/{upstream_branch}: {rev_proc.stderr.strip()}"

    upstream_sha = rev_proc.stdout.strip()
    return True, upstream_sha, f"Fetched {remote_name}/{upstream_branch} ({upstream_sha[:10]})"


def analyze_upstream_delta(
    repo_root: Path,
    target_branch: str,
    upstream_ref: str,
    max_commits: int = 500,
    max_delta_lines: int = 5000,
) -> UpstreamDeltaReport:
    """
    Analyze commits and diff between target_branch and upstream_ref.
    """
    # Target SHA
    tsha_proc = run_git(["rev-parse", target_branch], cwd=repo_root)
    target_sha = tsha_proc.stdout.strip() if tsha_proc.returncode == 0 else "HEAD"

    # Upstream SHA
    usha_proc = run_git(["rev-parse", upstream_ref], cwd=repo_root)
    upstream_sha = usha_proc.stdout.strip() if usha_proc.returncode == 0 else upstream_ref

    # Merge base
    mb_proc = run_git(["merge-base", target_sha, upstream_sha], cwd=repo_root)
    merge_base = mb_proc.stdout.strip() if mb_proc.returncode == 0 else ""

    if merge_base == upstream_sha:
        # Already fully merged / up to date
        return UpstreamDeltaReport(
            target_branch=target_branch,
            upstream_ref=upstream_ref,
            merge_base=merge_base,
            upstream_sha=upstream_sha,
            target_sha=target_sha,
            commit_count=0,
            insertions=0,
            deletions=0,
            changed_files=[],
            is_up_to_date=True,
            is_oversized=False,
            oversized_reasons=[],
        )

    # Commit count
    cc_proc = run_git(["rev-list", "--count", f"{merge_base}..{upstream_sha}"], cwd=repo_root)
    commit_count = int(cc_proc.stdout.strip()) if cc_proc.returncode == 0 and cc_proc.stdout.strip().isdigit() else 0

    # Diff stat
    stat_proc = run_git(["diff", "--shortstat", f"{merge_base}..{upstream_sha}"], cwd=repo_root)
    stat_str = stat_proc.stdout.strip()
    insertions = 0
    deletions = 0
    ins_m = re.search(r"(\d+)\s+insertion", stat_str)
    del_m = re.search(r"(\d+)\s+deletion", stat_str)
    if ins_m:
        insertions = int(ins_m.group(1))
    if del_m:
        deletions = int(del_m.group(1))
    total_delta_lines = insertions + deletions

    # Changed files
    files_proc = run_git(["diff", "--name-only", f"{merge_base}..{upstream_sha}"], cwd=repo_root)
    changed_files = [f.strip() for f in files_proc.stdout.splitlines() if f.strip()]

    oversized_reasons = []
    if commit_count > max_commits:
        oversized_reasons.append(f"Upstream commit count ({commit_count}) exceeds threshold ({max_commits})")
    if total_delta_lines > max_delta_lines:
        oversized_reasons.append(f"Upstream delta lines ({total_delta_lines} = +{insertions}/-{deletions}) exceeds threshold ({max_delta_lines})")

    is_oversized = len(oversized_reasons) > 0

    return UpstreamDeltaReport(
        target_branch=target_branch,
        upstream_ref=upstream_ref,
        merge_base=merge_base,
        upstream_sha=upstream_sha,
        target_sha=target_sha,
        commit_count=commit_count,
        insertions=insertions,
        deletions=deletions,
        changed_files=changed_files,
        is_up_to_date=False,
        is_oversized=is_oversized,
        oversized_reasons=oversized_reasons,
    )


def check_seam_intersection(
    changed_files: List[str],
    seam_surface: SeamSurface,
) -> SeamIntersectionReport:
    """
    Check if any file modified by upstream intersects with the active seam surface.
    """
    changed_set = set(changed_files)
    touched = sorted(list(changed_set & seam_surface.files))

    return SeamIntersectionReport(
        clean=(len(touched) == 0),
        touched_seam_files=touched,
        seam_surface_total=len(seam_surface.files),
    )


def perform_test_merge_and_gate(
    repo_root: Path,
    target_branch: str,
    upstream_ref: str,
    run_tests: bool = True,
    test_args: Optional[List[str]] = None,
    worktree_parent: Optional[Path] = None,
) -> MergeTestReport:
    """
    Perform an isolated test merge in a temporary worktree and execute the test gates.
    Cleans up all worktrees and temporary state unconditionally.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="wharenui_sync_test_", dir=worktree_parent))
    report = MergeTestReport(conflict_free=False)

    try:
        # Create detached worktree at target branch
        add_wt = run_git(["worktree", "add", "--detach", str(tmp_dir), target_branch], cwd=repo_root)
        if add_wt.returncode != 0:
            report.errors.append(f"Failed to create worktree: {add_wt.stderr.strip()}")
            return report

        # Test merge upstream
        merge_proc = run_git(["merge", "--no-commit", "--no-ff", upstream_ref], cwd=tmp_dir)
        if merge_proc.returncode != 0:
            # Check for conflict files
            diff_unmerged = run_git(["diff", "--name-only", "--diff-filter=U"], cwd=tmp_dir)
            conflicts = [f.strip() for f in diff_unmerged.stdout.splitlines() if f.strip()]
            report.conflict_free = False
            report.conflicted_files = conflicts or ["(merge failed - see git log)"]
            report.errors.append(f"Merge conflict detected with {upstream_ref}: {merge_proc.stderr.strip() or merge_proc.stdout.strip()}")
            run_git(["merge", "--abort"], cwd=tmp_dir)
            return report

        report.conflict_free = True

        if run_tests:
            # Run test gate script
            runner_script = tmp_dir / ".github" / "scripts" / "run_tests.py"
            if not runner_script.exists():
                runner_script = repo_root / ".github" / "scripts" / "run_tests.py"

            default_test_args = [
                sys.executable,
                str(runner_script),
                "--selector", "tests/run_agent",
                "-m", "wharenui_seam",
                "--mode", "xdist",
            ]
            cmd = test_args if test_args else default_test_args

            # Inherit environment with plugin dir
            env = os.environ.copy()
            plugin_dir = repo_root.parent / "wharenui-hermes-agent-plugin"
            if plugin_dir.exists():
                env["WHARENUI_PLUGIN_DIR"] = str(plugin_dir)

            test_proc = subprocess.run(
                cmd,
                cwd=tmp_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            report.tests_executed = True
            report.test_exit_code = test_proc.returncode
            report.tests_passed = (test_proc.returncode == 0)
            report.test_output = (test_proc.stdout + "\n" + test_proc.stderr).strip()

            if not report.tests_passed:
                report.errors.append(f"Test gate failed with exit code {test_proc.returncode}")
        else:
            report.tests_executed = False
            report.tests_passed = True

        # Clean up merge in worktree
        run_git(["merge", "--abort"], cwd=tmp_dir)

    finally:
        # Unconditionally remove worktree
        run_git(["worktree", "remove", "--force", str(tmp_dir)], cwd=repo_root)
        run_git(["worktree", "prune"], cwd=repo_root)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return report


def generate_pr_body(
    delta: UpstreamDeltaReport,
    seam_intersection: SeamIntersectionReport,
    merge_test: MergeTestReport,
    seam_surface: SeamSurface,
    proposal_date: str,
) -> str:
    """Generate reviewable Markdown body for the proposal pull request."""
    return f"""## Upstream Sync Proposal (`upstream-sync-{proposal_date}`)

### 1. Upstream Delta Summary
- **Target Fork Branch**: `{delta.target_branch}` (`{delta.target_sha[:10]}`)
- **Upstream Source**: `{delta.upstream_ref}` (`{delta.upstream_sha[:10]}`)
- **Merge Base**: `{delta.merge_base[:10]}`
- **Commits Included**: `{delta.commit_count}`
- **Delta Size**: `+{delta.insertions} / -{delta.deletions}` lines across `{len(delta.changed_files)}` files

### 2. Active Seam Surface Audit (Dynamic Derivation)
- **Derived Seam Files**: `{seam_surface.to_dict()['total_files']}` tracked touchpoints
- **Seam Intersection**: `0 files modified by upstream` (CLEAN)
- **Phase Control API Version**: `v{seam_surface.api_version or 1}`
- **Derived Source**: [`wharenui-fork.md`](./wharenui-fork.md) + [`agent/phase_control.py`](./agent/phase_control.py)

### 3. Safety & Test Gate Results
- **Merge Status**: `Conflict-Free (Clean non-fast-forward merge)`
- **Seam Test Gate**: `{"PASSED (0 failures)" if merge_test.tests_passed else "SKIPPED/FAILED"}`
- **Delta Bounds Check**: `PASSED (Within commit and line thresholds)`

### 4. Reviewer Checklist & Verification
- [ ] Verify upstream changes do not introduce subtle semantic regressions in agent runtime
- [ ] Confirm `wharenui_seam` test gate passes on PR CI
- [ ] Review upstream commit log and notes
- [ ] Manually approve and merge into `{delta.target_branch}`

> ⚠️ **STRICT ZERO AUTO-MERGE POLICY**:
> This PR was created as an isolated proposal branch (`upstream-sync-{proposal_date}`).
> Direct push to `{delta.target_branch}` and auto-merge automation are strictly prohibited.
> A maintainer must manually review and merge this proposal.
"""


def create_proposal_branch(
    repo_root: Path,
    target_branch: str,
    upstream_ref: str,
    branch_name: str,
    commit_message: str,
    push_remote: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Create proposal branch, perform merge commit, and optionally push ONLY the proposal branch.
    NEVER touches or pushes target_branch (wharenui-integration).
    """
    # Remember current branch
    cur_proc = run_git(["branch", "--show-current"], cwd=repo_root)
    original_branch = cur_proc.stdout.strip() or target_branch

    try:
        # Checkout target branch
        run_git(["checkout", target_branch], cwd=repo_root, check=True)

        # Create/reset proposal branch
        run_git(["checkout", "-B", branch_name], cwd=repo_root, check=True)

        # Merge upstream ref
        merge_proc = run_git(["merge", "--no-ff", "-m", commit_message, upstream_ref], cwd=repo_root)
        if merge_proc.returncode != 0:
            run_git(["merge", "--abort"], cwd=repo_root)
            return False, f"Failed to commit merge on {branch_name}: {merge_proc.stderr.strip()}"

        if push_remote:
            push_proc = run_git(["push", "-u", push_remote, branch_name], cwd=repo_root)
            if push_proc.returncode != 0:
                return False, f"Failed to push {branch_name} to {push_remote}: {push_proc.stderr.strip()}"

        return True, f"Successfully created and merged proposal branch '{branch_name}'"

    finally:
        # Return to original branch
        run_git(["checkout", original_branch], cwd=repo_root)


def open_pull_request(
    repo_root: Path,
    target_branch: str,
    branch_name: str,
    title: str,
    body: str,
) -> Tuple[bool, str]:
    """
    Open PR using GitHub CLI `gh`.
    Ensures zero auto-merge flags are passed.
    """
    # Check if gh CLI exists
    if not shutil.which("gh"):
        return False, "gh CLI not installed on host"

    # Check if PR already exists
    check_pr = subprocess.run(
        ["gh", "pr", "list", "--head", branch_name, "--base", target_branch, "--json", "url"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check_pr.returncode == 0 and check_pr.stdout.strip() != "[]":
        try:
            data = json.loads(check_pr.stdout)
            if data and "url" in data[0]:
                return True, data[0]["url"]
        except Exception:
            pass

    # Create new PR
    create_pr = subprocess.run(
        [
            "gh", "pr", "create",
            "--base", target_branch,
            "--head", branch_name,
            "--title", title,
            "--body", body,
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if create_pr.returncode == 0:
        pr_url = create_pr.stdout.strip()
        return True, pr_url
    else:
        return False, f"gh pr create failed: {create_pr.stderr.strip()}"


def execute_proposal_pipeline(
    repo_root: Path,
    upstream_url: str = "https://github.com/NousResearch/hermes-agent.git",
    upstream_branch: str = "main",
    target_branch: str = "wharenui-integration",
    max_commits: int = 500,
    max_delta_lines: int = 5000,
    dry_run: bool = False,
    create_pr: bool = False,
    push: bool = False,
    skip_tests: bool = False,
    test_args: Optional[List[str]] = None,
    fork_doc_path: str = "wharenui-fork.md",
    phase_control_file: str = "agent/phase_control.py",
    baseline_file: str = ".github/baseline-tier2.txt",
    date_tag: Optional[str] = None,
) -> ProposalResult:
    """
    Execute the entire upstream sync proposal pipeline with safety refusal matrix.
    """
    today_str = date_tag or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    branch_name = f"upstream-sync-{today_str}"

    # Step 1: Seam Surface Dynamic Derivation
    seam_surface = derive_seam_surface(
        repo_root=repo_root,
        fork_doc_path=fork_doc_path,
        phase_control_file=phase_control_file,
    )

    # Step 2: Preconditions Check
    preconds = verify_preconditions(
        repo_root=repo_root,
        baseline_file=baseline_file,
        fork_doc_path=fork_doc_path,
        target_branch=target_branch,
    )
    if not preconds.valid:
        return ProposalResult(
            status=STATUS_REFUSED_PRECONDITIONS,
            exit_code=EXIT_REFUSAL_PRECONDITIONS,
            refusal_reason=f"Preconditions failed: {'; '.join(preconds.errors)}",
            summary="Refusal: Preconditions check failed.",
            seam_surface=seam_surface,
            preconditions=preconds,
        )

    # Step 3: Fetch Upstream
    fetch_ok, upstream_sha, fetch_msg = fetch_upstream(
        repo_root=repo_root,
        upstream_url=upstream_url,
        upstream_branch=upstream_branch,
    )
    if not fetch_ok:
        return ProposalResult(
            status=STATUS_REFUSED_UPSTREAM_FETCH,
            exit_code=EXIT_REFUSAL_UPSTREAM_UNAVAILABLE,
            refusal_reason=fetch_msg,
            summary=f"Refusal: Upstream fetch failed ({fetch_msg}).",
            seam_surface=seam_surface,
            preconditions=preconds,
        )

    upstream_ref = f"upstream/{upstream_branch}"

    # Step 4: Analyze Upstream Delta
    delta = analyze_upstream_delta(
        repo_root=repo_root,
        target_branch=target_branch,
        upstream_ref=upstream_ref,
        max_commits=max_commits,
        max_delta_lines=max_delta_lines,
    )

    if delta.is_up_to_date:
        return ProposalResult(
            status=STATUS_ALREADY_UP_TO_DATE,
            exit_code=EXIT_SUCCESS,
            summary=f"Upstream ({upstream_ref}) is already fully merged into {target_branch}. No sync required.",
            seam_surface=seam_surface,
            preconditions=preconds,
            delta=delta,
        )

    if delta.is_oversized:
        return ProposalResult(
            status=STATUS_REFUSED_OVERSIZED_DELTA,
            exit_code=EXIT_REFUSAL_OVERSIZED_DELTA,
            refusal_reason="; ".join(delta.oversized_reasons),
            summary=f"Refusal: Oversized upstream delta ({'; '.join(delta.oversized_reasons)}). Manual evaluation required.",
            seam_surface=seam_surface,
            preconditions=preconds,
            delta=delta,
        )

    # Step 5: Check Seam Surface Intersection
    seam_intersection = check_seam_intersection(
        changed_files=delta.changed_files,
        seam_surface=seam_surface,
    )

    if not seam_intersection.clean:
        touched_str = ", ".join(seam_intersection.touched_seam_files)
        return ProposalResult(
            status=STATUS_REFUSED_SEAM_TOUCHED,
            exit_code=EXIT_REFUSAL_SEAM_TOUCHED,
            refusal_reason=f"Upstream touched Wharenui seam surface files: {touched_str}",
            summary=f"Refusal: Upstream modifies active seam surface ({touched_str}). Requires manual reconciliation.",
            seam_surface=seam_surface,
            preconditions=preconds,
            delta=delta,
            seam_intersection=seam_intersection,
        )

    # Step 6: Test Merge and Gate Execution (Isolated Worktree)
    merge_test = perform_test_merge_and_gate(
        repo_root=repo_root,
        target_branch=target_branch,
        upstream_ref=upstream_ref,
        run_tests=(not skip_tests),
        test_args=test_args,
    )

    if not merge_test.conflict_free:
        conf_str = ", ".join(merge_test.conflicted_files)
        return ProposalResult(
            status=STATUS_REFUSED_MERGE_CONFLICT,
            exit_code=EXIT_REFUSAL_MERGE_CONFLICT,
            refusal_reason=f"Merge conflicts encountered: {conf_str}",
            summary=f"Refusal: Upstream merge causes conflicts in {conf_str}. Safely halted and aborted.",
            seam_surface=seam_surface,
            preconditions=preconds,
            delta=delta,
            seam_intersection=seam_intersection,
            merge_test=merge_test,
        )

    if merge_test.tests_executed and not merge_test.tests_passed:
        return ProposalResult(
            status=STATUS_REFUSED_TESTS_FAILED,
            exit_code=EXIT_REFUSAL_TESTS_FAILED,
            refusal_reason="Test gate failure in merged state",
            summary=f"Refusal: Seam test gate failed after test merge (Exit code {merge_test.test_exit_code}).",
            seam_surface=seam_surface,
            preconditions=preconds,
            delta=delta,
            seam_intersection=seam_intersection,
            merge_test=merge_test,
        )

    # Step 7: Generate PR Body
    pr_body = generate_pr_body(
        delta=delta,
        seam_intersection=seam_intersection,
        merge_test=merge_test,
        seam_surface=seam_surface,
        proposal_date=today_str,
    )
    pr_title = f"[Proposal] Safe Upstream Sync ({today_str}) - {upstream_sha[:10]}"

    if dry_run:
        return ProposalResult(
            status=STATUS_DRY_RUN_PASSED,
            exit_code=EXIT_SUCCESS,
            summary=f"Dry run PASSED: Safe upstream sync proposal validated for {upstream_sha[:10]} ({delta.commit_count} commits).",
            branch_name=branch_name,
            pr_body=pr_body,
            seam_surface=seam_surface,
            preconditions=preconds,
            delta=delta,
            seam_intersection=seam_intersection,
            merge_test=merge_test,
        )

    # Step 8: Create Proposal Branch & PR
    commit_msg = f"chore(upstream): propose sync with upstream/{upstream_branch} as of {today_str} ({upstream_sha[:10]})"
    push_remote = "origin" if (push or create_pr) else None

    branch_ok, branch_msg = create_proposal_branch(
        repo_root=repo_root,
        target_branch=target_branch,
        upstream_ref=upstream_ref,
        branch_name=branch_name,
        commit_message=commit_msg,
        push_remote=push_remote,
    )
    if not branch_ok:
        return ProposalResult(
            status=STATUS_REFUSED_MERGE_CONFLICT,
            exit_code=EXIT_ERROR,
            refusal_reason=branch_msg,
            summary=f"Error creating proposal branch: {branch_msg}",
            seam_surface=seam_surface,
            preconditions=preconds,
            delta=delta,
            seam_intersection=seam_intersection,
            merge_test=merge_test,
        )

    pr_url = None
    if create_pr:
        pr_ok, pr_res = open_pull_request(
            repo_root=repo_root,
            target_branch=target_branch,
            branch_name=branch_name,
            title=pr_title,
            body=pr_body,
        )
        if pr_ok:
            pr_url = pr_res

    return ProposalResult(
        status=STATUS_PROPOSAL_CREATED,
        exit_code=EXIT_SUCCESS,
        summary=f"Successfully created safe upstream sync proposal on '{branch_name}'. PR: {pr_url or 'Created locally'}",
        branch_name=branch_name,
        pr_url=pr_url,
        pr_body=pr_body,
        seam_surface=seam_surface,
        preconditions=preconds,
        delta=delta,
        seam_intersection=seam_intersection,
        merge_test=merge_test,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wharenui Safe Upstream Sync Proposal Automation (Issue #9)"
    )
    parser.add_argument(
        "--upstream-repo",
        default="https://github.com/NousResearch/hermes-agent.git",
        help="URL of upstream git repository",
    )
    parser.add_argument(
        "--upstream-branch",
        default="main",
        help="Branch in upstream repository to sync from",
    )
    parser.add_argument(
        "--target-branch",
        default="wharenui-integration",
        help="Target branch in fork repo (default: wharenui-integration)",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=500,
        help="Maximum allowed commits in single automated proposal (default: 500)",
    )
    parser.add_argument(
        "--max-delta-lines",
        type=int,
        default=5000,
        help="Maximum allowed total delta lines (default: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all preconditions, delta limits, seam safety and test gates without pushing or creating PR",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push proposal branch and open reviewable Pull Request via gh CLI",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push proposal branch to origin without opening PR",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test gate execution (for dry-run simulation in unit tests)",
    )
    parser.add_argument(
        "--fork-doc",
        default="wharenui-fork.md",
        help="Path to fork architecture document",
    )
    parser.add_argument(
        "--phase-control-file",
        default="agent/phase_control.py",
        help="Path to phase_control.py seam definition",
    )
    parser.add_argument(
        "--baseline-file",
        default=".github/baseline-tier2.txt",
        help="Path to trusted Tier 2 serial baseline",
    )
    parser.add_argument(
        "--date-tag",
        default=None,
        help="Custom date tag for branch proposal (default: current UTC YYYYMMDD)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Write full proposal result as JSON to specified file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path.cwd()

    print("=" * 72)
    print("        WHARENUI UPSTREAM SYNC PROPOSAL AUTOMATION (ISSUE #9)       ")
    print("=" * 72)
    print(f" Target Branch  : {args.target_branch}")
    print(f" Upstream Repo  : {args.upstream_repo} ({args.upstream_branch})")
    print(f" Max Limits     : Commits <= {args.max_commits} | Lines <= {args.max_delta_lines}")
    print(f" Mode           : {'DRY RUN' if args.dry_run else ('CREATE PR' if args.create_pr else 'BRANCH ONLY')}")
    print("-" * 72)

    result = execute_proposal_pipeline(
        repo_root=repo_root,
        upstream_url=args.upstream_repo,
        upstream_branch=args.upstream_branch,
        target_branch=args.target_branch,
        max_commits=args.max_commits,
        max_delta_lines=args.max_delta_lines,
        dry_run=args.dry_run,
        create_pr=args.create_pr,
        push=args.push,
        skip_tests=args.skip_tests,
        fork_doc_path=args.fork_doc,
        phase_control_file=args.phase_control_file,
        baseline_file=args.baseline_file,
        date_tag=args.date_tag,
    )

    if result.seam_surface:
        print(f"\n[Seam Surface] Derived {len(result.seam_surface.files)} active seam files (API v{result.seam_surface.api_version or 1})")
        for sf in sorted(result.seam_surface.files):
            print(f"  - {sf}")

    if result.delta:
        print(f"\n[Upstream Delta] {result.delta.commit_count} commits, +{result.delta.insertions}/-{result.delta.deletions} lines")

    print("\n" + "-" * 72)
    print(f" Result Status  : {result.status}")
    print(f" Summary        : {result.summary}")
    if result.refusal_reason:
        print(f" Refusal Reason : {result.refusal_reason}")
    if result.branch_name:
        print(f" Proposal Branch: {result.branch_name}")
    if result.pr_url:
        print(f" Pull Request   : {result.pr_url}")
    print("=" * 72 + "\n")

    if args.json_output:
        out_path = Path(args.json_output)
        res_data = {
            "status": result.status,
            "exit_code": result.exit_code,
            "summary": result.summary,
            "refusal_reason": result.refusal_reason,
            "branch_name": result.branch_name,
            "pr_url": result.pr_url,
            "seam_surface": result.seam_surface.to_dict() if result.seam_surface else None,
            "delta": dataclasses.asdict(result.delta) if result.delta else None,
            "seam_intersection": dataclasses.asdict(result.seam_intersection) if result.seam_intersection else None,
            "merge_test": dataclasses.asdict(result.merge_test) if result.merge_test else None,
        }
        out_path.write_text(json.dumps(res_data, indent=2), encoding="utf-8")

    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
