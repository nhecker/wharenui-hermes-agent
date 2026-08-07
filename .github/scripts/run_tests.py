#!/usr/bin/env python3
"""
Hermes Agent Test Runner (WP-CI)
Features:
- Digestible summary output
- Count reconciliation (§6.4.1): collected == passed + failed + skipped + errors + xfailed
- Env-vs-real failure categorization
- Stateless CI gate: failures == 0
- Optional local dev baseline comparison via isolated git worktree
- Sharding support (--shard K/N or name)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Hermes Agent Digestible Test Runner")
    parser.add_argument(
        "--selector",
        nargs="*",
        default=None,
        help="Test directories or files to run (default: tests/agent tests/run_agent)",
    )
    parser.add_argument(
        "--mode",
        choices=["serial", "xdist"],
        default="serial",
        help="Execution mode: serial or xdist (-n auto)",
    )
    parser.add_argument(
        "--baseline",
        metavar="GIT_REF",
        default=None,
        help="Optional git ref to diff failures against using an isolated git worktree",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="Optional shard specifier (e.g. 1/2, 2/2, or shard label)",
    )
    parser.add_argument(
        "--junit",
        metavar="PATH",
        default=None,
        help="Path to output JUnit XML report",
    )
    parser.add_argument(
        "-m",
        "--marker",
        default=None,
        help="Pytest marker expression (e.g. 'not network')",
    )
    parser.add_argument(
        "--min-collected",
        type=int,
        default=0,
        help="Minimum number of collected tests expected (causes gate failure if collected count is lower)",
    )
    parser.add_argument(
        "--baseline-file",
        metavar="PATH",
        default=None,
        help="Path to a baseline file (sorted failing node IDs, one per line). "
             "Reports newly-failing/fixed/new/disappeared by node ID. "
             "Exits non-zero only on newly-failing.",
    )

    parser.add_argument("--baseline-collected-file", metavar="PATH", default=None, help="Path to the baseline complete collected node-ID set.")

    return parser.parse_known_args()


def normalize_selector(selector_args):
    """Flatten space-separated or list selectors into a clean list of paths."""
    if not selector_args:
        return ["tests/agent", "tests/run_agent"]
    paths = []
    for item in selector_args:
        for p in item.strip().split():
            if p:
                paths.append(p)
    return paths or ["tests/agent", "tests/run_agent"]


def collect_test_nodes(selectors, marker=None):
    """Run pytest --collect-only to get all test node IDs and check collection errors."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if marker:
        cmd.extend(["-m", marker])
    cmd.extend(selectors)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    nodes = []
    has_collection_errors = (
        proc.returncode == 2
        or "errors during collection" in proc.stdout
        or "errors during collection" in proc.stderr
    )
    
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("="):
            nodes.append(line)
            
    return nodes, has_collection_errors, proc.stdout + "\n" + proc.stderr


def apply_shard(test_nodes, shard_spec):
    """Partition test_nodes if shard_spec is in K/N format."""
    if not shard_spec or "/" not in shard_spec:
        return test_nodes, shard_spec

    match = re.match(r"^(\d+)/(\d+)$", shard_spec.strip())
    if not match:
        return test_nodes, shard_spec

    k = int(match.group(1))
    n = int(match.group(2))
    if k < 1 or k > n or n < 1:
        print(f"Invalid shard specifier {shard_spec}. Expected 1 <= K <= N.")
        sys.exit(1)

    total = len(test_nodes)
    if total == 0:
        return test_nodes, shard_spec

    chunk_size = (total + n - 1) // n
    start = (k - 1) * chunk_size
    end = min(start + chunk_size, total)
    
    sharded_nodes = test_nodes[start:end]
    print(f"Shard {k}/{n}: selected {len(sharded_nodes)} of {total} total test nodes.")
    return sharded_nodes, f"{k}/{n}"


def parse_junit_xml(xml_path):
    """Parse JUnit XML report produced by pytest."""
    if not os.path.exists(xml_path):
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse JUnit XML at {xml_path}: {e}")
        return None

    results = {
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "failures_details": [],
    }

    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

    for suite in suites:
        results["collected"] += int(suite.attrib.get("tests", 0))
        
        for case in suite.findall("testcase"):
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            node_id = f"{classname}::{name}"

            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")

            if failure is not None:
                message = failure.attrib.get("message", "")
                text = failure.text or ""
                if "XPASS" in message or "xpass" in message.lower():
                    results["xpassed"] += 1
                else:
                    results["failed"] += 1
                    results["failures_details"].append({
                        "node_id": node_id,
                        "type": "failure",
                        "message": message,
                        "text": text,
                    })
            elif error is not None:
                message = error.attrib.get("message", "")
                text = error.text or ""
                results["errors"] += 1
                results["failures_details"].append({
                    "node_id": node_id,
                    "type": "error",
                    "message": message,
                    "text": text,
                })
            elif skipped is not None:
                message = skipped.attrib.get("message", "")
                if "xfail" in message.lower() or skipped.attrib.get("type", "") == "pytest.xfail":
                    results["xfailed"] += 1
                else:
                    results["skipped"] += 1
            else:
                results["passed"] += 1

    return results


def categorize_failures(failures_details):
    """Bucket failures into ENV/DEP vs REAL."""
    env_dep_keywords = [
        "ModuleNotFoundError",
        "ImportError",
        "ConnectionError",
        "ConnectError",
        "APIConnectionError",
        "urllib3",
        "network",
        "socket.gai_error",
        "NameResolutionError",
    ]

    env_dep = []
    real = []

    for f in failures_details:
        content = f"{f['message']}\n{f['text']}"
        if any(kw in content for kw in env_dep_keywords):
            env_dep.append(f)
        else:
            real.append(f)

    return env_dep, real


def run_pytest(targets, mode, junit_path, marker=None, extra_args=None):
    """Execute pytest with specified options."""
    cmd = [sys.executable, "-m", "pytest", f"--junitxml={junit_path}"]
    
    if mode == "xdist":
        cmd.extend(["-n", "auto"])
        
    if marker:
        cmd.extend(["-m", marker])
        
    if extra_args:
        cmd.extend(extra_args)
        
    cmd.extend(targets)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_baseline_worktree(ref, selectors, mode, marker, extra_args):
    """Run tests against a baseline git ref in an isolated git worktree."""
    print(f"\n--- Setting up isolated git worktree for baseline ref '{ref}' ---")
    tmp_dir = tempfile.mkdtemp(prefix="hermes_baseline_")
    junit_path = os.path.join(tmp_dir, "baseline_junit.xml")

    try:
        add_proc = subprocess.run(
            ["git", "worktree", "add", "--detach", tmp_dir, ref],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if add_proc.returncode != 0:
            print(f"Failed to create git worktree for baseline '{ref}': {add_proc.stderr}")
            return None

        cmd = [sys.executable, "-m", "pytest", f"--junitxml={junit_path}"]
        if mode == "xdist":
            cmd.extend(["-n", "auto"])
        if marker:
            cmd.extend(["-m", marker])
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(selectors)

        subprocess.run(cmd, cwd=tmp_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return parse_junit_xml(junit_path)

    finally:
        subprocess.run(["git", "worktree", "remove", "--force", tmp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "worktree", "prune"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"--- Cleaned up worktree at {tmp_dir} ---")


def _normalize_node_id(node_id: str) -> str:
    """Normalize a node ID to the JUnit classname::name format."""
    if "/" in node_id and ".py::" in node_id:
        parts = node_id.split("::", 1)
        path_part = parts[0].replace("/", ".").removesuffix(".py")
        return path_part + "." + parts[1] if len(parts) == 2 else path_part
    return node_id


def main():
    args, extra_pytest_args = parse_args()
    selectors = normalize_selector(args.selector)
    marker = args.marker

    temp_junit_dir = None
    if args.junit:
        junit_path = os.path.abspath(args.junit)
    else:
        temp_junit_dir = tempfile.mkdtemp(prefix="hermes_junit_")
        junit_path = os.path.join(temp_junit_dir, "junit.xml")

    try:
        collected_nodes, collection_errored, collect_output = collect_test_nodes(selectors, marker=marker)
        expected_collected = len(collected_nodes)
        
        target_nodes, shard_label = apply_shard(collected_nodes if "/" in (args.shard or "") else selectors, args.shard)

        print(f"Running tests (Mode: {args.mode}, Selector: {selectors}, Shard: {shard_label or 'None'}, Marker: {marker or 'None'})...")
        ret_code, stdout, stderr = run_pytest(
            targets=target_nodes if "/" in (args.shard or "") else selectors,
            mode=args.mode,
            junit_path=junit_path,
            marker=marker,
            extra_args=extra_pytest_args,
        )

        results = parse_junit_xml(junit_path)
        if not results:
            print("ERROR: Failed to parse test results from JUnit XML.")
            sys.exit(1)

        total_executed = (
            results["passed"]
            + results["failed"]
            + results["errors"]
            + results["skipped"]
            + results["xfailed"]
            + results["xpassed"]
        )

        reconciled = (total_executed == results["collected"])
        subset_warning = False

        if "/" not in (args.shard or "") and expected_collected > 0:
            if total_executed < expected_collected * 0.9:
                subset_warning = True

        min_collected_failed = (args.min_collected > 0 and results["collected"] < args.min_collected)

        env_dep_failures, real_failures = categorize_failures(results["failures_details"])

        baseline_results = None
        if args.baseline:
            baseline_results = run_baseline_worktree(args.baseline, selectors, args.mode, marker, extra_pytest_args)

        print("\n" + "=" * 72)
        print("                      HERMES AGENT TEST SUMMARY                     ")
        print("=" * 72)
        print(f" Selector       : {' '.join(selectors)}")
        print(f" Mode           : {args.mode}")
        print(f" Shard          : {shard_label or 'None'}")
        print(f" Marker         : {marker or 'None'}")
        print(f" Reconciliation : {'PASS' if reconciled and not subset_warning and not collection_errored else 'FAIL'}")
        print(f"                  (Collected: {results['collected']} | Executed: {total_executed})")
        
        if collection_errored:
            print("                  [!] COLLECTION ERROR DETECTED DURING TEST DISCOVERY")
        if subset_warning:
            print(f"                  [!] WARNING: Executed {total_executed} tests, far below collected {expected_collected}!")
        if min_collected_failed:
            print(f"                  [!] ERROR: Collected {results['collected']} tests, below --min-collected {args.min_collected}!")
            
        print("-" * 72)
        print(f" Passed         : {results['passed']}")
        print(f" Failed         : {results['failed']}")
        print(f" Errors         : {results['errors']}")
        print(f" Skipped        : {results['skipped']}")
        print(f" XFailed        : {results['xfailed']}")
        print(f" XPassed        : {results['xpassed']}")
        print("-" * 72)
        print(" Failures Breakdown:")
        print(f"   ENV / DEP    : {len(env_dep_failures)}")
        print(f"   REAL         : {len(real_failures)}")

        if real_failures or env_dep_failures:
            print("\n Failure Details:")
            for f in real_failures:
                print(f"   [REAL] {f['node_id']}: {f['message']}")
            for f in env_dep_failures:
                print(f"   [ENV ] {f['node_id']}: {f['message']}")

        if baseline_results:
            curr_fail_set = {f["node_id"] for f in results["failures_details"]}
            base_fail_set = {f["node_id"] for f in baseline_results["failures_details"]}
            
            new_fails = curr_fail_set - base_fail_set
            fixed_fails = base_fail_set - curr_fail_set
            same_fails = curr_fail_set & base_fail_set

            print("-" * 72)
            print(f" Baseline Ref   : {args.baseline}")
            print(f"   New Failures : {len(new_fails)}")
            print(f"   Pre-existing : {len(same_fails)}")
            print(f"   Fixed        : {len(fixed_fails)}")

            if new_fails:
                print("\n   New Regressions vs Baseline:")
                for nf in sorted(new_fails):
                    print(f"     - {nf}")

        # --- B2: baseline-file delta mode ---
        baseline_file_newly_failing = []
        if args.baseline_file:
            baseline_path = Path(args.baseline_file)
            if not baseline_path.exists():
                print("\nERROR: baseline file not found: " + str(baseline_path))
                sys.exit(1)
            baseline_failures = set()
            for line in baseline_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    baseline_failures.add(_normalize_node_id(line))

            baseline_collected = set()
            if args.baseline_collected_file:
                collected_path = Path(args.baseline_collected_file)
                if not collected_path.exists():
                    print("\nERROR: baseline collected file not found: " + str(collected_path))
                    sys.exit(1)
                baseline_collected = {_normalize_node_id(line.strip()) for line in collected_path.read_text().splitlines() if line.strip() and not line.startswith("#")}

            curr_fail_set = {f["node_id"] for f in results["failures_details"]}
            curr_collected = set(_normalize_node_id(n) for n in collected_nodes)
            new_tests = sorted(curr_collected - baseline_collected) if baseline_collected else []
            newly_failing = sorted(curr_fail_set - baseline_failures - set(new_tests))
            newly_fixed = sorted(baseline_failures - curr_fail_set)
            disappeared = sorted(
                node for node in baseline_failures
                if node not in curr_collected
            )

            print("-" * 72)
            print(f" Baseline File  : {args.baseline_file}")
            print(f"   Newly failing: {len(newly_failing)}  <-- GATE")
            print(f"   New tests    : {len(new_tests)}  (non-blocking)")
            print(f"   Newly fixed  : {len(newly_fixed)}")
            print(f"   Disappeared  : {len(disappeared)}")
            if newly_failing:
                print("\n   *** NEWLY FAILING (gate) ***")
                for nf in newly_failing:
                    print(f"     + {nf}")
            if new_tests:
                print("\n   New tests (non-blocking):")
                for node in new_tests:
                    print(f"     ? {node}")
            if newly_fixed:
                print("\n   Newly fixed (informational):")
                for nf in newly_fixed:
                    print(f"     - {nf}")
            if disappeared:
                print("\n   Disappeared (informational):")
                for d in disappeared:
                    print(f"     * {d}")

        print("=" * 72)

        total_failures = results["failed"] + results["errors"]

        # If baseline-file mode, gate on newly-failing only
        if args.baseline_file:
            is_clean = (len(newly_failing) == 0 and not collection_errored and not min_collected_failed)
        else:
            is_clean = (total_failures == 0 and reconciled and not collection_errored and not subset_warning and not min_collected_failed)

        if is_clean:
            if args.baseline_file:
                print(" CI GATE RESULT  : CLEAN (0 newly-failing vs baseline)")
            else:
                print(" CI GATE RESULT  : CLEAN (0 failures)")
            print("=" * 72 + "\n")
            sys.exit(0)
        else:
            reasons = []
            if args.baseline_file:
                reasons.append(f"{len(newly_failing)} newly-failing vs baseline")
            elif total_failures > 0:
                reasons.append(f"{total_failures} failures")
            if collection_errored:
                reasons.append("collection error")
            if not args.baseline_file and (not reconciled or subset_warning):
                reasons.append("reconciliation mismatch")
            if min_collected_failed:
                reasons.append(f"collected {results['collected']} < min-collected {args.min_collected}")
            reason_str = " / ".join(reasons) if reasons else "failures detected"
            print(f" CI GATE RESULT  : FAILED ({reason_str})")
            print("=" * 72 + "\n")
            sys.exit(1)

    finally:
        if temp_junit_dir:
            shutil.rmtree(temp_junit_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
