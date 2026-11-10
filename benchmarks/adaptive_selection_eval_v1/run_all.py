
"""Portable MemoRoute final verification and adaptive evaluation runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "memoroute_final"

BENCHMARK_DIRS = [
    "benchmarks/adaptive_selection_eval_v1",
    "benchmarks/retrieval_baseline",
    "benchmarks/retrieval_challenge_v2",
    "benchmarks/retrieval_temporal_v1",
    "benchmarks/temporal_assessment_v1",
]

DATASET_CHECK = (
    "from benchmarks.retrieval_temporal_v1.dataset "
    "import build_records, build_relations, validate_dataset; "
    "validate_dataset(); "
    "records=tuple(build_records()); "
    "relations=tuple(build_relations()); "
    "print('Dataset validation: PASS'); "
    "print('Records:', len(records)); "
    "print('Relations:', len(relations))"
)

def run_stage(name, command, timeout=600):
    started = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        return {
            "name": name,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "returncode": result.returncode,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "command": command,
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:],
        }

    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "TIMEOUT",
            "returncode": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "command": command,
            "stdout": str(exc.stdout or "")[-12000:],
            "stderr": str(exc.stderr or "")[-12000:],
        }

    except Exception as exc:
        return {
            "name": name,
            "status": "ERROR",
            "returncode": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "command": command,
            "stdout": "",
            "stderr": repr(exc),
        }

def print_result(result):
    print(
        f"[{result['status']}] {result['name']} "
        f"({result['duration_seconds']}s)"
    )

    if result["stdout"].strip():
        print("--- stdout ---")
        print(result["stdout"])

    if result["stderr"].strip():
        print("--- stderr ---")
        print(result["stderr"])

def main():
    parser = argparse.ArgumentParser(
        description="Run MemoRoute's final validation pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Maximum number of parallel validation processes.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if not (ROOT / "pyproject.toml").is_file():
        raise SystemExit(f"Project root not found: {ROOT}")

    python = sys.executable

    scripts_dir = Path(sysconfig.get_path("scripts"))
    console_name = "memoroute.exe" if os.name == "nt" else "memoroute"
    console_path = scripts_dir / console_name

    if console_path.is_file():
        console_command = [str(console_path), "--help"]
    else:
        console_command = [
            python,
            "-c",
            (
                "raise SystemExit("
                "'Console executable not found in active Python environment'"
                ")"
            ),
        ]

    modules = [
        (
            "read_only_preflight",
            [python, "-m", "benchmarks.adaptive_selection_eval_v1.preflight"],
            180,
        ),
        (
            "source_contract_audit",
            [python, "-m", "benchmarks.adaptive_selection_eval_v1.source_audit"],
            300,
        ),
        (
            "focused_evaluation_tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_adaptive_selection_eval_v1.py",
            ],
            300,
        ),
        (
            "full_regression_tests",
            [python, "-m", "pytest", "-q"],
            600,
        ),
        (
            "core_ruff",
            [python, "-m", "ruff", "check", "src/memoroute", "tests"],
            180,
        ),
        (
            "core_format",
            [
                python,
                "-m",
                "ruff",
                "format",
                "--check",
                "src/memoroute",
                "tests",
            ],
            180,
        ),
        (
            "active_benchmark_ruff",
            [python, "-m", "ruff", "check", *BENCHMARK_DIRS],
            180,
        ),
        (
            "active_benchmark_format",
            [
                python,
                "-m",
                "ruff",
                "format",
                "--check",
                *BENCHMARK_DIRS,
            ],
            180,
        ),
        (
            "retrieval_temporal_dataset_contract",
            [python, "-c", DATASET_CHECK],
            120,
        ),
        (
            "temporal_assessment_challenge",
            [python, "-m", "benchmarks.temporal_assessment_v1.run_experiment"],
            300,
        ),
        (
            "module_cli",
            [python, "-m", "memoroute", "--help"],
            60,
        ),
        (
            "console_cli",
            console_command,
            60,
        ),
        (
            "working_tree_whitespace",
            ["git", "diff", "--check"],
            60,
        ),
        (
            "staged_whitespace",
            ["git", "diff", "--cached", "--check"],
            60,
        ),
    ]

    print("=" * 68)
    print("MEMOROUTE — FINAL VALIDATION PIPELINE")
    print("=" * 68)
    print("Repository:", ROOT)
    print("Python:", python)
    print("Parallel workers:", args.workers)
    print("Source files will not be auto-fixed.")
    print("Git staging, commits and pushes are not performed.")
    print()

    if args.dry_run:
        print("DRY RUN — planned parallel checks:")
        for name, command, _ in modules:
            print(f"\n{name}:")
            print(" ".join(command))

        print("\nSequential stage:")
        print(python, "-m", "benchmarks.adaptive_selection_eval_v1.run_experiment")
        print(python, "-m", "benchmarks.adaptive_selection_eval_v1.analyze")
        print("\nNo commands executed.")
        return 0

    # Phase 1: execute independent validation checks concurrently.
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_stage, name, command, timeout): name
            for name, command, timeout in modules
        }

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["name"])

    print("\n" + "=" * 68)
    print("PARALLEL VALIDATION RESULTS")
    print("=" * 68)

    for result in results:
        print_result(result)

    failures = [r for r in results if r["status"] != "PASS"]

    sequential_results = []

    # Do not generate new experiment evidence if validation is failing.
    if failures:
        print("\nDEPENDENT EXPERIMENTS SKIPPED.")
        print("Fix the reported validation failures, then rerun this pipeline.")

    else:
        # Phase 2: experiment and analysis must run in dependency order.
        sequential_stages = [
            (
                "real_bge_adaptive_experiment",
                [
                    python,
                    "-m",
                    "benchmarks.adaptive_selection_eval_v1.run_experiment",
                ],
                1800,
            ),
            (
                "adaptive_experiment_analysis",
                [
                    python,
                    "-m",
                    "benchmarks.adaptive_selection_eval_v1.analyze",
                ],
                600,
            ),
        ]

        for name, command, timeout in sequential_stages:
            result = run_stage(name, command, timeout)
            sequential_results.append(result)
            print_result(result)

            if result["status"] != "PASS":
                print(
                    f"\nStopping dependent stages after failure: {name}"
                )
                break

    # Preserve a timestamped execution report; never overwrite prior reports.
    RESULTS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = RESULTS / f"master_run_{timestamp}.json"

    all_results = results + sequential_results
    final_failures = [
        r for r in all_results if r["status"] != "PASS"
    ]

    report = {
        "project": "MemoRoute",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": str(ROOT),
        "python_executable": python,
        "parallel_workers": args.workers,
        "parallel_results": results,
        "sequential_results": sequential_results,
        "failure_count": len(final_failures),
        "failed_stages": [r["name"] for r in final_failures],
        "experiment_skipped": bool(failures),
        "git_mutations_performed": False,
        "source_auto_fixes_performed": False,
        "longmemeval_evaluated": False,
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("FINAL SUMMARY")
    print("=" * 68)
    print("Checks executed:", len(all_results))
    print("Failures:", len(final_failures))
    print("Report:", report_path)

    if final_failures:
        print("RESULT: NOT READY — inspect the failed stages.")
        return 1

    print("RESULT: ALL INCLUDED CHECKS PASSED.")
    print("LongMemEval remains a separate outstanding task.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
