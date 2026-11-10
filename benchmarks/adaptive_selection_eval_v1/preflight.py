"""Read-only preflight for the final MemoRoute adaptive evaluation."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "memoroute_final"


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    required_files = [
        "pyproject.toml",
        "benchmarks/retrieval_temporal_v1/dataset.py",
        "src/memoroute/memory/sqlite_store.py",
        "src/memoroute/retrieval/adaptive_selector.py",
        "src/memoroute/retrieval/temporal_store.py",
        "src/memoroute/retrieval/query_intent.py",
        "src/memoroute/retrieval/context_packet.py",
    ]
    missing_files = [
        item for item in required_files if not (ROOT / item).is_file()
    ]

    modules = [
        "benchmarks.retrieval_temporal_v1.dataset",
        "memoroute.memory.sqlite_store",
        "memoroute.retrieval.recent",
        "memoroute.retrieval.bm25",
        "memoroute.retrieval.vector",
        "memoroute.retrieval.hybrid",
        "memoroute.retrieval.temporal_store",
        "memoroute.retrieval.query_intent",
        "memoroute.retrieval.adaptive_selector",
    ]
    import_errors = {}
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            import_errors[module_name] = f"{type(exc).__name__}: {exc}"

    dataset_info = {}
    try:
        dataset = importlib.import_module(
            "benchmarks.retrieval_temporal_v1.dataset"
        )
        dataset.validate_dataset()
        dataset_info = {
            "records": len(dataset.build_records()),
            "relations": len(dataset.build_relations()),
            "queries": len(dataset.QUERIES),
        }
    except Exception as exc:
        dataset_info = {"error": f"{type(exc).__name__}: {exc}"}

    packages = {
        name: _version(name)
        for name in ("fastembed", "onnxruntime", "numpy", "pytest", "ruff")
    }
    missing_packages = [
        name for name in ("fastembed", "onnxruntime", "numpy", "pytest", "ruff")
        if packages[name] is None
    ]

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "repository": str(ROOT),
        "python": sys.version,
        "git": {
            "branch": _run_git("branch", "--show-current"),
            "head": _run_git("rev-parse", "--short", "HEAD"),
            "status": _run_git("status", "--short"),
        },
        "required_files_missing": missing_files,
        "import_errors": import_errors,
        "dataset": dataset_info,
        "packages": packages,
        "missing_required_packages": missing_packages,
        "safety": {
            "source_files_modified": False,
            "git_operations_performed": False,
            "network_or_model_download_attempted": False,
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = RESULTS / f"preflight_{stamp}.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("MEMOROUTE FINAL PREFLIGHT")
    print(f"Repository: {ROOT}")
    print(f"Branch: {report['git']['branch']}; HEAD: {report['git']['head']}")
    print(f"Dataset: {dataset_info}")
    print(f"Packages: {packages}")
    print(f"Missing files: {missing_files}")
    print(f"Import errors: {import_errors}")
    print(f"Report: {output}")

    if missing_files or import_errors or missing_packages or "error" in dataset_info:
        print("PREFLIGHT: FAIL — resolve the reported issues before running the experiment.")
        return 2

    if dataset_info.get("records") != 32 or dataset_info.get("queries") != 20:
        print(
            "PREFLIGHT: FAIL — the retrieval-temporal dataset shape differs "
            "from the registered 32/20 contract."
        )
        return 2

    print("PREFLIGHT: PASS")
    print("Existing Git modifications are reported, not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
