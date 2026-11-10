"""Read-only source/API contract audit for the adaptive evaluation pack."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "memoroute_final"

MODULES = {
    "dataset": "benchmarks.retrieval_temporal_v1.dataset",
    "store": "memoroute.memory.sqlite_store",
    "recent": "memoroute.retrieval.recent",
    "bm25": "memoroute.retrieval.bm25",
    "vector": "memoroute.retrieval.vector",
    "hybrid": "memoroute.retrieval.hybrid",
    "temporal_store": "memoroute.retrieval.temporal_store",
    "query_intent": "memoroute.retrieval.query_intent",
    "adaptive_selector": "memoroute.retrieval.adaptive_selector",
}

SOURCE_FILES = [
    "benchmarks/retrieval_temporal_v1/dataset.py",
    "src/memoroute/memory/sqlite_store.py",
    "src/memoroute/retrieval/recent.py",
    "src/memoroute/retrieval/bm25.py",
    "src/memoroute/retrieval/vector.py",
    "src/memoroute/retrieval/hybrid.py",
    "src/memoroute/retrieval/temporal_store.py",
    "src/memoroute/retrieval/query_intent.py",
    "src/memoroute/retrieval/adaptive_selector.py",
    "src/memoroute/retrieval/context_packet.py",
]


def _signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def main() -> int:
    loaded = {
        label: importlib.import_module(module_name)
        for label, module_name in MODULES.items()
    }
    dataset = loaded["dataset"]
    dataset.validate_dataset()

    signatures = {
        "SQLiteMemoryStore": _signature(loaded["store"].SQLiteMemoryStore),
        "save_record": _signature(loaded["store"].SQLiteMemoryStore.save_record),
        "save_relation": _signature(loaded["store"].SQLiteMemoryStore.save_relation),
        "BM25Retriever": _signature(loaded["bm25"].BM25Retriever),
        "BM25.retrieve": _signature(loaded["bm25"].BM25Retriever.retrieve),
        "VectorRetriever": _signature(loaded["vector"].VectorRetriever),
        "Vector.retrieve": _signature(loaded["vector"].VectorRetriever.retrieve),
        "HybridRetriever": _signature(loaded["hybrid"].HybridRetriever),
        "Hybrid.retrieve": _signature(loaded["hybrid"].HybridRetriever.retrieve),
        "assess_temporal_candidates_from_store": _signature(
            loaded["temporal_store"].assess_temporal_candidates_from_store
        ),
        "infer_query_intent": _signature(
            loaded["query_intent"].infer_query_intent
        ),
        "select_adaptive_context": _signature(
            loaded["adaptive_selector"].select_adaptive_context
        ),
    }

    selector_parameters = set(
        inspect.signature(
            loaded["adaptive_selector"].select_adaptive_context
        ).parameters
    )
    expected_selector_parameters = {
        "candidates", "assessment", "intent", "token_budget"
    }
    missing_selector_parameters = sorted(
        expected_selector_parameters - selector_parameters
    )

    file_hashes = {}
    missing_sources = []
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            missing_sources.append(relative)
            continue
        file_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {
            "records": len(dataset.build_records()),
            "relations": len(dataset.build_relations()),
            "queries": len(dataset.QUERIES),
            "query_fields": list(dataset.QUERIES[0].__dataclass_fields__),
        },
        "signatures": signatures,
        "selector_expected_parameters": sorted(expected_selector_parameters),
        "selector_missing_parameters": missing_selector_parameters,
        "source_sha256": file_hashes,
        "missing_source_files": missing_sources,
        "scope": "read-only; no source or Git mutation",
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = RESULTS / f"source_audit_{stamp}.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("SOURCE CONTRACT AUDIT")
    print(json.dumps(report["dataset"], indent=2))
    for name, signature in signatures.items():
        print(f"{name}: {signature}")
    print(f"Missing source files: {missing_sources}")
    print(f"Selector parameter mismatch: {missing_selector_parameters}")
    print(f"Report: {output}")

    if missing_sources or missing_selector_parameters:
        print("SOURCE AUDIT: FAIL")
        return 2
    print("SOURCE AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
