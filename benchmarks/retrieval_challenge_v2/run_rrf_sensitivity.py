
"""Evaluate MemoRoute hybrid RRF across multiple k values."""

from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.retrieval_challenge_v2.dataset import validate_dataset
from benchmarks.retrieval_challenge_v2.run_experiment import (
    LIMIT,
    build_records,
    evaluate,
)
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.hybrid import HybridRetriever
from memoroute.retrieval.vector import VectorRetriever


def main() -> None:
    validate_dataset()
    k_values = (10, 20, 40, 60, 80, 100)

    with TemporaryDirectory(prefix="memoroute-rrf-grid-") as temp_dir:
        store = SQLiteMemoryStore(Path(temp_dir) / "challenge.sqlite3")
        for record in build_records():
            store.save_record(record)

        depth = len(store.list_records())
        bm25 = BM25Retriever(store)
        vector = VectorRetriever(store)

        print(f"Records: {depth}")
        print(f"Candidate depth: {depth}")
        print(f"Final K: {LIMIT}")

        results = {}

        for k in k_values:
            hybrid = HybridRetriever(
                bm25,
                vector,
                rrf_k=k,
                candidate_limit=depth,
            )
            results[k] = evaluate(
                f"HYBRID RRF k={k}",
                lambda query, retriever=hybrid: retriever.retrieve(
                    query, limit=LIMIT
                ),
            )

        print("\n=== RRF SENSITIVITY COMPARISON ===")
        print(f"{'k':>6} {'Recall@1':>12} {'Recall@3':>12} {'MRR':>12}")

        for k, metrics in results.items():
            print(
                f"{k:>6} "
                f"{metrics['recall@1']:>12.3f} "
                f"{metrics['recall@3']:>12.3f} "
                f"{metrics['mrr']:>12.3f}"
            )

if __name__ == "__main__":
    main()
