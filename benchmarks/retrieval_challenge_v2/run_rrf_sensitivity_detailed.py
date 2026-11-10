
"""Detailed, reproducible RRF sensitivity experiment for MemoRoute v2."""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from benchmarks.retrieval_challenge_v2.dataset import (
    QUERIES,
    validate_dataset,
)
from benchmarks.retrieval_challenge_v2.run_experiment import (
    LIMIT,
    build_records,
)
from memoroute.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
)
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.hybrid import HybridRetriever
from memoroute.retrieval.vector import VectorRetriever

K_VALUES = (10, 20, 40, 60, 80, 100)
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def make_id(number: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{number:012d}")

class FixedRanking:
    """Expose a previously computed ranking through the retriever interface."""

    def __init__(self, query: str, ranking: list) -> None:
        self.query = query
        self.ranking = ranking

    def retrieve(self, query: str, limit: int = 10) -> list:
        if query != self.query:
            raise ValueError("Cached ranking received an unexpected query.")
        return self.ranking[:limit]

def package_version(package: str):
    try:
        return version(package)
    except PackageNotFoundError:
        return None

def git_metadata() -> dict:
    def run_git(*args):
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "working_tree_status": run_git("status", "--short"),
    }

def serialize_ranking(ranking: list, relevant_ids: set[UUID]) -> dict:
    positions = {
        record.id: rank
        for rank, record in enumerate(ranking, start=1)
    }

    relevant_ranks = {
        str(record_id): positions.get(record_id)
        for record_id in sorted(relevant_ids, key=str)
    }

    return {
        "records": [
            {
                "rank": rank,
                "id": str(record.id),
                "content": record.content,
            }
            for rank, record in enumerate(ranking, start=1)
        ],
        "relevant_memory_ranks": relevant_ranks,
        "first_relevant_rank": min(
            (rank for rank in relevant_ranks.values() if rank is not None),
            default=None,
        ),
    }

def calculate_metrics(query_results: list) -> dict:
    return {
        "recall@1": sum(
            recall_at_k(ids, relevant, k=1)
            for ids, relevant in query_results
        ) / len(query_results),
        "recall@3": sum(
            recall_at_k(ids, relevant, k=3)
            for ids, relevant in query_results
        ) / len(query_results),
        "mrr": mean_reciprocal_rank(query_results),
    }

def main() -> None:
    validate_dataset()

    with TemporaryDirectory(prefix="memoroute-rrf-detailed-") as temp_dir:
        store = SQLiteMemoryStore(Path(temp_dir) / "challenge.sqlite3")

        for record in build_records():
            store.save_record(record)

        depth = len(store.list_records())
        bm25 = BM25Retriever(store)
        vector = VectorRetriever(store, model_name=MODEL_NAME)

        query_details = []
        baseline_results = {"bm25": [], "bge_vector": []}
        hybrid_results = {k: [] for k in K_VALUES}

        for query_name, category, query, relevant_numbers in QUERIES:
            relevant_ids = {make_id(number) for number in relevant_numbers}

            # Compute each baseline ranking only once per query.
            bm25_ranking = bm25.retrieve(query, limit=depth)
            vector_ranking = vector.retrieve(query, limit=depth)

            bm25_ids = [record.id for record in bm25_ranking]
            vector_ids = [record.id for record in vector_ranking]

            baseline_results["bm25"].append((bm25_ids, relevant_ids))
            baseline_results["bge_vector"].append((vector_ids, relevant_ids))

            query_item = {
                "query_name": query_name,
                "category": category,
                "query": query,
                "relevant_memory_ids": [
                    str(record_id)
                    for record_id in sorted(relevant_ids, key=str)
                ],
                "bm25": serialize_ranking(bm25_ranking, relevant_ids),
                "bge_vector": serialize_ranking(vector_ranking, relevant_ids),
                "hybrid": {},
            }

            # Reuse the exact baseline rankings for every RRF k value.
            for k in K_VALUES:
                hybrid = HybridRetriever(
                    FixedRanking(query, bm25_ranking),
                    FixedRanking(query, vector_ranking),
                    rrf_k=k,
                    candidate_limit=depth,
                )

                ranking = hybrid.retrieve(query, limit=depth)
                ranking_ids = [record.id for record in ranking]

                hybrid_results[k].append((ranking_ids, relevant_ids))
                query_item["hybrid"][str(k)] = serialize_ranking(
                    ranking,
                    relevant_ids,
                )

            query_details.append(query_item)

        summaries = {
            "bm25": calculate_metrics(baseline_results["bm25"]),
            "bge_vector": calculate_metrics(baseline_results["bge_vector"]),
            "hybrid": {
                str(k): calculate_metrics(hybrid_results[k])
                for k in K_VALUES
            },
        }

        result = {
            "experiment": "MemoRoute v2 detailed RRF sensitivity",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "configuration": {
                "dataset": "retrieval_challenge_v2",
                "memory_count": depth,
                "query_count": len(QUERIES),
                "candidate_depth": depth,
                "final_evaluation_k": LIMIT,
                "rrf_k_values": list(K_VALUES),
                "embedding_model": MODEL_NAME,
                "ranking_strategy": (
                    "BM25 and BGE rankings computed once per query; "
                    "the same rankings reused for every RRF k."
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {
                    name: package_version(name)
                    for name in (
                        "memoroute",
                        "fastembed",
                        "onnxruntime",
                        "numpy",
                        "pytest",
                    )
                },
                "git": git_metadata(),
            },
            "aggregate_metrics": summaries,
            "queries": query_details,
        }

        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"rrf_sensitivity_detailed_{timestamp}.json"

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing evidence: {output_path}"
            )

        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print("\n=== DETAILED RRF SENSITIVITY RESULTS ===")
        print(f"Memories: {depth}")
        print(f"Queries: {len(QUERIES)}")
        print(f"Candidate depth: {depth}")
        print(f"Final evaluation K: {LIMIT}")
        print(f"Embedding model: {MODEL_NAME}")

        print("\nBaseline comparison:")
        print(f"{'Method':<16} {'Recall@1':>10} {'Recall@3':>10} {'MRR':>10}")
        for name in ("bm25", "bge_vector"):
            metrics = summaries[name]
            print(
                f"{name:<16} "
                f"{metrics['recall@1']:>10.3f} "
                f"{metrics['recall@3']:>10.3f} "
                f"{metrics['mrr']:>10.3f}"
            )

        print("\nRRF sensitivity:")
        print(f"{'RRF k':>8} {'Recall@1':>10} {'Recall@3':>10} {'MRR':>10}")
        for k in K_VALUES:
            metrics = summaries["hybrid"][str(k)]
            print(
                f"{k:>8} "
                f"{metrics['recall@1']:>10.3f} "
                f"{metrics['recall@3']:>10.3f} "
                f"{metrics['mrr']:>10.3f}"
            )

        print("\nSaved complete rankings and metadata to:")
        print(output_path)

if __name__ == "__main__":
    main()
