
"""Run the MemoRoute v2 synthetic retrieval challenge."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from benchmarks.retrieval_challenge_v2.dataset import (
    MEMORIES,
    QUERIES,
    validate_dataset,
)
from memoroute.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
)
from memoroute.memory.models import MemoryRecord, MemorySource
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.recent import RecentHistoryRetriever

BASE_TIME = datetime(2026, 10, 1, tzinfo=UTC)
SOURCE = MemorySource(kind="synthetic", reference="memoroute-challenge-v2")
LIMIT = 3

def make_id(number: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{number:012d}")

def build_records() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            id=make_id(number),
            content=content,
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=number - 1),
        )
        for number, _category, content in MEMORIES
    ]

def evaluate(name, retrieve):
    query_results = []
    details = []

    for query_name, category, query, relevant_numbers in QUERIES:
        results = retrieve(query)
        retrieved_ids = [record.id for record in results]
        relevant_ids = {make_id(number) for number in relevant_numbers}

        rank = next(
            (
                index
                for index, record_id in enumerate(retrieved_ids, start=1)
                if record_id in relevant_ids
            ),
            None,
        )

        query_results.append((retrieved_ids, relevant_ids))
        details.append(
            {
                "name": query_name,
                "category": category,
                "rank": rank,
                "results": [int(record.id.hex[-12:], 10) for record in results],
            }
        )

    metrics = {
        "recall@1": sum(
            recall_at_k(results, relevant, k=1)
            for results, relevant in query_results
        ) / len(query_results),
        "recall@3": sum(
            recall_at_k(results, relevant, k=3)
            for results, relevant in query_results
        ) / len(query_results),
        "mrr": mean_reciprocal_rank(query_results),
    }

    print(f"\n=== {name} ===")

    for item in details:
        rank_text = str(item["rank"]) if item["rank"] else "MISS"
        print(
            f"{item['name']} [{item['category']}]: "
            f"rank={rank_text}, top IDs={item['results']}"
        )

    print("\nOverall:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    print("\nBy category:")
    for category in sorted({item[1] for item in QUERIES}):
        category_results = [
            (results, relevant)
            for (results, relevant), item in zip(
                query_results, details, strict=True
            )
            if item["category"] == category
        ]
        category_mrr = mean_reciprocal_rank(category_results)
        category_recall = sum(
            recall_at_k(results, relevant, k=3)
            for results, relevant in category_results
        ) / len(category_results)

        print(
            f"{category}: "
            f"Recall@3={category_recall:.3f}, "
            f"MRR={category_mrr:.3f}"
        )

    return metrics

def main():
    validate_dataset()

    with TemporaryDirectory(prefix="memoroute-v2-") as temp_dir:
        store = SQLiteMemoryStore(Path(temp_dir) / "challenge.sqlite3")

        for record in build_records():
            store.save_record(record)

        print("Dataset records:", len(store.list_records()))
        print("Queries:", len(QUERIES))
        print("Retrieval limit K:", LIMIT)
        print("Relevant labels per query: 1 or more")

        recent = RecentHistoryRetriever(store)
        bm25 = BM25Retriever(store)

        recent_metrics = evaluate(
            "RECENT HISTORY",
            lambda _query: recent.retrieve(limit=LIMIT),
        )

        bm25_metrics = evaluate(
            "BM25",
            lambda query: bm25.retrieve(query, limit=LIMIT),
        )

        print("\n=== FINAL COMPARISON ===")
        print(f"{'Metric':<12} {'Recent':>10} {'BM25':>10}")

        for metric in ("recall@1", "recall@3", "mrr"):
            print(
                f"{metric:<12} "
                f"{recent_metrics[metric]:>10.3f} "
                f"{bm25_metrics[metric]:>10.3f}"
            )

if __name__ == "__main__":
    main()
