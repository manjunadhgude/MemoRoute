
"""Controlled synthetic comparison of recent-history and BM25 retrieval."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from memoroute.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
)
from memoroute.memory.models import MemoryRecord, MemorySource
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.recent import RecentHistoryRetriever

BASE_TIME = datetime(2026, 10, 1, tzinfo=UTC)
SOURCE = MemorySource(kind="synthetic", reference="memoroute-baseline-v1")
LIMIT = 3

def make_id(number: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{number:012d}")

MEMORIES = [
    (1, "Invoice INV-204 was paid on 2026-10-12 by bank transfer."),
    (2, "Customer prefers email receipts rather than printed copies."),
    (3, "The deployment uses PostgreSQL 17 with a Redis cache."),
    (4, "Refund for order ORD-882 was approved but bank settlement is pending."),
    (5, "Invoice INV-203 was issued on 2026-10-20 and is awaiting approval."),
    (6, "The customer changed their billing address; the updated statement is in the portal."),
    (7, "Deployment health checks passed after the Redis cache restart."),
    (8, "Refund for order ORD-881 was rejected because the return window expired."),
    (9, "The support team will send the maintenance notice on Monday."),
    (10, "The warehouse received 240 units against purchase order PO-771."),
    (11, "The API timeout was increased to 30 seconds during the release."),
    (12, "The analytics dashboard now refreshes every morning at 06:00 UTC."),
]

QUERIES = [
    (
        "invoice_payment",
        "When was invoice INV-204 paid?",
        make_id(1),
    ),
    (
        "receipt_preference",
        "How should customer receipts be delivered?",
        make_id(2),
    ),
    (
        "database_version",
        "Which PostgreSQL version does deployment use?",
        make_id(3),
    ),
    (
        "refund_status",
        "What is the status of refund ORD-882?",
        make_id(4),
    ),
]

def build_records() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            id=make_id(number),
            content=content,
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=number - 1),
        )
        for number, content in MEMORIES
    ]

def evaluate(name, retrieve, *, store):
    query_results = []
    per_query = []

    for query_name, query, relevant_id in QUERIES:
        results = retrieve(query)
        retrieved_ids = [record.id for record in results]
        relevant = {relevant_id}

        rank = next(
            (index for index, record_id in enumerate(retrieved_ids, start=1)
             if record_id == relevant_id),
            None,
        )

        query_results.append((retrieved_ids, relevant))
        per_query.append((query_name, rank, retrieved_ids))

    recall_1 = sum(
        recall_at_k(retrieved, relevant, k=1)
        for retrieved, relevant in query_results
    ) / len(query_results)

    recall_3 = sum(
        recall_at_k(retrieved, relevant, k=3)
        for retrieved, relevant in query_results
    ) / len(query_results)

    mrr = mean_reciprocal_rank(query_results)

    print(f"\n=== {name} ===")
    for query_name, rank, retrieved_ids in per_query:
        rank_text = str(rank) if rank is not None else "not retrieved"
        print(f"{query_name}: relevant rank = {rank_text}")
        print(f"  Top results: {[str(item) for item in retrieved_ids]}")

    print(f"Recall@1: {recall_1:.3f}")
    print(f"Recall@3: {recall_3:.3f}")
    print(f"MRR:      {mrr:.3f}")

    return {"recall@1": recall_1, "recall@3": recall_3, "mrr": mrr}

def main():
    with TemporaryDirectory(prefix="memoroute-baseline-") as temp_dir:
        database_path = Path(temp_dir) / "experiment.sqlite3"
        store = SQLiteMemoryStore(database_path)

        for record in build_records():
            store.save_record(record)

        print("Dataset records:", len(store.list_records()))
        print("Queries:", len(QUERIES))
        print("Retrieval limit K:", LIMIT)
        print("Relevant records per query: 1")

        recent = RecentHistoryRetriever(store)
        bm25 = BM25Retriever(store)

        recent_metrics = evaluate(
            "RECENT HISTORY",
            lambda _query: recent.retrieve(limit=LIMIT),
            store=store,
        )

        bm25_metrics = evaluate(
            "BM25",
            lambda query: bm25.retrieve(query, limit=LIMIT),
            store=store,
        )

        print("\n=== COMPARISON ===")
        print(f"{'Metric':<12} {'Recent':>10} {'BM25':>10}")
        for metric in ("recall@1", "recall@3", "mrr"):
            print(
                f"{metric:<12} "
                f"{recent_metrics[metric]:>10.3f} "
                f"{bm25_metrics[metric]:>10.3f}"
            )

if __name__ == "__main__":
    main()
