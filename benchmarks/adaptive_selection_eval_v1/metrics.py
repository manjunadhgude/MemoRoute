"""Transparent ranking and context-packet metrics for the adaptive evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

METRIC_KEYS = (
    "recall_at_1",
    "recall_at_3",
    "hit_rate_at_1",
    "hit_rate_at_3",
    "mrr",
)


def ranking_metrics(ranked_ids: Sequence[int], relevant_ids: Iterable[int]) -> dict[str, float]:
    """Calculate macro-ready per-query metrics from numeric memory IDs."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must not be empty")

    ranked = list(ranked_ids)

    def recall_at(k: int) -> float:
        return len(set(ranked[:k]) & relevant) / len(relevant)

    def hit_at(k: int) -> float:
        return float(bool(set(ranked[:k]) & relevant))

    reciprocal_rank = 0.0
    for rank, item_id in enumerate(ranked, start=1):
        if item_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "hit_rate_at_1": hit_at(1),
        "hit_rate_at_3": hit_at(3),
        "mrr": reciprocal_rank,
    }


def aggregate_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average per-query metrics; every query receives equal weight."""
    if not rows:
        raise ValueError("at least one metric row is required")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in METRIC_KEYS
    }


def packet_coverage(packet_ids: Iterable[int], relevant_ids: Iterable[int]) -> float:
    """Fraction of hand-labelled relevant memories retained in a context packet."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    return len(set(packet_ids) & relevant) / len(relevant)
