"""Standard ranking metrics for memory retrieval experiments."""

from collections.abc import Hashable, Sequence, Set


def _validate_relevant(relevant: Set[Hashable]) -> None:
    if not isinstance(relevant, Set):
        raise TypeError("relevant must be a set of identifiers")
    if not relevant:
        raise ValueError("relevant must contain at least one identifier")

def recall_at_k(
    retrieved: Sequence[Hashable],
    relevant: Set[Hashable],
    *,
    k: int,
) -> float:
    """Return the fraction of relevant identifiers found in the first k results."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    _validate_relevant(relevant)
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)

def reciprocal_rank(
    retrieved: Sequence[Hashable],
    relevant: Set[Hashable],
) -> float:
    """Return the reciprocal rank of the first relevant result for one query."""
    _validate_relevant(relevant)

    for rank, identifier in enumerate(retrieved, start=1):
        if identifier in relevant:
            return 1.0 / rank

    return 0.0

def mean_reciprocal_rank(
    query_results: Sequence[tuple[Sequence[Hashable], Set[Hashable]]],
) -> float:
    """Average reciprocal rank across a collection of queries."""
    if not query_results:
        raise ValueError("query_results must contain at least one query")

    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in query_results]
    return sum(scores) / len(scores)
