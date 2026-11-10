"""Hybrid memory retrieval using Reciprocal Rank Fusion (RRF)."""

from typing import Protocol

from memoroute.memory.models import MemoryRecord


class RankedRetriever(Protocol):
    """Minimal interface shared by ranked memory retrievers."""

    def retrieve(self, query: str, limit: int = 10) -> list[MemoryRecord]: ...

class HybridRetriever:
    """Fuse lexical and semantic rankings without comparing their raw scores."""

    def __init__(
        self,
        bm25_retriever: RankedRetriever,
        vector_retriever: RankedRetriever,
        *,
        rrf_k: int = 60,
        candidate_limit: int = 50,
    ) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
            raise TypeError("rrf_k must be an integer.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("candidate_limit must be an integer.")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive.")

        self._bm25 = bm25_retriever
        self._vector = vector_retriever
        self._rrf_k = rrf_k
        self._candidate_limit = candidate_limit

    def retrieve(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Return unique records ordered by their fused reciprocal-rank score."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if not query.strip():
            return []

        depth = max(self._candidate_limit, limit)
        rankings = (
            self._bm25.retrieve(query, limit=depth),
            self._vector.retrieve(query, limit=depth),
        )

        scores: dict[object, float] = {}
        best_ranks: dict[object, int] = {}
        records: dict[object, MemoryRecord] = {}

        for ranking in rankings:
            seen_in_ranking: set[object] = set()
            for rank, record in enumerate(ranking, start=1):
                record_id = record.id
                if record_id in seen_in_ranking:
                    continue
                seen_in_ranking.add(record_id)
                records.setdefault(record_id, record)
                scores[record_id] = scores.get(record_id, 0.0) + (1.0 / (self._rrf_k + rank))
                best_ranks[record_id] = min(best_ranks.get(record_id, rank), rank)

        ordered_ids = sorted(
            scores,
            key=lambda record_id: (
                -scores[record_id],
                best_ranks[record_id],
                -records[record_id].recorded_at.timestamp(),
                str(record_id),
            ),
        )
        return [records[record_id] for record_id in ordered_ids[:limit]]
