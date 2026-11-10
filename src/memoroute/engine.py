
"""End-to-end orchestration for MemoRoute retrieval and context selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memoroute.retrieval.adaptive_selector import select_adaptive_context
from memoroute.retrieval.query_intent import QueryIntent, infer_query_intent
from memoroute.retrieval.recent import RecentHistoryRetriever
from memoroute.retrieval.temporal_selector import TemporalQueryMode
from memoroute.retrieval.temporal_store import assess_temporal_candidates_from_store

@dataclass(frozen=True)
class MemoRouteResult:
    """Inspectable output of one complete MemoRoute retrieval run."""

    query: str
    intent: QueryIntent
    retrieved_candidates: tuple
    assessment: Any
    adaptive_selection: Any
    context_packet: Any

    @property
    def explanation(self):
        """Return the adaptive selector's explanation."""
        return self.adaptive_selection.explanation

class MemoRouteEngine:
    """Compose retrieval, temporal assessment and adaptive context selection.

    The engine accepts already-constructed dependencies. It does not own
    database initialization, embedding model setup or answer generation.
    """

    def __init__(self, *, store: Any, retriever: Any) -> None:
        if store is None:
            raise TypeError("store is required.")
        if retriever is None or not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must expose a callable retrieve method.")

        self.store = store
        self.retriever = retriever

    def run(
        self,
        query: str,
        *,
        mode: TemporalQueryMode | str = TemporalQueryMode.UNSPECIFIED,
        as_of: datetime | None = None,
        limit: int = 10,
        token_budget: int = 1200,
    ) -> MemoRouteResult:
        """Run retrieval through final budgeted context construction.

        CURRENT captures one UTC timestamp when the caller does not supply
        one. AS_OF requires an explicit timestamp. UNSPECIFIED has no timestamp.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if limit <= 0:
            raise ValueError("limit must be positive.")

        try:
            resolved_mode = (
                mode if isinstance(mode, TemporalQueryMode)
                else TemporalQueryMode(mode)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported temporal mode: {mode!r}") from exc

        resolved_as_of = as_of
        if resolved_mode == TemporalQueryMode.CURRENT and resolved_as_of is None:
            resolved_as_of = datetime.now(timezone.utc)

        intent = infer_query_intent(
            query,
            mode=resolved_mode,
            as_of=resolved_as_of,
        )

        if isinstance(self.retriever, RecentHistoryRetriever):
            retrieved = self.retriever.retrieve(limit=limit)
        else:
            retrieved = self.retriever.retrieve(query, limit=limit)

        candidates = tuple(retrieved)

        assessment = assess_temporal_candidates_from_store(
            self.store,
            candidates,
            mode=intent.mode,
            as_of=intent.as_of,
        )

        adaptive_selection = select_adaptive_context(
            candidates,
            assessment,
            intent,
            token_budget=token_budget,
        )

        return MemoRouteResult(
            query=query,
            intent=intent,
            retrieved_candidates=candidates,
            assessment=assessment,
            adaptive_selection=adaptive_selection,
            context_packet=adaptive_selection.context_packet,
        )
