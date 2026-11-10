"""SQLite-backed temporal assessment adapter."""

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from memoroute.memory.models import MemoryRecord, MemoryRelation
from memoroute.retrieval.temporal_selector import (
    TemporalQueryMode,
    TemporalSelectionResult,
    assess_temporal_candidates,
)


class RelationStore(Protocol):
    """Store interface required for bulk temporal relation lookup."""

    def get_relations_for_records(
        self, record_ids: Iterable[UUID]
    ) -> list[MemoryRelation]:
        """Return relations touching any supplied record IDs."""
        ...

def assess_temporal_candidates_from_store(
    store: RelationStore,
    candidates: Iterable[MemoryRecord],
    *,
    mode: TemporalQueryMode,
    as_of: datetime | None = None,
) -> TemporalSelectionResult:
    """Load candidate relations and return temporal assessments."""
    records = tuple(candidates)

    if any(not isinstance(record, MemoryRecord) for record in records):
        raise TypeError("Every candidate must be a MemoryRecord.")

    relations = (
        store.get_relations_for_records(record.id for record in records)
        if records
        else ()
    )

    return assess_temporal_candidates(
        records,
        mode=mode,
        as_of=as_of,
        relations=relations,
    )
