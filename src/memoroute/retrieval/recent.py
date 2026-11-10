"""Recent-history memory retrieval."""

from typing import Protocol

from memoroute.memory.models import MemoryRecord


class RecordStore(Protocol):
    """Storage interface required by the recent-history retriever."""

    def list_records(self, limit: int | None = None) -> list[MemoryRecord]:
        """Return stored records, newest first."""
        ...

class RecentHistoryRetriever:
    """Retrieve the most recently recorded memories."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def retrieve(self, limit: int = 10) -> list[MemoryRecord]:
        """Return up to ``limit`` memories, newest first."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")

        if limit <= 0:
            raise ValueError("limit must be positive.")

        return self._store.list_records(limit=limit)
