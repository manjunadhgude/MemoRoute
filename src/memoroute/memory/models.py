from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from memoroute.memory.temporal import TemporalInfo, TemporalKind


@dataclass(frozen=True)
class MemorySource:
    """Describes where a memory originated."""

    kind: str
    reference: str | None = None
    locator: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("Source kind must be a non-empty string.")

        for name, value in (
            ("reference", self.reference),
            ("locator", self.locator),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Source {name} must be non-empty when provided.")


@dataclass(frozen=True)
class MemoryRecord:
    """An immutable, atomic claim with provenance and temporal information."""

    content: str
    source: MemorySource

    id: UUID = field(default_factory=uuid4)
    source_at: datetime | None = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    temporal: TemporalInfo = field(
        default_factory=lambda: TemporalInfo(kind=TemporalKind.ATEMPORAL)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Memory content must be a non-empty string.")

        if not isinstance(self.source, MemorySource):
            raise TypeError("source must be a MemorySource instance.")

        if not isinstance(self.id, UUID):
            raise TypeError("Memory ID must be a UUID.")

        if not isinstance(self.temporal, TemporalInfo):
            raise TypeError("temporal must be a TemporalInfo instance.")

        _validate_timestamp("source_at", self.source_at, allow_none=True)
        _validate_timestamp("recorded_at", self.recorded_at)


class MemoryRelationKind(StrEnum):
    SUPERSEDES = "supersedes"
    CORRECTS = "corrects"
    CONTRADICTS = "contradicts"


@dataclass(frozen=True)
class MemoryRelation:
    """An immutable, directed relationship between two memory records."""

    source_id: UUID
    target_id: UUID
    kind: MemoryRelationKind

    id: UUID = field(default_factory=uuid4)
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, UUID):
            raise TypeError("source_id must be a UUID.")

        if not isinstance(self.target_id, UUID):
            raise TypeError("target_id must be a UUID.")

        if self.source_id == self.target_id:
            raise ValueError("A memory relation cannot reference itself.")

        if not isinstance(self.kind, MemoryRelationKind):
            raise TypeError("kind must be a MemoryRelationKind value.")

        if not isinstance(self.id, UUID):
            raise TypeError("Relation ID must be a UUID.")

        _validate_timestamp("recorded_at", self.recorded_at)


def _validate_timestamp(name: str, value: datetime | None, *, allow_none: bool = False) -> None:
    if value is None:
        if allow_none:
            return
        raise TypeError(f"{name} must be a datetime.")

    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
