
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TemporalKind(StrEnum):
    ATEMPORAL = "atemporal"
    EVENT = "event"
    STATE = "state"

class BoundaryKind(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    UNBOUNDED = "unbounded"

class TemporalResult(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INDETERMINATE = "indeterminate"

@dataclass(frozen=True)
class TemporalBoundary:
    """One boundary of a state validity interval."""

    kind: BoundaryKind
    value: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryKind):
            raise TypeError("kind must be a BoundaryKind value.")

        if self.kind == BoundaryKind.KNOWN:
            if not isinstance(self.value, datetime):
                raise TypeError("A known boundary requires a datetime.")
            _validate_aware_datetime("boundary value", self.value)
        elif self.value is not None:
            raise ValueError(
                "Unknown and unbounded boundaries cannot have a datetime value."
            )

    @classmethod
    def known(cls, value: datetime) -> "TemporalBoundary":
        return cls(kind=BoundaryKind.KNOWN, value=value)

    @classmethod
    def unknown(cls) -> "TemporalBoundary":
        return cls(kind=BoundaryKind.UNKNOWN)

    @classmethod
    def unbounded(cls) -> "TemporalBoundary":
        return cls(kind=BoundaryKind.UNBOUNDED)

@dataclass(frozen=True)
class TemporalInfo:
    """Temporal classification and timing information for one claim."""

    kind: TemporalKind
    occurred_at: datetime | None = None
    valid_from: TemporalBoundary | None = None
    valid_until: TemporalBoundary | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TemporalKind):
            raise TypeError("kind must be a TemporalKind value.")

        if self.kind == TemporalKind.ATEMPORAL:
            if any(
                value is not None
                for value in (
                    self.occurred_at,
                    self.valid_from,
                    self.valid_until,
                )
            ):
                raise ValueError("Atemporal claims cannot contain time fields.")

        elif self.kind == TemporalKind.EVENT:
            if self.valid_from is not None or self.valid_until is not None:
                raise ValueError("Events cannot have state validity boundaries.")

            if self.occurred_at is not None:
                _validate_aware_datetime("occurred_at", self.occurred_at)

        elif self.kind == TemporalKind.STATE:
            if self.occurred_at is not None:
                raise ValueError("States cannot have an event occurrence time.")

            if not isinstance(self.valid_from, TemporalBoundary):
                raise TypeError("States require a valid_from boundary.")

            if not isinstance(self.valid_until, TemporalBoundary):
                raise TypeError("States require a valid_until boundary.")

            start = self.valid_from
            end = self.valid_until

            if (
                start.kind == BoundaryKind.KNOWN
                and end.kind == BoundaryKind.KNOWN
                and end.value <= start.value
            ):
                raise ValueError(
                    "A known valid_until must be later than valid_from."
                )

    def evaluate_at(self, when: datetime) -> TemporalResult:
        """Evaluate whether a state applied at a particular time."""
        if self.kind != TemporalKind.STATE:
            raise TypeError("evaluate_at is only available for state claims.")

        _validate_aware_datetime("when", when)

        start = self.valid_from
        end = self.valid_until

        if start.kind == BoundaryKind.KNOWN and when < start.value:
            return TemporalResult.INVALID

        if end.kind == BoundaryKind.KNOWN and when >= end.value:
            return TemporalResult.INVALID

        if (
            start.kind == BoundaryKind.UNKNOWN
            or end.kind == BoundaryKind.UNKNOWN
        ):
            return TemporalResult.INDETERMINATE

        return TemporalResult.VALID

def _validate_aware_datetime(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
