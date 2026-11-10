"""Conservative query-intent representation; no hidden model inference."""
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from memoroute.retrieval.temporal_selector import TemporalQueryMode


class QueryPurpose(StrEnum):
    LOOKUP = "lookup"
    HISTORICAL = "historical"
    COMPARISON = "comparison"
    CORRECTION = "correction"
    CONTRADICTION = "contradiction"
    EVENT = "event"
    STATE_VERIFICATION = "state_verification"

@dataclass(frozen=True, slots=True)
class QueryIntent:
    mode: TemporalQueryMode
    as_of: datetime | None
    purpose: QueryPurpose
    explicit_temporal_cue: bool

def infer_query_intent(query: str, *, mode=None, as_of=None) -> QueryIntent:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    text = query.casefold()
    historical = any(
        x in text
        for x in ("as of", "at that time", "historically", "back then", "previously")
    )
    current = any(
        x in text
        for x in ("current", "currently", "latest", "right now", "today", "as of now")
    )
    if mode is None:
        resolved = TemporalQueryMode.UNSPECIFIED
    else:
        if not isinstance(mode, TemporalQueryMode):
            raise TypeError("mode must be a TemporalQueryMode")
        resolved = mode
    if resolved == TemporalQueryMode.UNSPECIFIED and as_of is not None:
        raise ValueError("UNSPECIFIED mode cannot include as_of")
    if resolved != TemporalQueryMode.UNSPECIFIED:
        if not isinstance(as_of, datetime):
            raise TypeError("CURRENT and AS_OF require an as_of datetime")
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
    if any(x in text for x in ("compare", "versus", " vs ", "difference between")):
        purpose = QueryPurpose.COMPARISON
    elif any(x in text for x in ("contradict", "conflict", "disagree", "inconsistent")):
        purpose = QueryPurpose.CONTRADICTION
    elif any(x in text for x in ("corrected", "correction", "amended", "revised")):
        purpose = QueryPurpose.CORRECTION
    elif historical:
        purpose = QueryPurpose.HISTORICAL
    elif any(x in text for x in ("when did", "when was", "occurred", "happened")):
        purpose = QueryPurpose.EVENT
    elif any(
        cue in text
        for cue in (
            "still active",
            "active on",
            "active at",
            "still available",
            "available on",
            "available at",
            "still valid",
            "valid on",
            "valid at",
            "expired",
            "in effect on",
            "in effect at",
        )
    ):
        purpose = QueryPurpose.STATE_VERIFICATION
    else:
        purpose = QueryPurpose.LOOKUP
    return QueryIntent(resolved, as_of, purpose, historical or current)
