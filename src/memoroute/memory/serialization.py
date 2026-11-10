
"""Versioned JSON serialization for MemoRoute memory models."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from memoroute.memory.models import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
)
from memoroute.memory.temporal import (
    BoundaryKind,
    TemporalBoundary,
    TemporalInfo,
    TemporalKind,
)

SCHEMA_VERSION = 1

def _datetime_to_string(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None

def _parse_uuid(value: Any, field: str) -> UUID:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a UUID string.")
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid UUID string.") from exc

def _parse_datetime(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO 8601 string or null.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO 8601 timestamp.") from exc

def _loads_json(payload: str, field: str) -> Any:
    if not isinstance(payload, str):
        raise TypeError(f"{field} payload must be a string.")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} payload must contain valid JSON.") from exc

def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object.")
    return value

def _require_keys(
    value: dict[str, Any],
    expected: set[str],
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"{field} has invalid fields; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )

def _check_version(data: dict[str, Any]) -> None:
    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {version!r}.")

def _boundary_to_dict(boundary: TemporalBoundary) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": boundary.kind.value}
    if boundary.kind == BoundaryKind.KNOWN:
        result["value"] = boundary.value.isoformat()
    return result

def _boundary_from_dict(data: Any, field: str) -> TemporalBoundary:
    value = _require_object(data, field)
    kind = BoundaryKind(value.get("kind"))

    if kind == BoundaryKind.KNOWN:
        _require_keys(value, {"kind", "value"}, field)
        timestamp = _parse_datetime(value["value"], f"{field}.value")
        return TemporalBoundary.known(timestamp)

    _require_keys(value, {"kind"}, field)

    if kind == BoundaryKind.UNKNOWN:
        return TemporalBoundary.unknown()
    return TemporalBoundary.unbounded()

def _temporal_to_dict(temporal: TemporalInfo) -> dict[str, Any]:
    if temporal.kind == TemporalKind.ATEMPORAL:
        return {"kind": temporal.kind.value}

    if temporal.kind == TemporalKind.EVENT:
        return {
            "kind": temporal.kind.value,
            "occurred_at": _datetime_to_string(temporal.occurred_at),
        }

    return {
        "kind": temporal.kind.value,
        "valid_from": _boundary_to_dict(temporal.valid_from),
        "valid_until": _boundary_to_dict(temporal.valid_until),
    }

def _temporal_from_dict(data: Any) -> TemporalInfo:
    value = _require_object(data, "temporal")
    kind = TemporalKind(value.get("kind"))

    if kind == TemporalKind.ATEMPORAL:
        _require_keys(value, {"kind"}, "temporal")
        return TemporalInfo(kind=kind)

    if kind == TemporalKind.EVENT:
        _require_keys(value, {"kind", "occurred_at"}, "temporal")
        return TemporalInfo(
            kind=kind,
            occurred_at=_parse_datetime(
                value["occurred_at"], "temporal.occurred_at"
            ),
        )

    _require_keys(
        value,
        {"kind", "valid_from", "valid_until"},
        "temporal",
    )
    return TemporalInfo(
        kind=kind,
        valid_from=_boundary_from_dict(value["valid_from"], "temporal.valid_from"),
        valid_until=_boundary_from_dict(value["valid_until"], "temporal.valid_until"),
    )

def memory_record_to_dict(record: MemoryRecord) -> dict[str, Any]:
    if not isinstance(record, MemoryRecord):
        raise TypeError("record must be a MemoryRecord.")

    return {
        "schema_version": SCHEMA_VERSION,
        "type": "memory_record",
        "id": str(record.id),
        "content": record.content,
        "source": {
            "kind": record.source.kind,
            "reference": record.source.reference,
            "locator": record.source.locator,
        },
        "source_at": _datetime_to_string(record.source_at),
        "recorded_at": record.recorded_at.isoformat(),
        "temporal": _temporal_to_dict(record.temporal),
    }

def memory_record_from_dict(data: Any) -> MemoryRecord:
    value = _require_object(data, "memory record")
    _require_keys(
        value,
        {
            "schema_version", "type", "id", "content", "source",
            "source_at", "recorded_at", "temporal",
        },
        "memory record",
    )
    _check_version(value)

    if value["type"] != "memory_record":
        raise ValueError("type must be 'memory_record'.")

    source_data = _require_object(value["source"], "source")
    _require_keys(
        source_data,
        {"kind", "reference", "locator"},
        "source",
    )

    return MemoryRecord(
        id=_parse_uuid(value["id"], "id"),
        content=value["content"],
        source=MemorySource(
            kind=source_data["kind"],
            reference=source_data["reference"],
            locator=source_data["locator"],
        ),
        source_at=_parse_datetime(value["source_at"], "source_at"),
        recorded_at=_parse_datetime(value["recorded_at"], "recorded_at"),
        temporal=_temporal_from_dict(value["temporal"]),
    )

def memory_relation_to_dict(relation: MemoryRelation) -> dict[str, Any]:
    if not isinstance(relation, MemoryRelation):
        raise TypeError("relation must be a MemoryRelation.")

    return {
        "schema_version": SCHEMA_VERSION,
        "type": "memory_relation",
        "id": str(relation.id),
        "source_id": str(relation.source_id),
        "target_id": str(relation.target_id),
        "kind": relation.kind.value,
        "recorded_at": relation.recorded_at.isoformat(),
    }

def memory_relation_from_dict(data: Any) -> MemoryRelation:
    value = _require_object(data, "memory relation")
    _require_keys(
        value,
        {
            "schema_version", "type", "id", "source_id",
            "target_id", "kind", "recorded_at",
        },
        "memory relation",
    )
    _check_version(value)

    if value["type"] != "memory_relation":
        raise ValueError("type must be 'memory_relation'.")

    return MemoryRelation(
        id=_parse_uuid(value["id"], "id"),
        source_id=_parse_uuid(value["source_id"], "source_id"),
        target_id=_parse_uuid(value["target_id"], "target_id"),
        kind=MemoryRelationKind(value["kind"]),
        recorded_at=_parse_datetime(value["recorded_at"], "recorded_at"),
    )

def dumps_memory_record(record: MemoryRecord) -> str:
    return json.dumps(memory_record_to_dict(record), ensure_ascii=False)

def loads_memory_record(payload: str) -> MemoryRecord:
    return memory_record_from_dict(_loads_json(payload, "Memory record"))

def dumps_memory_relation(relation: MemoryRelation) -> str:
    return json.dumps(memory_relation_to_dict(relation), ensure_ascii=False)

def loads_memory_relation(payload: str) -> MemoryRelation:
    return memory_relation_from_dict(_loads_json(payload, "Memory relation"))
