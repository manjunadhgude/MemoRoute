
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memoroute.memory.models import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
)
from memoroute.memory.serialization import (
    dumps_memory_record,
    dumps_memory_relation,
    loads_memory_record,
    loads_memory_relation,
    memory_record_from_dict,
    memory_record_to_dict,
)
from memoroute.memory.temporal import (
    TemporalBoundary,
    TemporalInfo,
    TemporalKind,
)

RECORDED_AT = datetime(2026, 11, 1, 10, 30, tzinfo=UTC)
SOURCE_AT = datetime(2026, 10, 31, 9, 15, tzinfo=UTC)

def make_record(temporal=None):
    return MemoryRecord(
        content="The user prefers Python for backend development.",
        source=MemorySource(
            kind="conversation",
            reference="conversation-123",
            locator="message-45",
        ),
        source_at=SOURCE_AT,
        recorded_at=RECORDED_AT,
        temporal=temporal or TemporalInfo(kind=TemporalKind.ATEMPORAL),
    )

def test_atemporal_record_round_trip():
    record = make_record()

    restored = memory_record_from_dict(memory_record_to_dict(record))

    assert restored == record

def test_event_record_round_trip():
    temporal = TemporalInfo(
        kind=TemporalKind.EVENT,
        occurred_at=datetime(2026, 10, 30, 14, 0, tzinfo=UTC),
    )
    record = make_record(temporal)

    restored = memory_record_from_dict(memory_record_to_dict(record))

    assert restored == record

def test_event_with_unknown_occurrence_time_round_trip():
    record = make_record(TemporalInfo(kind=TemporalKind.EVENT))

    restored = loads_memory_record(dumps_memory_record(record))

    assert restored == record
    assert restored.temporal.occurred_at is None

@pytest.mark.parametrize(
    ("start", "end"),
    [
        (
            TemporalBoundary.known(datetime(2026, 1, 1, tzinfo=UTC)),
            TemporalBoundary.known(datetime(2027, 1, 1, tzinfo=UTC)),
        ),
        (TemporalBoundary.unknown(), TemporalBoundary.known(
            datetime(2027, 1, 1, tzinfo=UTC)
        )),
        (TemporalBoundary.unbounded(), TemporalBoundary.unknown()),
        (TemporalBoundary.known(datetime(2026, 1, 1, tzinfo=UTC)),
         TemporalBoundary.unbounded()),
    ],
)
def test_state_record_round_trip_preserves_boundaries(start, end):
    temporal = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=start,
        valid_until=end,
    )
    record = make_record(temporal)

    restored = loads_memory_record(dumps_memory_record(record))

    assert restored == record

def test_record_json_preserves_provenance_and_timestamps():
    record = make_record()

    restored = loads_memory_record(dumps_memory_record(record))

    assert restored.id == record.id
    assert restored.source == record.source
    assert restored.source_at == SOURCE_AT
    assert restored.recorded_at == RECORDED_AT
    assert restored.recorded_at.tzinfo is not None

@pytest.mark.parametrize("kind", list(MemoryRelationKind))
def test_relation_json_round_trip(kind):
    relation = MemoryRelation(
        source_id=uuid4(),
        target_id=uuid4(),
        kind=kind,
        recorded_at=RECORDED_AT,
    )

    restored = loads_memory_relation(dumps_memory_relation(relation))

    assert restored == relation

def test_record_rejects_unsupported_schema_version():
    data = memory_record_to_dict(make_record())
    data["schema_version"] = 999

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        memory_record_from_dict(data)

def test_record_rejects_unexpected_fields():
    data = memory_record_to_dict(make_record())
    data["unexpected"] = "not allowed"

    with pytest.raises(ValueError, match="invalid fields"):
        memory_record_from_dict(data)

def test_record_rejects_naive_datetime():
    data = memory_record_to_dict(make_record())
    data["recorded_at"] = "2026-11-01T10:30:00"

    with pytest.raises(ValueError, match="timezone-aware"):
        memory_record_from_dict(data)

def test_state_boundary_rejects_missing_known_value():
    data = memory_record_to_dict(
        make_record(
            TemporalInfo(
                kind=TemporalKind.STATE,
                valid_from=TemporalBoundary.unknown(),
                valid_until=TemporalBoundary.unbounded(),
            )
        )
    )
    data["temporal"]["valid_from"] = {"kind": "known"}

    with pytest.raises(ValueError):
        memory_record_from_dict(data)

def test_record_rejects_malformed_uuid():
    data = memory_record_to_dict(make_record())
    data["id"] = "not-a-uuid"

    with pytest.raises(ValueError):
        memory_record_from_dict(data)

def test_record_rejects_non_string_uuid():
    data = memory_record_to_dict(make_record())
    data["id"] = 123

    with pytest.raises(TypeError):
        memory_record_from_dict(data)

def test_record_rejects_malformed_timestamp():
    data = memory_record_to_dict(make_record())
    data["recorded_at"] = "not-a-timestamp"

    with pytest.raises(ValueError):
        memory_record_from_dict(data)

def test_record_rejects_non_string_timestamp():
    data = memory_record_to_dict(make_record())
    data["recorded_at"] = 123

    with pytest.raises(TypeError):
        memory_record_from_dict(data)

def test_json_loader_rejects_malformed_json():
    with pytest.raises(ValueError):
        loads_memory_record("{invalid json}")

def test_json_loader_rejects_non_string_payload():
    with pytest.raises(TypeError):
        loads_memory_record(123)
