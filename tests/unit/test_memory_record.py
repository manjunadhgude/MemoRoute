
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memoroute.memory.models import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
)
from memoroute.memory.temporal import TemporalInfo, TemporalKind


def make_record(**overrides):
    values = {
        "content": "The user lives in Vijayawada.",
        "source": MemorySource(
            kind="conversation",
            reference="conversation-001",
        ),
    }
    values.update(overrides)
    return MemoryRecord(**values)

def test_memory_record_has_valid_defaults():
    record = make_record()

    assert isinstance(record.id, type(uuid4()))
    assert record.recorded_at.tzinfo is not None
    assert record.source_at is None
    assert record.temporal.kind == TemporalKind.ATEMPORAL

def test_memory_content_cannot_be_empty():
    with pytest.raises(ValueError):
        make_record(content="   ")

def test_source_kind_cannot_be_empty():
    with pytest.raises(ValueError):
        MemorySource(kind=" ")

def test_source_reference_cannot_be_blank():
    with pytest.raises(ValueError):
        MemorySource(kind="conversation", reference=" ")

def test_source_must_be_a_memory_source():
    with pytest.raises(TypeError):
        make_record(source="conversation")

def test_recorded_time_must_be_timezone_aware():
    with pytest.raises(ValueError):
        make_record(recorded_at=datetime(2026, 11, 1, 10, 0))

def test_source_time_must_be_timezone_aware():
    with pytest.raises(ValueError):
        make_record(source_at=datetime(2026, 11, 1, 10, 0))

def test_source_and_recorded_times_are_independent():
    source_time = datetime(2026, 1, 1, tzinfo=UTC)
    ingestion_time = datetime(2026, 3, 1, tzinfo=UTC)

    record = make_record(
        source_at=source_time,
        recorded_at=ingestion_time,
    )

    assert record.source_at == source_time
    assert record.recorded_at == ingestion_time

def test_memory_record_rejects_invalid_temporal_info():
    with pytest.raises(TypeError):
        make_record(temporal="state")

def test_memory_record_can_represent_a_state():
    from memoroute.memory.temporal import TemporalBoundary

    temporal = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.unknown(),
        valid_until=TemporalBoundary.unbounded(),
    )

    record = make_record(temporal=temporal)

    assert record.temporal == temporal

def test_memory_record_is_immutable():
    record = make_record()

    with pytest.raises(AttributeError):
        record.content = "Changed claim"

def test_relation_records_supersession():
    old_id = uuid4()
    new_id = uuid4()

    relation = MemoryRelation(
        source_id=new_id,
        target_id=old_id,
        kind=MemoryRelationKind.SUPERSEDES,
    )

    assert relation.source_id == new_id
    assert relation.target_id == old_id
    assert relation.kind == MemoryRelationKind.SUPERSEDES

def test_relation_records_correction():
    relation = MemoryRelation(
        source_id=uuid4(),
        target_id=uuid4(),
        kind=MemoryRelationKind.CORRECTS,
    )

    assert relation.kind == MemoryRelationKind.CORRECTS

def test_relation_records_contradiction():
    relation = MemoryRelation(
        source_id=uuid4(),
        target_id=uuid4(),
        kind=MemoryRelationKind.CONTRADICTS,
    )

    assert relation.kind == MemoryRelationKind.CONTRADICTS

def test_relation_cannot_reference_itself():
    memory_id = uuid4()

    with pytest.raises(ValueError):
        MemoryRelation(
            source_id=memory_id,
            target_id=memory_id,
            kind=MemoryRelationKind.CONTRADICTS,
        )

def test_relation_requires_uuid_references():
    with pytest.raises(TypeError):
        MemoryRelation(
            source_id="not-a-uuid",
            target_id=uuid4(),
            kind=MemoryRelationKind.SUPERSEDES,
        )

def test_relation_requires_valid_kind():
    with pytest.raises(TypeError):
        MemoryRelation(
            source_id=uuid4(),
            target_id=uuid4(),
            kind="supersedes",
        )

def test_relation_is_immutable():
    relation = MemoryRelation(
        source_id=uuid4(),
        target_id=uuid4(),
        kind=MemoryRelationKind.SUPERSEDES,
    )

    with pytest.raises(AttributeError):
        relation.kind = MemoryRelationKind.CORRECTS
