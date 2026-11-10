"""Tests for SQLite-backed temporal assessment."""


import pytest

from memoroute.memory.models import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
)
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.temporal_selector import (
    RelationDirection,
    RelationStatus,
    TemporalQueryMode,
)
from memoroute.retrieval.temporal_store import (
    assess_temporal_candidates_from_store,
)


def make_record(content: str) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        source=MemorySource(kind="test"),
    )


def test_adapter_preserves_candidate_order_and_relation_direction(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    first = make_record("Newer claim.")
    second = make_record("Older claim.")

    relation = MemoryRelation(
        source_id=first.id,
        target_id=second.id,
        kind=MemoryRelationKind.SUPERSEDES,
    )

    store.save_record(first)
    store.save_record(second)
    store.save_relation(relation)

    result = assess_temporal_candidates_from_store(
        store,
        [second, first],
        mode=TemporalQueryMode.UNSPECIFIED,
    )

    assert result.candidates == (second, first)
    assert [item.candidate_id for item in result.assessments] == [
        second.id,
        first.id,
    ]
    assert [item.original_rank for item in result.assessments] == [1, 2]

    second_assessment, first_assessment = result.assessments
    assert second_assessment.relation_status == RelationStatus.RESOLVED
    assert second_assessment.relations[0].candidate_direction == RelationDirection.TARGET
    assert first_assessment.relations[0].candidate_direction == RelationDirection.SOURCE


def test_adapter_returns_empty_result_for_empty_candidates(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    result = assess_temporal_candidates_from_store(
        store,
        iter(()),
        mode=TemporalQueryMode.UNSPECIFIED,
    )

    assert result.candidates == ()
    assert result.assessments == ()


def test_adapter_marks_relation_to_outside_candidate_as_partial(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    candidate = make_record("Candidate claim.")
    outside = make_record("Related claim.")

    relation = MemoryRelation(
        source_id=candidate.id,
        target_id=outside.id,
        kind=MemoryRelationKind.CONTRADICTS,
    )

    store.save_record(candidate)
    store.save_record(outside)
    store.save_relation(relation)

    result = assess_temporal_candidates_from_store(
        store,
        [candidate],
        mode=TemporalQueryMode.UNSPECIFIED,
    )

    assessment = result.assessments[0]
    assert assessment.relation_status == RelationStatus.PARTIAL
    assert len(assessment.relations) == 1
    assert assessment.relations[0].other_record_id == outside.id
    assert assessment.relations[0].other_endpoint_in_candidates is False


def test_adapter_rejects_invalid_candidate_type(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    with pytest.raises(TypeError, match="Every candidate must be a MemoryRecord"):
        assess_temporal_candidates_from_store(
            store,
            ["not a memory record"],
            mode=TemporalQueryMode.UNSPECIFIED,
        )


def test_adapter_preserves_selector_timestamp_validation(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    with pytest.raises(TypeError, match="require an as_of datetime"):
        assess_temporal_candidates_from_store(
            store,
            [],
            mode=TemporalQueryMode.CURRENT,
        )
