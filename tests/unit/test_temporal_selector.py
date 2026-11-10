from datetime import UTC, datetime

import pytest

from memoroute.memory import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
    TemporalBoundary,
    TemporalInfo,
    TemporalKind,
)
from memoroute.retrieval.temporal_selector import (
    CandidateDisposition,
    RelationDirection,
    RelationStatus,
    TemporalAssessmentStatus,
    TemporalQueryMode,
    assess_temporal_candidates,
)


def state(content, start, end):
    return MemoryRecord(
        content=content,
        source=MemorySource(kind="test"),
        temporal=TemporalInfo(
            kind=TemporalKind.STATE,
            valid_from=start,
            valid_until=end,
        ),
    )


def relation(source, target, kind):
    return MemoryRelation(source_id=source.id, target_id=target.id, kind=kind)


def test_valid_state_is_kept():
    record = state(
        "Current",
        TemporalBoundary.known(datetime(2026, 1, 1, tzinfo=UTC)),
        TemporalBoundary.unbounded(),
    )
    result = assess_temporal_candidates(
        [record], mode=TemporalQueryMode.CURRENT,
        as_of=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert result.assessments[0].temporal_status == TemporalAssessmentStatus.VALID
    assert result.assessments[0].disposition == CandidateDisposition.KEEP


def test_invalid_state_is_retained():
    record = state(
        "Expired",
        TemporalBoundary.unbounded(),
        TemporalBoundary.known(datetime(2025, 1, 1, tzinfo=UTC)),
    )
    result = assess_temporal_candidates(
        [record], mode=TemporalQueryMode.CURRENT,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert result.assessments[0].temporal_status == TemporalAssessmentStatus.INVALID
    assert result.candidates == (record,)


def test_historical_query_keeps_old_and_new_claims():
    old = state(
        "Old",
        TemporalBoundary.known(datetime(2020, 1, 1, tzinfo=UTC)),
        TemporalBoundary.known(datetime(2025, 1, 1, tzinfo=UTC)),
    )
    new = state(
        "New",
        TemporalBoundary.known(datetime(2025, 1, 1, tzinfo=UTC)),
        TemporalBoundary.unbounded(),
    )
    rel = relation(new, old, MemoryRelationKind.SUPERSEDES)
    result = assess_temporal_candidates(
        [old, new], mode=TemporalQueryMode.AS_OF,
        as_of=datetime(2024, 1, 1, tzinfo=UTC), relations=[rel],
    )
    assert result.assessments[0].temporal_status == TemporalAssessmentStatus.VALID
    assert result.assessments[1].temporal_status == TemporalAssessmentStatus.INVALID
    assert result.assessments[0].relations[0].candidate_direction == RelationDirection.TARGET
    assert result.assessments[1].relations[0].candidate_direction == RelationDirection.SOURCE


def test_unknown_boundary_is_indeterminate():
    record = state(
        "Unknown",
        TemporalBoundary.unknown(),
        TemporalBoundary.unbounded(),
    )
    result = assess_temporal_candidates(
        [record], mode=TemporalQueryMode.AS_OF,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert result.assessments[0].temporal_status == TemporalAssessmentStatus.INDETERMINATE


def test_exclusive_end_is_invalid():
    end = datetime(2026, 1, 1, tzinfo=UTC)
    record = state(
        "Ended",
        TemporalBoundary.unbounded(),
        TemporalBoundary.known(end),
    )
    result = assess_temporal_candidates(
        [record], mode=TemporalQueryMode.AS_OF, as_of=end,
    )
    assert result.assessments[0].temporal_status == TemporalAssessmentStatus.INVALID


@pytest.mark.parametrize(
    "mode,timestamp,error",
    [
        (TemporalQueryMode.CURRENT, None, TypeError),
        (TemporalQueryMode.AS_OF, None, TypeError),
        (TemporalQueryMode.CURRENT, datetime(2026, 1, 1), ValueError),
        (TemporalQueryMode.AS_OF, datetime(2026, 1, 1), ValueError),
    ],
)
def test_timestamp_validation(mode, timestamp, error):
    with pytest.raises(error):
        assess_temporal_candidates([], mode=mode, as_of=timestamp)


def test_unspecified_preserves_order_and_does_not_evaluate():
    first = state("First", TemporalBoundary.unknown(), TemporalBoundary.unbounded())
    second = MemoryRecord(content="Second", source=MemorySource(kind="test"))
    result = assess_temporal_candidates(
        [first, second], mode=TemporalQueryMode.UNSPECIFIED,
    )
    assert result.candidates == (first, second)
    assert result.assessments[0].temporal_status == TemporalAssessmentStatus.NOT_EVALUATED
    assert result.assessments[1].temporal_status == TemporalAssessmentStatus.NOT_APPLICABLE


def test_unspecified_rejects_timestamp():
    with pytest.raises(ValueError):
        assess_temporal_candidates(
            [], mode=TemporalQueryMode.UNSPECIFIED,
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "kind", [MemoryRelationKind.SUPERSEDES, MemoryRelationKind.CORRECTS],
)
def test_directed_relation_direction(kind):
    old = MemoryRecord(content="Old", source=MemorySource(kind="test"))
    new = MemoryRecord(content="New", source=MemorySource(kind="test"))
    rel = relation(new, old, kind)
    result = assess_temporal_candidates(
        [old, new], mode=TemporalQueryMode.UNSPECIFIED, relations=[rel],
    )
    assert result.assessments[0].relations[0].candidate_direction == RelationDirection.TARGET
    assert result.assessments[1].relations[0].candidate_direction == RelationDirection.SOURCE


def test_contradiction_keeps_both_candidates():
    a = MemoryRecord(content="A", source=MemorySource(kind="test"))
    b = MemoryRecord(content="B", source=MemorySource(kind="test"))
    rel = relation(a, b, MemoryRelationKind.CONTRADICTS)
    result = assess_temporal_candidates(
        [a, b], mode=TemporalQueryMode.UNSPECIFIED, relations=[rel],
    )
    assert len(result.candidates) == 2
    assert all(x.relation_status == RelationStatus.RESOLVED for x in result.assessments)


def test_outside_endpoint_is_partial():
    a = MemoryRecord(content="A", source=MemorySource(kind="test"))
    b = MemoryRecord(content="B", source=MemorySource(kind="test"))
    rel = relation(a, b, MemoryRelationKind.SUPERSEDES)
    result = assess_temporal_candidates(
        [a], mode=TemporalQueryMode.UNSPECIFIED, relations=[rel],
    )
    assert result.assessments[0].relation_status == RelationStatus.PARTIAL
    assert result.candidates == (a,)


def test_relation_order_does_not_change_assessments():
    a = MemoryRecord(content="A", source=MemorySource(kind="test"))
    b = MemoryRecord(content="B", source=MemorySource(kind="test"))
    c = MemoryRecord(content="C", source=MemorySource(kind="test"))
    r1 = relation(a, b, MemoryRelationKind.CONTRADICTS)
    r2 = relation(c, a, MemoryRelationKind.CORRECTS)

    left = assess_temporal_candidates(
        [a, b, c], mode=TemporalQueryMode.UNSPECIFIED, relations=[r1, r2],
    )
    right = assess_temporal_candidates(
        [a, b, c], mode=TemporalQueryMode.UNSPECIFIED, relations=[r2, r1],
    )
    assert left.assessments == right.assessments


def test_duplicate_candidate_ids_rejected():
    a = MemoryRecord(content="A", source=MemorySource(kind="test"))
    with pytest.raises(ValueError):
        assess_temporal_candidates(
            [a, a], mode=TemporalQueryMode.UNSPECIFIED,
        )
