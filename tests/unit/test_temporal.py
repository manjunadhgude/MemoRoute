
from datetime import UTC, datetime

import pytest

from memoroute.memory.temporal import (
    BoundaryKind,
    TemporalBoundary,
    TemporalInfo,
    TemporalKind,
)


def test_atemporal_claim_has_no_time_fields():
    info = TemporalInfo(kind=TemporalKind.ATEMPORAL)

    assert info.kind == TemporalKind.ATEMPORAL

def test_event_can_have_a_known_occurrence_time():
    occurred_at = datetime(2026, 3, 10, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.EVENT,
        occurred_at=occurred_at,
    )

    assert info.occurred_at == occurred_at

def test_event_can_have_an_unknown_occurrence_time():
    info = TemporalInfo(kind=TemporalKind.EVENT)

    assert info.occurred_at is None

def test_state_can_have_known_and_unbounded_boundaries():
    start = datetime(2022, 1, 1, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.known(start),
        valid_until=TemporalBoundary.unbounded(),
    )

    assert info.valid_from.kind == BoundaryKind.KNOWN
    assert info.valid_until.kind == BoundaryKind.UNBOUNDED

def test_state_can_preserve_unknown_boundaries():
    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.unknown(),
        valid_until=TemporalBoundary.unknown(),
    )

    assert info.valid_from.kind == BoundaryKind.UNKNOWN
    assert info.valid_until.kind == BoundaryKind.UNKNOWN

def test_known_state_end_must_be_after_known_start():
    boundary = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValueError):
        TemporalInfo(
            kind=TemporalKind.STATE,
            valid_from=TemporalBoundary.known(boundary),
            valid_until=TemporalBoundary.known(boundary),
        )

def test_known_boundary_must_be_timezone_aware():
    naive_time = datetime(2026, 1, 1)

    with pytest.raises(ValueError):
        TemporalBoundary.known(naive_time)

def test_unknown_boundary_cannot_contain_a_datetime():
    with pytest.raises(ValueError):
        TemporalBoundary(
            kind=BoundaryKind.UNKNOWN,
            value=datetime(2026, 1, 1, tzinfo=UTC),
        )

def test_atemporal_claim_rejects_event_time():
    with pytest.raises(ValueError):
        TemporalInfo(
            kind=TemporalKind.ATEMPORAL,
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

def test_state_is_valid_inside_known_interval():
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    query = datetime(2023, 6, 1, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.known(start),
        valid_until=TemporalBoundary.known(end),
    )

    from memoroute.memory.temporal import TemporalResult

    assert info.evaluate_at(query) == TemporalResult.VALID

def test_state_is_invalid_before_known_start():
    start = datetime(2022, 1, 1, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.known(start),
        valid_until=TemporalBoundary.unbounded(),
    )

    from memoroute.memory.temporal import TemporalResult

    assert info.evaluate_at(datetime(2021, 1, 1, tzinfo=UTC)) == TemporalResult.INVALID

def test_state_is_invalid_at_exclusive_end():
    end = datetime(2025, 1, 1, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.unbounded(),
        valid_until=TemporalBoundary.known(end),
    )

    from memoroute.memory.temporal import TemporalResult

    assert info.evaluate_at(end) == TemporalResult.INVALID

def test_unknown_start_produces_indeterminate_result():
    end = datetime(2025, 1, 1, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.unknown(),
        valid_until=TemporalBoundary.known(end),
    )

    from memoroute.memory.temporal import TemporalResult

    assert (
        info.evaluate_at(datetime(2023, 1, 1, tzinfo=UTC))
        == TemporalResult.INDETERMINATE
    )

def test_known_end_can_prove_invalidity_even_if_start_is_unknown():
    end = datetime(2025, 1, 1, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.unknown(),
        valid_until=TemporalBoundary.known(end),
    )

    from memoroute.memory.temporal import TemporalResult

    assert info.evaluate_at(datetime(2026, 1, 1, tzinfo=UTC)) == TemporalResult.INVALID

def test_unknown_end_produces_indeterminate_result():
    start = datetime(2022, 1, 1, tzinfo=UTC)

    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.known(start),
        valid_until=TemporalBoundary.unknown(),
    )

    from memoroute.memory.temporal import TemporalResult

    assert (
        info.evaluate_at(datetime(2023, 1, 1, tzinfo=UTC))
        == TemporalResult.INDETERMINATE
    )

def test_two_unbounded_boundaries_mean_valid_at_any_time():
    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.unbounded(),
        valid_until=TemporalBoundary.unbounded(),
    )

    from memoroute.memory.temporal import TemporalResult

    assert (
        info.evaluate_at(datetime(1900, 1, 1, tzinfo=UTC))
        == TemporalResult.VALID
    )

def test_temporal_evaluation_rejects_naive_query_time():
    info = TemporalInfo(
        kind=TemporalKind.STATE,
        valid_from=TemporalBoundary.unbounded(),
        valid_until=TemporalBoundary.unbounded(),
    )

    with pytest.raises(ValueError):
        info.evaluate_at(datetime(2026, 1, 1))

def test_temporal_evaluation_is_only_for_states():
    info = TemporalInfo(kind=TemporalKind.EVENT)

    with pytest.raises(TypeError):
        info.evaluate_at(datetime(2026, 1, 1, tzinfo=UTC))
