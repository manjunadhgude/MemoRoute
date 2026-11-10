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
from memoroute.retrieval.adaptive_selector import select_adaptive_context
from memoroute.retrieval.context_packet import build_context_packet
from memoroute.retrieval.query_intent import QueryIntent, QueryPurpose
from memoroute.retrieval.temporal_selector import (
    TemporalQueryMode,
    assess_temporal_candidates,
)


def record(content, start=None, end=None):
    temporal = (
        TemporalInfo(kind=TemporalKind.ATEMPORAL)
        if start is None and end is None
        else TemporalInfo(
            kind=TemporalKind.STATE,
            valid_from=start,
            valid_until=end,
        )
    )
    return MemoryRecord(content=content, source=MemorySource(kind="test"), temporal=temporal)


def intent(mode, purpose=QueryPurpose.LOOKUP, as_of=None):
    return QueryIntent(mode, as_of, purpose, False)


def test_unspecified_keeps_retrieval_order_and_ignores_temporal_tiering():
    invalid = record(
        "Expired deployment",
        TemporalBoundary.unbounded(),
        TemporalBoundary.known(datetime(2025, 1, 1, tzinfo=UTC)),
    )
    valid = record(
        "Current deployment",
        TemporalBoundary.known(datetime(2025, 1, 1, tzinfo=UTC)),
        TemporalBoundary.unbounded(),
    )
    candidates = (invalid, valid)
    assessment = assess_temporal_candidates(candidates, mode=TemporalQueryMode.UNSPECIFIED)
    result = select_adaptive_context(
        candidates, assessment, intent(TemporalQueryMode.UNSPECIFIED), token_budget=100
    )
    assert result.ranked_candidates == candidates


def test_current_mode_orders_valid_before_invalid_but_retains_both():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    invalid = record(
        "Old deployment",
        TemporalBoundary.unbounded(),
        TemporalBoundary.known(datetime(2025, 1, 1, tzinfo=UTC)),
    )
    valid = record(
        "Current deployment",
        TemporalBoundary.known(datetime(2025, 1, 1, tzinfo=UTC)),
        TemporalBoundary.unbounded(),
    )
    candidates = (invalid, valid)
    assessment = assess_temporal_candidates(
        candidates, mode=TemporalQueryMode.CURRENT, as_of=now
    )
    result = select_adaptive_context(
        candidates, assessment, intent(TemporalQueryMode.CURRENT, as_of=now), token_budget=100
    )
    assert result.ranked_candidates == (valid, invalid)
    assert {item.record.id for item in result.context_packet.items} == {invalid.id, valid.id}


@pytest.mark.parametrize("kind", [MemoryRelationKind.SUPERSEDES, MemoryRelationKind.CORRECTS])
def test_directed_relation_roles_retain_source_target_direction(kind):
    source = record("Replacement claim")
    target = record("Prior claim")
    relation = MemoryRelation(source_id=source.id, target_id=target.id, kind=kind)
    candidates = (target, source)
    assessment = assess_temporal_candidates(
        candidates, mode=TemporalQueryMode.UNSPECIFIED, relations=(relation,)
    )
    result = select_adaptive_context(
        candidates, assessment, intent(TemporalQueryMode.UNSPECIFIED), token_budget=100
    )
    roles = dict(result.roles)
    assert roles[source.id] == "directed_replacement_candidate"
    assert roles[target.id] == "directed_prior_claim_context"
    assert result.ranked_candidates == candidates


def test_contradiction_pair_is_labelled_for_comparison_without_claiming_truth():
    left = record("Production uses port 8080")
    right = record("Production uses port 9090")
    relation = MemoryRelation(
        source_id=left.id, target_id=right.id, kind=MemoryRelationKind.CONTRADICTS
    )
    candidates = (left, right)
    assessment = assess_temporal_candidates(
        candidates, mode=TemporalQueryMode.UNSPECIFIED, relations=(relation,)
    )
    result = select_adaptive_context(
        candidates,
        assessment,
        intent(TemporalQueryMode.UNSPECIFIED, QueryPurpose.COMPARISON),
        token_budget=100,
    )
    assert dict(result.roles) == {
        left.id: "paired_contradiction_evidence",
        right.id: "paired_contradiction_evidence",
    }
    assert any(
        "neither side is treated as automatically correct" in line for line in result.explanation
    )


def test_partial_directed_relation_is_disclosed():
    candidate = record("A corrected claim")
    outside = record("Earlier claim")
    relation = MemoryRelation(
        source_id=candidate.id, target_id=outside.id, kind=MemoryRelationKind.CORRECTS
    )
    assessment = assess_temporal_candidates(
        (candidate,), mode=TemporalQueryMode.UNSPECIFIED, relations=(relation,)
    )
    result = select_adaptive_context(
        (candidate,), assessment, intent(TemporalQueryMode.UNSPECIFIED), token_budget=100
    )
    assert dict(result.roles)[candidate.id] == "partial_directed_relation_evidence"
    assert any("endpoint outside" in line for line in result.explanation)


def test_intent_and_assessment_temporal_contract_must_match():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    candidate = record("Current claim")
    assessment = assess_temporal_candidates(
        (candidate,), mode=TemporalQueryMode.CURRENT, as_of=now
    )
    with pytest.raises(ValueError, match="must match"):
        select_adaptive_context(
            (candidate,),
            assessment,
            intent(TemporalQueryMode.AS_OF, as_of=now),
            token_budget=20,
        )


def test_packet_preserves_original_rank_after_adaptive_reordering():
    first = record("First candidate")
    second = record("Second candidate")
    packet = build_context_packet(
        (second, first),
        token_budget=100,
        original_ranks={first.id: 1, second.id: 2},
    )
    assert [item.original_rank for item in packet.items] == [2, 1]


def test_packet_skips_oversized_item_and_still_considers_later_small_item():
    large = record(" ".join(["large"] * 20))
    small = record("tiny")
    packet = build_context_packet((large, small), token_budget=2)
    assert [item.record.id for item in packet.items] == [small.id]
    assert packet.omitted_candidate_ids == (large.id,)
    assert packet.budget_exhausted is True


def test_packet_rejects_duplicate_candidate_ids():
    item = record("Repeated")
    with pytest.raises(ValueError, match="unique"):
        build_context_packet((item, item), token_budget=10)
