
import pytest

from benchmarks.retrieval_temporal_v1.dataset import (
    QUERIES,
    build_records,
    build_relations,
    make_id,
)
from memoroute.retrieval.adaptive_selector import select_adaptive_context
from memoroute.retrieval.query_intent import QueryPurpose, infer_query_intent
from memoroute.retrieval.temporal_selector import (
    TemporalAssessmentStatus,
    TemporalQueryMode,
    assess_temporal_candidates,
)

SCENARIOS = {scenario.name: scenario for scenario in QUERIES}
RECORDS = {record.id: record for record in build_records()}
RELATIONS = build_relations()

def run_scenario(name, candidate_ids=None):
    scenario = SCENARIOS[name]
    ordered_ids = (
        scenario.assessment_candidate_ids
        if candidate_ids is None
        else candidate_ids
    )
    candidates = tuple(RECORDS[make_id(number)] for number in ordered_ids)
    candidate_set = {record.id for record in candidates}

    relations = tuple(
        relation
        for relation in RELATIONS
        if relation.source_id in candidate_set
        or relation.target_id in candidate_set
    )

    mode = TemporalQueryMode(scenario.mode)
    query_intent = infer_query_intent(
        scenario.question,
        mode=mode,
        as_of=scenario.as_of,
    )
    assessment = assess_temporal_candidates(
        candidates,
        mode=mode,
        as_of=scenario.as_of,
        relations=relations,
    )
    result = select_adaptive_context(
        candidates,
        assessment,
        query_intent,
        token_budget=256,
    )

    return scenario, query_intent, assessment, result

@pytest.mark.parametrize(
    ("name", "expected_role"),
    [
        (
            "expired_subscription",
            "state_verification_temporal_mismatch_evidence",
        ),
        (
            "future_feature",
            "state_verification_temporal_mismatch_evidence",
        ),
        (
            "unknown_retention_end",
            "state_verification_uncertain_evidence",
        ),
        (
            "unknown_rotation_start",
            "state_verification_uncertain_evidence",
        ),
    ],
)
def test_recovered_state_verification_scenarios_retain_relevant_evidence(
    name, expected_role
):
    scenario, query_intent, assessment, result = run_scenario(name)

    assert query_intent.purpose == QueryPurpose.STATE_VERIFICATION

    expected_statuses = {
        make_id(number): TemporalAssessmentStatus(status)
        for number, status in scenario.expected_temporal_statuses
    }
    actual_statuses = {
        item.candidate_id: item.temporal_status
        for item in assessment.assessments
    }
    assert actual_statuses == expected_statuses

    assert tuple(result.ranked_candidates) == assessment.candidates

    roles = dict(result.roles)
    relevant_ids = {make_id(number) for number in scenario.relevant_ids}
    packet_ids = {item.record.id for item in result.context_packet.items}

    assert relevant_ids.issubset(packet_ids)
    assert all(roles[record_id] == expected_role for record_id in relevant_ids)
    assert result.context_packet.estimated_tokens <= result.context_packet.token_budget

@pytest.mark.parametrize(
    ("name", "replacement_id", "prior_id"),
    [
        ("invoice_preference_current", 5, 6),
        ("production_database_current", 1, 2),
        ("superseded_database_claim", 1, 2),
    ],
)
def test_retained_current_claim_wins_and_directed_roles_are_preserved(
    name, replacement_id, prior_id
):
    scenario, query_intent, assessment, result = run_scenario(
        name,
        candidate_ids=(
            prior_id,
            replacement_id,
        ),
    )

    replacement_uuid = make_id(replacement_id)
    prior_uuid = make_id(prior_id)

    # The older claim is deliberately supplied first.
    assert assessment.candidates[0].id == prior_uuid
    assert result.ranked_candidates[0].id == replacement_uuid

    roles = dict(result.roles)
    assert roles[replacement_uuid] == "directed_replacement_candidate"
    assert roles[prior_uuid] == "directed_prior_claim_context"

    packet_ids = {item.record.id for item in result.context_packet.items}
    relevant_ids = {make_id(number) for number in scenario.relevant_ids}

    assert relevant_ids.issubset(packet_ids)
    assert result.context_packet.estimated_tokens <= result.context_packet.token_budget

def test_production_cache_remains_a_valid_valid_entity_disambiguation_case():
    scenario, query_intent, assessment, result = run_scenario(
        "production_cache_current"
    )

    assert query_intent.purpose == QueryPurpose.LOOKUP

    statuses = {
        item.candidate_id: item.temporal_status
        for item in assessment.assessments
    }
    assert statuses == {
        make_id(3): TemporalAssessmentStatus.VALID,
        make_id(4): TemporalAssessmentStatus.VALID,
    }

    # Temporal validity alone must not invent an environment-based winner.
    assert [record.id for record in result.ranked_candidates] == [
        make_id(3),
        make_id(4),
    ]

    packet_ids = {item.record.id for item in result.context_packet.items}
    assert {make_id(3), make_id(4)}.issubset(packet_ids)
    assert make_id(4) in packet_ids
