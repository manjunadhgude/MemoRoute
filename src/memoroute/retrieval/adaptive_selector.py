"""Conservative, relation-aware adaptive context selection.

This layer consumes the existing retrieval and temporal-assessment outputs.
It does not alter retrieval scores, temporal assessment, or memory relations.
"""
from dataclasses import dataclass
from uuid import UUID

from memoroute.memory.models import MemoryRecord, MemoryRelationKind
from memoroute.retrieval.context_packet import ContextPacket, build_context_packet
from memoroute.retrieval.query_intent import QueryIntent, QueryPurpose
from memoroute.retrieval.temporal_selector import (
    TemporalAssessmentStatus,
    TemporalQueryMode,
    TemporalSelectionResult,
)


@dataclass(frozen=True, slots=True)
class AdaptiveSelectionResult:
    ranked_candidates: tuple[MemoryRecord, ...]
    context_packet: ContextPacket
    roles: tuple[tuple[UUID, str], ...]
    explanation: tuple[str, ...]


def select_adaptive_context(
    candidates,
    assessment: TemporalSelectionResult,
    intent: QueryIntent,
    *,
    token_budget: int,
) -> AdaptiveSelectionResult:
    """Rank already-retrieved candidates conservatively and build a packet.

    Relation direction is descriptive evidence, not proof that a correction
    or superseding claim is true. No candidate is removed.
    """
    records = tuple(candidates)
    if not isinstance(assessment, TemporalSelectionResult):
        raise TypeError("assessment must be a TemporalSelectionResult")
    if not isinstance(intent, QueryIntent):
        raise TypeError("intent must be a QueryIntent")
    if records != assessment.candidates:
        raise ValueError("assessment must correspond to candidates in the same order")
    if assessment.mode != intent.mode or assessment.as_of != intent.as_of:
        raise ValueError("intent temporal mode and timestamp must match the assessment")
    if not isinstance(intent.purpose, QueryPurpose):
        raise TypeError("intent.purpose must be a QueryPurpose")

    by_id = {item.candidate_id: item for item in assessment.assessments}
    if len(by_id) != len(assessment.assessments):
        raise ValueError("assessment contains duplicate candidate IDs")
    if set(by_id) != {record.id for record in records}:
        raise ValueError("assessment IDs do not match candidate IDs")

    # [tier, original rank, record, role]. Lower tier is placed earlier.
    rows = []
    for record in records:
        item = by_id[record.id]
        status = item.temporal_status
        if intent.mode == TemporalQueryMode.UNSPECIFIED:
            tier, role = 0, "primary"
        elif intent.purpose == QueryPurpose.STATE_VERIFICATION:
            if status == TemporalAssessmentStatus.VALID:
                role = "state_verification_valid_evidence"
            elif status == TemporalAssessmentStatus.INDETERMINATE:
                role = "state_verification_uncertain_evidence"
            elif status == TemporalAssessmentStatus.INVALID:
                role = "state_verification_temporal_mismatch_evidence"
            else:
                role = "non_state_or_not_evaluated"
            tier = 0
        elif status == TemporalAssessmentStatus.VALID:
            tier, role = 0, "temporally_valid"
        elif status == TemporalAssessmentStatus.INDETERMINATE:
            tier, role = 1, "uncertain_temporal_evidence"
        elif status == TemporalAssessmentStatus.INVALID:
            tier, role = 2, "temporal_mismatch_context"
        else:
            tier, role = 0, "non_state_or_not_evaluated"
        rows.append({
            "tier": tier,
            "rank": item.original_rank,
            "record": record,
            "role": role,
        })

    row_by_id = {row["record"].id: row for row in rows}
    candidate_ids = set(row_by_id)
    relation_seen = set()
    relation_notes = set()

    for item in assessment.assessments:
        for evidence in item.relations:
            if evidence.relation_id in relation_seen:
                continue
            relation_seen.add(evidence.relation_id)

            source_present = evidence.source_id in candidate_ids
            target_present = evidence.target_id in candidate_ids
            if evidence.kind == MemoryRelationKind.CONTRADICTS:
                if intent.purpose in (QueryPurpose.COMPARISON, QueryPurpose.CONTRADICTION):
                    if source_present and target_present:
                        for endpoint in (evidence.source_id, evidence.target_id):
                            row_by_id[endpoint]["tier"] = 0
                            row_by_id[endpoint]["role"] = "paired_contradiction_evidence"
                        relation_notes.add(
                            "A retrieved contradiction pair is kept together in ranking; "
                            "neither side is treated as automatically correct."
                        )
                    else:
                        present = evidence.source_id if source_present else evidence.target_id
                        if present in row_by_id:
                            row_by_id[present]["role"] = "partial_contradiction_evidence"
                        relation_notes.add(
                            "A contradiction relation has an endpoint outside "
                            "the retrieved candidates; "
                            "the missing endpoint was not fabricated or fetched."
                        )
                continue

            if evidence.kind in (MemoryRelationKind.SUPERSEDES, MemoryRelationKind.CORRECTS):
                if source_present:
                    row_by_id[evidence.source_id]["role"] = "directed_replacement_candidate"
                if target_present:
                    row_by_id[evidence.target_id]["role"] = "directed_prior_claim_context"
                if source_present and target_present:
                    relation_notes.add(
                        "Directed SUPERSEDES/CORRECTS evidence is labelled source-to-target; "
                        "direction alone does not establish truth."
                    )
                else:
                    present = evidence.source_id if source_present else evidence.target_id
                    if present in row_by_id:
                        row_by_id[present]["role"] = "partial_directed_relation_evidence"
                    relation_notes.add(
                        "A directed relation has an endpoint outside the retrieved candidates; "
                        "its available direction is retained as partial evidence."
                    )

    rows.sort(key=lambda row: (row["tier"], row["rank"]))
    ordered = tuple(row["record"] for row in rows)
    role_map = {row["record"].id: row["role"] for row in rows}
    original_ranks = {row["record"].id: row["rank"] for row in rows}
    packet = build_context_packet(
        ordered,
        token_budget=token_budget,
        roles=role_map,
        original_ranks=original_ranks,
    )

    explanation = [
        (
            "Candidate relevance is inherited from upstream retrieval order; "
            "this layer does not calculate semantic similarity or combine incomparable "
            "BM25/vector/RRF scores."
        ),
        (
            "Temporal status changes ordering only for timestamped CURRENT/AS_OF intents; "
            "UNSPECIFIED does not apply a temporal penalty."
        ),
        (
            "Directed correction/supersession relations are exposed as evidence, "
            "not treated as proof that the source claim is true."
        ),
        (
            "Only already-retrieved candidates are considered; "
            "this selector does not fetch missing relation endpoints."
        ),
        (
            "The context packet scans all ranked candidates and may include a later smaller item "
            "when an earlier item does not fit."
        ),
        "Token counts use the documented rough whitespace-word estimate, not a model tokenizer.",
    ]
    explanation.extend(sorted(relation_notes))
    if packet.omitted_candidate_ids:
        explanation.append(
            "Some candidates were omitted because the estimated context budget was exhausted."
        )
    return AdaptiveSelectionResult(
        ranked_candidates=ordered,
        context_packet=packet,
        roles=tuple((record.id, role_map[record.id]) for record in ordered),
        explanation=tuple(explanation),
    )
