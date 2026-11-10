from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from memoroute.memory.models import MemoryRecord, MemoryRelation, MemoryRelationKind
from memoroute.memory.temporal import TemporalKind, TemporalResult


class TemporalQueryMode(StrEnum):
    CURRENT = "current"
    AS_OF = "as_of"
    UNSPECIFIED = "unspecified"


class TemporalAssessmentStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"


class CandidateDisposition(StrEnum):
    KEEP = "keep"


class RelationStatus(StrEnum):
    NONE = "none"
    RESOLVED = "resolved"
    PARTIAL = "partial"


class RelationDirection(StrEnum):
    SOURCE = "source"
    TARGET = "target"


@dataclass(frozen=True, slots=True)
class RelationEvidence:
    relation_id: UUID
    kind: MemoryRelationKind
    source_id: UUID
    target_id: UUID
    candidate_direction: RelationDirection
    other_record_id: UUID
    other_endpoint_in_candidates: bool


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    candidate_id: UUID
    original_rank: int
    temporal_status: TemporalAssessmentStatus
    relation_status: RelationStatus
    relations: tuple[RelationEvidence, ...]
    reason_code: str
    disposition: CandidateDisposition = CandidateDisposition.KEEP


@dataclass(frozen=True, slots=True)
class TemporalSelectionResult:
    candidates: tuple[MemoryRecord, ...]
    assessments: tuple[CandidateAssessment, ...]
    mode: TemporalQueryMode
    as_of: datetime | None


def _aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def assess_temporal_candidates(
    candidates: Iterable[MemoryRecord],
    *,
    mode: TemporalQueryMode,
    as_of: datetime | None = None,
    relations: Iterable[MemoryRelation] = (),
) -> TemporalSelectionResult:
    if not isinstance(mode, TemporalQueryMode):
        raise TypeError("mode must be a TemporalQueryMode.")

    if mode == TemporalQueryMode.UNSPECIFIED:
        if as_of is not None:
            raise ValueError("UNSPECIFIED mode cannot include as_of.")
    else:
        if not isinstance(as_of, datetime):
            raise TypeError("CURRENT and AS_OF require an as_of datetime.")
        if not _aware(as_of):
            raise ValueError("as_of must be timezone-aware.")

    records = tuple(candidates)
    if any(not isinstance(r, MemoryRecord) for r in records):
        raise TypeError("Every candidate must be a MemoryRecord.")

    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique.")

    id_set = set(ids)
    relation_map = {}

    for relation in relations:
        if not isinstance(relation, MemoryRelation):
            raise TypeError("Every relation must be a MemoryRelation.")
        previous = relation_map.get(relation.id)
        if previous is not None and previous != relation:
            raise ValueError("Conflicting relations share an ID.")
        relation_map[relation.id] = relation

    evidence_map = {record_id: [] for record_id in ids}

    for relation in sorted(
        relation_map.values(),
        key=lambda r: (
            r.kind.value,
            r.source_id.hex,
            r.target_id.hex,
            r.id.hex,
        ),
    ):
        source_present = relation.source_id in id_set
        target_present = relation.target_id in id_set

        if source_present:
            evidence_map[relation.source_id].append(
                RelationEvidence(
                    relation_id=relation.id,
                    kind=relation.kind,
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    candidate_direction=RelationDirection.SOURCE,
                    other_record_id=relation.target_id,
                    other_endpoint_in_candidates=target_present,
                )
            )

        if target_present:
            evidence_map[relation.target_id].append(
                RelationEvidence(
                    relation_id=relation.id,
                    kind=relation.kind,
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    candidate_direction=RelationDirection.TARGET,
                    other_record_id=relation.source_id,
                    other_endpoint_in_candidates=source_present,
                )
            )

    assessments = []

    for rank, record in enumerate(records, start=1):
        if record.temporal.kind != TemporalKind.STATE:
            status = TemporalAssessmentStatus.NOT_APPLICABLE
            reason = "NON_STATE_NOT_APPLICABLE"
        elif mode == TemporalQueryMode.UNSPECIFIED:
            status = TemporalAssessmentStatus.NOT_EVALUATED
            reason = "STATE_NOT_EVALUATED"
        else:
            result = record.temporal.evaluate_at(as_of)
            status = {
                TemporalResult.VALID: TemporalAssessmentStatus.VALID,
                TemporalResult.INVALID: TemporalAssessmentStatus.INVALID,
                TemporalResult.INDETERMINATE: TemporalAssessmentStatus.INDETERMINATE,
            }[result]
            reason = {
                TemporalAssessmentStatus.VALID: "STATE_VALID",
                TemporalAssessmentStatus.INVALID: "STATE_INVALID_RETAINED",
                TemporalAssessmentStatus.INDETERMINATE: "STATE_INDETERMINATE",
            }[status]

        evidence = tuple(evidence_map[record.id])

        if not evidence:
            relation_status = RelationStatus.NONE
        elif all(e.other_endpoint_in_candidates for e in evidence):
            relation_status = RelationStatus.RESOLVED
        else:
            relation_status = RelationStatus.PARTIAL

        assessments.append(
            CandidateAssessment(
                candidate_id=record.id,
                original_rank=rank,
                temporal_status=status,
                relation_status=relation_status,
                relations=evidence,
                reason_code=reason,
            )
        )

    return TemporalSelectionResult(
        candidates=records,
        assessments=tuple(assessments),
        mode=mode,
        as_of=as_of,
    )
