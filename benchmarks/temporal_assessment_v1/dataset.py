
"""Deterministic temporal assessment challenge v1 for MemoRoute."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from memoroute.memory import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
    TemporalBoundary,
    TemporalInfo,
    TemporalKind,
)
from memoroute.retrieval.temporal_selector import TemporalQueryMode

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
SOURCE = MemorySource(
    kind="synthetic",
    reference="memoroute-temporal-assessment-v1",
)

def make_id(number: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{number:012d}")

def make_relation_id(number: int) -> UUID:
    return make_id(1000 + number)

def timestamp(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)

def state(
    content: str,
    number: int,
    start: TemporalBoundary,
    end: TemporalBoundary,
) -> MemoryRecord:
    return MemoryRecord(
        id=make_id(number),
        content=content,
        source=SOURCE,
        recorded_at=BASE_TIME + timedelta(days=number),
        temporal=TemporalInfo(
            kind=TemporalKind.STATE,
            valid_from=start,
            valid_until=end,
        ),
    )

def build_records() -> list[MemoryRecord]:
    """Build 24 deterministic memories."""

    records = [
        state(
            "The active support plan is Standard.",
            1,
            TemporalBoundary.known(timestamp(2026, 1, 1)),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The historical support plan was Basic.",
            2,
            TemporalBoundary.known(timestamp(2024, 1, 1)),
            TemporalBoundary.known(timestamp(2025, 1, 1)),
        ),
        state(
            "The future support plan will be Premium.",
            3,
            TemporalBoundary.known(timestamp(2027, 1, 1)),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The previous office location was Building A.",
            4,
            TemporalBoundary.unbounded(),
            TemporalBoundary.known(timestamp(2026, 6, 1)),
        ),
        state(
            "The start date of the customer's current subscription is unknown.",
            5,
            TemporalBoundary.unknown(),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The end date of the customer's current contract is unknown.",
            6,
            TemporalBoundary.known(timestamp(2025, 1, 1)),
            TemporalBoundary.unknown(),
        ),
        state(
            "The customer used the Basic service plan.",
            7,
            TemporalBoundary.known(timestamp(2024, 1, 1)),
            TemporalBoundary.known(timestamp(2025, 1, 1)),
        ),
        state(
            "The customer uses the Premium service plan.",
            8,
            TemporalBoundary.known(timestamp(2025, 1, 1)),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The registered billing address was 10 North Street.",
            9,
            TemporalBoundary.known(timestamp(2024, 1, 1)),
            TemporalBoundary.known(timestamp(2026, 1, 1)),
        ),
        state(
            "The corrected billing address is 20 South Street.",
            10,
            TemporalBoundary.known(timestamp(2026, 1, 1)),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The customer account is marked as active.",
            11,
            TemporalBoundary.unknown(),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The customer account is marked as suspended.",
            12,
            TemporalBoundary.known(timestamp(2025, 1, 1)),
            TemporalBoundary.unbounded(),
        ),
        MemoryRecord(
            id=make_id(13),
            content="The revised delivery instruction supersedes an earlier instruction.",
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=13),
        ),
        MemoryRecord(
            id=make_id(14),
            content="A customer support call occurred on 10 February 2026.",
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=14),
            temporal=TemporalInfo(
                kind=TemporalKind.EVENT,
                occurred_at=timestamp(2026, 2, 10),
            ),
        ),
        MemoryRecord(
            id=make_id(15),
            content="The company name is Example Services.",
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=15),
        ),
        state(
            "The current project owner is Team Alpha.",
            16,
            TemporalBoundary.known(timestamp(2025, 1, 1)),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The previous project owner was Team Beta.",
            17,
            TemporalBoundary.known(timestamp(2024, 1, 1)),
            TemporalBoundary.known(timestamp(2025, 1, 1)),
        ),
        MemoryRecord(
            id=make_id(18),
            content="The project review meeting occurred on 15 March 2026.",
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=18),
            temporal=TemporalInfo(
                kind=TemporalKind.EVENT,
                occurred_at=timestamp(2026, 3, 15),
            ),
        ),
        state(
            "The new deployment region will be Region West.",
            19,
            TemporalBoundary.known(timestamp(2027, 1, 1)),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The previous deployment region was Region East.",
            20,
            TemporalBoundary.unbounded(),
            TemporalBoundary.known(timestamp(2025, 12, 1)),
        ),
        state(
            "The start date of the maintenance window is unknown.",
            21,
            TemporalBoundary.unknown(),
            TemporalBoundary.unbounded(),
        ),
        state(
            "The maintenance window started on 1 January 2025, but its end date is unknown.",
            22,
            TemporalBoundary.known(timestamp(2025, 1, 1)),
            TemporalBoundary.unknown(),
        ),
        MemoryRecord(
            id=make_id(23),
            content="The organisation operates in the software services industry.",
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=23),
        ),
        MemoryRecord(
            id=make_id(24),
            content="The earlier delivery instruction was to leave parcels at reception.",
            source=SOURCE,
            recorded_at=BASE_TIME + timedelta(days=24),
        ),
    ]

    return records

def build_relations() -> list[MemoryRelation]:
    """Build directed relations with deterministic IDs."""

    definitions = (
        (1, 8, 7, MemoryRelationKind.SUPERSEDES),
        (2, 10, 9, MemoryRelationKind.CORRECTS),
        (3, 11, 12, MemoryRelationKind.CONTRADICTS),
        (4, 13, 24, MemoryRelationKind.SUPERSEDES),
        (5, 16, 17, MemoryRelationKind.SUPERSEDES),
        (6, 19, 20, MemoryRelationKind.CONTRADICTS),
    )

    return [
        MemoryRelation(
            id=make_relation_id(relation_number),
            source_id=make_id(source_number),
            target_id=make_id(target_number),
            kind=kind,
            recorded_at=BASE_TIME + timedelta(days=30 + relation_number),
        )
        for relation_number, source_number, target_number, kind in definitions
    ]

@dataclass(frozen=True, slots=True)
class QueryCase:
    name: str
    question: str
    mode: TemporalQueryMode
    as_of: datetime | None
    candidate_ids: tuple[int, ...]
    relation_ids: tuple[int, ...]
    expected_statuses: tuple[tuple[int, str], ...]
    # (candidate_id, relation_number, direction, other_record_id, other_in_candidates)
    expected_relations: tuple[tuple[int, int, str, int, bool], ...] = ()

QUERIES = (
    QueryCase(
        name="current_valid_state",
        question="Which support plan is active?",
        mode=TemporalQueryMode.CURRENT,
        as_of=timestamp(2026, 11, 3),
        candidate_ids=(1,),
        relation_ids=(),
        expected_statuses=((1, "valid"),),
    ),
    QueryCase(
        name="historical_valid_state",
        question="Which support plan applied on 1 June 2024?",
        mode=TemporalQueryMode.AS_OF,
        as_of=timestamp(2024, 6, 1),
        candidate_ids=(2,),
        relation_ids=(),
        expected_statuses=((2, "valid"),),
    ),
    QueryCase(
        name="future_state_not_yet_valid",
        question="Which support plan applied on 1 June 2026?",
        mode=TemporalQueryMode.AS_OF,
        as_of=timestamp(2026, 6, 1),
        candidate_ids=(3,),
        relation_ids=(),
        expected_statuses=((3, "invalid"),),
    ),
    QueryCase(
        name="expired_state",
        question="Was Building A still the office location in November 2026?",
        mode=TemporalQueryMode.CURRENT,
        as_of=timestamp(2026, 11, 3),
        candidate_ids=(4,),
        relation_ids=(),
        expected_statuses=((4, "invalid"),),
    ),
    QueryCase(
        name="unknown_start_boundary",
        question="Is the subscription valid now?",
        mode=TemporalQueryMode.CURRENT,
        as_of=timestamp(2026, 11, 3),
        candidate_ids=(5,),
        relation_ids=(),
        expected_statuses=((5, "indeterminate"),),
    ),
    QueryCase(
        name="unspecified_time",
        question="What is known about the customer's contract?",
        mode=TemporalQueryMode.UNSPECIFIED,
        as_of=None,
        candidate_ids=(6,),
        relation_ids=(),
        expected_statuses=((6, "not_evaluated"),),
    ),
    QueryCase(
        name="historical_supersession",
        question="Which service plan applied on 1 June 2024?",
        mode=TemporalQueryMode.AS_OF,
        as_of=timestamp(2024, 6, 1),
        candidate_ids=(7, 8),
        relation_ids=(1,),
        expected_statuses=((7, "valid"), (8, "invalid")),
        expected_relations=(
            (7, 1, "target", 8, True),
            (8, 1, "source", 7, True),
        ),
    ),
    QueryCase(
        name="correction_direction",
        question="How are the old and corrected billing addresses related?",
        mode=TemporalQueryMode.UNSPECIFIED,
        as_of=None,
        candidate_ids=(9, 10),
        relation_ids=(2,),
        expected_statuses=((9, "not_evaluated"), (10, "not_evaluated")),
        expected_relations=(
            (9, 2, "target", 10, True),
            (10, 2, "source", 9, True),
        ),
    ),
    QueryCase(
        name="contradiction_preserves_both",
        question="What conflicting account status claims are recorded?",
        mode=TemporalQueryMode.CURRENT,
        as_of=timestamp(2026, 11, 3),
        candidate_ids=(11, 12),
        relation_ids=(3,),
        expected_statuses=((11, "indeterminate"), (12, "valid")),
        expected_relations=(
            (11, 3, "source", 12, True),
            (12, 3, "target", 11, True),
        ),
    ),
    QueryCase(
        name="partial_relation_endpoint",
        question="What is the revised delivery instruction?",
        mode=TemporalQueryMode.UNSPECIFIED,
        as_of=None,
        candidate_ids=(13,),
        relation_ids=(4,),
        expected_statuses=((13, "not_applicable"),),
        expected_relations=((13, 4, "source", 24, False),),
    ),
    QueryCase(
        name="event_and_atemporal",
        question="What event and general fact are recorded?",
        mode=TemporalQueryMode.CURRENT,
        as_of=timestamp(2026, 11, 3),
        candidate_ids=(14, 15, 23),
        relation_ids=(),
        expected_statuses=(
            (14, "not_applicable"),
            (15, "not_applicable"),
            (23, "not_applicable"),
        ),
    ),
    QueryCase(
        name="combined_temporal_and_relation_case",
        question="What project ownership and deployment claims apply now?",
        mode=TemporalQueryMode.CURRENT,
        as_of=timestamp(2026, 11, 3),
        candidate_ids=(16, 17, 18, 19, 20, 21, 22),
        relation_ids=(5, 6),
        expected_statuses=(
            (16, "valid"),
            (17, "invalid"),
            (18, "not_applicable"),
            (19, "invalid"),
            (20, "invalid"),
            (21, "indeterminate"),
            (22, "indeterminate"),
        ),
        expected_relations=(
            (16, 5, "source", 17, True),
            (17, 5, "target", 16, True),
            (19, 6, "source", 20, True),
            (20, 6, "target", 19, True),
        ),
    ),
)

def validate_dataset() -> None:
    records = build_records()
    relations = build_relations()

    record_ids = {record.id for record in records}
    relation_ids = {relation.id for relation in relations}

    if len(records) != 24 or len(record_ids) != 24:
        raise ValueError("Expected exactly 24 unique memory records.")

    if len(relations) != 6 or len(relation_ids) != 6:
        raise ValueError("Expected exactly 6 unique relations.")

    if len(QUERIES) != 12:
        raise ValueError("Expected exactly 12 query cases.")

    names = [query.name for query in QUERIES]
    if len(names) != len(set(names)):
        raise ValueError("Query names must be unique.")

    for query in QUERIES:
        if not query.question.strip():
            raise ValueError(f"{query.name} has an empty question.")

        if query.mode == TemporalQueryMode.UNSPECIFIED and query.as_of is not None:
            raise ValueError(f"{query.name}: UNSPECIFIED cannot include as_of.")

        if query.mode != TemporalQueryMode.UNSPECIFIED and query.as_of is None:
            raise ValueError(f"{query.name}: this mode requires as_of.")

        if len(query.candidate_ids) != len(set(query.candidate_ids)):
            raise ValueError(f"{query.name}: duplicate candidate IDs.")

        if not set(query.candidate_ids).issubset(
            {int(record.id.hex[-12:], 10) for record in records}
        ):
            raise ValueError(f"{query.name}: unknown candidate ID.")

        expected_ids = {candidate_id for candidate_id, _ in query.expected_statuses}
        if expected_ids != set(query.candidate_ids):
            raise ValueError(f"{query.name}: expected statuses do not cover candidates.")

        for relation_number in query.relation_ids:
            if make_relation_id(relation_number) not in relation_ids:
                raise ValueError(f"{query.name}: unknown relation ID.")

        for candidate_id, relation_number, _direction, other_id, _present in (
            query.expected_relations
        ):
            if candidate_id not in query.candidate_ids:
                raise ValueError(f"{query.name}: relation expectation has unknown candidate.")
            if relation_number not in query.relation_ids:
                raise ValueError(f"{query.name}: relation expectation is not included.")
            if other_id not in {int(record.id.hex[-12:], 10) for record in records}:
                raise ValueError(f"{query.name}: relation expectation has unknown endpoint.")

    # Ensure all relation endpoints refer to known records.
    for relation in relations:
        if relation.source_id not in record_ids or relation.target_id not in record_ids:
            raise ValueError("A relation references an unknown memory record.")

if __name__ == "__main__":
    validate_dataset()
    print("Temporal Assessment Challenge v1 dataset: VALID")
    print("Memory records:", len(build_records()))
    print("Relations:", len(build_relations()))
    print("Queries:", len(QUERIES))
