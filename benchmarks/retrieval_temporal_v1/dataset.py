
"""Deterministic synthetic dataset for Retrieval–Temporal Challenge v1."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from memoroute.memory.models import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
)
from memoroute.memory.temporal import (
    TemporalBoundary,
    TemporalInfo,
    TemporalKind,
)

BASE_TIME = datetime(2026, 10, 1, tzinfo=UTC)
CURRENT_AS_OF = datetime(2026, 11, 3, 12, tzinfo=UTC)
SOURCE = MemorySource(
    kind="synthetic",
    reference="memoroute-retrieval-temporal-v1",
)

# Each record:
# (numeric ID, category, content, temporal kind,
#  start boundary kind, start date,
#  end boundary kind, end date, event date)
#
# Boundary kinds: known, unknown, unbounded.
# Dates are ISO-8601 UTC dates. Unused fields are None.

MEMORIES = (
    (1, "database", "Production services use PostgreSQL 17 as their primary relational database.", "state", "known", "2026-10-01", "unbounded", None, None),
    (2, "database", "Production services previously used PostgreSQL 16 as their primary relational database.", "state", "known", "2026-06-01", "known", "2026-10-01", None),
    (3, "cache", "The staging environment uses Redis 6 for caching.", "state", "known", "2026-08-01", "unbounded", None, None),
    (4, "cache", "The production environment uses Redis 7 for caching.", "state", "known", "2026-10-10", "unbounded", None, None),
    (5, "invoice_preference", "Customer C-104 currently prefers invoices with itemized tax breakdowns.", "state", "known", "2026-10-25", "unbounded", None, None),
    (6, "invoice_preference", "Customer C-104 previously preferred summary-only invoices.", "state", "known", "2026-09-01", "known", "2026-10-25", None),
    (7, "receipt_preference", "Customer C-104 wants electronic receipts delivered by email after each purchase.", "state", "known", "2026-05-01", "unbounded", None, None),
    (8, "receipt_preference", "Customer C-140 prefers printed receipts rather than email copies.", "state", "known", "2026-04-01", "unbounded", None, None),
    (9, "refund", "Refund ORD-882 was approved, but its bank transfer had not settled as of 28 October.", "event", None, None, None, None, "2026-10-28"),
    (10, "refund", "Refund ORD-828 was approved and the bank transfer settled on 27 October.", "event", None, None, None, None, "2026-10-27"),
    (11, "invoice_payment", "Supplier invoice INV-204 was paid by bank transfer on 12 October 2026.", "event", None, None, None, None, "2026-10-12"),
    (12, "invoice_payment", "Supplier invoice INV-240 was paid by bank transfer on 12 October 2026.", "event", None, None, None, None, "2026-10-12"),
    (13, "procurement", "The original invoice for purchase order PO-410 listed 100 units.", "state", "known", "2026-10-05", "known", "2026-10-20", None),
    (14, "procurement", "The goods receipt for PO-410 recorded 96 units received on 18 October.", "event", None, None, None, None, "2026-10-18"),
    (15, "procurement", "The corrected invoice for PO-410 lists 96 units, correcting the original 100-unit invoice.", "state", "known", "2026-10-20", "unbounded", None, None),
    (16, "warehouse", "The North warehouse was the designated dispatch location before November.", "state", "known", "2026-07-01", "known", "2026-11-01", None),
    (17, "warehouse", "The East warehouse became the designated dispatch location from 1 November.", "state", "known", "2026-11-01", "unbounded", None, None),
    (18, "project_event", "The Atlas migration launch review took place on 22 October 2026.", "event", None, None, None, None, "2026-10-22"),
    (19, "office_schedule", "The office is normally closed on Fridays under its weekly operating schedule.", "atemporal", None, None, None, None, None),
    (20, "office_event", "The office closed for a one-time electrical inspection on 30 October 2026.", "event", None, None, None, None, "2026-10-30"),
    (21, "security_policy", "The start date of the latest API-key rotation policy has not yet been confirmed.", "state", "unknown", None, "unbounded", None, None),
    (22, "retention_policy", "The data-retention policy is active, but its expiry date is not known.", "state", "known", "2026-07-01", "unknown", None, None),
    (23, "support_sla", "Enterprise support tickets have a four-hour first-response target.", "state", "unbounded", None, "unbounded", None, None),
    (24, "support_sla", "The previous enterprise support policy had an eight-hour first-response target and ended on 1 October.", "state", "unbounded", None, "known", "2026-10-01", None),
    (25, "deployment_region", "The production payment service is deployed in the us-east region.", "state", "known", "2026-09-01", "unbounded", None, None),
    (26, "deployment_region", "The production payment service is deployed in the eu-west region.", "state", "known", "2026-09-01", "unbounded", None, None),
    (27, "approval_policy", "Refunds above 5000 INR require manager approval.", "state", "known", "2026-08-01", "unbounded", None, None),
    (28, "approval_policy", "Refunds above 10000 INR require manager approval.", "state", "known", "2026-08-01", "unbounded", None, None),
    (29, "tax_identifier", "The supplier's registered business identifier is GSTIN 37ABCDE1234F1Z5.", "atemporal", None, None, None, None, None),
    (30, "contract", "The Acme support contract is active from 1 September 2026.", "state", "known", "2026-09-01", "unbounded", None, None),
    (31, "product_feature", "The new analytics export feature is scheduled to become available from 1 December 2026.", "state", "known", "2026-12-01", "unbounded", None, None),
    (32, "subscription", "The legacy analytics subscription was active from 1 January through 1 October 2026.", "state", "known", "2026-01-01", "known", "2026-10-01", None),
)

# Relations are directed: source_id -> target_id.
# SUPERSEDES and CORRECTS use the newer/corrected claim as source.

RELATIONS = (
    (1, 1, 2, MemoryRelationKind.SUPERSEDES),
    (2, 5, 6, MemoryRelationKind.SUPERSEDES),
    (3, 15, 13, MemoryRelationKind.CORRECTS),
    (4, 25, 26, MemoryRelationKind.CONTRADICTS),
    (5, 27, 28, MemoryRelationKind.CONTRADICTS),
    (6, 17, 16, MemoryRelationKind.SUPERSEDES),
    (8, 23, 24, MemoryRelationKind.SUPERSEDES),
)


# Hand-authored relation-evidence expectations.
# Entry format: (candidate ID, relation fixture number, kind, direction,
# other endpoint ID, other endpoint included in this query's candidate set).
# These labels are independent of the temporal selector and SQLite adapter.
EXPECTED_RELATION_EVIDENCE = {
    "production_database_current": (
        (1, 1, "supersedes", "source", 2, True),
        (2, 1, "supersedes", "target", 1, True),
    ),
    "production_database_historical": (
        (1, 1, "supersedes", "source", 2, True),
        (2, 1, "supersedes", "target", 1, True),
    ),
    "invoice_preference_current": (
        (5, 2, "supersedes", "source", 6, True),
        (6, 2, "supersedes", "target", 5, True),
    ),
    "invoice_preference_historical": (
        (5, 2, "supersedes", "source", 6, True),
        (6, 2, "supersedes", "target", 5, True),
    ),
    "superseded_database_claim": (
        (1, 1, "supersedes", "source", 2, True),
        (2, 1, "supersedes", "target", 1, True),
    ),
    "superseded_warehouse": (
        (17, 6, "supersedes", "source", 16, True),
        (16, 6, "supersedes", "target", 17, True),
    ),
    "corrected_invoice": (
        (15, 3, "corrects", "source", 13, True),
        (13, 3, "corrects", "target", 15, True),
    ),
    "historical_correction": (
        (15, 3, "corrects", "source", 13, True),
        (13, 3, "corrects", "target", 15, True),
    ),
    "contradictory_regions": (
        (25, 4, "contradicts", "source", 26, True),
        (26, 4, "contradicts", "target", 25, True),
    ),
    "contradictory_approval_threshold": (
        (27, 5, "contradicts", "source", 28, True),
        (28, 5, "contradicts", "target", 27, True),
    ),
    "unbounded_support_sla": (
        (23, 8, "supersedes", "source", 24, False),
    ),
    "unspecified_temporal_claims": (
        (23, 8, "supersedes", "source", 24, True),
        (24, 8, "supersedes", "target", 23, True),
    ),
}

@dataclass(frozen=True)
class QueryScenario:
    name: str
    category: str
    question: str
    mode: str
    as_of: datetime | None
    relevant_ids: tuple[int, ...]
    assessment_candidate_ids: tuple[int, ...]
    expected_temporal_statuses: tuple[tuple[int, str], ...]

# Expected statuses are hand-authored.
# Status values: valid, invalid, indeterminate,
# not_applicable, not_evaluated.

QUERIES = (
    QueryScenario(
        "production_database_current", "current_state",
        "Which PostgreSQL version is currently used in production?",
        "current", CURRENT_AS_OF, (1,), (1, 2),
        ((1, "valid"), (2, "invalid")),
    ),
    QueryScenario(
        "production_database_historical", "historical_state",
        "Which PostgreSQL version was used in production in August?",
        "as_of", datetime(2026, 8, 15, tzinfo=UTC), (2,), (1, 2),
        ((1, "invalid"), (2, "valid")),
    ),
    QueryScenario(
        "production_cache_current", "entity_confusion",
        "Which Redis version is configured in production, not staging?",
        "current", CURRENT_AS_OF, (4,), (3, 4),
        ((3, "valid"), (4, "valid")),
    ),
    QueryScenario(
        "invoice_preference_current", "current_state",
        "What invoice format does customer C-104 currently prefer?",
        "current", CURRENT_AS_OF, (5,), (5, 6),
        ((5, "valid"), (6, "invalid")),
    ),
    QueryScenario(
        "invoice_preference_historical", "historical_state",
        "What invoice format did customer C-104 prefer on 10 October?",
        "as_of", datetime(2026, 10, 10, tzinfo=UTC), (6,), (5, 6),
        ((5, "invalid"), (6, "valid")),
    ),
    QueryScenario(
        "future_feature", "future_state",
        "Is the analytics export feature available on 3 November?",
        "as_of", CURRENT_AS_OF, (31,), (31,),
        ((31, "invalid"),),
    ),
    QueryScenario(
        "expired_subscription", "expired_state",
        "Was the legacy analytics subscription still active on 3 November?",
        "as_of", CURRENT_AS_OF, (32,), (32,),
        ((32, "invalid"),),
    ),
    QueryScenario(
        "unknown_rotation_start", "unknown_boundary",
        "Was the latest API-key rotation policy active on 3 November?",
        "current", CURRENT_AS_OF, (21,), (21,),
        ((21, "indeterminate"),),
    ),
    QueryScenario(
        "unknown_retention_end", "unknown_boundary",
        "Can we confirm the retention policy is still valid on 3 November?",
        "current", CURRENT_AS_OF, (22,), (22,),
        ((22, "indeterminate"),),
    ),
    QueryScenario(
        "unbounded_support_sla", "unbounded_boundary",
        "What is the enterprise support first-response target?",
        "current", CURRENT_AS_OF, (23,), (23,),
        ((23, "valid"),),
    ),
    QueryScenario(
        "superseded_database_claim", "supersession",
        "Which production database claim supersedes the older version?",
        "current", CURRENT_AS_OF, (1,), (1, 2),
        ((1, "valid"), (2, "invalid")),
    ),
    QueryScenario(
        "superseded_warehouse", "supersession",
        "Which warehouse is the designated dispatch location now?",
        "current", CURRENT_AS_OF, (17,), (16, 17),
        ((16, "invalid"), (17, "valid")),
    ),
    QueryScenario(
        "corrected_invoice", "correction",
        "What is the corrected quantity on the PO-410 invoice?",
        "current", CURRENT_AS_OF, (15,), (13, 15),
        ((13, "invalid"), (15, "valid")),
    ),
    QueryScenario(
        "historical_correction", "correction",
        "Which PO-410 invoice quantity applied on 10 October?",
        "as_of", datetime(2026, 10, 10, tzinfo=UTC), (13,), (13, 15),
        ((13, "valid"), (15, "invalid")),
    ),
    QueryScenario(
        "contradictory_regions", "contradiction",
        "Which region is recorded for the production payment service?",
        "current", CURRENT_AS_OF, (25, 26), (25, 26),
        ((25, "valid"), (26, "valid")),
    ),
    QueryScenario(
        "contradictory_approval_threshold", "contradiction",
        "What refund amount requires manager approval?",
        "current", CURRENT_AS_OF, (27, 28), (27, 28),
        ((27, "valid"), (28, "valid")),
    ),
    QueryScenario(
        "event_and_atemporal", "event_atemporal",
        "When was the Atlas migration launch review, and what is the supplier GSTIN?",
        "unspecified", None, (18, 29), (18, 19, 29),
        ((18, "not_applicable"), (19, "not_applicable"), (29, "not_applicable")),
    ),
    QueryScenario(
        "refund_event", "event",
        "When was refund ORD-882 approved, and had the transfer settled?",
        "unspecified", None, (9,), (9, 10),
        ((9, "not_applicable"), (10, "not_applicable")),
    ),
    QueryScenario(
        "office_schedule_vs_event", "event_atemporal",
        "Is Friday closure the normal office schedule or a one-time event?",
        "unspecified", None, (19, 20), (19, 20),
        ((19, "not_applicable"), (20, "not_applicable")),
    ),
    QueryScenario(
        "unspecified_temporal_claims", "unspecified_time",
        "What support response targets and contract terms are recorded?",
        "unspecified", None, (23, 24, 30), (23, 24, 30),
        ((23, "not_evaluated"), (24, "not_evaluated"), (30, "not_evaluated")),
    ),
)

def make_id(number: int) -> UUID:
    """Return a stable UUID for a numeric fixture ID."""
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ValueError("Fixture IDs must be positive integers.")
    return UUID(f"00000000-0000-0000-0000-{number:012d}")

def _date(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=UTC)

def _boundary(kind: str | None, value: str | None) -> TemporalBoundary | None:
    if kind is None:
        return None
    if kind == "known":
        return TemporalBoundary.known(_date(value))
    if kind == "unknown":
        return TemporalBoundary.unknown()
    if kind == "unbounded":
        return TemporalBoundary.unbounded()
    raise ValueError(f"Unknown boundary kind: {kind}")

def build_records() -> list[MemoryRecord]:
    """Build all memory fixtures with deterministic IDs and timestamps."""
    records = []

    for (
        number,
        _category,
        content,
        temporal_kind,
        start_kind,
        start_date,
        end_kind,
        end_date,
        event_date,
    ) in MEMORIES:
        if temporal_kind == "atemporal":
            temporal = TemporalInfo(kind=TemporalKind.ATEMPORAL)
        elif temporal_kind == "event":
            temporal = TemporalInfo(
                kind=TemporalKind.EVENT,
                occurred_at=_date(event_date),
            )
        elif temporal_kind == "state":
            temporal = TemporalInfo(
                kind=TemporalKind.STATE,
                valid_from=_boundary(start_kind, start_date),
                valid_until=_boundary(end_kind, end_date),
            )
        else:
            raise ValueError(f"Unknown temporal kind: {temporal_kind}")

        records.append(
            MemoryRecord(
                id=make_id(number),
                content=content,
                source=SOURCE,
                recorded_at=BASE_TIME + timedelta(days=number - 1),
                temporal=temporal,
            )
        )

    return records

def build_relations() -> list[MemoryRelation]:
    """Build directed relation fixtures with deterministic IDs."""
    return [
        MemoryRelation(
            id=make_id(1000 + relation_number),
            source_id=make_id(source_id),
            target_id=make_id(target_id),
            kind=kind,
            recorded_at=BASE_TIME + timedelta(days=relation_number),
        )
        for relation_number, source_id, target_id, kind in RELATIONS
    ]

def validate_dataset() -> None:
    """Validate fixture counts, references, timestamps, and expected labels."""
    if len(MEMORIES) != 32:
        raise ValueError(f"Expected 32 memories, found {len(MEMORIES)}.")
    if len(RELATIONS) != 7:
        raise ValueError(f"Expected 7 relations, found {len(RELATIONS)}.")
    if len(QUERIES) != 20:
        raise ValueError(f"Expected 20 queries, found {len(QUERIES)}.")

    memory_ids = [item[0] for item in MEMORIES]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("Memory IDs must be unique.")

    known_ids = set(memory_ids)
    relation_ids = [item[0] for item in RELATIONS]

    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("Relation fixture IDs must be unique.")

    for relation_number, source_id, target_id, kind in RELATIONS:
        if source_id not in known_ids or target_id not in known_ids:
            raise ValueError(f"Relation {relation_number} references an unknown record.")
        if source_id == target_id:
            raise ValueError(f"Relation {relation_number} cannot reference itself.")
        if not isinstance(kind, MemoryRelationKind):
            raise TypeError(f"Relation {relation_number} has an invalid kind.")

    query_by_name = {query.name: query for query in QUERIES}
    if set(EXPECTED_RELATION_EVIDENCE) - set(query_by_name):
        raise ValueError("Relation oracle references an unknown query scenario.")

    relation_map = {number: (source, target, kind.value) for number, source, target, kind in RELATIONS}
    for query_name, entries in EXPECTED_RELATION_EVIDENCE.items():
        query = query_by_name[query_name]
        candidate_ids = set(query.assessment_candidate_ids)
        seen_pairs = set()
        for candidate, relation_number, kind, direction, other_id, other_present in entries:
            if candidate not in candidate_ids:
                raise ValueError(f"{query_name}: oracle candidate {candidate} is not in scenario candidates.")
            if relation_number not in relation_map:
                raise ValueError(f"{query_name}: unknown relation fixture {relation_number}.")
            source_id, target_id, actual_kind = relation_map[relation_number]
            expected = (
                (source_id, "source", target_id)
                if direction == "source"
                else (target_id, "target", source_id)
                if direction == "target"
                else None
            )
            if expected is None or (candidate, direction, other_id) != (expected[0], expected[1], expected[2]):
                raise ValueError(f"{query_name}: incorrect relation direction for candidate {candidate}.")
            if kind != actual_kind:
                raise ValueError(f"{query_name}: relation kind mismatch for fixture {relation_number}.")
            if bool(other_present) != (other_id in candidate_ids):
                raise ValueError(f"{query_name}: incorrect in-candidate flag for relation {relation_number}.")
            pair = (candidate, relation_number)
            if pair in seen_pairs:
                raise ValueError(f"{query_name}: duplicate relation evidence {pair}.")
            seen_pairs.add(pair)

        expected_pairs = set()
        for relation_number, source_id, target_id, _kind in RELATIONS:
            if source_id in candidate_ids:
                expected_pairs.add((source_id, relation_number))
            if target_id in candidate_ids:
                expected_pairs.add((target_id, relation_number))
        if seen_pairs != expected_pairs:
            raise ValueError(f"{query_name}: relation evidence is incomplete or excessive.")

    query_names = [query.name for query in QUERIES]
    if len(query_names) != len(set(query_names)):
        raise ValueError("Query names must be unique.")

    for query in QUERIES:
        if not query.question.strip():
            raise ValueError(f"{query.name} has an empty question.")

        if query.mode not in {"current", "as_of", "unspecified"}:
            raise ValueError(f"{query.name} has an invalid query mode.")

        if query.mode == "unspecified" and query.as_of is not None:
            raise ValueError(f"{query.name}: unspecified mode cannot have as_of.")

        if query.mode in {"current", "as_of"}:
            if query.as_of is None or query.as_of.tzinfo is None:
                raise ValueError(f"{query.name}: timestamped mode requires aware as_of.")

        if not query.relevant_ids:
            raise ValueError(f"{query.name} requires retrieval relevance labels.")

        if not set(query.relevant_ids).issubset(known_ids):
            raise ValueError(f"{query.name} references unknown relevant IDs.")

        if not query.assessment_candidate_ids:
            raise ValueError(f"{query.name} requires assessment candidates.")

        if not set(query.assessment_candidate_ids).issubset(known_ids):
            raise ValueError(f"{query.name} references unknown assessment candidates.")

        expected_ids = tuple(item[0] for item in query.expected_temporal_statuses)

        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError(f"{query.name} has duplicate expected status IDs.")

        if set(expected_ids) != set(query.assessment_candidate_ids):
            raise ValueError(
                f"{query.name}: expected temporal labels must cover exactly "
                "the assessment candidate IDs."
            )

    # Independently check that hand-authored temporal labels agree with
    # the temporal model's documented interval semantics.
    record_map = {record.id: record for record in build_records()}

    for query in QUERIES:
        for numeric_id, expected in query.expected_temporal_statuses:
            record = record_map[make_id(numeric_id)]

            if record.temporal.kind != TemporalKind.STATE:
                actual = "not_applicable"
            elif query.mode == "unspecified":
                actual = "not_evaluated"
            else:
                actual = record.temporal.evaluate_at(query.as_of).value

            if actual != expected:
                raise ValueError(
                    f"{query.name}: record {numeric_id} expected {expected}, "
                    f"but temporal model returned {actual}."
                )

if __name__ == "__main__":
    validate_dataset()
    print("Dataset validation: SUCCESS")
    print("Memory records:", len(MEMORIES))
    print("Relations:", len(RELATIONS))
    print("Query scenarios:", len(QUERIES))
    print(
        "Assessment checks:",
        sum(len(query.expected_temporal_statuses) for query in QUERIES),
    )
    print("Temporal labels: internally consistent")
