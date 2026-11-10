"""Integration tests for Retrieval-Temporal Challenge v1."""

from benchmarks.retrieval_temporal_v1.dataset import (
    EXPECTED_RELATION_EVIDENCE,
    QUERIES,
    build_records,
    build_relations,
    validate_dataset,
)
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.temporal_selector import RelationStatus, TemporalQueryMode
from memoroute.retrieval.temporal_store import (
    assess_temporal_candidates_from_store,
)


def numeric_id(value):
    """Recover the deterministic numeric suffix from a fixture UUID."""
    return int(value.hex[-12:])


def test_relation_expectations_are_complete_and_valid():
    validate_dataset()

    assert len(EXPECTED_RELATION_EVIDENCE) == 12
    assert sum(map(len, EXPECTED_RELATION_EVIDENCE.values())) == 23


def test_sqlite_adapter_matches_hand_authored_relation_evidence_for_all_scenarios(
    tmp_path,
):
    records = build_records()
    relations = build_relations()
    record_map = {numeric_id(record.id): record for record in records}

    for query in QUERIES:
        store = SQLiteMemoryStore(tmp_path / f"{query.name}.sqlite3")

        for record in records:
            store.save_record(record)

        for relation in relations:
            store.save_relation(relation)

        candidates = [record_map[number] for number in query.assessment_candidate_ids]

        result = assess_temporal_candidates_from_store(
            store,
            candidates,
            mode=TemporalQueryMode(query.mode),
            as_of=query.as_of,
        )

        # Adapter must preserve the exact input candidate order.
        assert tuple(candidates) == result.candidates, query.name

        assert [numeric_id(item.candidate_id) for item in result.assessments] == list(
            query.assessment_candidate_ids
        ), query.name

        authored = EXPECTED_RELATION_EVIDENCE.get(query.name, ())

        for assessment in result.assessments:
            candidate_id = numeric_id(assessment.candidate_id)

            expected = sorted(
                (
                    candidate,
                    relation_number + 1000,
                    kind,
                    direction,
                    other_id,
                    other_in_candidates,
                )
                for (
                    candidate,
                    relation_number,
                    kind,
                    direction,
                    other_id,
                    other_in_candidates,
                ) in authored
                if candidate == candidate_id
            )

            actual = sorted(
                (
                    candidate_id,
                    numeric_id(evidence.relation_id),
                    evidence.kind.value,
                    evidence.candidate_direction.value,
                    numeric_id(evidence.other_record_id),
                    evidence.other_endpoint_in_candidates,
                )
                for evidence in assessment.relations
            )

            assert actual == expected, (
                query.name,
                candidate_id,
                actual,
                expected,
            )

            if not expected:
                expected_status = RelationStatus.NONE
            elif all(entry[5] for entry in expected):
                expected_status = RelationStatus.RESOLVED
            else:
                expected_status = RelationStatus.PARTIAL

            assert assessment.relation_status == expected_status, (
                query.name,
                candidate_id,
            )

            assert assessment.disposition.value == "keep"
