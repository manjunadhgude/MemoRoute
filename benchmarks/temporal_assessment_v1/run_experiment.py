
"""Evaluate MemoRoute's temporal assessment selector against challenge v1."""

from collections import Counter

from benchmarks.temporal_assessment_v1.dataset import (
    QUERIES,
    build_records,
    build_relations,
    validate_dataset,
)
from memoroute.retrieval.temporal_selector import (
    RelationStatus,
    assess_temporal_candidates,
)


def _number(value) -> int:
    return int(value.hex[-12:], 10)

def run() -> int:
    validate_dataset()

    records = { _number(record.id): record for record in build_records() }
    relations = { _number(relation.id) - 1000: relation for relation in build_relations() }

    total_checks = 0
    passed_checks = 0
    failures = []
    status_counts = Counter()

    for query in QUERIES:
        candidates = [records[number] for number in query.candidate_ids]
        selected_relations = [relations[number] for number in query.relation_ids]

        result = assess_temporal_candidates(
            candidates,
            mode=query.mode,
            as_of=query.as_of,
            relations=selected_relations,
        )

        actual_ids = tuple(_number(record.id) for record in result.candidates)
        expected_ids = query.candidate_ids

        checks = []

        checks.append(("candidate_order", actual_ids == expected_ids))

        expected_statuses = dict(query.expected_statuses)
        expected_relation_rows = {}
        for candidate_id, relation_number, direction, other_id, other_present in (
            query.expected_relations
        ):
            expected_relation_rows.setdefault(candidate_id, []).append(
                (
                    relation_number,
                    direction,
                    other_id,
                    other_present,
                )
            )

        for assessment in result.assessments:
            candidate_id = _number(assessment.candidate_id)
            expected_status = expected_statuses[candidate_id]
            actual_status = assessment.temporal_status.value
            status_counts[actual_status] += 1

            checks.append(
                (
                    f"temporal_status[{candidate_id}]",
                    actual_status == expected_status,
                )
            )

            actual_relation_rows = tuple(
                sorted(
                    (
                        _number(evidence.relation_id) - 1000,
                        evidence.candidate_direction.value,
                        _number(evidence.other_record_id),
                        evidence.other_endpoint_in_candidates,
                    )
                    for evidence in assessment.relations
                )
            )
            expected_rows = tuple(
                sorted(expected_relation_rows.get(candidate_id, []))
            )

            checks.append(
                (
                    f"relation_evidence[{candidate_id}]",
                    actual_relation_rows == expected_rows,
                )
            )

            if not expected_rows:
                expected_relation_status = RelationStatus.NONE.value
            elif all(row[3] for row in expected_rows):
                expected_relation_status = RelationStatus.RESOLVED.value
            else:
                expected_relation_status = RelationStatus.PARTIAL.value

            checks.append(
                (
                    f"relation_status[{candidate_id}]",
                    assessment.relation_status.value == expected_relation_status,
                )
            )

            checks.append(
                (
                    f"original_rank[{candidate_id}]",
                    assessment.original_rank == query.candidate_ids.index(candidate_id) + 1,
                )
            )
            checks.append(
                (
                    f"disposition[{candidate_id}]",
                    assessment.disposition.value == "keep",
                )
            )

        for check_name, passed in checks:
            total_checks += 1
            if passed:
                passed_checks += 1
            else:
                failures.append(f"{query.name}: {check_name}")

        query_passed = all(passed for _, passed in checks)
        print(f"{'PASS' if query_passed else 'FAIL'} {query.name}")

    print("\n========== TEMPORAL ASSESSMENT SUMMARY ==========")
    print(f"Queries: {len(QUERIES)}")
    print(f"Checks passed: {passed_checks}/{total_checks}")
    print(f"Assessment check accuracy: {passed_checks / total_checks:.3%}")
    print("Observed temporal statuses:", dict(sorted(status_counts.items())))

    if failures:
        print("\nFailures:")
        for failure in failures:
            print("-", failure)
        return 1

    print("\nAll temporal assessment checks passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
