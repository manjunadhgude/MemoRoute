
"""Run the Retrieval–Temporal Challenge v1 experiment."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.retrieval_temporal_v1.dataset import (
    BASE_TIME,
    CURRENT_AS_OF,
    EXPECTED_RELATION_EVIDENCE,
    QUERIES,
    RELATIONS,
    build_records,
    build_relations,
    make_id,
    validate_dataset,
)
from memoroute.evaluation.metrics import recall_at_k, reciprocal_rank
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.hybrid import HybridRetriever
from memoroute.retrieval.recent import RecentHistoryRetriever
from memoroute.retrieval.temporal_selector import TemporalQueryMode
from memoroute.retrieval.temporal_store import assess_temporal_candidates_from_store
from memoroute.retrieval.vector import VectorRetriever

OUTPUT_K = 3
CANDIDATE_DEPTH = 32
RRF_K = 60
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def _evidence_tuple(evidence, relation_numbers):
    return (
        next(
            number for relation_id, number in relation_numbers.items()
            if relation_id == evidence.relation_id
        ),
        evidence.kind.value,
        evidence.candidate_direction.value,
        next(
            number for number in range(1, 100)
            if make_id(number) == evidence.other_record_id
        ),
        evidence.other_endpoint_in_candidates,
    )

def _assessment_json(assessment, relation_numbers):
    return {
        "candidate_id": next(
            number for number in range(1, 100)
            if make_id(number) == assessment.candidate_id
        ),
        "original_rank": assessment.original_rank,
        "temporal_status": assessment.temporal_status.value,
        "relation_status": assessment.relation_status.value,
        "reason_code": assessment.reason_code,
        "disposition": assessment.disposition.value,
        "relations": [
            {
                "relation_number": relation_numbers[item.relation_id],
                "kind": item.kind.value,
                "direction": item.candidate_direction.value,
                "other_record_id": next(
                    number for number in range(1, 100)
                    if make_id(number) == item.other_record_id
                ),
                "other_endpoint_in_candidates": item.other_endpoint_in_candidates,
            }
            for item in assessment.relations
        ],
    }

def run_experiment(*, vector_model=None):
    """Run retrieval and temporal assessment without changing rankings."""
    validate_dataset()

    with TemporaryDirectory(prefix="memoroute-retrieval-temporal-v1-") as temp:
        store = SQLiteMemoryStore(Path(temp) / "challenge.sqlite3")

        records = build_records()
        relations = build_relations()
        for record in records:
            store.save_record(record)
        for relation in relations:
            store.save_relation(relation)

        record_by_number = {
            number: next(record for record in records if record.id == make_id(number))
            for number in range(1, 33)
        }
        relation_numbers = {
            make_id(1000 + number): number
            for number, *_ in RELATIONS
        }

        recent = RecentHistoryRetriever(store)
        bm25 = BM25Retriever(store)
        vector = VectorRetriever(
            store,
            model_name=MODEL_NAME,
            model=vector_model,
        )
        hybrid = HybridRetriever(
            bm25,
            vector,
            rrf_k=RRF_K,
            candidate_limit=CANDIDATE_DEPTH,
        )

        retrievers = {
            "recent_history": lambda query: recent.retrieve(limit=OUTPUT_K),
            "bm25": lambda query: bm25.retrieve(query.question, limit=OUTPUT_K),
            "vector": lambda query: vector.retrieve(query.question, limit=OUTPUT_K),
            "hybrid_rrf": lambda query: hybrid.retrieve(query.question, limit=OUTPUT_K),
        }

        output = {
            "experiment": "retrieval_temporal_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": {
                "records": len(records),
                "relations": len(relations),
                "queries": len(QUERIES),
                "synthetic_and_hand_labelled": True,
            },
            "configuration": {
                "output_k": OUTPUT_K,
                "candidate_depth": CANDIDATE_DEPTH,
                "rrf_k": RRF_K,
                "embedding_model": (
                    "injected_test_model" if vector_model is not None else MODEL_NAME
                ),
                "base_time": BASE_TIME.isoformat(),
                "current_as_of": CURRENT_AS_OF.isoformat(),
                "temporal_policy": "post_retrieval_annotation_only",
            },
            "retrieval": {},
            "fixed_temporal_assessment": {},
        }

        for name, retrieve in retrievers.items():
            query_rows = []
            metric_rows = []

            for query in QUERIES:
                retrieved = retrieve(query)
                retrieved_ids = [record.id for record in retrieved]
                relevant_ids = {make_id(number) for number in query.relevant_ids}

                query_rows.append({
                    "query": query.name,
                    "retrieved_ids": [
                        next(number for number in range(1, 33)
                             if make_id(number) == record.id)
                        for record in retrieved
                    ],
                    "relevant_ids": list(query.relevant_ids),
                    "recall_at_1": recall_at_k(retrieved_ids, relevant_ids, k=1),
                    "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, k=3),
                    "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
                })
                metric_rows.append(query_rows[-1])

                # Annotate actual retrieved results. Do not reorder or filter.
                actual_assessment = assess_temporal_candidates_from_store(
                    store,
                    retrieved,
                    mode=TemporalQueryMode(query.mode),
                    as_of=query.as_of,
                )
                query_rows[-1]["temporal_annotations"] = [
                    _assessment_json(item, relation_numbers)
                    for item in actual_assessment.assessments
                ]

            output["retrieval"][name] = {
                "metrics": {
                    "recall_at_1": sum(row["recall_at_1"] for row in metric_rows)
                    / len(metric_rows),
                    "recall_at_3": sum(row["recall_at_3"] for row in metric_rows)
                    / len(metric_rows),
                    "mrr": sum(row["reciprocal_rank"] for row in metric_rows)
                    / len(metric_rows),
                },
                "queries": query_rows,
            }

        # Independently score the fixed, hand-labelled temporal probes.
        status_checks = []
        relation_status_checks = []
        rank_checks = []
        disposition_checks = []
        evidence_scenario_checks = []
        probe_results = []

        for query in QUERIES:
            candidates = [
                record_by_number[number]
                for number in query.assessment_candidate_ids
            ]
            result = assess_temporal_candidates_from_store(
                store,
                candidates,
                mode=TemporalQueryMode(query.mode),
                as_of=query.as_of,
            )

            expected_statuses = dict(query.expected_temporal_statuses)
            expected_evidence = EXPECTED_RELATION_EVIDENCE.get(query.name, ())
            actual_evidence = []

            for assessment in result.assessments:
                candidate_number = next(
                    number for number in range(1, 33)
                    if make_id(number) == assessment.candidate_id
                )

                status_checks.append(
                    assessment.temporal_status.value
                    == expected_statuses[candidate_number]
                )
                rank_checks.append(
                    assessment.original_rank
                    == query.assessment_candidate_ids.index(candidate_number) + 1
                )
                disposition_checks.append(
                    assessment.disposition.value == "keep"
                )

                candidate_evidence = [
                    entry for entry in expected_evidence
                    if entry[0] == candidate_number
                ]
                expected_relation_status = (
                    "none" if not candidate_evidence
                    else "resolved" if all(entry[5] for entry in candidate_evidence)
                    else "partial"
                )
                relation_status_checks.append(
                    assessment.relation_status.value == expected_relation_status
                )

                for item in assessment.relations:
                    actual_evidence.append((
                        candidate_number,
                        *_evidence_tuple(item, relation_numbers),
                    ))

            expected_evidence_set = set(expected_evidence)
            actual_evidence_set = set(actual_evidence)
            evidence_scenario_checks.append(
                actual_evidence_set == expected_evidence_set
            )

            probe_results.append({
                "query": query.name,
                "mode": query.mode,
                "assessment_candidate_ids": list(query.assessment_candidate_ids),
                "assessments": [
                    _assessment_json(item, relation_numbers)
                    for item in result.assessments
                ],
                "expected_relation_evidence_entries": len(expected_evidence),
                "actual_relation_evidence_entries": len(actual_evidence),
                "relation_evidence_match": (
                    actual_evidence_set == expected_evidence_set
                ),
            })

        check_groups = {
            "temporal_status": status_checks,
            "relation_status": relation_status_checks,
            "original_rank": rank_checks,
            "keep_disposition": disposition_checks,
            "relation_evidence_per_scenario": evidence_scenario_checks,
        }
        output["fixed_temporal_assessment"] = {
            "checks": {
                name: {
                    "passed": sum(checks),
                    "total": len(checks),
                }
                for name, checks in check_groups.items()
            },
            "expected_relation_evidence_entries": sum(
                len(entries) for entries in EXPECTED_RELATION_EVIDENCE.values()
            ),
            "all_checks_passed": all(
                all(checks) for checks in check_groups.values()
            ),
            "queries": probe_results,
        }

        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path. Existing files are never overwritten.",
    )
    args = parser.parse_args()

    result = run_experiment()
    output_path = args.output or Path("results") / (
        "retrieval_temporal_v1_"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents accidental overwrites.
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")

    print("Output:", output_path)
    print("Retrieval metrics:")
    for name, data in result["retrieval"].items():
        metrics = data["metrics"]
        print(
            f"  {name:<16} "
            f"R@1={metrics['recall_at_1']:.4f} "
            f"R@3={metrics['recall_at_3']:.4f} "
            f"MRR={metrics['mrr']:.4f}"
        )

    print("\nFixed temporal assessment:")
    for name, counts in result["fixed_temporal_assessment"]["checks"].items():
        print(f"  {name:<32} {counts['passed']}/{counts['total']}")

    print(
        "  Overall:",
        "PASS" if result["fixed_temporal_assessment"]["all_checks_passed"]
        else "FAIL",
    )

if __name__ == "__main__":
    main()
