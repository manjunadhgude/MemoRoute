"""Run separate real-BGE adaptive-selection evaluation and ablations.

The existing retrieval-temporal baseline runner is not imported or modified.
This runner reproduces its four retrieval methods at candidate depth 32 / K=3,
then evaluates adaptive selection and controlled ablations on those same pools.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from benchmarks.adaptive_selection_eval_v1.metrics import (
    aggregate_metrics,
    packet_coverage,
    ranking_metrics,
)
from benchmarks.retrieval_temporal_v1.dataset import (
    QUERIES,
    build_records,
    build_relations,
    validate_dataset,
)
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.adaptive_selector import select_adaptive_context
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.hybrid import HybridRetriever
from memoroute.retrieval.query_intent import infer_query_intent
from memoroute.retrieval.recent import RecentHistoryRetriever
from memoroute.retrieval.temporal_selector import (
    TemporalQueryMode,
    assess_temporal_candidates,
)
from memoroute.retrieval.temporal_store import assess_temporal_candidates_from_store
from memoroute.retrieval.vector import VectorRetriever

CANDIDATE_DEPTH = 32
OUTPUT_K = 3
RRF_K = 60
DEFAULT_TOKEN_BUDGET = 256
BUDGET_SWEEP = (64, 128, 256, 512)


def numeric_id(value: Any) -> int:
    """Convert deterministic fixture UUIDs back to their numeric IDs."""
    return int(value.hex[-12:])


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_exclusive(output_dir: Path, stem: str, content: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"{stem}_{stamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path


def _assessment_and_intent(
    store: SQLiteMemoryStore,
    candidates: tuple,
    scenario: Any,
    *,
    use_temporal: bool,
    use_relations: bool,
):
    if use_temporal:
        mode = TemporalQueryMode(scenario.mode)
        as_of = scenario.as_of
    else:
        mode = TemporalQueryMode.UNSPECIFIED
        as_of = None

    intent = infer_query_intent(
        scenario.question,
        mode=mode,
        as_of=as_of,
    )

    if use_relations:
        assessment = assess_temporal_candidates_from_store(
            store,
            candidates,
            mode=mode,
            as_of=as_of,
        )
    else:
        assessment = assess_temporal_candidates(
            candidates,
            mode=mode,
            as_of=as_of,
            relations=(),
        )
    return assessment, intent


def _select(
    store: SQLiteMemoryStore,
    candidates: tuple,
    scenario: Any,
    *,
    token_budget: int,
    use_temporal: bool = True,
    use_relations: bool = True,
):
    assessment, intent = _assessment_and_intent(
        store,
        candidates,
        scenario,
        use_temporal=use_temporal,
        use_relations=use_relations,
    )
    return select_adaptive_context(
        candidates,
        assessment,
        intent,
        token_budget=token_budget,
    )


def _record_ids(records: Any) -> list[int]:
    return [numeric_id(record.id) for record in records]


def _packet_summary(result: Any, relevant_ids: tuple[int, ...]) -> dict[str, Any]:
    packet_items = result.context_packet.items
    packet_ids = [numeric_id(item.record.id) for item in packet_items]
    return {
        "packet_ids": packet_ids,
        "packet_recall": packet_coverage(packet_ids, relevant_ids),
        "estimated_tokens": result.context_packet.estimated_tokens,
        "token_budget": result.context_packet.token_budget,
        "budget_compliant": (
            result.context_packet.estimated_tokens
            <= result.context_packet.token_budget
        ),
        "selected_item_count": len(packet_items),
        "omitted_candidate_count": len(
            result.context_packet.omitted_candidate_ids
        ),
        "estimator": result.context_packet.estimator,
    }


def _temporal_probe(store: SQLiteMemoryStore, record_map: dict[int, Any]) -> dict[str, Any]:
    checks = []
    relation_evidence_count = 0
    relation_status_counts: Counter[str] = Counter()

    for scenario in QUERIES:
        candidates = tuple(record_map[number] for number in scenario.assessment_candidate_ids)
        assessment = assess_temporal_candidates_from_store(
            store,
            candidates,
            mode=TemporalQueryMode(scenario.mode),
            as_of=scenario.as_of,
        )
        expected = dict(scenario.expected_temporal_statuses)
        for item in assessment.assessments:
            number = numeric_id(item.candidate_id)
            if number not in expected:
                continue
            actual = item.temporal_status.value
            checks.append(
                {
                    "query": scenario.name,
                    "record_id": number,
                    "expected": expected[number],
                    "actual": actual,
                    "pass": actual == expected[number],
                }
            )
            relation_evidence_count += len(item.relations)
            relation_status_counts[item.relation_status.value] += 1

    passed = sum(bool(item["pass"]) for item in checks)
    return {
        "passed": passed,
        "total": len(checks),
        "accuracy": passed / len(checks) if checks else 0.0,
        "checks": checks,
        "relation_evidence_entries_observed": relation_evidence_count,
        "relation_status_counts": dict(relation_status_counts),
    }


def run_experiment(
    *,
    model: Any = None,
    output_dir: Path | None = None,
    candidate_depth: int = CANDIDATE_DEPTH,
    output_k: int = OUTPUT_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    budget_sweep: tuple[int, ...] = BUDGET_SWEEP,
) -> dict[str, Any]:
    """Execute the experiment; tests may inject a deterministic embedding model."""
    validate_dataset()
    records = build_records()
    relations = build_relations()
    if candidate_depth <= 0 or candidate_depth > len(records):
        raise ValueError("candidate_depth must be between 1 and the record count")
    if output_k != 3:
        raise ValueError("this registered experiment uses output K=3")
    if token_budget <= 0 or any(value <= 0 for value in budget_sweep):
        raise ValueError("all context token budgets must be positive")

    output_dir = output_dir or (
        Path(__file__).resolve().parents[2] / "results" / "memoroute_final"
    )
    record_map = {numeric_id(record.id): record for record in records}

    with TemporaryDirectory(prefix="memoroute-adaptive-eval-") as temp_dir:
        store = SQLiteMemoryStore(Path(temp_dir) / "retrieval_temporal.sqlite3")
        for record in records:
            store.save_record(record)
        for relation in relations:
            store.save_relation(relation)

        recent = RecentHistoryRetriever(store)
        bm25 = BM25Retriever(store)
        vector = (
            VectorRetriever(store, model=model)
            if model is not None
            else VectorRetriever(store)
        )
        vector.refresh()
        hybrid = HybridRetriever(
            bm25,
            vector,
            rrf_k=RRF_K,
            candidate_limit=candidate_depth,
        )

        retrievers = {
            "recent_history": lambda query: recent.retrieve(limit=candidate_depth),
            "bm25": lambda query: bm25.retrieve(query, limit=candidate_depth),
            "bge_vector": lambda query: vector.retrieve(query, limit=candidate_depth),
            "hybrid_rrf": lambda query: hybrid.retrieve(query, limit=candidate_depth),
        }

        base_rows: dict[str, list[dict[str, float]]] = {
            name: [] for name in retrievers
        }
        adaptive_rows: dict[str, list[dict[str, float]]] = {
            name: [] for name in retrievers
        }
        packet_rows: dict[str, list[dict[str, float]]] = {
            name: [] for name in retrievers
        }
        ablation_rows: dict[str, list[dict[str, float]]] = {
            "hybrid_identity": [],
            "hybrid_full": [],
            "hybrid_no_temporal": [],
            "hybrid_no_relations": [],
        }
        budget_rows: dict[int, list[dict[str, float]]] = {
            value: [] for value in budget_sweep
        }
        per_query: dict[str, Any] = {}

        for scenario in QUERIES:
            relevant_ids = tuple(scenario.relevant_ids)
            candidate_pools = {
                name: tuple(retrieve(scenario.question))
                for name, retrieve in retrievers.items()
            }
            query_result: dict[str, Any] = {
                "category": scenario.category,
                "question": scenario.question,
                "mode": scenario.mode,
                "as_of": (
                    scenario.as_of.isoformat()
                    if scenario.as_of is not None
                    else None
                ),
                "relevant_ids": list(relevant_ids),
                "methods": {},
            }

            for method, candidates in candidate_pools.items():
                base_ids = _record_ids(candidates)
                base_metric = ranking_metrics(base_ids, relevant_ids)
                base_rows[method].append(base_metric)

                adaptive = _select(
                    store,
                    candidates,
                    scenario,
                    token_budget=token_budget,
                )
                adaptive_ids = _record_ids(adaptive.ranked_candidates)
                adaptive_metric = ranking_metrics(adaptive_ids, relevant_ids)
                packet_metric = _packet_summary(adaptive, relevant_ids)
                adaptive_rows[method].append(adaptive_metric)
                packet_rows[method].append(
                    {
                        "packet_recall": packet_metric["packet_recall"],
                        "budget_compliant": float(packet_metric["budget_compliant"]),
                        "estimated_tokens": float(packet_metric["estimated_tokens"]),
                    }
                )
                query_result["methods"][method] = {
                    "candidate_ids": base_ids,
                    "baseline_metrics": base_metric,
                    "adaptive_ids": adaptive_ids,
                    "adaptive_metrics": adaptive_metric,
                    "context_packet": packet_metric,
                    "adaptive_roles": [
                        {"record_id": numeric_id(record_id), "role": role}
                        for record_id, role in adaptive.roles
                    ],
                    "explanation": list(adaptive.explanation),
                }

            hybrid_candidates = candidate_pools["hybrid_rrf"]
            hybrid_ids = _record_ids(hybrid_candidates)
            ablation_rows["hybrid_identity"].append(
                ranking_metrics(hybrid_ids, relevant_ids)
            )

            full = _select(
                store,
                hybrid_candidates,
                scenario,
                token_budget=token_budget,
            )
            no_temporal = _select(
                store,
                hybrid_candidates,
                scenario,
                token_budget=token_budget,
                use_temporal=False,
                use_relations=True,
            )
            no_relations = _select(
                store,
                hybrid_candidates,
                scenario,
                token_budget=token_budget,
                use_temporal=True,
                use_relations=False,
            )
            ablation_results = {
                "hybrid_full": full,
                "hybrid_no_temporal": no_temporal,
                "hybrid_no_relations": no_relations,
            }
            query_result["hybrid_ablations"] = {}
            for name, result in ablation_results.items():
                ids = _record_ids(result.ranked_candidates)
                metric = ranking_metrics(ids, relevant_ids)
                ablation_rows[name].append(metric)
                query_result["hybrid_ablations"][name] = {
                    "ranked_ids": ids,
                    "metrics": metric,
                    "context_packet": _packet_summary(result, relevant_ids),
                }

            query_result["hybrid_ablations"]["hybrid_identity"] = {
                "ranked_ids": hybrid_ids,
                "metrics": ablation_rows["hybrid_identity"][-1],
            }

            query_result["hybrid_budget_sweep"] = {}
            for budget in budget_sweep:
                result = _select(
                    store,
                    hybrid_candidates,
                    scenario,
                    token_budget=budget,
                )
                summary = _packet_summary(result, relevant_ids)
                budget_rows[budget].append(
                    {
                        "packet_recall": summary["packet_recall"],
                        "budget_compliant": float(summary["budget_compliant"]),
                        "estimated_tokens": float(summary["estimated_tokens"]),
                    }
                )
                query_result["hybrid_budget_sweep"][str(budget)] = summary

            per_query[scenario.name] = query_result

        temporal_probe = _temporal_probe(store, record_map)

    aggregate_base = {
        name: aggregate_metrics(rows) for name, rows in base_rows.items()
    }
    aggregate_adaptive = {
        name: aggregate_metrics(rows) for name, rows in adaptive_rows.items()
    }
    aggregate_ablations = {
        name: aggregate_metrics(rows) for name, rows in ablation_rows.items()
    }

    packet_summary = {}
    for name, rows in packet_rows.items():
        packet_summary[name] = {
            "mean_packet_recall": sum(row["packet_recall"] for row in rows) / len(rows),
            "budget_compliance_rate": sum(
                row["budget_compliant"] for row in rows
            ) / len(rows),
            "mean_estimated_tokens": sum(
                row["estimated_tokens"] for row in rows
            ) / len(rows),
        }

    budget_summary = {}
    for budget, rows in budget_rows.items():
        budget_summary[str(budget)] = {
            "mean_packet_recall": sum(row["packet_recall"] for row in rows) / len(rows),
            "budget_compliance_rate": sum(
                row["budget_compliant"] for row in rows
            ) / len(rows),
            "mean_estimated_tokens": sum(
                row["estimated_tokens"] for row in rows
            ) / len(rows),
        }

    result = {
        "schema_version": 1,
        "experiment": "MemoRoute Adaptive Selection Evaluation v1",
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "dataset_module": "benchmarks.retrieval_temporal_v1.dataset",
            "dataset_validation": "validate_dataset() passed",
            "synthetic_ground_truth_warning": (
                "Hand-authored synthetic labels; not production ground truth."
            ),
            "baseline_runner_policy": (
                "Existing retrieval_temporal_v1 baseline runner is untouched."
            ),
        },
        "configuration": {
            "record_count": len(records),
            "relation_count": len(relations),
            "query_count": len(QUERIES),
            "candidate_depth": candidate_depth,
            "output_k": output_k,
            "rrf_k": RRF_K,
            "token_budget": token_budget,
            "budget_sweep": list(budget_sweep),
            "embedding_model": "BAAI/bge-small-en-v1.5 (VectorRetriever default)",
            "embedding_backend": "real FastEmbed unless a test model is injected",
        },
        "environment": {
            "python": __import__("sys").version,
            "fastembed": _version("fastembed"),
            "onnxruntime": _version("onnxruntime"),
            "numpy": _version("numpy"),
            "pytest": _version("pytest"),
            "ruff": _version("ruff"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_head": _git_value("rev-parse", "--short", "HEAD"),
            "git_status": _git_value("status", "--short"),
        },
        "retrieval_baselines": aggregate_base,
        "adaptive_by_retriever": aggregate_adaptive,
        "hybrid_ablations": aggregate_ablations,
        "context_packets_by_retriever": packet_summary,
        "hybrid_context_budget_sweep": budget_summary,
        "fixed_temporal_probe": temporal_probe,
        "per_query": per_query,
        "interpretation_guardrails": [
            "Do not combine raw BM25, cosine, or RRF scores; the experiment uses ranks.",
            "Adaptive selector metrics are measured on the same retrieved candidate pools.",
            "The no-temporal ablation uses UNSPECIFIED mode while retaining relation evidence.",
            (
                "The no-relations ablation retains temporal assessment but supplies "
                "no relation evidence."
            ),
            "Relation direction is evidence, not proof that a source claim is true.",
            "A resolved relation endpoint is not a semantically resolved contradiction.",
            "Token counts are rough estimates, not tokenizer-exact counts.",
            "A synthetic benchmark result does not establish production generalization.",
        ],
    }

    json_path = _write_exclusive(
        output_dir,
        "adaptive_eval",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    print("ADAPTIVE EVALUATION COMPLETE")
    print(f"Records: {len(records)}")
    print(f"Relations: {len(relations)}")
    print(f"Queries: {len(QUERIES)}")
    print(f"Candidate depth: {candidate_depth}; output K: {output_k}; RRF k: {RRF_K}")
    print(f"Fixed temporal probes: {temporal_probe['passed']}/{temporal_probe['total']}")
    print("\nBaseline metrics:")
    for name, values in aggregate_base.items():
        print(
            f"  {name:<14} "
            f"R@1={values['recall_at_1']:.4f} "
            f"R@3={values['recall_at_3']:.4f} "
            f"Hit@3={values['hit_rate_at_3']:.4f} "
            f"MRR={values['mrr']:.4f}"
        )
    print("\nAdaptive metrics:")
    for name, values in aggregate_adaptive.items():
        print(
            f"  {name:<14} "
            f"R@1={values['recall_at_1']:.4f} "
            f"R@3={values['recall_at_3']:.4f} "
            f"Hit@3={values['hit_rate_at_3']:.4f} "
            f"MRR={values['mrr']:.4f}"
        )
    print(f"\nJSON artifact: {json_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-depth", type=int, default=CANDIDATE_DEPTH)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    args = parser.parse_args()
    run_experiment(
        candidate_depth=args.candidate_depth,
        token_budget=args.token_budget,
    )


if __name__ == "__main__":
    main()
