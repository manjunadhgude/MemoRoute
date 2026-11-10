"""Analyze the latest adaptive evaluation artifact without changing source files."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _latest_artifact(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("adaptive_eval_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No adaptive_eval_*.json artifacts found in {results_dir}"
        )
    return candidates[-1]


def _metric_table(title: str, values: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Method | Recall@1 | Recall@3 | Hit Rate@3 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in values.items():
        lines.append(
            f"| {name} | {metrics['recall_at_1']:.4f} | "
            f"{metrics['recall_at_3']:.4f} | "
            f"{metrics['hit_rate_at_3']:.4f} | {metrics['mrr']:.4f} |"
        )
    lines.append("")
    return lines


def analyze(path: Path | None = None) -> Path:
    results_dir = Path(__file__).resolve().parents[2] / "results" / "memoroute_final"
    artifact = path or _latest_artifact(results_dir)
    data: dict[str, Any] = json.loads(artifact.read_text(encoding="utf-8"))

    lines = [
        "# MemoRoute Adaptive Selection — Experiment Analysis",
        "",
        f"- Source artifact: `{artifact}`",
        f"- Experiment time: `{data['created_at']}`",
        f"- Dataset: {data['configuration']['record_count']} records, "
        f"{data['configuration']['relation_count']} relations, "
        f"{data['configuration']['query_count']} queries",
        f"- Candidate depth: {data['configuration']['candidate_depth']}",
        f"- Output K: {data['configuration']['output_k']}",
        f"- Context budget: {data['configuration']['token_budget']} estimated tokens",
        "",
    ]
    lines.extend(_metric_table("Retrieval baselines", data["retrieval_baselines"]))
    lines.extend(
        _metric_table(
            "Adaptive selector by retrieval source",
            data["adaptive_by_retriever"],
        )
    )
    lines.extend(_metric_table("Hybrid ablations", data["hybrid_ablations"]))

    baseline = data["retrieval_baselines"]["hybrid_rrf"]
    adaptive = data["hybrid_ablations"]["hybrid_full"]
    lines.extend(
        [
            "## Hybrid RRF versus full adaptive selection",
            "",
            "| Metric | Baseline | Adaptive | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("recall_at_1", "Recall@1"),
        ("recall_at_3", "Recall@3"),
        ("hit_rate_at_3", "Hit Rate@3"),
        ("mrr", "MRR"),
    ):
        delta = adaptive[key] - baseline[key]
        lines.append(
            f"| {label} | {baseline[key]:.4f} | {adaptive[key]:.4f} | {delta:+.4f} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Ablation deltas versus full adaptive selection",
            "",
            "| Variant | Recall@3 delta | MRR delta |",
            "|---|---:|---:|",
        ]
    )
    for name, metrics in data["hybrid_ablations"].items():
        if name == "hybrid_full":
            continue
        lines.append(
            f"| {name} | "
            f"{metrics['recall_at_3'] - adaptive['recall_at_3']:+.4f} | "
            f"{metrics['mrr'] - adaptive['mrr']:+.4f} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Context-budget sweep",
            "",
            "| Budget | Mean relevant evidence coverage | Budget compliance | "
            "Mean estimated tokens |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget, metrics in data["hybrid_context_budget_sweep"].items():
        lines.append(
            f"| {budget} | {metrics['mean_packet_recall']:.4f} | "
            f"{metrics['budget_compliance_rate']:.4f} | "
            f"{metrics['mean_estimated_tokens']:.1f} |"
        )
    lines.append("")

    temporal = data["fixed_temporal_probe"]
    lines.extend(
        [
            "## Fixed temporal-probe validation",
            "",
            f"- Correct statuses: **{temporal['passed']}/{temporal['total']}** "
            f"({temporal['accuracy']:.2%})",
            "- Relation evidence entries observed: "
            f"{temporal['relation_evidence_entries_observed']}",
            f"- Relation status counts: `{temporal['relation_status_counts']}`",
            "",
        ]
    )
    mismatches = [item for item in temporal["checks"] if not item["pass"]]
    if mismatches:
        lines.extend(["### Temporal-probe mismatches", ""])
        for item in mismatches:
            lines.append(
                f"- `{item['query']}` record {item['record_id']}: "
                f"expected `{item['expected']}`, got `{item['actual']}`"
            )
        lines.append("")
    else:
        lines.extend(["No fixed temporal-probe status mismatches were observed.", ""])

    query_changes = []
    for query_name, query in data["per_query"].items():
        before = query["hybrid_ablations"]["hybrid_identity"]["metrics"]["mrr"]
        after = query["hybrid_ablations"]["hybrid_full"]["metrics"]["mrr"]
        if after > before:
            outcome = "improved"
        elif after < before:
            outcome = "regressed"
        else:
            outcome = "unchanged"
        query_changes.append((query_name, outcome, before, after, query["category"]))

    lines.extend(
        [
            "## Query-level hybrid changes",
            "",
            "| Query | Category | Outcome | Baseline MRR | Adaptive MRR |",
            "|---|---|---|---:|---:|",
        ]
    )
    for name, outcome, before, after, category in query_changes:
        lines.append(
            f"| {name} | {category} | {outcome} | {before:.4f} | {after:.4f} |"
        )
    lines.append("")

    improved = sum(item[1] == "improved" for item in query_changes)
    regressed = sum(item[1] == "regressed" for item in query_changes)
    unchanged = sum(item[1] == "unchanged" for item in query_changes)
    lines.extend(
        [
            "## Interpretation safeguards",
            "",
            f"- Query MRR improved: {improved}; regressed: {regressed}; unchanged: {unchanged}.",
            "- Treat these outcomes as evidence on this hand-labelled synthetic challenge only.",
            "- A higher aggregate score does not establish production generalization.",
            "- Inspect query-level changes, especially production/staging, "
            "current/historical reversals, "
            "contradictions, and event/atemporal cases.",
            "- Do not claim that temporal validity resolves two simultaneously valid "
            "but semantically "
            "different entities.",
            "- Token counts are approximate whitespace estimates, not "
            "model-tokenizer measurements.",
            "- The no-temporal and no-relations variants are controlled ablations, not independent "
            "retrieval systems.",
            "",
        ]
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = results_dir / f"adaptive_analysis_{stamp}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(f"Analysis report: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    analyze(args.input)


if __name__ == "__main__":
    main()
