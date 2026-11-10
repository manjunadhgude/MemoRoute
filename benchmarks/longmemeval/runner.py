
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memoroute.memory.models import MemoryRecord, MemorySource
from memoroute.memory.temporal import TemporalInfo, TemporalKind
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.hybrid import HybridRetriever
from memoroute.retrieval.vector import VectorRetriever

EXPECTED_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
DEFAULT_DATASET = Path("data/raw/longmemeval_s_cleaned.json")
TOP_K = (1, 3, 5, 10)
RETRIEVERS = ("bm25", "vector", "hybrid")
RECORD_NAMESPACE = uuid.UUID("9d45e0d7-9d6b-4db2-9f2d-55a67e4d9b32")

def parse_dataset_datetime(value: str) -> datetime:
    """Parse LongMemEval wall-clock timestamps and normalize them to UTC."""
    if not isinstance(value, str):
        raise TypeError("Dataset timestamp must be a string.")

    patterns = (
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d (%a)",
        "%Y/%m/%d",
    )
    for pattern in patterns:
        try:
            parsed = datetime.strptime(value.strip(), pattern)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported LongMemEval timestamp: {value!r}")

def is_abstention(question: dict[str, Any]) -> bool:
    return str(question["question_id"]).endswith("_abs")

def serialize_session(turns: list[dict[str, Any]]) -> str:
    """Preserve the complete session as role-prefixed plain text."""
    if not isinstance(turns, list) or not turns:
        raise ValueError("A session must contain at least one turn.")

    parts = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise TypeError(f"Turn {index} must be an object.")
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Turn {index} requires string role and content.")
        parts.append(f"{role.strip()}: {content.strip()}")
    return "\n\n".join(parts)

def build_record(
    question_id: str,
    session_id: str,
    session_date: datetime,
    turns: list[dict[str, Any]],
) -> MemoryRecord:
    """Map one LongMemEval session to one deterministic MemoRoute record."""
    identity = f"longmemeval:{question_id}:{session_id}"
    record_id = uuid.uuid5(RECORD_NAMESPACE, identity)
    content = serialize_session(turns)

    return MemoryRecord(
        content=content,
        source=MemorySource(
            kind="longmemeval_session",
            reference=question_id,
            locator=session_id,
        ),
        id=record_id,
        source_at=session_date,
        # Dataset has no ingestion timestamp. Reuse the source date as a
        # deterministic tie-break timestamp; do not interpret it as real
        # system ingestion time.
        recorded_at=session_date,
        temporal=TemporalInfo(
            kind=TemporalKind.EVENT,
            occurred_at=session_date,
        ),
    )

class QuestionStore:
    """Small isolated store implementing MemoRoute's RecordStore contract."""

    def __init__(self, records: list[MemoryRecord]):
        self._records = list(records)

    def list_records(self, limit: int | None = None) -> list[MemoryRecord]:
        records = sorted(
            self._records,
            key=lambda item: (item.recorded_at, str(item.id)),
            reverse=True,
        )
        return records if limit is None else records[:limit]

class CachedFastEmbed:
    """One shared FastEmbed model with document/query vector caching."""

    def __init__(self, model_name: str, threads: int = 2):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self._cache: dict[tuple[str, str], Any] = {}
        self.query_mode = (
            "query_embed"
            if callable(getattr(self._model, "query_embed", None))
            else "embed_fallback"
        )

    def _encode(self, texts: list[str], mode: str) -> list[Any]:
        import numpy as np

        keys = [(mode, text) for text in texts]
        missing = list(dict.fromkeys(text for key, text in zip(keys, texts) if key not in self._cache))

        if missing:
            method = getattr(self._model, "embed")
            if mode == "query" and self.query_mode == "query_embed":
                method = getattr(self._model, "query_embed")

            vectors = list(method(missing))
            if len(vectors) != len(missing):
                raise RuntimeError("Embedding model returned an unexpected vector count.")

            for text, vector in zip(missing, vectors, strict=True):
                array = np.asarray(vector, dtype=np.float32)
                if array.ndim != 1 or not np.isfinite(array).all():
                    raise ValueError("Embedding model returned an invalid vector.")
                self._cache[(mode, text)] = array

        return [self._cache[key] for key in keys]

    def embed(self, documents: list[str]) -> list[Any]:
        return self._encode(documents, "document")

    def query_embed(self, queries: list[str]) -> list[Any]:
        return self._encode(queries, "query")

def make_question_records(row: dict[str, Any]) -> list[MemoryRecord]:
    ids = row["haystack_session_ids"]
    dates = row["haystack_dates"]
    sessions = row["haystack_sessions"]

    if not (len(ids) == len(dates) == len(sessions)):
        raise ValueError(f"Session alignment failure: {row['question_id']}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate session IDs: {row['question_id']}")

    records = [
        build_record(
            row["question_id"],
            session_id,
            parse_dataset_datetime(session_date),
            turns,
        )
        for session_id, session_date, turns in zip(
            ids, dates, sessions, strict=True
        )
    ]

    available = {record.source.locator for record in records}
    missing = set(row["answer_session_ids"]) - available
    if missing:
        raise ValueError(
            f"Gold sessions absent from history for {row['question_id']}: {sorted(missing)}"
        )
    return records

def eligible_records(
    records: list[MemoryRecord], question_date: datetime
) -> list[MemoryRecord]:
    return [
        record
        for record in records
        if record.source_at is not None and record.source_at <= question_date
    ]

def metrics_for(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> dict[str, float | int]:
    if not relevant_ids:
        raise ValueError("Metrics require at least one relevant session.")

    ranked = retrieved_ids
    first_rank = next(
        (index for index, item in enumerate(ranked, start=1) if item in relevant_ids),
        None,
    )
    output: dict[str, float | int] = {}

    for k in TOP_K:
        hits = len(set(ranked[:k]) & relevant_ids)
        output[f"recall@{k}"] = hits / len(relevant_ids)
        output[f"hit@{k}"] = int(hits > 0)

    output["mrr"] = 0.0 if first_rank is None else 1.0 / first_rank
    output["retrieved_count"] = len(ranked)
    return output

def retrieve_all(
    records: list[MemoryRecord],
    query: str,
    embedding_model: CachedFastEmbed | None,
) -> dict[str, list[str]]:
    if not records:
        raise ValueError("Cannot retrieve from an empty history.")

    store = QuestionStore(records)
    bm25 = BM25Retriever(store)
    results = {
        "bm25": [record.source.locator for record in bm25.retrieve(query, limit=10)]
    }

    if embedding_model is not None:
        vector = VectorRetriever(store, model=embedding_model)
        results["vector"] = [
            record.source.locator for record in vector.retrieve(query, limit=10)
        ]
        hybrid = HybridRetriever(
            bm25,
            vector,
            rrf_k=60,
            candidate_limit=max(50, len(records)),
        )
        results["hybrid"] = [
            record.source.locator for record in hybrid.retrieve(query, limit=10)
        ]

    return results

def dataset_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    abstentions = [row for row in rows if is_abstention(row)]
    future_history = 0
    future_gold = 0
    future_gold_questions: set[str] = set()
    abstention_with_has_answer = 0

    for row in rows:
        question_date = parse_dataset_datetime(row["question_date"])
        for session_id, raw_date, session in zip(
            row["haystack_session_ids"],
            row["haystack_dates"],
            row["haystack_sessions"],
            strict=True,
        ):
            session_date = parse_dataset_datetime(raw_date)
            if session_date > question_date:
                future_history += 1
                if session_id in row["answer_session_ids"]:
                    future_gold += 1
                    future_gold_questions.add(row["question_id"])

        if is_abstention(row) and any(
            turn.get("has_answer") is True
            for session in row["haystack_sessions"]
            for turn in session
        ):
            abstention_with_has_answer += 1

    return {
        "question_count": len(rows),
        "abstention_count": len(abstentions),
        "answerable_count": len(rows) - len(abstentions),
        "abstention_with_has_answer": abstention_with_has_answer,
        "future_history_session_references": future_history,
        "future_gold_session_references": future_gold,
        "future_gold_affected_questions": len(future_gold_questions),
    }

def aggregate(
    question_results: list[dict[str, Any]],
    scope: str,
    retriever: str,
) -> dict[str, Any] | None:
    selected = [
        row[scope][retriever]
        for row in question_results
        if row[scope] is not None and retriever in row[scope]
    ]
    if not selected:
        return None

    keys = [*(f"recall@{k}" for k in TOP_K), *(f"hit@{k}" for k in TOP_K), "mrr"]
    return {
        "question_count": len(selected),
        **{
            key: round(sum(item[key] for item in selected) / len(selected), 6)
            for key in keys
        },
    }

def category_aggregates(
    question_results: list[dict[str, Any]], scope: str, retriever: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in question_results:
        if row[scope] is not None and retriever in row[scope]:
            grouped[row["question_type"]].append(row)

    output = {}
    for category, items in sorted(grouped.items()):
        values = [item[scope][retriever] for item in items]
        output[category] = {
            "question_count": len(values),
            "recall@1": round(sum(x["recall@1"] for x in values) / len(values), 6),
            "recall@3": round(sum(x["recall@3"] for x in values) / len(values), 6),
            "recall@5": round(sum(x["recall@5"] for x in values) / len(values), 6),
            "recall@10": round(sum(x["recall@10"] for x in values) / len(values), 6),
            "mrr": round(sum(x["mrr"] for x in values) / len(values), 6),
        }
    return output

def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def run(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset)
    raw = dataset_path.read_bytes()
    dataset_sha = hashlib.sha256(raw).hexdigest()
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset checksum mismatch: expected {EXPECTED_DATASET_SHA256}, got {dataset_sha}"
        )

    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 500:
        raise ValueError("Expected the official 500-question LongMemEval-S dataset.")

    audit = dataset_audit(rows)
    selected_rows = rows if args.limit is None else rows[:args.limit]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(args.output_root) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    embedding_model = None
    embedding_status = "SKIPPED"
    embedding_error = None
    try:
        embedding_model = CachedFastEmbed(args.model_name, threads=args.threads)
        embedding_status = "PASS"
    except Exception as exc:
        embedding_error = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: Vector and Hybrid will be SKIPPED: {embedding_error}")

    question_results: list[dict[str, Any]] = []
    abstention_ids = []
    processed = 0
    temporal_no_eligible_gold = 0
    checkpoint = output_dir / "checkpoint.jsonl"

    try:
        with checkpoint.open("w", encoding="utf-8") as checkpoint_file:
            for index, row in enumerate(selected_rows, start=1):
                question_id = row["question_id"]
                processed += 1

                if is_abstention(row):
                    abstention_ids.append(question_id)
                    continue

                question_date = parse_dataset_datetime(row["question_date"])
                records = make_question_records(row)
                gold = set(row["answer_session_ids"])

                original_rankings = retrieve_all(
                    records, row["question"], embedding_model
                )
                original_metrics = {
                    name: metrics_for(ranking, gold)
                    for name, ranking in original_rankings.items()
                }

                temporal_corpus = eligible_records(records, question_date)
                temporal_ids = {
                    record.source.locator for record in temporal_corpus
                }
                temporal_gold = gold & temporal_ids

                temporal_metrics = None
                temporal_rankings = None
                temporal_status = "NO_ELIGIBLE_GOLD"

                if temporal_gold:
                    temporal_rankings = retrieve_all(
                        temporal_corpus, row["question"], embedding_model
                    )
                    temporal_metrics = {
                        name: metrics_for(ranking, temporal_gold)
                        for name, ranking in temporal_rankings.items()
                    }
                    temporal_status = "SCORED"
                else:
                    temporal_no_eligible_gold += 1

                result = {
                    "question_id": question_id,
                    "question_type": row["question_type"],
                    "original": original_metrics,
                    "original_rankings": original_rankings,
                    "temporal": temporal_metrics,
                    "temporal_rankings": temporal_rankings,
                    "temporal_status": temporal_status,
                    "history_session_count": len(records),
                    "temporal_session_count": len(temporal_corpus),
                    "gold_session_count": len(gold),
                    "temporal_gold_session_count": len(temporal_gold),
                }
                question_results.append(result)
                checkpoint_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                checkpoint_file.flush()

                if index % 10 == 0 or index == len(selected_rows):
                    print(
                        f"Processed {index}/{len(selected_rows)} dataset rows; "
                        f"answerable scored: {len(question_results)}"
                    )

    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "processed_rows": processed,
            "checkpoint": str(checkpoint),
            "dataset_sha256": dataset_sha,
        }
        write_json(output_dir / "failure.json", failure)
        raise

    model_status = {
        "bm25": "PASS",
        "vector": embedding_status,
        "hybrid": embedding_status,
    }
    if embedding_error:
        model_status["vector_error"] = embedding_error
        model_status["hybrid_error"] = embedding_error

    summary = {
        "run_id": run_id,
        "status": "PASS" if question_results else "FAIL",
        "mode": "SMOKE" if args.limit is not None else "FULL",
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "dataset_revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "embedding_model": args.model_name,
        "embedding_query_mode": (
            embedding_model.query_mode if embedding_model else None
        ),
        "embedding_status": embedding_status,
        "retriever_status": model_status,
        "dataset_audit": audit,
        "processed_rows": processed,
        "answerable_questions_scored": len(question_results),
        "abstention_questions_encountered": len(abstention_ids),
        "abstention_question_ids": abstention_ids,
        "temporal_questions_scored": sum(
            row["temporal_status"] == "SCORED" for row in question_results
        ),
        "temporal_questions_without_eligible_gold": temporal_no_eligible_gold,
        "original_metrics": {},
        "temporal_metrics": {},
        "original_category_metrics": {},
        "temporal_category_metrics": {},
        "question_results": question_results,
    }

    for retriever in RETRIEVERS:
        summary["original_metrics"][retriever] = aggregate(
            question_results, "original", retriever
        )
        summary["temporal_metrics"][retriever] = aggregate(
            question_results, "temporal", retriever
        )
        summary["original_category_metrics"][retriever] = category_aggregates(
            question_results, "original", retriever
        )
        summary["temporal_category_metrics"][retriever] = category_aggregates(
            question_results, "temporal", retriever
        )

    write_json(output_dir / "results.json", summary)

    lines = [
        "# MemoRoute — LongMemEval-S retrieval benchmark",
        "",
        f"- Run: `{run_id}`",
        f"- Mode: **{summary['mode']}**",
        f"- Dataset SHA-256: `{dataset_sha}`",
        f"- Processed rows: {processed}",
        f"- Answerable questions scored: {len(question_results)}",
        f"- Abstention questions encountered: {len(abstention_ids)}",
        f"- Temporal questions scored: {summary['temporal_questions_scored']}",
        f"- Temporal questions without eligible gold: {temporal_no_eligible_gold}",
        f"- Embedding status: {embedding_status}",
        "",
        "## Original-label results",
        "",
        "| Retriever | Questions | R@1 | R@3 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in RETRIEVERS:
        value = summary["original_metrics"][name]
        if value is None:
            lines.append(f"| {name} | SKIPPED | — | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | {value['question_count']} | "
                f"{value['recall@1']:.4f} | {value['recall@3']:.4f} | "
                f"{value['recall@5']:.4f} | {value['recall@10']:.4f} | "
                f"{value['mrr']:.4f} |"
            )

    lines += [
        "",
        "## Temporally admissible results",
        "",
        "| Retriever | Questions | R@1 | R@3 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RETRIEVERS:
        value = summary["temporal_metrics"][name]
        if value is None:
            lines.append(f"| {name} | SKIPPED | — | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | {value['question_count']} | "
                f"{value['recall@1']:.4f} | {value['recall@3']:.4f} | "
                f"{value['recall@5']:.4f} | {value['recall@10']:.4f} | "
                f"{value['mrr']:.4f} |"
            )

    lines += [
        "",
        "## Dataset integrity notes",
        "",
        f"- Abstention IDs: {audit['abstention_count']}",
        f"- Abstention IDs with `has_answer` turns: {audit['abstention_with_has_answer']}",
        f"- Future-dated history references: {audit['future_history_session_references']}",
        f"- Future-dated gold references: {audit['future_gold_session_references']}",
        f"- Questions affected by future gold: {audit['future_gold_affected_questions']}",
        "",
        "## Interpretation limits",
        "",
        "- Retrieval-only evaluation; no answer-generation accuracy is measured.",
        "- Original-label metrics retain the supplied gold labels.",
        "- Temporal metrics exclude future sessions from the corpus and future gold from eligible labels.",
        "- Temporal questions with no eligible gold evidence are excluded from temporal metric denominators and counted separately.",
        "- Dataset timestamps are timezone-naive; this runner normalizes them to UTC for consistent comparisons.",
        "- `recorded_at` is set to the source session timestamp solely for deterministic tie-breaking because ingestion timestamps are not supplied.",
        "- A smoke run is not a full benchmark result.",
        "",
        "## Artifacts",
        "",
        "- `results.json`: complete machine-readable output and per-question rankings.",
        "- `checkpoint.jsonl`: completed answerable question results.",
        "- `summary.md`: concise report.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== BENCHMARK SUMMARY ===")
    print("Status:", summary["status"])
    print("Mode:", summary["mode"])
    print("Answerable scored:", len(question_results))
    print("Abstentions encountered:", len(abstention_ids))
    print("Temporal scored:", summary["temporal_questions_scored"])
    print("Temporal no eligible gold:", temporal_no_eligible_gold)
    for name in RETRIEVERS:
        value = summary["original_metrics"][name]
        if value is None:
            print(f"{name}: SKIPPED")
        else:
            print(
                f"{name}: R@1={value['recall@1']:.4f} "
                f"R@3={value['recall@3']:.4f} "
                f"R@5={value['recall@5']:.4f} "
                f"R@10={value['recall@10']:.4f} "
                f"MRR={value['mrr']:.4f}"
            )
    print("Output:", output_dir)
    return output_dir

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the MemoRoute retrieval-only LongMemEval-S benchmark."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default="results/longmemeval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.threads <= 0:
        parser.error("--threads must be positive.")

    try:
        run(args)
    except Exception as exc:
        print(f"BENCHMARK FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
