
"""Command-line interface for MemoRoute."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from memoroute.engine import MemoRouteEngine
from memoroute.memory.models import MemoryRecord, MemorySource
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.bm25 import BM25Retriever
from memoroute.retrieval.recent import RecentHistoryRetriever
from memoroute.retrieval.temporal_selector import TemporalQueryMode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memoroute",
        description="Retrieve, assess and budget evidence from a MemoRoute memory store.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="Add a memory to a SQLite database.")
    add.add_argument("--db", required=True, help="SQLite database path.")
    add.add_argument("--content", required=True, help="Memory text.")
    add.add_argument(
        "--source-kind",
        default="cli",
        help="Source type label (default: cli).",
    )
    add.add_argument("--source-reference", help="Optional source reference.")
    add.add_argument("--source-locator", help="Optional source locator.")

    query = commands.add_parser("query", help="Retrieve a budgeted context packet.")
    query.add_argument("--db", required=True, help="SQLite database path.")
    query.add_argument("--query", required=True, help="Retrieval query.")
    query.add_argument(
        "--retriever",
        choices=("bm25", "recent"),
        default="bm25",
        help="Retrieval method (default: bm25).",
    )
    query.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in TemporalQueryMode),
        default=TemporalQueryMode.UNSPECIFIED.value,
    )
    query.add_argument(
        "--as-of",
        help="Timezone-aware ISO-8601 timestamp; required for as_of mode.",
    )
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--token-budget", type=int, default=1200)

    return parser

def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "--as-of must be an ISO-8601 timestamp, e.g. "
            "2025-01-01T12:00:00+00:00"
        ) from exc

def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)

def _add_memory(args: argparse.Namespace) -> int:
    database = Path(args.db)
    database.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteMemoryStore(database)

    record = MemoryRecord(
        content=args.content,
        source=MemorySource(
            kind=args.source_kind,
            reference=args.source_reference,
            locator=args.source_locator,
        ),
    )
    store.save_record(record)

    print(json.dumps({
        "status": "saved",
        "database": str(database),
        "record_id": str(record.id),
        "content": record.content,
        "source_kind": record.source.kind,
    }, indent=2))
    return 0

def _query_memories(args: argparse.Namespace) -> int:
    store = SQLiteMemoryStore(args.db)
    if args.retriever == "recent":
        retriever = RecentHistoryRetriever(store)
    else:
        retriever = BM25Retriever(store)

    engine = MemoRouteEngine(store=store, retriever=retriever)
    result = engine.run(
        args.query,
        mode=TemporalQueryMode(args.mode),
        as_of=_parse_datetime(args.as_of),
        limit=args.limit,
        token_budget=args.token_budget,
    )

    assessment_by_id = {
        str(item.candidate_id): item
        for item in result.assessment.assessments
    }

    retrieved = []
    for rank, record in enumerate(result.retrieved_candidates, start=1):
        assessment = assessment_by_id[str(record.id)]
        retrieved.append({
            "rank": rank,
            "id": str(record.id),
            "content": record.content,
            "temporal_status": assessment.temporal_status.value,
            "relation_status": assessment.relation_status.value,
            "reason": assessment.reason_code,
            "relation_count": len(assessment.relations),
        })

    packet = [{
        "id": str(item.record.id),
        "content": item.record.content,
        "estimated_tokens": item.estimated_tokens,
        "original_rank": item.original_rank,
        "role": _json_default(item.role),
    } for item in result.context_packet.items]

    output = {
        "query": result.query,
        "intent": {
            "mode": result.intent.mode.value,
            "as_of": (
                result.intent.as_of.isoformat()
                if result.intent.as_of is not None else None
            ),
            "purpose": result.intent.purpose.value,
            "explicit_temporal_cue": result.intent.explicit_temporal_cue,
        },
        "retriever": args.retriever,
        "retrieved_count": len(retrieved),
        "retrieved_candidates": retrieved,
        "context_packet": {
            "token_budget": result.context_packet.token_budget,
            "estimated_tokens": result.context_packet.estimated_tokens,
            "estimator": result.context_packet.estimator,
            "budget_exhausted": result.context_packet.budget_exhausted,
            "items": packet,
            "omitted_candidate_ids": [
                str(record_id)
                for record_id in result.context_packet.omitted_candidate_ids
            ],
        },
        "explanation": result.explanation,
    }
    print(json.dumps(output, indent=2, default=_json_default))
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "add":
            return _add_memory(args)
        if args.command == "query":
            return _query_memories(args)
        parser.error("Unknown command.")
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    return 2

if __name__ == "__main__":
    raise SystemExit(main())
