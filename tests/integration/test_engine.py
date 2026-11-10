
from datetime import UTC, datetime

import pytest

from memoroute.engine import MemoRouteEngine
from memoroute.memory.models import MemoryRecord, MemorySource
from memoroute.retrieval.recent import RecentHistoryRetriever
from memoroute.retrieval.temporal_selector import (
    TemporalQueryMode,
)


class FakeStore:
    def __init__(self, records):
        self.records = list(records)
        self.relation_lookups = []

    def get_relations_for_records(self, record_ids):
        ids = tuple(record_ids)
        self.relation_lookups.append(ids)
        return []

    def list_records(self, limit=None):
        records = list(reversed(self.records))
        return records if limit is None else records[:limit]

class FakeRetriever:
    def __init__(self, records):
        self.records = list(records)
        self.calls = []

    def retrieve(self, query, limit=10):
        self.calls.append((query, limit))
        return self.records[:limit]

def make_record(content):
    return MemoryRecord(
        content=content,
        source=MemorySource(kind="integration-test"),
    )

def make_engine(records):
    store = FakeStore(records)
    retriever = FakeRetriever(records)
    return MemoRouteEngine(store=store, retriever=retriever), store, retriever

def test_engine_runs_complete_unspecified_pipeline():
    records = [
        make_record("The production database uses PostgreSQL."),
        make_record("The service runs in the production environment."),
    ]
    engine, store, retriever = make_engine(records)

    result = engine.run(
        "What database does production use?",
        token_budget=100,
    )

    assert result.retrieved_candidates == tuple(records)
    assert result.assessment.candidates == tuple(records)
    assert result.intent.mode == TemporalQueryMode.UNSPECIFIED
    assert result.context_packet is result.adaptive_selection.context_packet
    assert len(store.relation_lookups) == 1
    assert retriever.calls == [("What database does production use?", 10)]

def test_current_mode_supplies_one_aware_timestamp():
    records = [make_record("Production uses PostgreSQL.")]
    engine, _, _ = make_engine(records)

    result = engine.run("What is the current database?", mode="current")

    assert result.intent.mode == TemporalQueryMode.CURRENT
    assert isinstance(result.intent.as_of, datetime)
    assert result.intent.as_of.tzinfo is not None
    assert result.assessment.as_of == result.intent.as_of

def test_as_of_requires_explicit_timestamp():
    engine, _, _ = make_engine([make_record("A historical fact.")])

    with pytest.raises((TypeError, ValueError)):
        engine.run("What was true?", mode=TemporalQueryMode.AS_OF)

def test_unspecified_rejects_as_of_timestamp():
    engine, _, _ = make_engine([make_record("A fact.")])

    with pytest.raises((TypeError, ValueError)):
        engine.run(
            "What do we know?",
            mode=TemporalQueryMode.UNSPECIFIED,
            as_of=datetime(2025, 1, 1, tzinfo=UTC),
        )

def test_recent_history_retriever_uses_query_free_interface():
    records = [
        make_record("Older memory."),
        make_record("Newer memory."),
    ]
    store = FakeStore(records)
    retriever = RecentHistoryRetriever(store)
    engine = MemoRouteEngine(store=store, retriever=retriever)

    result = engine.run("Show recent memories.", limit=1)

    assert len(result.retrieved_candidates) == 1
    assert result.retrieved_candidates[0].content == "Newer memory."
    assert result.assessment.candidates == result.retrieved_candidates

def test_empty_retrieval_produces_empty_result():
    engine, _, _ = make_engine([])

    result = engine.run("Find a relevant memory.")

    assert result.retrieved_candidates == ()
    assert result.context_packet.items == ()

def test_invalid_query_and_limit_are_rejected():
    engine, _, _ = make_engine([])

    with pytest.raises(ValueError):
        engine.run("   ")

    with pytest.raises(ValueError):
        engine.run("valid query", limit=0)

    with pytest.raises(TypeError):
        engine.run("valid query", limit=True)
