
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from memoroute.memory.models import MemoryRecord, MemorySource
from memoroute.retrieval.bm25 import BM25Retriever


class FakeStore:
    def __init__(self, records):
        self.records = records

    def list_records(self, limit=None):
        records = sorted(
            self.records,
            key=lambda record: (record.recorded_at, str(record.id)),
            reverse=True,
        )
        return records if limit is None else records[:limit]

def make_record(content, day):
    timestamp = datetime(2026, 1, day, tzinfo=UTC)
    record = MemoryRecord(
        content=content,
        source=MemorySource(kind="test"),
    )
    return replace(record, recorded_at=timestamp)

def test_ranks_relevant_memory_above_newer_unrelated_memory():
    relevant = make_record(
        "The invoice payment was received from the customer.",
        1,
    )
    unrelated = make_record(
        "The weather was sunny during the weekend.",
        3,
    )
    store = FakeStore([unrelated, relevant])

    results = BM25Retriever(store).retrieve("invoice payment")

    assert results[0] == relevant

def test_returns_only_requested_number_of_results():
    records = [
        make_record("invoice payment received", 1),
        make_record("invoice payment pending", 2),
        make_record("invoice payment processed", 3),
    ]

    results = BM25Retriever(FakeStore(records)).retrieve(
        "invoice payment",
        limit=2,
    )

    assert len(results) == 2

def test_returns_empty_list_for_blank_query():
    record = make_record("invoice payment received", 1)

    results = BM25Retriever(FakeStore([record])).retrieve("   ")

    assert results == []

def test_returns_empty_list_when_no_terms_match():
    record = make_record("invoice payment received", 1)

    results = BM25Retriever(FakeStore([record])).retrieve("astronomy")

    assert results == []

def test_returns_empty_list_for_empty_store():
    results = BM25Retriever(FakeStore([])).retrieve("invoice")

    assert results == []

def test_equal_scores_use_newest_record_first():
    older = make_record("invoice payment", 1)
    newer = make_record("invoice payment", 2)

    results = BM25Retriever(FakeStore([older, newer])).retrieve("invoice")

    assert results == [newer, older]

@pytest.mark.parametrize("limit", [0, -1])
def test_rejects_non_positive_limit(limit):
    retriever = BM25Retriever(FakeStore([]))

    with pytest.raises(ValueError):
        retriever.retrieve("invoice", limit=limit)

@pytest.mark.parametrize("limit", [True, 1.5, "2"])
def test_rejects_non_integer_limit(limit):
    retriever = BM25Retriever(FakeStore([]))

    with pytest.raises(TypeError):
        retriever.retrieve("invoice", limit=limit)

def test_rejects_non_string_query():
    retriever = BM25Retriever(FakeStore([]))

    with pytest.raises(TypeError):
        retriever.retrieve(None)
