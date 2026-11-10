from dataclasses import replace
from datetime import UTC, datetime

import pytest

from memoroute.memory.models import MemoryRecord, MemorySource
from memoroute.retrieval.vector import VectorRetriever


class FakeStore:
    def __init__(self, records):
        self.records = records

    def list_records(self, limit=None):
        records = sorted(
            self.records,
            key=lambda r: (r.recorded_at, str(r.id)),
            reverse=True,
        )
        return records if limit is None else records[:limit]

class FakeModel:
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed(self, documents):
        self.document_calls += 1
        vectors = {
            "invoice payment received": [1.0, 0.0],
            "sunny weekend weather": [0.0, 1.0],
            "refund pending": [0.8, 0.2],
        }
        return [vectors[text] for text in documents]

    def query_embed(self, queries):
        self.query_calls += 1
        return [[1.0, 0.0] for _ in queries]

def make_record(content, day):
    record = MemoryRecord(content=content, source=MemorySource(kind="test"))
    return replace(record, recorded_at=datetime(2026, 1, day, tzinfo=UTC))

def test_ranks_by_cosine_similarity():
    relevant = make_record("invoice payment received", 1)
    unrelated = make_record("sunny weekend weather", 2)
    retriever = VectorRetriever(
        FakeStore([unrelated, relevant]), model=FakeModel()
    )
    assert retriever.retrieve("invoice payment")[0] == relevant

def test_limits_results():
    records = [
        make_record("invoice payment received", 1),
        make_record("sunny weekend weather", 2),
    ]
    retriever = VectorRetriever(FakeStore(records), model=FakeModel())
    assert len(retriever.retrieve("invoice", limit=1)) == 1

def test_empty_query_returns_empty():
    record = make_record("invoice payment received", 1)
    retriever = VectorRetriever(FakeStore([record]), model=FakeModel())
    assert retriever.retrieve("   ") == []

def test_empty_store_returns_empty():
    retriever = VectorRetriever(FakeStore([]), model=FakeModel())
    assert retriever.retrieve("invoice") == []

def test_document_embeddings_are_cached():
    record = make_record("invoice payment received", 1)
    model = FakeModel()
    retriever = VectorRetriever(FakeStore([record]), model=model)
    retriever.retrieve("first query")
    retriever.retrieve("second query")
    assert model.document_calls == 1
    assert model.query_calls == 2

def test_refresh_embeds_new_records_only():
    first = make_record("invoice payment received", 1)
    second = make_record("sunny weekend weather", 2)
    store = FakeStore([first])
    model = FakeModel()
    retriever = VectorRetriever(store, model=model)
    store.records.append(second)
    retriever.refresh()
    assert model.document_calls == 2
    assert len(retriever.retrieve("invoice")) == 2

@pytest.mark.parametrize("limit", [0, -1])
def test_rejects_non_positive_limit(limit):
    retriever = VectorRetriever(FakeStore([]), model=FakeModel())
    with pytest.raises(ValueError):
        retriever.retrieve("invoice", limit=limit)

@pytest.mark.parametrize("limit", [True, 1.5, "2"])
def test_rejects_invalid_limit(limit):
    retriever = VectorRetriever(FakeStore([]), model=FakeModel())
    with pytest.raises(TypeError):
        retriever.retrieve("invoice", limit=limit)

def test_rejects_non_string_query():
    retriever = VectorRetriever(FakeStore([]), model=FakeModel())
    with pytest.raises(TypeError):
        retriever.retrieve(None)
