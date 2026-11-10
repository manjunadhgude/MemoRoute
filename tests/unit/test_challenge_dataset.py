
import pytest

from benchmarks.retrieval_challenge_v2 import dataset


def test_accepts_valid_challenge_dataset():
    dataset.validate_dataset()

def test_rejects_duplicate_memory_ids(monkeypatch):
    memories = list(dataset.MEMORIES)
    memories[1] = (
        memories[0][0],
        *memories[1][1:],
    )
    monkeypatch.setattr(dataset, "MEMORIES", tuple(memories))

    with pytest.raises(ValueError, match="Memory IDs must be unique"):
        dataset.validate_dataset()

def test_rejects_incorrect_memory_count(monkeypatch):
    monkeypatch.setattr(dataset, "MEMORIES", dataset.MEMORIES[:-1])

    with pytest.raises(ValueError, match="Expected exactly 16 memories"):
        dataset.validate_dataset()

def test_rejects_incorrect_query_count(monkeypatch):
    monkeypatch.setattr(dataset, "QUERIES", dataset.QUERIES[:-1])

    with pytest.raises(ValueError, match="Expected exactly 8 queries"):
        dataset.validate_dataset()

def test_rejects_empty_relevance_labels(monkeypatch):
    queries = list(dataset.QUERIES)
    name, category, query, _ = queries[0]
    queries[0] = (name, category, query, ())
    monkeypatch.setattr(dataset, "QUERIES", tuple(queries))

    with pytest.raises(ValueError, match="one or more relevance labels"):
        dataset.validate_dataset()

def test_rejects_unknown_memory_references(monkeypatch):
    queries = list(dataset.QUERIES)
    name, category, query, _ = queries[0]
    queries[0] = (name, category, query, (999,))
    monkeypatch.setattr(dataset, "QUERIES", tuple(queries))

    with pytest.raises(ValueError, match="unknown memory ID"):
        dataset.validate_dataset()

def test_rejects_duplicate_relevance_labels(monkeypatch):
    queries = list(dataset.QUERIES)
    name, category, query, _ = queries[0]
    queries[0] = (name, category, query, (1, 1))
    monkeypatch.setattr(dataset, "QUERIES", tuple(queries))

    with pytest.raises(ValueError, match="duplicate relevance labels"):
        dataset.validate_dataset()

def test_rejects_blank_queries(monkeypatch):
    queries = list(dataset.QUERIES)
    name, category, _, relevant_ids = queries[0]
    queries[0] = (name, category, "   ", relevant_ids)
    monkeypatch.setattr(dataset, "QUERIES", tuple(queries))

    with pytest.raises(ValueError, match="empty query"):
        dataset.validate_dataset()
