from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from memoroute.retrieval.hybrid import HybridRetriever


class Record:
    def __init__(self, number: int):
        self.id = UUID(f"00000000-0000-0000-0000-{number:012d}")
        self.recorded_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=number)

class FakeRetriever:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def retrieve(self, query, limit=10):
        self.calls.append((query, limit))
        return self.records[:limit]

def test_rrf_rewards_records_ranked_by_both_retrievers():
    consensus = Record(1)
    lexical_only = Record(2)
    bm25 = FakeRetriever([lexical_only, consensus])
    vector = FakeRetriever([consensus])

    result = HybridRetriever(bm25, vector, candidate_limit=10).retrieve("query", limit=2)

    assert [record.id for record in result] == [consensus.id, lexical_only.id]

def test_rrf_handles_records_missing_from_one_ranking():
    first = Record(1)
    second = Record(2)
    result = HybridRetriever(
        FakeRetriever([first]), FakeRetriever([second]), candidate_limit=5
    ).retrieve("query", limit=5)

    assert {record.id for record in result} == {first.id, second.id}

def test_duplicate_ids_within_one_ranking_count_only_once():
    first = Record(1)
    duplicate = Record(1)
    second = Record(2)
    result = HybridRetriever(
        FakeRetriever([first, duplicate]), FakeRetriever([second]), candidate_limit=5
    ).retrieve("query", limit=5)

    assert len(result) == 2
    assert len({record.id for record in result}) == 2

def test_retrievers_receive_candidate_depth_and_requested_query():
    bm25 = FakeRetriever([])
    vector = FakeRetriever([])
    HybridRetriever(bm25, vector, candidate_limit=12).retrieve("hello", limit=3)

    assert bm25.calls == [("hello", 12)]
    assert vector.calls == [("hello", 12)]

def test_requested_limit_can_exceed_candidate_depth():
    bm25 = FakeRetriever([])
    vector = FakeRetriever([])
    HybridRetriever(bm25, vector, candidate_limit=3).retrieve("hello", limit=7)

    assert bm25.calls == [("hello", 7)]
    assert vector.calls == [("hello", 7)]

def test_blank_query_returns_empty_without_calling_retrievers():
    bm25 = FakeRetriever([Record(1)])
    vector = FakeRetriever([Record(2)])

    assert HybridRetriever(bm25, vector).retrieve("   ") == []
    assert bm25.calls == []
    assert vector.calls == []

@pytest.mark.parametrize("kwargs", [{"rrf_k": 0}, {"candidate_limit": 0}])
def test_positive_configuration_values_are_required(kwargs):
    with pytest.raises(ValueError):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), **kwargs)

@pytest.mark.parametrize("kwargs", [{"rrf_k": True}, {"candidate_limit": True}])
def test_boolean_configuration_values_are_rejected(kwargs):
    with pytest.raises(TypeError):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), **kwargs)

def test_limit_validation_matches_other_retrievers():
    hybrid = HybridRetriever(FakeRetriever([]), FakeRetriever([]))
    with pytest.raises(TypeError):
        hybrid.retrieve("query", limit=True)
    with pytest.raises(ValueError):
        hybrid.retrieve("query", limit=0)
