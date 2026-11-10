
import pytest

from memoroute.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_relevant_results():
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, k=3) == 0.5

def test_recall_at_k_returns_zero_when_no_relevant_result_is_found():
    assert recall_at_k(["a", "b"], {"c"}, k=2) == 0.0

def test_recall_at_k_returns_one_when_all_relevant_results_are_found():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0

def test_reciprocal_rank_uses_first_relevant_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == 0.5

def test_reciprocal_rank_returns_zero_when_no_relevant_result_is_found():
    assert reciprocal_rank(["a", "b"], {"c"}) == 0.0

def test_mean_reciprocal_rank_averages_multiple_queries():
    results = [
        (["a", "b"], {"b"}),
        (["c", "d"], {"c"}),
        (["e"], {"z"}),
    ]

    assert mean_reciprocal_rank(results) == pytest.approx(0.5)

@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_recall_at_k_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=k)

def test_metrics_reject_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)

    with pytest.raises(ValueError):
        reciprocal_rank(["a"], set())

def test_mean_reciprocal_rank_rejects_empty_query_collection():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])

def test_multi_relevance_returns_partial_recall_when_one_of_two_is_found():
    retrieved = ["memory-5"]
    relevant = {"memory-1", "memory-5"}

    assert recall_at_k(retrieved, relevant, k=1) == 0.5

def test_multi_relevance_returns_full_recall_when_both_are_found():
    retrieved = ["memory-5", "memory-1", "memory-9"]
    relevant = {"memory-1", "memory-5"}

    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 1.0
