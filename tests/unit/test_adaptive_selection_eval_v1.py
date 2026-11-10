"""Tests for the separate adaptive evaluation harness; no real model downloads."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.adaptive_selection_eval_v1.metrics import (
    aggregate_metrics,
    packet_coverage,
    ranking_metrics,
)
from benchmarks.adaptive_selection_eval_v1.run_experiment import run_experiment


class FakeEmbeddingModel:
    """Small deterministic adapter used only by tests."""

    @staticmethod
    def _vector(text: str):
        return np.array(
            [len(text) + 1, sum(map(ord, text)) % 97 + 1],
            dtype=float,
        )

    def embed(self, documents):
        return [self._vector(document) for document in documents]

    def query_embed(self, queries):
        return [self._vector(query) for query in queries]


def test_ranking_metrics_distinguish_recall_and_hit_rate():
    result = ranking_metrics([1, 2, 3], [1, 2])
    assert result["recall_at_1"] == pytest.approx(0.5)
    assert result["recall_at_3"] == pytest.approx(1.0)
    assert result["hit_rate_at_1"] == pytest.approx(1.0)
    assert result["hit_rate_at_3"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)


def test_ranking_metrics_returns_zero_when_no_relevant_item_is_found():
    result = ranking_metrics([1, 2, 3], [9])
    assert result["recall_at_3"] == 0.0
    assert result["hit_rate_at_3"] == 0.0
    assert result["mrr"] == 0.0


def test_aggregate_metrics_macro_averages_queries():
    rows = [
        ranking_metrics([1], [1]),
        ranking_metrics([2], [9]),
    ]
    result = aggregate_metrics(rows)
    assert result["recall_at_1"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)


def test_packet_coverage_uses_relevant_set_denominator():
    assert packet_coverage([1, 3], [1, 2]) == pytest.approx(0.5)


def test_fake_model_experiment_runs_all_methods_and_ablations(tmp_path):
    result = run_experiment(
        model=FakeEmbeddingModel(),
        output_dir=tmp_path,
        candidate_depth=32,
        output_k=3,
        token_budget=256,
        budget_sweep=(64, 128),
    )
    assert result["configuration"]["record_count"] == 32
    assert result["configuration"]["query_count"] == 20
    assert set(result["retrieval_baselines"]) == {
        "recent_history",
        "bm25",
        "bge_vector",
        "hybrid_rrf",
    }
    assert set(result["adaptive_by_retriever"]) == {
        "recent_history",
        "bm25",
        "bge_vector",
        "hybrid_rrf",
    }
    assert set(result["hybrid_ablations"]) == {
        "hybrid_identity",
        "hybrid_full",
        "hybrid_no_temporal",
        "hybrid_no_relations",
    }
    assert result["fixed_temporal_probe"]["total"] > 0
    assert len(list(tmp_path.glob("adaptive_eval_*.json"))) == 1
