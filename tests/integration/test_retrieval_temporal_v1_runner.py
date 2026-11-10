
"""Integration tests for the retrieval-temporal experiment runner."""

import hashlib
import re

from benchmarks.retrieval_temporal_v1.run_experiment import run_experiment


class FakeEmbeddingModel:
    """Small deterministic embedding model for tests; no network required."""

    @staticmethod
    def _vectors(texts):
        vectors = []

        for text in texts:
            vector = [0.0] * 64

            for token in re.findall(r"\b\w+\b", text.casefold()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % 64
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign

            vectors.append(vector)

        return vectors

    def embed(self, documents):
        return self._vectors(documents)

    def query_embed(self, queries):
        return self._vectors(queries)

def test_retrieval_temporal_experiment_with_fake_embeddings():
    result = run_experiment(vector_model=FakeEmbeddingModel())

    assert result["dataset"]["records"] == 32
    assert result["dataset"]["relations"] == 7
    assert result["dataset"]["queries"] == 20

    assert set(result["retrieval"]) == {
        "recent_history",
        "bm25",
        "vector",
        "hybrid_rrf",
    }

    for retriever in result["retrieval"].values():
        assert len(retriever["queries"]) == 20
        assert 0.0 <= retriever["metrics"]["recall_at_1"] <= 1.0
        assert 0.0 <= retriever["metrics"]["recall_at_3"] <= 1.0
        assert 0.0 <= retriever["metrics"]["mrr"] <= 1.0

    temporal = result["fixed_temporal_assessment"]

    assert temporal["all_checks_passed"] is True
    assert temporal["checks"]["temporal_status"] == {
        "passed": 37,
        "total": 37,
    }
    assert temporal["expected_relation_evidence_entries"] == 23

    assert all(
        query["relation_evidence_match"]
        for query in temporal["queries"]
    )
