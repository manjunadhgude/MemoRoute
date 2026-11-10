
import pytest

from benchmarks.retrieval_challenge_v2.dataset import validate_dataset
from benchmarks.retrieval_challenge_v2.run_experiment import (
    LIMIT,
    build_records,
    evaluate,
)
from memoroute.memory.sqlite_store import SQLiteMemoryStore
from memoroute.retrieval.bm25 import BM25Retriever


def test_v2_benchmark_pipeline_reproduces_bm25_results(tmp_path, capsys):
    validate_dataset()

    store = SQLiteMemoryStore(tmp_path / "challenge.sqlite3")

    for record in build_records():
        store.save_record(record)

    assert len(store.list_records()) == 16

    retriever = BM25Retriever(store)

    metrics = evaluate(
        "BM25 INTEGRATION TEST",
        lambda query: retriever.retrieve(query, limit=LIMIT),
    )

    capsys.readouterr()

    assert metrics["recall@1"] == pytest.approx(0.5625)
    assert metrics["recall@3"] == pytest.approx(0.9375)
    assert metrics["mrr"] == pytest.approx(0.7916666667)
