
"""Tests for the synthetic temporal assessment benchmark v1."""

from benchmarks.temporal_assessment_v1.dataset import (
    QUERIES,
    build_records,
    build_relations,
    validate_dataset,
)
from benchmarks.temporal_assessment_v1.run_experiment import run


def test_dataset_has_expected_counts_and_validates():
    validate_dataset()

    assert len(build_records()) == 24
    assert len(build_relations()) == 6
    assert len(QUERIES) == 12

def test_dataset_builders_are_deterministic():
    assert build_records() == build_records()
    assert build_relations() == build_relations()

def test_every_query_has_unique_name_and_candidates():
    names = [query.name for query in QUERIES]

    assert len(names) == len(set(names))

    for query in QUERIES:
        assert query.candidate_ids
        assert len(query.candidate_ids) == len(set(query.candidate_ids))

def test_temporal_assessment_experiment_passes(capsys):
    assert run() == 0

    output = capsys.readouterr().out

    assert "Queries: 12" in output
    assert "Checks passed: 127/127" in output
    assert "All temporal assessment checks passed." in output
