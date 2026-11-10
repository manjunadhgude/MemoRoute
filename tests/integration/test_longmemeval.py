
from datetime import datetime, timezone

import pytest

from benchmarks.longmemeval.runner import (
    build_record,
    eligible_records,
    is_abstention,
    metrics_for,
    parse_dataset_datetime,
    serialize_session,
)

def test_parse_longmemeval_timestamp():
    parsed = parse_dataset_datetime("2023/05/30 (Tue) 23:40")
    assert parsed == datetime(2023, 5, 30, 23, 40, tzinfo=timezone.utc)

def test_session_serialization_preserves_roles_and_content():
    text = serialize_session(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    )
    assert text == "user: Hello\n\nassistant: Hi there"

def test_session_serialization_rejects_malformed_turn():
    with pytest.raises(ValueError):
        serialize_session([{"role": "user"}])

def test_abstention_is_classified_by_question_id_suffix():
    assert is_abstention({"question_id": "example_abs"})
    assert not is_abstention({"question_id": "example"})

def test_metrics_support_multiple_gold_sessions():
    result = metrics_for(["s1", "s2", "s4"], {"s2", "s3"})
    assert result["recall@1"] == 0.0
    assert result["recall@3"] == 0.5
    assert result["hit@3"] == 1
    assert result["mrr"] == 0.5

def test_metrics_reject_empty_gold():
    with pytest.raises(ValueError):
        metrics_for(["s1"], set())

def test_temporal_filter_excludes_future_sessions():
    past = datetime(2023, 5, 1, tzinfo=timezone.utc)
    future = datetime(2023, 6, 1, tzinfo=timezone.utc)
    question_date = datetime(2023, 5, 15, tzinfo=timezone.utc)

    records = [
        build_record("q1", "past", past, [{"role": "user", "content": "past"}]),
        build_record("q1", "future", future, [{"role": "user", "content": "future"}]),
    ]
    eligible = eligible_records(records, question_date)
    assert [record.source.locator for record in eligible] == ["past"]

def test_record_identity_is_deterministic():
    date = datetime(2023, 5, 1, tzinfo=timezone.utc)
    turns = [{"role": "user", "content": "hello"}]
    first = build_record("q1", "s1", date, turns)
    second = build_record("q1", "s1", date, turns)
    assert first.id == second.id
    assert first.source.locator == "s1"
