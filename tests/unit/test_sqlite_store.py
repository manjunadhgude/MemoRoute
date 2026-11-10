
"""Tests for SQLite persistence of immutable memory records and relations."""

import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memoroute.memory.models import (
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    MemorySource,
)
from memoroute.memory.sqlite_store import SQLiteMemoryStore


def make_record(content: str = "A test memory.") -> MemoryRecord:
    return MemoryRecord(
        content=content,
        source=MemorySource(kind="test"),
    )

def make_relation(
    source_id,
    target_id,
    kind=MemoryRelationKind.SUPERSEDES,
    relation_id=None,
    recorded_at: datetime | None = None,
) -> MemoryRelation:
    return MemoryRelation(
        source_id=source_id,
        target_id=target_id,
        kind=kind,
        id=relation_id or uuid4(),
        **({"recorded_at": recorded_at} if recorded_at is not None else {}),
    )

def test_record_survives_store_reopening(tmp_path):
    database_path = tmp_path / "memory.sqlite3"
    record = make_record()
    first_store = SQLiteMemoryStore(database_path)

    first_store.save_record(record)

    reopened_store = SQLiteMemoryStore(database_path)
    assert reopened_store.get_record(record.id) == record

def test_missing_record_returns_none(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    assert store.get_record(uuid4()) is None

def test_duplicate_record_id_is_rejected(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    record = make_record()

    store.save_record(record)

    with pytest.raises(sqlite3.IntegrityError):
        store.save_record(record)

@pytest.mark.parametrize("kind", list(MemoryRelationKind))
def test_relation_round_trip_and_endpoint_lookup(tmp_path, kind):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    source = make_record("Source memory.")
    target = make_record("Target memory.")
    relation = make_relation(source.id, target.id, kind)

    store.save_record(source)
    store.save_record(target)
    store.save_relation(relation)

    assert store.get_relation(relation.id) == relation
    assert relation in store.get_relations_for_record(source.id)
    assert relation in store.get_relations_for_record(target.id)

def test_missing_relation_returns_none(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    assert store.get_relation(uuid4()) is None

def test_relations_for_record_returns_both_endpoint_directions(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    first = make_record("First memory.")
    middle = make_record("Middle memory.")
    last = make_record("Last memory.")

    first_relation = make_relation(
        first.id,
        middle.id,
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second_relation = make_relation(
        last.id,
        middle.id,
        recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    for record in (first, middle, last):
        store.save_record(record)

    store.save_relation(first_relation)
    store.save_relation(second_relation)

    assert store.get_relations_for_record(middle.id) == [
        first_relation,
        second_relation,
    ]

@pytest.mark.parametrize("missing_endpoint", ["source", "target"])
def test_relation_requires_both_records(tmp_path, missing_endpoint):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    existing_record = make_record()
    store.save_record(existing_record)
    missing_id = uuid4()

    source_id = (
        missing_id if missing_endpoint == "source" else existing_record.id
    )
    target_id = (
        missing_id if missing_endpoint == "target" else existing_record.id
    )
    relation = make_relation(source_id, target_id)

    with pytest.raises(sqlite3.IntegrityError):
        store.save_relation(relation)

    assert store.get_relation(relation.id) is None

def test_duplicate_relation_id_is_rejected(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    first = make_record("First memory.")
    second = make_record("Second memory.")
    third = make_record("Third memory.")

    for record in (first, second, third):
        store.save_record(record)

    relation_id = uuid4()
    original = make_relation(
        first.id,
        second.id,
        relation_id=relation_id,
    )
    duplicate = make_relation(
        first.id,
        third.id,
        relation_id=relation_id,
    )

    store.save_relation(original)

    with pytest.raises(sqlite3.IntegrityError):
        store.save_relation(duplicate)
