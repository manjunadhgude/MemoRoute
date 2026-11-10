"""SQLite persistence for immutable MemoRoute memory records and relations."""

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from uuid import UUID

from memoroute.memory.models import MemoryRecord, MemoryRelation
from memoroute.memory.serialization import (
    dumps_memory_record,
    dumps_memory_relation,
    loads_memory_record,
    loads_memory_relation,
)

_SCHEMA_VERSION = 1

class SQLiteMemoryStore:
    """Persist memory records and relations in a local SQLite database."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]

            if version not in (0, _SCHEMA_VERSION):
                raise RuntimeError(f"Unsupported database schema version: {version}.")

            if version == _SCHEMA_VERSION:
                return

            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_records (
                        id TEXT PRIMARY KEY NOT NULL,
                        payload TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_relations (
                        id TEXT PRIMARY KEY NOT NULL,
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        CHECK (source_id <> target_id),
                        FOREIGN KEY (source_id)
                            REFERENCES memory_records(id) ON DELETE RESTRICT,
                        FOREIGN KEY (target_id)
                            REFERENCES memory_records(id) ON DELETE RESTRICT
                    )
                    """
                )
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def save_record(self, record: MemoryRecord) -> None:
        """Insert a record. Existing IDs are never overwritten."""
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord.")

        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO memory_records (id, payload)
                    VALUES (?, ?)
                    """,
                    (str(record.id), dumps_memory_record(record)),
                )

    def list_records(self, limit: int | None = None) -> list[MemoryRecord]:
        """Return records newest first, optionally restricted to a positive limit."""
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None.")
            if limit <= 0:
                raise ValueError("limit must be positive.")

        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT payload FROM memory_records").fetchall()

        records = [loads_memory_record(row["payload"]) for row in rows]
        records.sort(
            key=lambda record: (record.recorded_at, str(record.id)),
            reverse=True,
        )

        return records if limit is None else records[:limit]

    def get_record(self, record_id: UUID) -> MemoryRecord | None:
        """Return a record by ID, or None when it does not exist."""
        if not isinstance(record_id, UUID):
            raise TypeError("record_id must be a UUID.")

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM memory_records WHERE id = ?",
                (str(record_id),),
            ).fetchone()

        return loads_memory_record(row["payload"]) if row is not None else None

    def save_relation(self, relation: MemoryRelation) -> None:
        """Insert a relation between two existing records."""
        if not isinstance(relation, MemoryRelation):
            raise TypeError("relation must be a MemoryRelation.")

        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO memory_relations (
                        id, source_id, target_id, kind, recorded_at, payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(relation.id),
                        str(relation.source_id),
                        str(relation.target_id),
                        relation.kind.value,
                        relation.recorded_at.isoformat(),
                        dumps_memory_relation(relation),
                    ),
                )

    def get_relation(self, relation_id: UUID) -> MemoryRelation | None:
        """Return a relation by ID, or None when it does not exist."""
        if not isinstance(relation_id, UUID):
            raise TypeError("relation_id must be a UUID.")

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM memory_relations WHERE id = ?",
                (str(relation_id),),
            ).fetchone()

        return loads_memory_relation(row["payload"]) if row is not None else None

    def get_relations_for_record(self, record_id: UUID) -> list[MemoryRelation]:
        """Return relations where the record is either endpoint."""
        if not isinstance(record_id, UUID):
            raise TypeError("record_id must be a UUID.")

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM memory_relations
                WHERE source_id = ? OR target_id = ?
                ORDER BY recorded_at, id
                """,
                (str(record_id), str(record_id)),
            ).fetchall()

        return [loads_memory_relation(row["payload"]) for row in rows]

    def get_relations_for_records(
        self, record_ids: Iterable[UUID]
    ) -> list[MemoryRelation]:
        """Return unique relations touching any supplied record IDs."""
        try:
            supplied_ids = tuple(record_ids)
        except TypeError as exc:
            raise TypeError("record_ids must be an iterable of UUIDs.") from exc

        for record_id in supplied_ids:
            if not isinstance(record_id, UUID):
                raise TypeError("Every record ID must be a UUID.")

        unique_ids = tuple(dict.fromkeys(supplied_ids))
        if not unique_ids:
            return []

        relations_by_id: dict[UUID, MemoryRelation] = {}

        # Stay below SQLite's conservative 999-variable parameter limit.
        chunk_size = 900

        with closing(self._connect()) as connection:
            for start in range(0, len(unique_ids), chunk_size):
                chunk = unique_ids[start : start + chunk_size]
                placeholders = ", ".join("?" for _ in chunk)
                parameters = tuple(str(record_id) for record_id in chunk)

                rows = connection.execute(
                    f"""
                    SELECT payload
                    FROM memory_relations
                    WHERE source_id IN ({placeholders})
                       OR target_id IN ({placeholders})
                    """,
                    parameters + parameters,
                ).fetchall()

                for row in rows:
                    relation = loads_memory_relation(row["payload"])
                    relations_by_id[relation.id] = relation

        return sorted(
            relations_by_id.values(),
            key=lambda relation: (relation.recorded_at, str(relation.id)),
        )
