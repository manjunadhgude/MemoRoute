"""Local embedding-based memory retrieval."""

from typing import Protocol
from uuid import UUID

from memoroute.memory.models import MemoryRecord
from memoroute.retrieval.recent import RecordStore


class EmbeddingModel(Protocol):
    """Interface required from an embedding model."""

    def embed(self, documents): ...
    def query_embed(self, queries): ...

def _numpy():
    # Keep NumPy optional for users who only import/use non-vector retrieval.
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Vector retrieval requires NumPy. Install MemoRoute with: "
            "pip install -e '.[embeddings]'"
        ) from exc
    return np

class VectorRetriever:
    """Rank memories by cosine similarity using cached local embeddings."""

    def __init__(
        self,
        store: RecordStore,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        threads: int = 2,
        model: EmbeddingModel | None = None,
    ) -> None:
        if model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "Install MemoRoute with the embeddings extra: pip install -e '.[embeddings]'"
                ) from exc
            model = TextEmbedding(model_name=model_name, threads=threads)

        self._store = store
        self._model = model
        self._vectors: dict[UUID, object] = {}
        self._contents: dict[UUID, str] = {}
        self._records: dict[UUID, MemoryRecord] = {}
        self.refresh()

    def refresh(self) -> None:
        """Load current records and embed only records not already cached."""
        np = _numpy()
        records = self._store.list_records()
        current_ids = {record.id for record in records}

        for record_id in set(self._vectors) - current_ids:
            del self._vectors[record_id]
            self._contents.pop(record_id, None)

        changed_records = [
            record
            for record in records
            if record.id not in self._vectors or self._contents.get(record.id) != record.content
        ]

        if changed_records:
            vectors = np.asarray(
                list(self._model.embed([record.content for record in changed_records])),
                dtype=np.float32,
            )
            if vectors.ndim != 2 or vectors.shape[0] != len(changed_records):
                raise ValueError("Embedding model returned an invalid document matrix.")
            if vectors.shape[1] == 0:
                raise ValueError("Embedding model returned empty document vectors.")
            if not np.isfinite(vectors).all():
                raise ValueError("Embedding model returned non-finite document values.")
            norms = np.linalg.norm(vectors, axis=1)
            if np.any(norms == 0):
                raise ValueError("Embedding model returned a zero document vector.")

            dimensions = vectors.shape[1]
            existing_dimensions = {vector.shape[0] for vector in self._vectors.values()}
            if existing_dimensions and existing_dimensions != {dimensions}:
                raise ValueError("Embedding model returned inconsistent document dimensions.")

            for record, vector in zip(changed_records, vectors, strict=True):
                self._vectors[record.id] = vector
                self._contents[record.id] = record.content

        self._records = {record.id: record for record in records}

    def retrieve(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Return memories ranked by cosine similarity."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if not query.strip():
            return []

        self.refresh()
        if not self._records:
            return []

        np = _numpy()
        dimensions = next(iter(self._vectors.values())).shape[0]
        query_vectors = np.asarray(
            list(self._model.query_embed([query])),
            dtype=np.float32,
        )
        if query_vectors.shape != (1, dimensions):
            raise ValueError("Embedding model returned an invalid query vector.")
        query_vector = query_vectors[0]

        if not np.isfinite(query_vector).all():
            raise ValueError("Embedding model returned non-finite query values.")

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Embedding model returned a zero query vector.")

        scored = []
        for record_id, record in self._records.items():
            vector = self._vectors[record_id]
            score = float(np.dot(query_vector, vector) / (query_norm * np.linalg.norm(vector)))
            scored.append((score, record))

        scored.sort(
            key=lambda item: (item[0], item[1].recorded_at, str(item[1].id)),
            reverse=True,
        )
        return [record for _, record in scored[:limit]]
