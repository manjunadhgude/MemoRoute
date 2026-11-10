"""BM25 lexical memory retrieval."""

import math
import re

from memoroute.memory.models import MemoryRecord
from memoroute.retrieval.recent import RecordStore


class BM25Retriever:
    """Rank memories by lexical relevance to a query using BM25."""

    def __init__(
        self,
        store: RecordStore,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self._store = store
        self._k1 = k1
        self._b = b

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.casefold())

    def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Return the most relevant memories, newest first when scores tie."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")

        if limit <= 0:
            raise ValueError("limit must be positive.")

        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        records = self._store.list_records()
        if not records:
            return []

        documents = [self._tokenize(record.content) for record in records]
        document_count = len(documents)
        average_length = sum(map(len, documents)) / document_count

        if average_length == 0:
            return []

        document_frequencies = {
            term: sum(term in set(document) for document in documents) for term in query_terms
        }

        scored_records = []

        for record, document in zip(records, documents, strict=True):
            term_frequencies = {term: document.count(term) for term in query_terms}

            document_length = len(document)
            score = 0.0

            for term, frequency in term_frequencies.items():
                if frequency == 0:
                    continue

                df = document_frequencies[term]
                inverse_document_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))

                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * document_length / average_length
                )

                score += inverse_document_frequency * frequency * (self._k1 + 1) / denominator

            if score > 0:
                scored_records.append((score, record))

        scored_records.sort(
            key=lambda item: (
                item[0],
                item[1].recorded_at,
                str(item[1].id),
            ),
            reverse=True,
        )

        return [record for _, record in scored_records[:limit]]
