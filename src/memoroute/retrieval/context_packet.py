"""Budgeted context packets with a transparent rough token estimate."""
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from memoroute.memory.models import MemoryRecord


@dataclass(frozen=True, slots=True)
class ContextItem:
    record: MemoryRecord
    estimated_tokens: int
    original_rank: int
    role: str


@dataclass(frozen=True, slots=True)
class ContextPacket:
    items: tuple[ContextItem, ...]
    token_budget: int
    estimated_tokens: int
    omitted_candidate_ids: tuple[UUID, ...]
    budget_exhausted: bool
    estimator: str = "ceil(whitespace_words * 1.3); estimate_only"


def estimate_tokens(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return math.ceil(len(re.findall(r"\S+", text)) * 1.3)


def build_context_packet(
    records,
    *,
    token_budget: int,
    roles=None,
    original_ranks: Mapping[UUID, int] | None = None,
) -> ContextPacket:
    """Build a first-fit packet while preserving upstream retrieval ranks."""
    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise TypeError("token_budget must be an integer")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    roles = {} if roles is None else roles
    if not isinstance(roles, Mapping):
        raise TypeError("roles must be a mapping")
    if original_ranks is not None and not isinstance(original_ranks, Mapping):
        raise TypeError("original_ranks must be a mapping")

    items, omitted, total = [], [], 0
    seen_ids = set()
    for packet_rank, record in enumerate(records, 1):
        if not isinstance(record, MemoryRecord):
            raise TypeError("every item must be a MemoryRecord")
        if record.id in seen_ids:
            raise ValueError("candidate record IDs must be unique")
        seen_ids.add(record.id)

        original_rank = packet_rank if original_ranks is None else original_ranks.get(record.id)
        if (
            isinstance(original_rank, bool)
            or not isinstance(original_rank, int)
            or original_rank <= 0
        ):
            raise ValueError("every included candidate must have a positive integer original rank")

        cost = estimate_tokens(record.content)
        if total + cost <= token_budget:
            items.append(
                ContextItem(
                    record=record,
                    estimated_tokens=cost,
                    original_rank=original_rank,
                    role=roles.get(record.id, "candidate"),
                )
            )
            total += cost
        else:
            omitted.append(record.id)
    return ContextPacket(tuple(items), token_budget, total, tuple(omitted), bool(omitted))
