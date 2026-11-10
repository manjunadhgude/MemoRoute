
# MemoRoute LongMemEval-S Retrieval Benchmark

## Purpose

This benchmark measures session-level retrieval quality. It does not generate
answers and does not claim answer-generation accuracy.

Dataset:
- Official cleaned LongMemEval-S JSON.
- Upstream revision: `98d7416c24c778c2fee6e6f3006e7a073259d48f`.
- Expected SHA-256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.

## Mapping

Each conversation session becomes one `MemoryRecord`.
The complete turn sequence is serialized as role-prefixed text.
Session ID is retained in `MemorySource.locator`; question ID is retained
in `MemorySource.reference`. Session date populates `source_at` and
`TemporalInfo(EVENT, occurred_at=...)`.

A separate in-memory store is created for each question. No records from
another question can enter its retrieval corpus.

`recorded_at` is set to the session timestamp only to make existing recency
tie-breaking deterministic. It is not a real ingestion timestamp.

## Evaluation

Retrievers:
- BM25
- BGE Vector (`BAAI/bge-small-en-v1.5`)
- Hybrid Reciprocal Rank Fusion (`rrf_k=60`)

Metrics:
- Recall@1/3/5/10 (fraction of all gold sessions retrieved)
- Hit@1/3/5/10
- Mean reciprocal rank (MRR)

Original-label evaluation uses all supplied history sessions and gold
`answer_session_ids` for answerable questions.

Temporal evaluation removes sessions later than the question timestamp and
removes future-dated gold sessions from the eligible gold set. Questions
with no remaining eligible gold are excluded from temporal metric
denominators and counted separately.

The 30 IDs ending in `_abs` are treated as abstention cases and excluded from
retrieval-recall metrics. Their inconsistent `has_answer` metadata is
reported separately.

Dataset timestamps do not include timezone offsets. The runner interprets
them as UTC for deterministic comparisons; this is a normalization
convention, not a claim about the original locale.

## Commands

Smoke test:

```bash
python -m benchmarks.longmemeval.runner --limit 3
```

Full benchmark:

```bash
python -m benchmarks.longmemeval.runner
```

Results are written to a unique timestamped directory under
`results/longmemeval/`. Existing results are never overwritten.

## Interpretation

The original-label and temporal metrics answer different questions and
must be reported separately. The original evaluation preserves the supplied
benchmark labels. The temporal evaluation asks what could be retrieved from
sessions available by the question date. Neither is answer-generation QA.
