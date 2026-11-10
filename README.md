# MemoRoute

**Adaptive, temporal, and contradiction-aware memory retrieval for LLM applications.**

MemoRoute is a Python project that explores how an AI application can retrieve relevant information from a persistent memory store, account for temporal validity and relationships between claims, and construct a token-budgeted context packet.

Instead of treating every stored memory as equally useful, MemoRoute separates retrieval, temporal assessment, relation-aware selection, and context construction.

> **Project status:** Reconstructed recovery snapshot. Local package installation, CLI startup, SQLite ingestion and retrieval have been verified on the Acer development environment. This is not a production-ready LLM application or a byte-identical backup of the original Codespaces workspace.

## 1. What MemoRoute does

MemoRoute provides a memory retrieval and context-selection pipeline:

1. Store memory records in SQLite.
2. Retrieve candidate memories using a supported retrieval method.
3. Infer query intent and assess temporal relevance.
4. Inspect available relations between retrieved memories.
5. Adapt candidate ordering and assign evidence roles.
6. Construct a context packet within a configurable token budget.

The result is structured evidence that an application can pass to a language model.

**MemoRoute does not generate the final natural-language answer.** It focuses on memory retrieval and context preparation.

## 2. Architecture

```text
User Query
    |
    v
MemoRoute Engine
    |
    +---- Query Intent
    |
    +---- Candidate Retrieval
    |         |
    |         +---- BM25
    |         +---- Recent Retrieval
    |         +---- Vector Retrieval Module
    |         +---- Hybrid Retrieval Module
    |
    +---- Temporal Assessment
    |
    +---- Relation-Aware Adaptive Selection
    |
    v
Context Packet Builder
    |
    v
Budgeted Evidence Context
```

The engine coordinates the retrieval and selection stages. The context packet builder produces the final structured evidence package.

### Main components

| Component | Responsibility |
|---|---|
| `memory/models.py` | Memory records, relations and temporal data models |
| `memory/sqlite_store.py` | Persistent SQLite storage |
| `memory/temporal.py` | Temporal information and boundaries |
| `retrieval/query_intent.py` | Query intent representation |
| `retrieval/bm25.py` | BM25 retrieval |
| `retrieval/recent.py` | Recent-history retrieval |
| `retrieval/vector.py` | Vector retrieval implementation |
| `retrieval/hybrid.py` | Hybrid retrieval implementation |
| `retrieval/temporal_selector.py` | Temporal candidate assessment |
| `retrieval/adaptive_selector.py` | Temporal and relation-aware candidate selection |
| `retrieval/context_packet.py` | Token-budgeted context construction |
| `engine.py` | Coordinates the retrieval pipeline |
| `cli.py` | Command-line interface |

The current CLI exposes BM25 and recent retrieval. The existence of vector and hybrid modules does not mean that every retrieval backend is exposed through the CLI or has been validated in every environment.

## 3. Key design ideas

### Temporal awareness

Memories can contain temporal information, such as when a fact became valid or when an event occurred.

Temporal assessment can distinguish valid, invalid and indeterminate evidence for time-aware queries.

### Contradiction and relation awareness

Memory relations include:

- `CONTRADICTS`
- `SUPERSEDES`
- `CORRECTS`

The adaptive selector can identify related claims and assign evidence roles.

A contradiction does not automatically establish which claim is correct. A correction or supersession relation also does not independently prove that the replacement claim is true.

The selector works with the retrieved candidate set. It does not automatically fetch missing relation endpoints.

### Adaptive context selection

Candidate ordering and evidence roles depend on query intent and temporal assessment.

For example:
- Unspecified temporal queries preserve upstream retrieval order.
- Current-state queries can prioritize temporally valid evidence.
- Comparison queries can identify paired contradiction evidence.

### Token-budgeted context packets

The context packet builder selects evidence within a configurable budget.

The current token estimator is a rough whitespace-based estimate, not an exact tokenizer for a particular language model.

## 4. Installation

### Requirements

- Python 3.11 or newer
- pip
- Windows, Linux or macOS
- SQLite support through Python's standard library

Clone or download the project and open a terminal in the project directory.

### Windows PowerShell

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Install the project and development dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

If virtual-environment activation is preferred, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then the corresponding `python` commands can be used directly.

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

The package declares no mandatory third-party runtime dependencies. The development extra installs pytest and Ruff.

Vector and embedding experiments may require additional model and runtime setup. They are not included in the minimal installation.

## 5. Quickstart: store and retrieve a memory

MemoRoute provides two CLI commands: `add` and `query`.

### Step 1: Add a memory

```powershell
.\.venv\Scripts\python.exe -m memoroute add --db demo\quickstart.sqlite3 --content "The production cache uses Redis version 7." --source-kind demo
```

Expected behaviour: the CLI saves the memory to a SQLite database and returns its record ID.

### Step 2: Retrieve the memory

```powershell
.\.venv\Scripts\python.exe -m memoroute query --db demo\quickstart.sqlite3 --query "Which Redis version does production use?" --retriever bm25 --mode unspecified --limit 5 --token-budget 100
```

The output is JSON containing information such as:

- Query intent
- Retrieval method
- Retrieved candidates
- Temporal assessment
- Relation assessment
- Context packet
- Selection explanation

The example demonstrates memory ingestion, retrieval and context construction. It does not demonstrate LLM answer generation.

### CLI help

```powershell
.\.venv\Scripts\python.exe -m memoroute --help
```

## 6. Testing and verification

The recovered project contains unit and integration tests covering memory models, serialization, SQLite storage, retrieval, temporal assessment, adaptive selection, context packets, CLI behaviour and benchmark components.

The Acer recovery environment has recorded a full test result of:

**201 passed**

This is a local verification result for the recovered snapshot, not a claim that every possible configuration or external benchmark has been tested. Re-run the test suite after installation.

Additional local checks have verified:

- Editable package build and installation.
- Installed package metadata.
- Direct Python package import.
- CLI startup and help output.
- SQLite memory ingestion and BM25 retrieval.

Ruff is configured in `pyproject.toml`. The recovered project has outstanding lint findings; zero lint findings are not claimed.

## 7. Evaluation frameworks

The repository contains evaluation code for several retrieval and temporal experiments:

- Retrieval baseline
- Retrieval challenge v2
- Retrieval temporal v1
- Temporal assessment v1
- Adaptive selection evaluation v1
- LongMemEval-S retrieval benchmark

These are evaluation implementations, not proof that all experiments have completed successfully.

### LongMemEval-S

The LongMemEval runner is designed to evaluate session-level retrieval quality.

Its documented retrieval methods include:

- BM25
- BGE Vector (`BAAI/bge-small-en-v1.5`)
- Hybrid Reciprocal Rank Fusion

Its documented metrics include:

- Recall@1/3/5/10
- Hit@1/3/5/10
- Mean Reciprocal Rank (MRR)

The benchmark separates original-label evaluation from temporal evaluation.

It measures retrieval quality, not answer-generation accuracy.

**Important:** The official cleaned LongMemEval-S dataset is not included in this recovered package. A full LongMemEval run has not been verified. No LongMemEval scores are claimed here.

The benchmark's own documentation is available at `benchmarks/longmemeval/README.md`.

## 8. Project structure

```text
MemoRoute/
|
+-- src/
|   +-- memoroute/
|       +-- memory/
|       +-- retrieval/
|       +-- evaluation/
|       +-- cli.py
|       +-- engine.py
|       +-- __main__.py
|
+-- benchmarks/
|   +-- adaptive_selection_eval_v1/
|   +-- longmemeval/
|   +-- retrieval_baseline/
|   +-- retrieval_challenge_v2/
|   +-- retrieval_temporal_v1/
|   +-- temporal_assessment_v1/
|
+-- tests/
|   +-- unit/
|   +-- integration/
|
+-- demo/
+-- pyproject.toml
+-- README.md
```

## 9. Recovery and provenance notice

This project directory is a reconstructed recovery snapshot assembled from saved conversation exports and source captures after the original GitHub Codespaces working tree became unavailable.

It is **not proven byte-identical or complete** relative to that original working tree.

The recovery manifests record:
- Source-capture provenance for 55 files.
- 17 packaged files without individual source-capture mappings.
- Ten source-capture-mapped files with packaged byte sizes different from their capture records.
- File hashes and sizes for the recorded package payload.

Some files were corrected during recovery. The manifests preserve the available provenance and package-integrity information.

Detailed recovery manifests and package-integrity records are retained separately and are not included in this public release.

Historical recovery validation and current Acer-local verification should be treated as separate evidence.

## 10. Known limitations

- MemoRoute does not generate final natural-language answers.
- The CLI currently exposes BM25 and recent retrieval, not every retrieval module.
- Token counts are estimates, not model-tokenizer measurements.
- Relation-aware selection does not prove which conflicting claim is true.
- Missing relation endpoints are not automatically retrieved.
- The official LongMemEval-S dataset is not included in the recovered package.
- Full LongMemEval evaluation results are not available here.
- External embedding models require separate setup.
- The recovered snapshot is not proven identical to the original Codespaces repository.
- Ruff lint and formatting are not claimed to be clean.

## 11. Research direction

MemoRoute explores the design of memory retrieval systems that account for:

- Temporal validity
- Contradictory information
- Directed correction and supersession relations
- Query intent
- Adaptive context selection
- Limited context budgets

The broader research question is how an AI system can select useful, time-appropriate and traceable evidence from a growing memory store before passing that evidence to a language model.

## 12. Project status

MemoRoute is an experimental retrieval and context-selection project.

Its current value is in the modular retrieval pipeline, temporal and relation-aware selection logic, context packet construction, and the associated test and evaluation infrastructure.

The project should be evaluated on the basis of implemented behaviour and reproducible evidence, not unsupported claims of production readiness or benchmark performance.
