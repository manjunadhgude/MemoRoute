
import json

from memoroute.cli import main


def test_cli_add_then_query_uses_real_sqlite(tmp_path, capsys):
    database = tmp_path / "cli-demo.sqlite3"

    exit_code = main([
        "add",
        "--db", str(database),
        "--content", "The production cache uses Redis version 7.",
        "--source-kind", "integration-test",
    ])
    assert exit_code == 0
    added = json.loads(capsys.readouterr().out)
    assert added["status"] == "saved"
    assert database.exists()

    exit_code = main([
        "query",
        "--db", str(database),
        "--query", "Which Redis version does production use?",
        "--retriever", "bm25",
        "--mode", "unspecified",
        "--limit", "5",
        "--token-budget", "100",
    ])
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)

    assert result["retrieved_count"] == 1
    assert result["retrieved_candidates"][0]["content"] == (
        "The production cache uses Redis version 7."
    )
    assert len(result["context_packet"]["items"]) == 1
    assert result["context_packet"]["estimator"]

def test_cli_recent_query_and_current_mode(tmp_path, capsys):
    database = tmp_path / "recent-demo.sqlite3"

    main([
        "add", "--db", str(database),
        "--content", "The service uses PostgreSQL.",
    ])
    capsys.readouterr()

    exit_code = main([
        "query",
        "--db", str(database),
        "--query", "Show recent memories.",
        "--retriever", "recent",
        "--mode", "current",
    ])
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)

    assert result["intent"]["mode"] == "current"
    assert result["intent"]["as_of"] is not None
    assert result["retrieved_count"] == 1

def test_cli_as_of_requires_timestamp(tmp_path):
    import pytest

    with pytest.raises(SystemExit) as exc:
        main([
            "query",
            "--db", str(tmp_path / "empty.sqlite3"),
            "--query", "What was true?",
            "--mode", "as_of",
        ])
    assert exc.value.code == 2
