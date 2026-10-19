"""Tests for migration discovery and the connection helper.

Applying migrations needs Postgres, which CI does not have. What can be tested
without one is the part that decides *what* to apply and in what order — which
is the part that would silently corrupt a schema if it got it wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fraud_mlops.db import MIGRATIONS_DIR, connect, discover_migrations


def _write(directory: Path, name: str, body: str = "SELECT 1;") -> Path:
    path = directory / name
    path.write_text(body)
    return path


# --- Discovery and ordering ---


def test_migrations_are_returned_in_numeric_order(tmp_path: Path) -> None:
    """Lexical order on a zero-padded prefix is numeric order.

    Written out as a test because an unpadded `10_...` would sort before
    `2_...`, and migrations applied out of order fail in ways that are hard to
    unpick afterwards.
    """
    for name in ("003_third.sql", "001_first.sql", "010_tenth.sql", "002_second.sql"):
        _write(tmp_path, name)

    versions = [m.version for m in discover_migrations(tmp_path)]
    assert versions == ["001", "002", "003", "010"]


def test_badly_named_migrations_are_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "001_fine.sql")
    _write(tmp_path, "add_labels.sql")  # no numeric prefix

    with pytest.raises(ValueError, match="does not match"):
        discover_migrations(tmp_path)


def test_duplicate_versions_are_rejected(tmp_path: Path) -> None:
    """Two people adding `004_` on separate branches is the realistic case."""
    _write(tmp_path, "004_labels.sql")
    _write(tmp_path, "004_latency.sql")

    with pytest.raises(ValueError, match="Duplicate migration versions"):
        discover_migrations(tmp_path)


def test_missing_directory_yields_no_migrations(tmp_path: Path) -> None:
    assert discover_migrations(tmp_path / "nope") == []


def test_migration_exposes_its_sql(tmp_path: Path) -> None:
    _write(tmp_path, "001_thing.sql", "CREATE TABLE thing (id INT);")
    migration = discover_migrations(tmp_path)[0]

    assert migration.version == "001"
    assert migration.name == "001_thing"
    assert "CREATE TABLE thing" in migration.sql


# --- The real migration directory ---


def test_repo_migrations_are_discoverable_and_ordered() -> None:
    migrations = discover_migrations()

    assert [m.name for m in migrations] == [
        "001_decisions",
        "002_labels",
        "003_decision_latency",
    ]
    assert MIGRATIONS_DIR.exists()


def test_repo_migrations_are_rerunnable() -> None:
    """Every statement must tolerate being applied against an existing schema.

    The `decisions` table already exists in RDS — it was created by hand long
    before there was any DDL for it — so migration 001 has to be a no-op there
    rather than an error.
    """
    for migration in discover_migrations():
        sql = migration.sql.upper()
        for statement, guard in (
            ("CREATE TABLE", "IF NOT EXISTS"),
            ("CREATE INDEX", "IF NOT EXISTS"),
            ("ADD COLUMN", "IF NOT EXISTS"),
        ):
            if statement in sql:
                assert guard in sql, f"{migration.name}: {statement} without {guard}"


def test_the_tables_the_rest_of_the_code_depends_on_exist_in_migrations() -> None:
    """A cheap contract check between the SQL files and the code that reads them."""
    combined = " ".join(m.sql.lower() for m in discover_migrations())

    # `schema_migrations` is created by ensure_migrations_table(), not by a
    # migration file — it has to exist before the first one can be recorded.
    for table in ("decisions", "labels", "ground_truth"):
        assert table in combined, f"no migration mentions {table}"

    # The columns the performance flow and the latency scripts select by name.
    for column in ("label_available_at", "ingested_at", "latency_ms", "scored_at"):
        assert column in combined, f"no migration defines {column}"


# --- Connection helper ---


def test_connect_reports_missing_settings_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear error beats psycopg2's 'could not translate host name "None"'."""
    for key in ("RDS_HOST", "RDS_USER", "RDS_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="Missing database settings"):
        connect()


def test_connect_names_every_missing_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDS_HOST", "localhost")
    monkeypatch.delenv("RDS_USER", raising=False)
    monkeypatch.delenv("RDS_PASSWORD", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        connect()

    message = str(excinfo.value)
    assert "user" in message
    assert "password" in message
    assert "host" not in message.split("Missing database settings:")[1].split("]")[0]
