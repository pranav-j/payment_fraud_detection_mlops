"""Postgres connection and schema migrations.

Before this module, the `decisions` table existed only as a shape implied by
three `INSERT` statements in three different files. There was no DDL anywhere in
the repo: the table had been created by hand against RDS, so a fresh environment
could not reproduce it, and nobody could say what its columns were without
reading the inserts.

So this module does two things:

- **One connection helper.** `connect()` replaces the four hand-rolled
  `psycopg2.connect(...)` blocks that had drifted apart (different defaults,
  different SSL settings).
- **Migrations.** Plain numbered `.sql` files under `migrations/`, applied in
  order and recorded in a `schema_migrations` table. No Alembic: there is no ORM
  here and nothing to autogenerate from, so a directory of SQL that anyone can
  read and `psql -f` by hand is the honest tool.

Usage:

    python -m fraud_mlops.db status     # what's applied, what's pending
    python -m fraud_mlops.db migrate    # apply everything pending
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraud_mlops.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MIGRATIONS_DIR: Path = PROJECT_ROOT / "migrations"

# Filenames must start with a zero-padded number so lexical order is apply order.
MIGRATION_PATTERN = re.compile(r"^(\d{3,})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class Migration:
    """One `.sql` file on disk."""

    version: str
    name: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text()


def connect(**overrides: Any) -> Any:
    """Open a Postgres connection from the environment.

    Reads `RDS_HOST`, `RDS_PORT`, `RDS_DB`, `RDS_USER`, `RDS_PASSWORD`. Keyword
    arguments override any of them, which is how tests point at a local
    instance.

    `sslmode=require` by default: RDS is reachable from outside the VPC in this
    build (a documented limitation), so the connection should at least be
    encrypted. A local Postgres with no TLS needs `sslmode="prefer"`.
    """
    import psycopg2

    params: dict[str, Any] = {
        "host": os.environ.get("RDS_HOST"),
        "port": int(os.environ.get("RDS_PORT", 5432)),
        "dbname": os.environ.get("RDS_DB", "mlflow"),
        "user": os.environ.get("RDS_USER"),
        "password": os.environ.get("RDS_PASSWORD"),
        "sslmode": "require",
    }
    params.update(overrides)

    missing = [k for k in ("host", "user", "password") if not params.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing database settings: {missing}. Set RDS_HOST/RDS_USER/RDS_PASSWORD "
            f"(see .env.example) or pass them to connect()."
        )

    return psycopg2.connect(**params)


def discover_migrations(directory: Path | None = None) -> list[Migration]:
    """Every migration file on disk, in apply order."""
    directory = directory or MIGRATIONS_DIR
    if not directory.exists():
        return []

    found: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_PATTERN.match(path.name)
        if not match:
            raise ValueError(
                f"Migration {path.name!r} does not match NNN_lower_snake_case.sql — "
                f"the numeric prefix is what defines apply order."
            )
        found.append(Migration(version=match.group(1), name=path.stem, path=path))

    versions = [m.version for m in found]
    duplicates = {v for v in versions if versions.count(v) > 1}
    if duplicates:
        raise ValueError(f"Duplicate migration versions: {sorted(duplicates)}")

    return found


def ensure_migrations_table(conn: Any) -> None:
    """Create the bookkeeping table. Safe to call repeatedly."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    conn.commit()


def applied_versions(conn: Any) -> set[str]:
    ensure_migrations_table(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def pending_migrations(conn: Any, directory: Path | None = None) -> list[Migration]:
    done = applied_versions(conn)
    return [m for m in discover_migrations(directory) if m.version not in done]


def apply_migrations(conn: Any, directory: Path | None = None) -> list[Migration]:
    """Apply every pending migration, each in its own transaction.

    Returns the migrations that were applied. Idempotent: running it twice in a
    row applies nothing the second time.
    """
    pending = pending_migrations(conn, directory)
    if not pending:
        logger.info("Schema is up to date; nothing to apply")
        return []

    for migration in pending:
        logger.info("Applying %s", migration.name)
        with conn.cursor() as cur:
            cur.execute(migration.sql)
            cur.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                (migration.version, migration.name),
            )
        conn.commit()
        logger.info("Applied %s", migration.name)

    return pending


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Database schema management")
    parser.add_argument("command", choices=["migrate", "status"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv

    load_dotenv()

    conn = connect()
    try:
        if args.command == "status":
            done = applied_versions(conn)
            for migration in discover_migrations():
                mark = "applied" if migration.version in done else "PENDING"
                print(f"  [{mark:>7}] {migration.name}")
        else:
            applied = apply_migrations(conn)
            print(f"Applied {len(applied)} migration(s).")
            for migration in applied:
                print(f"  - {migration.name}")
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
