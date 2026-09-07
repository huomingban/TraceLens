"""Print a safe, non-destructive database connectivity and schema check."""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Allow both `python scripts/check_database.py` and `python -m scripts.check_database`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracelens.storage import engine


def main() -> int:
    # Never print DATABASE_URL because it may contain a database password.
    print(f"dialect: {engine.dialect.name}")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            tables = inspect(connection).get_table_names()
        print("connection: OK")
        print(f"tables: {', '.join(sorted(tables)) or '(none)'}")
        if "alembic_version" not in tables:
            print("migration: missing (run python -m alembic upgrade head)")
            return 2
        print("migration: version table found")
        return 0
    except SQLAlchemyError as error:
        print(f"connection: FAILED ({error.__class__.__name__})")
        print("Check that DATABASE_URL, database name, username, and password are correct.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
