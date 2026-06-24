"""Create database tables (idempotent).

Usage:
    python scripts/init_db.py

Uses DATABASE_URL from the environment if set, otherwise the local SQLite
fallback (app.db). Safe to run repeatedly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ssd import config, db  # noqa: E402


def main() -> None:
    url = config.database_url()
    masked = url.split("@")[-1] if "@" in url else url
    print(f"Initialising database -> {masked}")
    db.init_db()
    print("Done. Tables created (or already present).")


if __name__ == "__main__":
    main()
