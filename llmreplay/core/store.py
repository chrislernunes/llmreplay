"""Append-only event log. SQLite by default; JSONL or S3 via backends."""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator

from .event import Event


_DEFAULT_DIR = Path(os.environ.get("LLMREPLAY_DIR", "~/.llmreplay")).expanduser()


def _db_path(run_id: str, base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{run_id}.db"


@contextmanager
def _conn(path: Path) -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        yield con
    finally:
        con.close()


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            step    INTEGER PRIMARY KEY,
            kind    TEXT    NOT NULL,
            ts      REAL    NOT NULL,
            payload TEXT    NOT NULL
        )
    """)
    con.commit()


class EventStore:
    """Thread-safe, append-only store for a single run."""

    def __init__(self, run_id: str, base_dir: Path | None = None, read_only: bool = False):
        self.run_id   = run_id
        self.base_dir = base_dir or _DEFAULT_DIR
        self.path     = _db_path(run_id, self.base_dir)
        self._read_only = read_only

        if read_only and not self.path.exists():
            raise FileNotFoundError(f"No recorded run found: {run_id}")
        if not read_only:
            with _conn(self.path) as con:
                _ensure_schema(con)

    # ------------------------------------------------------------------ write

    def append(self, event: Event) -> None:
        if self._read_only:
            raise RuntimeError("Store opened read-only")
        with _conn(self.path) as con:
            con.execute(
                "INSERT INTO events(step, kind, ts, payload) VALUES(?,?,?,?)",
                (event.step, event.kind.value, event.ts, json.dumps(event.payload)),
            )
            con.commit()

    # ------------------------------------------------------------------ read

    def get(self, step: int) -> Event | None:
        with _conn(self.path) as con:
            row = con.execute(
                "SELECT step,kind,ts,payload FROM events WHERE step=?", (step,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def iter_from(self, start: int = 0) -> Iterator[Event]:
        with _conn(self.path) as con:
            rows = con.execute(
                "SELECT step,kind,ts,payload FROM events WHERE step>=? ORDER BY step",
                (start,),
            ).fetchall()
        for row in rows:
            yield self._row_to_event(row)

    def count(self) -> int:
        with _conn(self.path) as con:
            return con.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def all_run_ids(self) -> list[str]:
        return [p.stem for p in self.base_dir.glob("*.db")]

    # ------------------------------------------------------------------ export

    def export_jsonl(self, dest: Path, compress: bool = True) -> Path:
        dest = Path(str(dest) + ".gz") if compress else dest
        opener = gzip.open if compress else open
        with opener(dest, "wt", encoding="utf-8") as fh:  # type: ignore[arg-type]
            for ev in self.iter_from():
                fh.write(json.dumps(ev.to_dict()) + "\n")
        return dest

    # ------------------------------------------------------------------ utils

    def _row_to_event(self, row: tuple) -> Event:
        step, kind_str, ts, payload_json = row
        from .event import EventKind
        return Event(
            run_id=self.run_id,
            step=step,
            kind=EventKind(kind_str),
            payload=json.loads(payload_json),
            ts=ts,
        )

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
