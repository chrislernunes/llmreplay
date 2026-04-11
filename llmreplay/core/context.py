"""record() context manager — instruments the runtime and writes an event log."""
from __future__ import annotations

import contextvars
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np  # optional; guarded below

from .event import Event, EventKind
from .store import EventStore


# ── active session (thread + async safe) ──────────────────────────────────────
_session_var: contextvars.ContextVar["RecordSession | None"] = contextvars.ContextVar(
    "_session_var", default=None
)


def current_session() -> "RecordSession | None":
    return _session_var.get()


class RecordSession:
    def __init__(self, run_id: str, store: EventStore, seed: int):
        self.run_id  = run_id
        self.store   = store
        self.seed    = seed
        self._step   = 0
        self._lock   = threading.Lock()
        self.metadata: dict[str, Any] = {}

    # ── step counter ──────────────────────────────────────────────────────────
    def next_step(self) -> int:
        with self._lock:
            step = self._step
            self._step += 1
            return step

    # ── generic emit ──────────────────────────────────────────────────────────
    def emit(self, kind: EventKind, payload: dict) -> Event:
        ev = Event(run_id=self.run_id, step=self.next_step(), kind=kind, payload=payload)
        self.store.append(ev)
        return ev

    # ── convenience helpers (called by hooks) ─────────────────────────────────
    def record_llm_request(self, provider: str, model: str, params: dict, messages: list) -> Event:
        return self.emit(EventKind.LLM_REQUEST, {
            "provider": provider, "model": model,
            "params": params, "messages": messages,
        })

    def record_llm_response(self, raw: dict, cost_usd: float = 0.0) -> Event:
        return self.emit(EventKind.LLM_RESPONSE, {"raw": raw, "cost_usd": cost_usd})

    def record_tool_call(self, name: str, inputs: Any) -> Event:
        return self.emit(EventKind.TOOL_CALL, {"name": name, "inputs": inputs})

    def record_tool_result(self, name: str, result: Any) -> Event:
        return self.emit(EventKind.TOOL_RESULT, {"name": name, "result": result})

    def record_memory_read(self, query: str, results: list) -> Event:
        return self.emit(EventKind.MEMORY_READ, {"query": query, "results": results})

    def record_memory_write(self, key: str, value: Any) -> Event:
        return self.emit(EventKind.MEMORY_WRITE, {"key": key, "value": value})

    def record_exception(self, exc_type: str, message: str) -> Event:
        return self.emit(EventKind.EXCEPTION, {"exc_type": exc_type, "message": message})


# ── public API ────────────────────────────────────────────────────────────────

@contextmanager
def record(
    run_id: str,
    *,
    seed: int | None = None,
    base_dir: Path | None = None,
    overwrite: bool = False,
    metadata: dict | None = None,
) -> Generator[RecordSession, None, None]:
    """
    Record every LLM call, tool use, and stochastic event to an immutable log.

    Usage::

        async with record("my_run") as session:
            result = await my_agent.run(query)
    """
    from llmreplay.hooks import install_all, uninstall_all

    store = EventStore(run_id, base_dir)
    if overwrite:
        store.delete()
        store = EventStore(run_id, base_dir)

    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2 ** 31)

    session = RecordSession(run_id=run_id, store=store, seed=actual_seed)
    session.metadata = metadata or {}

    # Persist seed + metadata as step-0 event
    session.emit(EventKind.RANDOM_SEED, {"seed": actual_seed})
    session.emit(EventKind.METADATA, {"metadata": session.metadata})

    # Seed RNGs
    random.seed(actual_seed)
    try:
        np.random.seed(actual_seed)
    except Exception:  # pragma: no cover
        pass

    token = _session_var.set(session)
    install_all()
    try:
        yield session
    except Exception as exc:
        session.record_exception(type(exc).__name__, str(exc))
        raise
    finally:
        uninstall_all()
        _session_var.reset(token)
