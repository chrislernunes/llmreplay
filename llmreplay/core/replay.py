"""Replay engine — reads a recorded trace with zero network calls."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .event import Event, EventKind
from .store import EventStore


class ReplayStubClient:
    """
    A drop-in response queue for replay.

    During replay, instead of calling a real LLM API, consume the
    pre-recorded responses in order::

        session = replay("my_run")
        response = session.stub_client.create(model="gpt-4o", messages=[...])

    The ``model`` and ``messages`` arguments are accepted but ignored —
    the next recorded response is returned regardless.
    """

    def __init__(self, response_queue: deque[dict]):
        self._q = response_queue

    def create(self, **kwargs) -> dict:
        if not self._q:
            raise RuntimeError(
                "ReplayStubClient: response queue exhausted — "
                "called more times than were recorded."
            )
        return self._q.popleft()

    async def acreate(self, **kwargs) -> dict:
        """Async mirror of create() for use with async agent code."""
        return self.create(**kwargs)


def _build_response_queue(store: EventStore, start: int) -> deque[dict]:
    q: deque[dict] = deque()
    for ev in store.iter_from(start):
        if ev.kind == EventKind.LLM_RESPONSE:
            raw = ev.payload.get("raw", {})
            q.append(raw)
    return q


@dataclass
class ReplaySession:
    """
    A read-only view over a recorded run.

    Attributes
    ----------
    stub_client:
        A ``ReplayStubClient`` pre-loaded with the recorded LLM responses.
        Use it as a drop-in replacement for your LLM client in replay code
        so no network calls are made.

    Typical usage::

        session = replay("my_run")

        # Inspect the trace
        for ev in session.events():
            print(ev.step, ev.kind.value, ev.payload)

        # Re-run agent logic without network calls
        response = session.stub_client.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "..."}],
        )
    """
    run_id:       str
    store:        EventStore
    start:        int
    overrides:    dict[str, Any]       = field(default_factory=dict)
    stub_client:  ReplayStubClient     = field(init=False)

    def __post_init__(self) -> None:
        self.stub_client = ReplayStubClient(_build_response_queue(self.store, self.start))

    def events(self) -> Iterator[Event]:
        """Iterate every recorded event from ``start``."""
        return self.store.iter_from(self.start)

    def total_steps(self) -> int:
        return self.store.count()

    def cost_by_step(self) -> dict[int, float]:
        costs = {}
        for ev in self.store.iter_from():
            if ev.kind == EventKind.LLM_RESPONSE:
                costs[ev.step] = ev.payload.get("cost_usd", 0.0)
        return costs

    def total_cost(self) -> float:
        return sum(self.cost_by_step().values())


def replay(
    run_id: str,
    *,
    step: int = 0,
    base_dir: Path | None = None,
    overrides: dict | None = None,
) -> ReplaySession:
    """
    Load a recorded run and return a ReplaySession.

    ``overrides`` is stored on the session for caller-controlled
    parameterisation. It does not affect the stored event log.
    """
    store = EventStore(run_id, base_dir, read_only=True)
    if store.count() == 0:  # pragma: no cover
        raise FileNotFoundError(f"No recorded run found: {run_id}")

    # Restore the original RNG seed for determinism
    seed_event = store.get(0)
    if seed_event and seed_event.kind == EventKind.RANDOM_SEED:
        random.seed(seed_event.payload["seed"])
        try:
            import numpy as np  # noqa: PLC0415
            np.random.seed(seed_event.payload["seed"])
        except ImportError:  # pragma: no cover
            pass

    return ReplaySession(
        run_id=run_id,
        store=store,
        start=step,
        overrides=overrides or {},
    )


class ReplayMismatchError(AssertionError):
    """Raised when a replayed event sequence doesn't match expectations."""


class ReplayValidator:
    """
    Strict event-sequence validator for recorded traces.

    Usage::

        store = EventStore(run_id, base_dir, read_only=True)
        v = ReplayValidator(store, start=2)   # skip seed + metadata events

        v.expect(EventKind.LLM_REQUEST, provider="openai", model="gpt-4o")
        v.expect(EventKind.LLM_RESPONSE)
        v.expect(EventKind.TOOL_CALL, name="search")
        v.expect(EventKind.TOOL_RESULT, name="search")
        v.assert_done()   # fails if any events remain
    """

    def __init__(self, store: EventStore, start: int = 0):
        self._events = list(store.iter_from(start))
        self._pos = 0

    def expect(self, kind: EventKind, **payload_subset) -> Event:
        if self._pos >= len(self._events):
            raise ReplayMismatchError(
                f"trace ended: expected kind={kind.value} but no more events"
            )
        ev = self._events[self._pos]
        if ev.kind != kind:
            raise ReplayMismatchError(
                f"step {ev.step}: expected kind={kind.value}, got kind={ev.kind.value}"
            )
        for key, expected in payload_subset.items():
            actual = ev.payload.get(key)
            if actual != expected:
                raise ReplayMismatchError(
                    f"step {ev.step}: payload[{key!r}] expected {expected!r}, got {actual!r}"
                )
        self._pos += 1
        return ev

    def assert_done(self) -> None:
        remaining = self._events[self._pos:]
        if remaining:
            kinds = [e.kind.value for e in remaining]
            raise ReplayMismatchError(
                f"{len(remaining)} unmatched event(s) remaining in trace: {kinds}"
            )


def fork(
    run_id: str,
    new_run_id: str,
    *,
    at_step: int,
    base_dir: Path | None = None,
) -> EventStore:
    """
    Copy a run's event log up to ``at_step`` into a new run_id, then write
    a BRANCH event marking the divergence point.

    The new store can be extended with different agent logic — counterfactual
    debugging without modifying the original trace.
    """
    src = EventStore(run_id, base_dir, read_only=True)
    dst = EventStore(new_run_id, base_dir)
    for ev in src.iter_from():
        if ev.step >= at_step:
            break
        dst.append(Event(
            run_id=new_run_id,
            step=ev.step,
            kind=ev.kind,
            payload=ev.payload,
            ts=ev.ts,
        ))
    # Mark the divergence point so the new trace is self-documenting
    dst.append(Event(
        run_id=new_run_id,
        step=at_step,
        kind=EventKind.BRANCH,
        payload={"forked_from": run_id, "at_step": at_step},
    ))
    return dst
