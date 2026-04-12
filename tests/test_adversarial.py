"""
Adversarial / production-hardening tests.

Covers:
- Mutation / code-drift: change tool impl after recording → replay still returns recorded data
- Concurrency: two parallel record() sessions must not bleed into each other
- ReplayValidator: exact event-sequence validation + mismatch raises
- Stub client exhaustion: calling stub more times than recorded raises cleanly
- ToolMocker exhaustion and assert_exhausted
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from llmreplay.core.context import record
from llmreplay.core.event import EventKind
from llmreplay.core.replay import replay, ReplayValidator, ReplayMismatchError, ReplayStubClient
from llmreplay.core.store import EventStore
from llmreplay.mocking.tools import record_tool, ToolMocker


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Mutation / code-drift test
# ═══════════════════════════════════════════════════════════════════════════════

def test_code_drift_tool_output_unchanged(tmp_dir):
    """
    Record with tool_v1. Then swap in tool_v2 (different logic).
    Replay must still return the *recorded* output from v1 — not call v2.
    This is the key guarantee: replay is immutable w.r.t. tool implementation.
    """
    # --- v1: returns correct price ---
    def fetch_price_v1(ticker: str) -> dict:
        return {"price": 2850.50, "source": "v1"}

    with record("drift_test", base_dir=tmp_dir, seed=42) as session:
        session.record_tool_call("fetch_price", {"ticker": "RELIANCE"})
        result_v1 = fetch_price_v1("RELIANCE")
        session.record_tool_result("fetch_price", result_v1)

    # --- v2: broken implementation (would return wrong data) ---
    def fetch_price_v2(ticker: str) -> dict:
        return {"price": 0.0, "source": "BROKEN_v2"}

    store  = EventStore("drift_test", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="fetch_price")
    def fetch_price_v2_mocked(ticker: str) -> dict:
        return fetch_price_v2(ticker)    # this body never runs

    replayed = fetch_price_v2_mocked("RELIANCE")

    assert replayed == {"price": 2850.50, "source": "v1"}, (
        "Replay must return recorded v1 output, not the new v2 logic"
    )


def test_code_drift_llm_response_unchanged(tmp_dir):
    """LLM responses from recording must survive even if prompts change in replay code."""
    from collections import deque

    recorded_raw = {"choices": [{"message": {"content": "original answer"}}]}

    with record("llm_drift", base_dir=tmp_dir, seed=1) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [{"role": "user", "content": "original prompt"}])
        session.record_llm_response(recorded_raw, 0.001)

    # Replay — stub returns the *recorded* response regardless of what prompt we'd send
    rs     = replay("llm_drift", base_dir=tmp_dir)
    stubbed = rs.stub_client.create(model="gpt-4o", messages=[{"role": "user", "content": "DIFFERENT prompt"}])

    assert stubbed["choices"][0]["message"]["content"] == "original answer"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Concurrency — session isolation via contextvars
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_concurrent_sessions_isolated(tmp_dir):
    """
    Two async tasks run record() simultaneously.
    Events from task A must NOT appear in task B's store and vice versa.
    """
    async def agent_a():
        with record("concurrent_a", base_dir=tmp_dir, seed=10) as session:
            await asyncio.sleep(0.01)   # yield — lets task B start
            session.record_tool_call("tool_a", {"data": "A"})
            await asyncio.sleep(0.01)
            session.record_tool_result("tool_a", {"out": "A"})

    async def agent_b():
        await asyncio.sleep(0.005)     # start slightly after A
        with record("concurrent_b", base_dir=tmp_dir, seed=20) as session:
            session.record_tool_call("tool_b", {"data": "B"})
            await asyncio.sleep(0.02)
            session.record_tool_result("tool_b", {"out": "B"})

    await asyncio.gather(agent_a(), agent_b())

    store_a = EventStore("concurrent_a", tmp_dir, read_only=True)
    store_b = EventStore("concurrent_b", tmp_dir, read_only=True)

    events_a = list(store_a.iter_from())
    events_b = list(store_b.iter_from())

    # A must only have tool_a events
    tool_calls_a = [e for e in events_a if e.kind == EventKind.TOOL_CALL]
    assert all(e.payload["name"] == "tool_a" for e in tool_calls_a), (
        "Session A contaminated by session B's events"
    )

    # B must only have tool_b events
    tool_calls_b = [e for e in events_b if e.kind == EventKind.TOOL_CALL]
    assert all(e.payload["name"] == "tool_b" for e in tool_calls_b), (
        "Session B contaminated by session A's events"
    )

    # Both stores must have events (not empty / merged)
    assert len(events_a) > 0
    assert len(events_b) > 0


@pytest.mark.asyncio
async def test_parallel_llm_calls_ordered(tmp_dir):
    """
    Multiple LLM calls fired sequentially inside one async session must be
    recorded in call order — no interleaving.
    """
    with record("parallel_order", base_dir=tmp_dir, seed=99) as session:
        for i in range(5):
            session.record_llm_request("openai", "gpt-4o", {}, [{"role": "user", "content": f"q{i}"}])
            await asyncio.sleep(0)   # yield between calls
            session.record_llm_response({"idx": i}, 0.001)

    store  = EventStore("parallel_order", tmp_dir, read_only=True)
    reqs   = [e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST]
    steps  = [e.step for e in reqs]
    assert steps == sorted(steps), "LLM_REQUEST steps are not monotonically increasing"
    contents = [e.payload["messages"][0]["content"] for e in reqs]
    assert contents == [f"q{i}" for i in range(5)], "Request order corrupted"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ReplayValidator — strict event-sequence assertion
# ═══════════════════════════════════════════════════════════════════════════════

def test_replay_validator_happy_path(tmp_dir):
    """Validator passes when actual sequence matches expected."""
    with record("validator_ok", base_dir=tmp_dir, seed=0) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({}, 0.001)
        session.record_tool_call("search", {"q": "test"})
        session.record_tool_result("search", {"r": []})

    store     = EventStore("validator_ok", tmp_dir, read_only=True)
    validator = ReplayValidator(store, start=2)   # skip seed + metadata

    ev = validator.expect(EventKind.LLM_REQUEST,  provider="openai", model="gpt-4o")
    assert ev.kind == EventKind.LLM_REQUEST

    validator.expect(EventKind.LLM_RESPONSE)
    validator.expect(EventKind.TOOL_CALL,   name="search")
    validator.expect(EventKind.TOOL_RESULT, name="search")
    validator.assert_done()   # must not raise


def test_replay_validator_wrong_kind_raises(tmp_dir):
    """Validator raises ReplayMismatchError when event kind doesn't match."""
    with record("validator_wrong_kind", base_dir=tmp_dir, seed=0) as session:
        session.record_tool_call("search", {"q": "test"})

    store     = EventStore("validator_wrong_kind", tmp_dir, read_only=True)
    validator = ReplayValidator(store, start=2)

    with pytest.raises(ReplayMismatchError, match="expected kind=llm_request"):
        validator.expect(EventKind.LLM_REQUEST)   # actual is tool_call


def test_replay_validator_wrong_payload_raises(tmp_dir):
    """Validator raises when payload subset doesn't match."""
    with record("validator_wrong_payload", base_dir=tmp_dir, seed=0) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])

    store     = EventStore("validator_wrong_payload", tmp_dir, read_only=True)
    validator = ReplayValidator(store, start=2)

    with pytest.raises(ReplayMismatchError, match="model"):
        validator.expect(EventKind.LLM_REQUEST, model="gpt-3.5-turbo")  # recorded gpt-4o


def test_replay_validator_extra_events_raises(tmp_dir):
    """assert_done raises if there are unmatched events remaining."""
    with record("validator_extra", base_dir=tmp_dir, seed=0) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({}, 0.001)

    store     = EventStore("validator_extra", tmp_dir, read_only=True)
    validator = ReplayValidator(store, start=2)
    validator.expect(EventKind.LLM_REQUEST)
    # Don't consume LLM_RESPONSE — assert_done must catch it
    with pytest.raises(ReplayMismatchError, match="unmatched"):
        validator.assert_done()


def test_replay_validator_trace_exhausted_raises(tmp_dir):
    """Expecting more events than exist raises with a clear message."""
    with record("validator_exhaust", base_dir=tmp_dir, seed=0) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])

    store     = EventStore("validator_exhaust", tmp_dir, read_only=True)
    validator = ReplayValidator(store, start=2)
    validator.expect(EventKind.LLM_REQUEST)
    with pytest.raises(ReplayMismatchError, match="trace ended"):
        validator.expect(EventKind.LLM_RESPONSE)   # nothing here


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Stub client exhaustion
# ═══════════════════════════════════════════════════════════════════════════════

def test_stub_client_exhaustion_raises(tmp_dir):
    """ReplayStubClient raises RuntimeError if called more times than recorded."""
    with record("stub_exhaust", base_dir=tmp_dir, seed=0) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({"answer": 1}, 0.001)

    rs = replay("stub_exhaust", base_dir=tmp_dir)
    rs.stub_client.create(model="gpt-4o", messages=[])   # consumes the one response

    with pytest.raises(RuntimeError, match="exhausted"):
        rs.stub_client.create(model="gpt-4o", messages=[])   # second call → boom


@pytest.mark.asyncio
async def test_stub_client_async_exhaustion_raises(tmp_dir):
    """Async stub also raises on exhaustion."""
    with record("stub_async_exhaust", base_dir=tmp_dir, seed=0) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({"answer": 1}, 0.001)

    rs = replay("stub_async_exhaust", base_dir=tmp_dir)
    await rs.stub_client.acreate(model="gpt-4o", messages=[])

    with pytest.raises(RuntimeError, match="exhausted"):
        await rs.stub_client.acreate(model="gpt-4o", messages=[])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ToolMocker exhaustion + assert_exhausted
# ═══════════════════════════════════════════════════════════════════════════════

def test_tool_mocker_exhaustion_raises(tmp_dir):
    """ToolMocker raises RuntimeError when queue is exhausted."""
    with record("mocker_exhaust", base_dir=tmp_dir, seed=0) as session:
        session.record_tool_call("ping", {})
        session.record_tool_result("ping", {"ok": True})

    store  = EventStore("mocker_exhaust", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="ping")
    def ping(): ...

    ping()   # consumes the one result

    with pytest.raises(RuntimeError, match="no recorded result"):
        ping()   # second call → empty queue


def test_tool_mocker_assert_exhausted_passes(tmp_dir):
    """assert_exhausted passes when all recorded results are consumed."""
    with record("mocker_consumed", base_dir=tmp_dir, seed=0) as session:
        session.record_tool_call("echo", {"v": 1})
        session.record_tool_result("echo", 1)
        session.record_tool_call("echo", {"v": 2})
        session.record_tool_result("echo", 2)

    store  = EventStore("mocker_consumed", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="echo")
    def echo(v): ...

    echo(1)
    echo(2)
    mocker.assert_exhausted("echo")   # must not raise


def test_tool_mocker_assert_exhausted_fails_when_leftovers(tmp_dir):
    """assert_exhausted raises AssertionError when queue has unconsumed results."""
    with record("mocker_leftover", base_dir=tmp_dir, seed=0) as session:
        session.record_tool_call("echo", {"v": 1})
        session.record_tool_result("echo", 1)
        session.record_tool_call("echo", {"v": 2})
        session.record_tool_result("echo", 2)

    store  = EventStore("mocker_leftover", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="echo")
    def echo(v): ...

    echo(1)   # consume only one — leftover 2 remains

    with pytest.raises(AssertionError, match="unconsumed"):
        mocker.assert_exhausted("echo")


@pytest.mark.asyncio
async def test_tool_mocker_async_mock(tmp_dir):
    """ToolMocker.mock works transparently for async tool functions."""
    with record("mocker_async", base_dir=tmp_dir, seed=0) as session:
        session.record_tool_call("async_fetch", {"url": "https://example.com"})
        session.record_tool_result("async_fetch", {"status": 200, "body": "hello"})

    store  = EventStore("mocker_async", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="async_fetch")
    async def async_fetch(url: str) -> dict:
        raise RuntimeError("should never run in replay")

    result = await async_fetch("https://anything.com")
    assert result == {"status": 200, "body": "hello"}
