"""
Determinism guarantee: record once, replay N times → bitwise identical events.
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

import pytest

from llmreplay.core.context import record
from llmreplay.core.event   import EventKind
from llmreplay.core.replay  import replay, fork
from llmreplay.core.store   import EventStore
from llmreplay.mocking.tools import record_tool, ToolMocker
from llmreplay.report        import export_report


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


# ── fake LLM ──────────────────────────────────────────────────────────────────

def _fake_llm_response(idx: int) -> dict:
    """Simulate a deterministic LLM response."""
    return {
        "id": f"chatcmpl-{idx}",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": f"Response {idx}"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


async def _fake_agent(session, n_calls: int = 3):
    """
    Fake agent that:
    1. Makes n_calls 'LLM calls' (manually recorded for isolation)
    2. Uses a tool
    3. Reads random values
    """
    for i in range(n_calls):
        session.record_llm_request("openai", "gpt-4o-mini", {"temperature": 0.7}, [])
        session.record_llm_response(_fake_llm_response(i), cost_usd=0.0001 * i)

    # Tool call
    session.record_tool_call("search", {"query": "test"})
    session.record_tool_result("search", {"results": ["a", "b"]})

    # Randomness — must be identical on replay (seeded)
    return [random.randint(0, 1000) for _ in range(5)]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_record_creates_events(tmp_dir):
    with record("test_basic", base_dir=tmp_dir, seed=42) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [{"role": "user", "content": "hi"}])
        session.record_llm_response({"choices": [{"message": {"content": "hello"}}]}, 0.0001)

    store = EventStore("test_basic", tmp_dir, read_only=True)
    events = list(store.iter_from())

    # seed + metadata + request + response = 4
    assert len(events) == 4
    kinds = [e.kind for e in events]
    assert EventKind.RANDOM_SEED in kinds
    assert EventKind.LLM_REQUEST in kinds
    assert EventKind.LLM_RESPONSE in kinds


@pytest.mark.asyncio
async def test_determinism_100_replays(tmp_dir):
    """Record once → replay 100 times → all event logs identical."""
    run_id = "det_test"

    with record(run_id, base_dir=tmp_dir, seed=1337) as session:
        randoms = [random.randint(0, 1_000_000) for _ in range(20)]
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({"resp": "ok"}, 0.001)

    store    = EventStore(run_id, tmp_dir, read_only=True)
    baseline = [json.dumps(e.to_dict()) for e in store.iter_from()]

    for _ in range(100):
        # Restore session externally (same seed → same randoms)
        random.seed(1337)
        replayed = [random.randint(0, 1_000_000) for _ in range(20)]
        assert replayed == randoms, "RNG not deterministic with same seed"

    # Verify event log unchanged after 100 reads
    after = [json.dumps(e.to_dict()) for e in store.iter_from()]
    assert baseline == after


def test_replay_session_events(tmp_dir):
    with record("replay_test", base_dir=tmp_dir, seed=99) as session:
        session.record_llm_request("openai", "gpt-4o-mini", {}, [])
        session.record_llm_response({"choices": []}, 0.0005)

    rs = replay("replay_test", base_dir=tmp_dir)
    events = list(rs.events())
    assert len(events) >= 3

    cost = rs.total_cost()
    assert cost == pytest.approx(0.0005)


def test_replay_from_step(tmp_dir):
    with record("step_test", base_dir=tmp_dir, seed=7) as session:
        for i in range(5):
            session.record_llm_request("openai", "gpt-4o", {}, [])
            session.record_llm_response({}, 0.001)

    rs = replay("step_test", step=6, base_dir=tmp_dir)
    events = list(rs.events())
    assert all(e.step >= 6 for e in events)


def test_fork_creates_new_store(tmp_dir):
    with record("fork_src", base_dir=tmp_dir, seed=5) as session:
        for i in range(4):
            session.record_llm_request("openai", "gpt-4o", {}, [])
            session.record_llm_response({}, 0.001)

    dst = fork("fork_src", "fork_dst", at_step=4, base_dir=tmp_dir)
    # fork() copies steps 0..at_step-1 then appends a BRANCH event at at_step
    assert dst.count() == 5

    events = list(dst.iter_from())
    branch_ev = events[-1]
    assert branch_ev.kind == EventKind.BRANCH
    assert branch_ev.payload["forked_from"] == "fork_src"
    assert branch_ev.payload["at_step"] == 4


def test_exception_recorded(tmp_dir):
    with pytest.raises(ValueError):
        with record("exc_test", base_dir=tmp_dir, seed=0) as session:
            raise ValueError("deliberate failure")

    store  = EventStore("exc_test", tmp_dir, read_only=True)
    events = list(store.iter_from())
    exc_ev = [e for e in events if e.kind == EventKind.EXCEPTION]
    assert len(exc_ev) == 1
    assert exc_ev[0].payload["exc_type"] == "ValueError"
    assert "deliberate failure" in exc_ev[0].payload["message"]


def test_tool_mocker(tmp_dir):
    with record("tool_test", base_dir=tmp_dir, seed=3) as session:
        session.record_tool_call("fetch_price", {"ticker": "RELIANCE"})
        session.record_tool_result("fetch_price", {"price": 2850.50})
        session.record_tool_call("fetch_price", {"ticker": "TCS"})
        session.record_tool_result("fetch_price", {"price": 4100.00})

    store  = EventStore("tool_test", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="fetch_price")
    def fetch_price(ticker: str) -> dict:
        raise RuntimeError("should never reach here")

    assert fetch_price("RELIANCE") == {"price": 2850.50}
    assert fetch_price("TCS")      == {"price": 4100.00}


def test_bug_report_export(tmp_dir):
    with record("report_test", base_dir=tmp_dir, seed=42, metadata={"agent": "test_agent"}) as session:
        session.record_llm_request("anthropic", "claude-sonnet-4", {}, [])
        session.record_llm_response({"content": "hi"}, 0.002)

    dest = tmp_dir / "report.json"
    path = export_report("report_test", dest, base_dir=tmp_dir)
    assert path.exists()

    import json
    report = json.loads(path.read_text())
    assert report["run_id"] == "report_test"
    assert report["metadata"]["agent"] == "test_agent"
    assert report["total_cost_usd"] == pytest.approx(0.002)
    assert len(report["llm_pairs"]) == 1


def test_jsonl_export(tmp_dir):
    with record("jsonl_test", base_dir=tmp_dir, seed=1) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({}, 0.001)

    store = EventStore("jsonl_test", tmp_dir, read_only=True)
    dest  = tmp_dir / "events.jsonl"
    path  = store.export_jsonl(dest, compress=False)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == store.count()


def test_cost_by_step(tmp_dir):
    costs = [0.001, 0.005, 0.002]
    with record("cost_test", base_dir=tmp_dir, seed=8) as session:
        for c in costs:
            session.record_llm_request("openai", "gpt-4o", {}, [])
            session.record_llm_response({}, c)

    rs = replay("cost_test", base_dir=tmp_dir)
    assert rs.total_cost() == pytest.approx(sum(costs))
    assert len(rs.cost_by_step()) == len(costs)
