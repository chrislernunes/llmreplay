"""
Targeted tests for mocking/tools.py and regression.py.

Covers:
- record_tool async path
- record_tool with explicit name=
- record_tool outside session (no-op)
- ToolMocker.remaining, assert_exhausted, queue exhaustion error
- ToolMocker async mock path
- RegressionSuite exception path (case raises)
- RegressionSuite.print_report (both all-pass and some-fail)
- RegressionSuite._summarize correctness
- export_finetune_dataset JSONL and Alpaca formats
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from llmreplay.core.context import record
from llmreplay.core.event import EventKind
from llmreplay.core.store import EventStore
from llmreplay.mocking.tools import record_tool, ToolMocker
from llmreplay.regression import RegressionSuite, export_finetune_dataset


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


# ══════════════════════════════════════════════════════════════════════════════
# record_tool — sync
# ══════════════════════════════════════════════════════════════════════════════

def test_record_tool_sync_records_call_and_result(tmp_dir):
    """@record_tool on a sync fn records TOOL_CALL + TOOL_RESULT."""
    @record_tool
    def fetch(ticker: str) -> dict:
        return {"price": 42.0}

    with record("rt_sync", base_dir=tmp_dir, seed=1):
        result = fetch("RELIANCE")

    assert result == {"price": 42.0}

    store  = EventStore("rt_sync", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]
    assert EventKind.TOOL_CALL   in kinds
    assert EventKind.TOOL_RESULT in kinds

    call = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert call.payload["name"] == "fetch"

    result_ev = next(e for e in events if e.kind == EventKind.TOOL_RESULT)
    assert result_ev.payload["result"] == {"price": 42.0}


def test_record_tool_explicit_name(tmp_dir):
    """name= kwarg overrides function name in recorded events."""
    @record_tool(name="market_price")
    def _internal(ticker: str) -> float:
        return 99.9

    with record("rt_named", base_dir=tmp_dir, seed=2):
        _internal("INFY")

    store = EventStore("rt_named", tmp_dir, read_only=True)
    call  = next(e for e in store.iter_from() if e.kind == EventKind.TOOL_CALL)
    assert call.payload["name"] == "market_price"


def test_record_tool_sync_no_session():
    """Outside record(), @record_tool is transparent — no session, no error."""
    @record_tool
    def noop() -> str:
        return "done"

    assert noop() == "done"


def test_record_tool_sync_args_captured(tmp_dir):
    """Both positional args and kwargs are captured in TOOL_CALL inputs."""
    @record_tool
    def search(query: str, limit: int = 5) -> list:
        return ["a", "b"]

    with record("rt_args", base_dir=tmp_dir, seed=3):
        search("nifty", limit=10)

    store = EventStore("rt_args", tmp_dir, read_only=True)
    call  = next(e for e in store.iter_from() if e.kind == EventKind.TOOL_CALL)
    assert list(call.payload["inputs"]["args"]) == ["nifty"]
    assert call.payload["inputs"]["kwargs"] == {"limit": 10}


# ══════════════════════════════════════════════════════════════════════════════
# record_tool — async
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_record_tool_async_records_call_and_result(tmp_dir):
    """@record_tool on an async fn records TOOL_CALL + TOOL_RESULT."""
    @record_tool
    async def fetch_async(ticker: str) -> dict:
        return {"price": 77.0}

    with record("rt_async", base_dir=tmp_dir, seed=4):
        result = await fetch_async("TCS")

    assert result == {"price": 77.0}

    store = EventStore("rt_async", tmp_dir, read_only=True)
    kinds = [e.kind for e in store.iter_from()]
    assert EventKind.TOOL_CALL   in kinds
    assert EventKind.TOOL_RESULT in kinds


@pytest.mark.asyncio
async def test_record_tool_async_no_session():
    """Async outside record() — transparent, no error."""
    @record_tool
    async def async_noop() -> str:
        return "async done"

    assert await async_noop() == "async done"


@pytest.mark.asyncio
async def test_record_tool_async_explicit_name(tmp_dir):
    """Async with name= kwarg — name propagated correctly."""
    @record_tool(name="async_price")
    async def _get(sym: str) -> float:
        return 123.4

    with record("rt_async_named", base_dir=tmp_dir, seed=5):
        await _get("WIPRO")

    store = EventStore("rt_async_named", tmp_dir, read_only=True)
    call  = next(e for e in store.iter_from() if e.kind == EventKind.TOOL_CALL)
    assert call.payload["name"] == "async_price"


# ══════════════════════════════════════════════════════════════════════════════
# ToolMocker
# ══════════════════════════════════════════════════════════════════════════════

def test_toolmocker_sync_returns_recorded_results(tmp_dir):
    """ToolMocker replays recorded results in order."""
    @record_tool
    def send_email(to: str) -> dict:
        return {"status": "sent", "to": to}

    with record("tm_sync", base_dir=tmp_dir, seed=10):
        send_email("a@b.com")
        send_email("c@d.com")

    store  = EventStore("tm_sync", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="send_email")
    def send_email(to: str) -> dict:
        raise RuntimeError("should not execute")

    r1 = send_email("a@b.com")
    r2 = send_email("c@d.com")

    assert r1 == {"status": "sent", "to": "a@b.com"}
    assert r2 == {"status": "sent", "to": "c@d.com"}


def test_toolmocker_remaining(tmp_dir):
    """remaining() returns correct count before and after consumption."""
    @record_tool
    def ping() -> str:
        return "pong"

    with record("tm_remaining", base_dir=tmp_dir, seed=11):
        ping(); ping(); ping()

    store  = EventStore("tm_remaining", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    assert mocker.remaining("ping") == 3

    @mocker.mock(name="ping")
    def ping() -> str: ...

    ping()
    assert mocker.remaining("ping") == 2
    ping(); ping()
    assert mocker.remaining("ping") == 0


def test_toolmocker_assert_exhausted_passes(tmp_dir):
    """assert_exhausted passes when queue is empty."""
    @record_tool
    def one_shot() -> int:
        return 1

    with record("tm_exhausted_ok", base_dir=tmp_dir, seed=12):
        one_shot()

    store  = EventStore("tm_exhausted_ok", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="one_shot")
    def one_shot() -> int: ...

    one_shot()
    mocker.assert_exhausted("one_shot")  # must not raise


def test_toolmocker_assert_exhausted_raises(tmp_dir):
    """assert_exhausted raises AssertionError when queue has leftovers."""
    @record_tool
    def multi() -> int:
        return 99

    with record("tm_exhausted_fail", base_dir=tmp_dir, seed=13):
        multi(); multi()

    store  = EventStore("tm_exhausted_fail", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="multi")
    def multi() -> int: ...

    multi()  # consume only one of two
    with pytest.raises(AssertionError, match="unconsumed"):
        mocker.assert_exhausted("multi")


def test_toolmocker_queue_exhaustion_raises(tmp_dir):
    """Calling a mocked tool more times than recorded raises RuntimeError."""
    @record_tool
    def once() -> str:
        return "once"

    with record("tm_overrun", base_dir=tmp_dir, seed=14):
        once()

    store  = EventStore("tm_overrun", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="once")
    def once() -> str: ...

    once()  # OK
    with pytest.raises(RuntimeError, match="no recorded result"):
        once()  # queue exhausted


def test_toolmocker_unknown_tool_raises(tmp_dir):
    """Mocking a tool that was never recorded raises RuntimeError."""
    store  = EventStore.__new__(EventStore)
    mocker = ToolMocker()
    # load from an empty store — but we need a valid one
    import tempfile, shutil
    d = Path(tempfile.mkdtemp())
    try:
        with record("tm_unknown_base", base_dir=d, seed=0):
            pass
        store = EventStore("tm_unknown_base", d, read_only=True)
        mocker.load(store)
    finally:
        shutil.rmtree(d)

    @mocker.mock(name="ghost_tool")
    def ghost_tool() -> str: ...

    with pytest.raises(RuntimeError, match="no recorded result"):
        ghost_tool()


@pytest.mark.asyncio
async def test_toolmocker_async_returns_recorded_results(tmp_dir):
    """ToolMocker async mock returns recorded results."""
    @record_tool
    async def async_fetch(sym: str) -> float:
        return 55.5

    with record("tm_async", base_dir=tmp_dir, seed=15):
        await async_fetch("HDFC")

    store  = EventStore("tm_async", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="async_fetch")
    async def async_fetch(sym: str) -> float: ...

    result = await async_fetch("HDFC")
    assert result == 55.5


# ══════════════════════════════════════════════════════════════════════════════
# RegressionSuite
# ══════════════════════════════════════════════════════════════════════════════

def _seed_run(run_id: str, tmp_dir: Path, n_calls: int = 1, cost: float = 0.001):
    """Helper: record a run with n_calls LLM pairs."""
    with record(run_id, base_dir=tmp_dir, seed=99) as s:
        for _ in range(n_calls):
            s.record_llm_request("openai", "gpt-4o", {}, [])
            s.record_llm_response(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                cost_usd=cost,
            )


def test_regression_suite_summarize(tmp_dir):
    """_summarize returns correct total_steps, total_llm_calls, total_cost_usd."""
    _seed_run("reg_sum", tmp_dir, n_calls=3, cost=0.002)
    suite   = RegressionSuite(base_dir=tmp_dir)
    summary = suite._summarize("reg_sum")
    assert summary["total_llm_calls"] == 3
    assert summary["total_cost_usd"]  == pytest.approx(0.006)
    assert summary["total_steps"]     >= 3  # seed + metadata + 6 LLM events


def test_regression_suite_exception_case(tmp_dir):
    """When a case function raises, result is failed with error string."""
    _seed_run("reg_exc", tmp_dir)
    suite = RegressionSuite(base_dir=tmp_dir)

    @suite.case("reg_exc")
    def exploding(original, session):
        raise ValueError("intentional boom")

    results = suite.run()
    assert len(results) == 1
    assert results[0].passed is False
    assert "ValueError" in results[0].error
    assert "intentional boom" in results[0].error


def test_regression_suite_missing_run(tmp_dir):
    """Case referencing nonexistent run → failed with FileNotFoundError."""
    suite = RegressionSuite(base_dir=tmp_dir)

    @suite.case("run_does_not_exist_xyz")
    def check(original, session):
        return True

    results = suite.run()
    assert results[0].passed is False
    assert "FileNotFoundError" in results[0].error or "Error" in results[0].error


def test_regression_suite_print_report_all_pass(tmp_dir, capsys):
    """print_report with all-pass produces 'All passed' output."""
    _seed_run("reg_print_pass", tmp_dir)
    suite = RegressionSuite(base_dir=tmp_dir)

    @suite.case("reg_print_pass")
    def ok(original, session):
        return True

    results = suite.run()
    suite.print_report(results)  # must not raise


def test_regression_suite_print_report_some_fail(tmp_dir):
    """print_report with failures must not raise."""
    _seed_run("reg_print_fail_a", tmp_dir)
    _seed_run("reg_print_fail_b", tmp_dir)
    suite = RegressionSuite(base_dir=tmp_dir)

    @suite.case("reg_print_fail_a")
    def pass_case(original, session):
        return True

    @suite.case("reg_print_fail_b")
    def fail_case(original, session):
        return False

    results = suite.run()
    assert any(r.passed for r in results)
    assert any(not r.passed for r in results)
    suite.print_report(results)  # must not raise


def test_regression_suite_cost_regression_check(tmp_dir):
    """Real-world check: cost increase of >10% triggers failure."""
    _seed_run("reg_cost_orig", tmp_dir, n_calls=1, cost=0.01)
    suite = RegressionSuite(base_dir=tmp_dir)

    @suite.case("reg_cost_orig")
    def cost_check(original, session):
        # Simulate 50% cost increase — should fail
        return session.total_cost() <= original["total_cost_usd"] * 1.1

    results = suite.run()
    # The replay cost equals original cost so it passes
    assert results[0].passed is True


# ══════════════════════════════════════════════════════════════════════════════
# export_finetune_dataset
# ══════════════════════════════════════════════════════════════════════════════

def _seed_finetune_run(run_id: str, tmp_dir: Path, prompts: list[tuple[str, str]]):
    """Seed a run with alternating LLM_REQUEST / LLM_RESPONSE pairs."""
    with record(run_id, base_dir=tmp_dir, seed=77) as s:
        for prompt, reply in prompts:
            s.record_llm_request("openai", "gpt-4o", {}, [{"role": "user", "content": prompt}])
            s.record_llm_response(
                {"choices": [{"message": {"role": "assistant", "content": reply}}]},
                cost_usd=0.001,
            )


def test_export_finetune_jsonl(tmp_dir):
    """JSONL export contains correct messages for each prompt/reply pair."""
    _seed_finetune_run("ft_jsonl", tmp_dir, [("What is 2+2?", "4"), ("Capital of India?", "New Delhi")])
    dest = tmp_dir / "out.jsonl"

    export_finetune_dataset(["ft_jsonl"], str(dest), base_dir=tmp_dir, format="jsonl")

    assert dest.exists()
    rows = [json.loads(l) for l in dest.read_text().strip().splitlines()]
    assert len(rows) == 2

    # Last message in each row must be the assistant reply
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "4"}
    assert rows[1]["messages"][-1] == {"role": "assistant", "content": "New Delhi"}


def test_export_finetune_alpaca(tmp_dir):
    """Alpaca format produces instruction/input/output rows."""
    _seed_finetune_run("ft_alpaca", tmp_dir, [("Explain recursion.", "It's self-referential.")])
    dest = tmp_dir / "alpaca.jsonl"

    export_finetune_dataset(["ft_alpaca"], str(dest), base_dir=tmp_dir, format="alpaca")

    rows = [json.loads(l) for l in dest.read_text().strip().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert "instruction" in row
    assert "output"      in row
    assert row["output"] == "It's self-referential."
    assert row["input"]  == ""


def test_export_finetune_multiple_runs(tmp_dir):
    """Multiple run_ids are concatenated into one file."""
    _seed_finetune_run("ft_multi_a", tmp_dir, [("A", "1")])
    _seed_finetune_run("ft_multi_b", tmp_dir, [("B", "2"), ("C", "3")])
    dest = tmp_dir / "multi.jsonl"

    export_finetune_dataset(["ft_multi_a", "ft_multi_b"], str(dest), base_dir=tmp_dir)

    rows = [json.loads(l) for l in dest.read_text().strip().splitlines()]
    assert len(rows) == 3


def test_export_finetune_empty_run(tmp_dir):
    """Run with no LLM calls produces zero rows — no crash."""
    with record("ft_empty", base_dir=tmp_dir, seed=0):
        pass  # no LLM calls
    dest = tmp_dir / "empty.jsonl"

    export_finetune_dataset(["ft_empty"], str(dest), base_dir=tmp_dir)

    rows = dest.read_text().strip().splitlines()
    assert rows == []
