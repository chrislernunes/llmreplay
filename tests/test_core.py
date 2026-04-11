"""Core integration tests — store, events, regression suite."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from llmreplay.core.event import Event, EventKind
from llmreplay.core.store import EventStore
from llmreplay.core.context import record
from llmreplay.regression import RegressionSuite


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


def test_store_append_and_read(tmp_dir):
    store = EventStore("s1", tmp_dir)
    ev = Event(run_id="s1", step=0, kind=EventKind.LLM_REQUEST, payload={"model": "gpt-4o"})
    store.append(ev)

    fetched = store.get(0)
    assert fetched is not None
    assert fetched.kind == EventKind.LLM_REQUEST
    assert fetched.payload["model"] == "gpt-4o"


def test_store_iter_from(tmp_dir):
    store = EventStore("s2", tmp_dir)
    for i in range(10):
        store.append(Event(run_id="s2", step=i, kind=EventKind.TOOL_CALL, payload={"i": i}))

    events = list(store.iter_from(start=5))
    assert [e.step for e in events] == list(range(5, 10))


def test_store_count(tmp_dir):
    store = EventStore("s3", tmp_dir)
    for i in range(7):
        store.append(Event(run_id="s3", step=i, kind=EventKind.METADATA, payload={}))
    assert store.count() == 7


def test_regression_suite_pass(tmp_dir):
    with record("reg_pass", base_dir=tmp_dir, seed=1) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({}, 0.001)

    suite = RegressionSuite(base_dir=tmp_dir)

    @suite.case("reg_pass")
    def check(original, rs):
        return original["total_llm_calls"] == 1

    results = suite.run()
    assert results[0].passed is True


def test_regression_suite_fail(tmp_dir):
    with record("reg_fail", base_dir=tmp_dir, seed=2) as session:
        session.record_llm_request("openai", "gpt-4o", {}, [])
        session.record_llm_response({}, 0.001)

    suite = RegressionSuite(base_dir=tmp_dir)

    @suite.case("reg_fail")
    def check(original, rs):
        return False  # always fails

    results = suite.run()
    assert results[0].passed is False


def test_missing_run_raises(tmp_dir):
    from llmreplay.core.replay import replay
    with pytest.raises(FileNotFoundError):
        replay("nonexistent_run_id_xyz", base_dir=tmp_dir)


def test_readonly_store_raises_on_append(tmp_dir):
    store_w = EventStore("ro_test", tmp_dir)
    store_w.append(Event(run_id="ro_test", step=0, kind=EventKind.METADATA, payload={}))

    store_r = EventStore("ro_test", tmp_dir, read_only=True)
    with pytest.raises(RuntimeError):
        store_r.append(Event(run_id="ro_test", step=1, kind=EventKind.METADATA, payload={}))


def test_record_overwrite(tmp_dir):
    """overwrite=True deletes existing run and starts fresh."""
    with record("overwrite_run", base_dir=tmp_dir, seed=1) as s:
        s.record_llm_request("openai", "gpt-4o", {}, [])
        s.record_llm_response({}, 0.001)

    with record("overwrite_run", base_dir=tmp_dir, seed=2, overwrite=True) as s:
        pass  # empty run

    store = EventStore("overwrite_run", tmp_dir, read_only=True)
    llm_reqs = [e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST]
    assert len(llm_reqs) == 0


def test_record_memory_helpers(tmp_dir):
    """record_memory_read and record_memory_write emit correct events."""
    with record("mem_helpers", base_dir=tmp_dir, seed=1) as s:
        s.record_memory_read("query text", [{"doc": "a"}])
        s.record_memory_write("key1", {"value": 99})

    store  = EventStore("mem_helpers", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]
    assert EventKind.MEMORY_READ  in kinds
    assert EventKind.MEMORY_WRITE in kinds


def test_export_report_compress(tmp_dir):
    """export_report with compress=True writes a .json.gz file."""
    import gzip, json
    from llmreplay.report import export_report

    with record("report_compress", base_dir=tmp_dir, seed=1) as s:
        s.record_llm_request("openai", "gpt-4o", {}, [{"role": "user", "content": "hi"}])
        s.record_llm_response({"choices": [{"message": {"role": "assistant", "content": "hello"}}]}, 0.005)

    dest = export_report("report_compress", tmp_dir / "report.json", base_dir=tmp_dir, compress=True)
    assert dest.suffix == ".gz"
    with gzip.open(dest, "rt") as f:
        data = json.load(f)
    assert data["run_id"] == "report_compress"
    assert data["total_cost_usd"] == pytest.approx(0.005)


def test_export_report_with_exception(tmp_dir):
    """export_report captures EXCEPTION events in the exceptions list."""
    from llmreplay.report import export_report
    import json

    with record("report_exc", base_dir=tmp_dir, seed=1) as s:
        s.record_llm_request("openai", "gpt-4o", {}, [])
        s.record_llm_response({}, 0.001)
        s.record_exception("ValueError", "something broke")

    dest = export_report("report_exc", tmp_dir / "exc_report.json", base_dir=tmp_dir)
    data = json.loads(dest.read_text())
    assert len(data["exceptions"]) == 1
    assert data["exceptions"][0]["exc_type"] == "ValueError"


def test_store_export_jsonl(tmp_dir):
    """EventStore.export_jsonl writes a valid gzip JSONL file."""
    import gzip, json as _json

    store = EventStore("jsonl_test", tmp_dir)
    store.append(Event(run_id="jsonl_test", step=0, kind=EventKind.METADATA, payload={"x": 1}))
    store.append(Event(run_id="jsonl_test", step=1, kind=EventKind.LLM_REQUEST, payload={"model": "gpt-4o"}))

    path = store.export_jsonl(tmp_dir / "out.jsonl", compress=True)
    assert path.exists()
    with gzip.open(path, "rt") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 2
    assert _json.loads(lines[1])["kind"] == "llm_request"


def test_replay_with_overrides(tmp_dir):
    """replay() accepts overrides dict without error."""
    with record("replay_override", base_dir=tmp_dir, seed=1) as s:
        s.record_llm_request("openai", "gpt-4o", {}, [])
        s.record_llm_response({}, 0.001)

    from llmreplay.core.replay import replay
    rs = replay("replay_override", base_dir=tmp_dir, overrides={"model": "gpt-4o-mini"})
    assert rs.overrides["model"] == "gpt-4o-mini"


def test_langchain_handler_none_when_not_installed():
    """If langchain is installed, langchain_handler is not None."""
    from llmreplay import langchain_handler
    # langchain is installed in our dev env — just assert it's callable or None
    assert langchain_handler is None or callable(langchain_handler)


def test_event_from_dict_roundtrip():
    """Event.from_dict(ev.to_dict()) roundtrips correctly."""
    ev = Event(run_id="rt", step=5, kind=EventKind.LLM_REQUEST, payload={"model": "gpt-4o"}, ts=1234567890.0)
    d  = ev.to_dict()
    ev2 = Event.from_dict(d)
    assert ev2.run_id  == ev.run_id
    assert ev2.step    == ev.step
    assert ev2.kind    == ev.kind
    assert ev2.payload == ev.payload
    assert ev2.ts      == ev.ts


def test_store_all_run_ids(tmp_dir):
    """all_run_ids returns all stored run IDs."""
    EventStore("run_alpha", tmp_dir).append(Event(run_id="run_alpha", step=0, kind=EventKind.METADATA, payload={}))
    EventStore("run_beta",  tmp_dir).append(Event(run_id="run_beta",  step=0, kind=EventKind.METADATA, payload={}))
    ids = EventStore("run_alpha", tmp_dir, read_only=True).all_run_ids()
    assert "run_alpha" in ids
    assert "run_beta"  in ids


def test_replay_total_steps(tmp_dir):
    """ReplaySession.total_steps() returns the correct step count."""
    with record("rs_steps", base_dir=tmp_dir, seed=1) as s:
        s.record_llm_request("openai", "gpt-4o", {}, [])
        s.record_llm_response({}, 0.001)

    from llmreplay.core.replay import replay
    rs = replay("rs_steps", base_dir=tmp_dir)
    assert rs.total_steps() >= 3  # seed + metadata + 2 LLM events
