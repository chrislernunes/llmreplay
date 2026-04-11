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
