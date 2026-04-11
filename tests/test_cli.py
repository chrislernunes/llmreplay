"""
CLI tests using Click's test runner — exercises list, view, step, cost, export, delete.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from llmreplay.cli import cli
from llmreplay.core.context import record
from llmreplay.core.event import EventKind


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


@pytest.fixture
def populated_dir(tmp_dir):
    """A temp dir with one pre-recorded run."""
    with record("cli_run_001", base_dir=tmp_dir, seed=77) as session:
        session.record_llm_request("openai", "gpt-4o", {"temperature": 0.5},
                                   [{"role": "user", "content": "What is Kelly criterion?"}])
        session.record_llm_response(
            {"choices": [{"message": {"content": "Kelly criterion..."}}]},
            cost_usd=0.0042,
        )
        session.record_tool_call("search", {"query": "Kelly"})
        session.record_tool_result("search", {"results": ["r1", "r2"]})
    return tmp_dir


def run(args, base_dir):
    """Helper: invoke CLI with --dir override."""
    runner = CliRunner()
    return runner.invoke(cli, [*args, "--dir", str(base_dir)], catch_exceptions=False)


# ── list ──────────────────────────────────────────────────────────────────────

def test_cli_list_shows_run(populated_dir):
    result = run(["list"], populated_dir)
    assert result.exit_code == 0
    assert "cli_run_001" in result.output


def test_cli_list_empty_dir(tmp_dir):
    result = run(["list"], tmp_dir)
    assert result.exit_code == 0
    assert "No recorded runs found" in result.output


# ── view ──────────────────────────────────────────────────────────────────────

def test_cli_view_shows_events(populated_dir):
    result = run(["view", "cli_run_001"], populated_dir)
    assert result.exit_code == 0
    assert "llm_request" in result.output
    assert "llm_response" in result.output
    assert "tool_call" in result.output


def test_cli_view_unknown_run(tmp_dir):
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "nonexistent", "--dir", str(tmp_dir)], catch_exceptions=False)
    assert result.exit_code == 0
    assert "not found" in result.output.lower()


def test_cli_view_step_filter(populated_dir):
    result = run(["view", "cli_run_001", "--step", "3"], populated_dir)
    assert result.exit_code == 0
    # Only steps >= 3 should appear; steps 0,1,2 should not
    assert "0000" not in result.output


def test_cli_view_kind_filter(populated_dir):
    result = run(["view", "cli_run_001", "--kind", "tool_call"], populated_dir)
    assert result.exit_code == 0
    assert "tool_call" in result.output
    assert "llm_request" not in result.output


# ── step ──────────────────────────────────────────────────────────────────────

def test_cli_step_shows_payload(populated_dir):
    result = run(["step", "cli_run_001", "0"], populated_dir)
    assert result.exit_code == 0
    assert "random_seed" in result.output   # step 0 is always the seed event


def test_cli_step_not_found(populated_dir):
    result = run(["step", "cli_run_001", "999"], populated_dir)
    assert result.exit_code == 0
    assert "not found" in result.output.lower()


# ── cost ──────────────────────────────────────────────────────────────────────

def test_cli_cost_shows_total(populated_dir):
    result = run(["cost", "cli_run_001"], populated_dir)
    assert result.exit_code == 0
    assert "0.0042" in result.output
    assert "gpt-4o" in result.output


# ── export ────────────────────────────────────────────────────────────────────

def test_cli_export_json_bug_report(populated_dir):
    out = populated_dir / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["export", "cli_run_001", "--json", "--out", str(out), "--dir", str(populated_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert out.exists()

    report = json.loads(out.read_text())
    assert report["run_id"] == "cli_run_001"
    assert report["total_cost_usd"] == pytest.approx(0.0042)
    assert len(report["llm_pairs"]) == 1


def test_cli_export_jsonl(populated_dir):
    out = populated_dir / "events.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["export", "cli_run_001", "--jsonl", "--out", str(out), "--dir", str(populated_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # export_jsonl appends nothing extra — file path is returned
    real_out = populated_dir / "events.jsonl"
    assert real_out.exists()
    lines = real_out.read_text().strip().splitlines()
    assert len(lines) > 0
    # Each line must be valid JSON
    for line in lines:
        obj = json.loads(line)
        assert "kind" in obj
        assert "step" in obj


# ── delete ────────────────────────────────────────────────────────────────────

def test_cli_delete_removes_run(populated_dir):
    db = populated_dir / "cli_run_001.db"
    assert db.exists()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["delete", "cli_run_001", "--dir", str(populated_dir), "--yes"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert not db.exists()
