"""
Phase 3: Regression testing framework.

Run a suite of recorded traces through updated agent code and assert
that outputs or costs haven't regressed.

Usage::

    from llmreplay.regression import RegressionSuite

    suite = RegressionSuite()

    @suite.case("run_abc123")
    def check(original, replay_session):
        # assert final response matches
        resp = [e for e in replay_session.events() if e.kind.value == "llm_response"]
        return len(resp) == original["total_llm_calls"]

    results = suite.run()
    suite.print_report(results)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from rich.console import Console
from rich.table import Table
from rich import box

from llmreplay.core.event import EventKind
from llmreplay.core.replay import replay, ReplaySession
from llmreplay.core.store import EventStore

console = Console()


@dataclass
class CaseResult:
    run_id:   str
    passed:   bool
    duration: float
    error:    str | None = None
    details:  Any = None


class RegressionSuite:
    """Collect regression cases and run them as a batch."""

    def __init__(self, base_dir=None):
        self._cases: list[tuple[str, Callable]] = []
        self._base_dir = base_dir

    def case(self, run_id: str):
        """Register a replay case. The decorated function receives (original_summary, session)."""
        def decorator(fn: Callable) -> Callable:
            self._cases.append((run_id, fn))
            return fn
        return decorator

    def run(self) -> list[CaseResult]:
        results = []
        for run_id, fn in self._cases:
            t0 = time.perf_counter()
            try:
                session  = replay(run_id, base_dir=self._base_dir)
                original = self._summarize(run_id)
                passed   = fn(original, session)
                results.append(CaseResult(
                    run_id=run_id,
                    passed=bool(passed),
                    duration=time.perf_counter() - t0,
                ))
            except Exception as exc:
                results.append(CaseResult(
                    run_id=run_id,
                    passed=False,
                    duration=time.perf_counter() - t0,
                    error=f"{type(exc).__name__}: {exc}",
                ))
        return results

    def print_report(self, results: list[CaseResult]) -> None:
        table = Table(title="Regression Results", box=box.ROUNDED, show_lines=True)
        table.add_column("Run ID",   style="cyan")
        table.add_column("Status",   justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Error")

        passed = sum(1 for r in results if r.passed)
        for r in results:
            status = "[green]✓ PASS[/green]" if r.passed else "[red]✗ FAIL[/red]"
            table.add_row(r.run_id, status, f"{r.duration:.3f}s", r.error or "")

        console.print(table)
        console.print(
            f"\n[bold]{'[green]All passed' if passed == len(results) else '[red]Some failed'}[/bold]"
            f"  {passed}/{len(results)} cases"
        )

    def _summarize(self, run_id: str) -> dict:
        store = EventStore(run_id, self._base_dir, read_only=True)
        total_cost = 0.0
        llm_calls  = 0
        for ev in store.iter_from():
            if ev.kind == EventKind.LLM_RESPONSE:
                total_cost += ev.payload.get("cost_usd", 0.0)
                llm_calls  += 1
        return {
            "total_steps":     store.count(),
            "total_llm_calls": llm_calls,
            "total_cost_usd":  total_cost,
        }


# ── fine-tuning dataset export ────────────────────────────────────────────────

def export_finetune_dataset(
    run_ids: list[str],
    dest: str,
    *,
    base_dir=None,
    format: str = "jsonl",  # "jsonl" (OpenAI) or "alpaca"
) -> None:
    """
    Export prompt/response pairs from recorded runs into a fine-tuning dataset.

    Supports OpenAI JSONL format and Alpaca JSON format.
    """
    import json
    from pathlib import Path

    rows = []
    for run_id in run_ids:
        store = EventStore(run_id, base_dir, read_only=True)
        pending = None
        for ev in store.iter_from():
            if ev.kind == EventKind.LLM_REQUEST:
                pending = ev.payload
            elif ev.kind == EventKind.LLM_RESPONSE and pending:
                messages = pending.get("messages", [])
                raw      = ev.payload.get("raw", {})
                choices  = raw.get("choices") or []
                reply    = choices[0]["message"]["content"] if choices else ""
                if format == "jsonl":
                    rows.append({"messages": messages + [{"role": "assistant", "content": reply}]})
                else:  # alpaca
                    prompt = " ".join(m.get("content", "") for m in messages)
                    rows.append({"instruction": prompt, "input": "", "output": reply})
                pending = None

    dest_path = Path(dest)
    with open(dest_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    console.print(f"[green]Fine-tune dataset → {dest_path}  ({len(rows)} rows)[/green]")
