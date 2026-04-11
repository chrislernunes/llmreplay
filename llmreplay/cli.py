"""llmreplay CLI — view runs, inspect steps, export bug reports."""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from llmreplay.core.event import EventKind
from llmreplay.core.store import EventStore, _DEFAULT_DIR

console = Console()

# ── colour map ────────────────────────────────────────────────────────────────
_KIND_STYLE = {
    EventKind.LLM_REQUEST:  "bold cyan",
    EventKind.LLM_RESPONSE: "bold green",
    EventKind.TOOL_CALL:    "bold yellow",
    EventKind.TOOL_RESULT:  "yellow",
    EventKind.MEMORY_READ:  "blue",
    EventKind.MEMORY_WRITE: "bold blue",
    EventKind.RANDOM_SEED:  "dim",
    EventKind.METADATA:     "dim",
    EventKind.BRANCH:       "magenta",
    EventKind.EXCEPTION:    "bold red",
}


@click.group()
def cli():
    """llmreplay — deterministic replay debugger for LLM agents."""


# ── list ──────────────────────────────────────────────────────────────────────
@cli.command("list")
@click.option("--dir", "base_dir", default=None, help="Override storage directory")
def list_runs(base_dir):
    """List all recorded runs."""
    base = Path(base_dir) if base_dir else _DEFAULT_DIR
    dbs  = sorted(base.glob("*.db"))
    if not dbs:
        console.print("[dim]No recorded runs found.[/dim]")
        return

    table = Table(title="Recorded Runs", box=box.ROUNDED, show_lines=True)
    table.add_column("Run ID",   style="cyan", no_wrap=True)
    table.add_column("Steps",    justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("File")

    for db in dbs:
        run_id = db.stem
        store  = EventStore(run_id, base, read_only=True)
        n      = store.count()
        cost   = 0.0
        for ev in store.iter_from():
            if ev.kind == EventKind.LLM_RESPONSE:
                cost += ev.payload.get("cost_usd", 0.0)
        table.add_row(run_id, str(n), f"${cost:.4f}", str(db))

    console.print(table)


# ── view ──────────────────────────────────────────────────────────────────────
@cli.command("view")
@click.argument("run_id")
@click.option("--step", "-s", default=None, type=int, help="Jump to a specific step")
@click.option("--kind", "-k", default=None, help="Filter by event kind (e.g. llm_request)")
@click.option("--dir",  "base_dir", default=None)
def view_run(run_id, step, kind, base_dir):
    """View all events in a run, optionally filtered."""
    base  = Path(base_dir) if base_dir else _DEFAULT_DIR
    try:
        store = EventStore(run_id, base, read_only=True)
    except FileNotFoundError:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        return

    if store.count() == 0:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        return

    start       = step or 0
    filter_kind = EventKind(kind) if kind else None

    for ev in store.iter_from(start):
        if filter_kind and ev.kind != filter_kind:
            continue

        style   = _KIND_STYLE.get(ev.kind, "white")
        header  = Text(f"[{ev.step:04d}] {ev.kind.value}", style=style)
        payload = json.dumps(ev.payload, indent=2)

        # Truncate huge payloads in terminal view
        if len(payload) > 2000:
            payload = payload[:2000] + "\n  ... [truncated] ..."

        console.print(Panel(
            Syntax(payload, "json", theme="dracula", word_wrap=True),
            title=header,
            expand=False,
        ))


# ── step ──────────────────────────────────────────────────────────────────────
@cli.command("step")
@click.argument("run_id")
@click.argument("step_num", type=int)
@click.option("--dir", "base_dir", default=None)
def view_step(run_id, step_num, base_dir):
    """Inspect a single step in full detail."""
    base  = Path(base_dir) if base_dir else _DEFAULT_DIR
    store = EventStore(run_id, base, read_only=True)
    ev    = store.get(step_num)
    if ev is None:
        console.print(f"[red]Step {step_num} not found in run '{run_id}'.[/red]")
        return

    style  = _KIND_STYLE.get(ev.kind, "white")
    header = Text(f"[{ev.step:04d}] {ev.kind.value}", style=style)
    console.print(Panel(
        Syntax(json.dumps(ev.payload, indent=2), "json", theme="dracula"),
        title=header,
    ))


# ── cost ──────────────────────────────────────────────────────────────────────
@cli.command("cost")
@click.argument("run_id")
@click.option("--dir", "base_dir", default=None)
def show_cost(run_id, base_dir):
    """Show per-step cost breakdown for a run."""
    base  = Path(base_dir) if base_dir else _DEFAULT_DIR
    store = EventStore(run_id, base, read_only=True)

    table = Table(title=f"Cost breakdown: {run_id}", box=box.SIMPLE)
    table.add_column("Step",  justify="right")
    table.add_column("Model")
    table.add_column("Cost (USD)", justify="right", style="green")

    total = 0.0
    req_model: dict[int, str] = {}
    for ev in store.iter_from():
        if ev.kind == EventKind.LLM_REQUEST:
            req_model[ev.step + 1] = ev.payload.get("model", "?")   # next step is the response
        if ev.kind == EventKind.LLM_RESPONSE:
            cost = ev.payload.get("cost_usd", 0.0)
            total += cost
            table.add_row(str(ev.step), req_model.get(ev.step, "?"), f"${cost:.6f}")

    console.print(table)
    console.print(f"[bold green]Total: ${total:.6f}[/bold green]")


# ── export ────────────────────────────────────────────────────────────────────
@cli.command("export")
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, help="Export full bug report JSON")
@click.option("--jsonl", is_flag=True, help="Export raw event log as JSONL")
@click.option("--compress", "-z", is_flag=True, help="Gzip output")
@click.option("--out", "-o", default=None, help="Output file path")
@click.option("--dir", "base_dir", default=None)
def export_run(run_id, as_json, jsonl, compress, out, base_dir):
    """Export a run to JSON bug report or JSONL event log."""
    base = Path(base_dir) if base_dir else _DEFAULT_DIR

    if jsonl:
        store = EventStore(run_id, base, read_only=True)
        dest  = Path(out) if out else Path(f"{run_id}_events.jsonl")
        path  = store.export_jsonl(dest, compress=compress)
        console.print(f"[green]Exported JSONL → {path}[/green]")
    else:
        from llmreplay.report import export_report
        dest = Path(out) if out else Path(f"{run_id}_bugreport.json")
        path = export_report(run_id, dest, base_dir=base, compress=compress)
        console.print(f"[green]Bug report → {path}[/green]")


# ── delete ────────────────────────────────────────────────────────────────────
@cli.command("delete")
@click.argument("run_id")
@click.option("--dir", "base_dir", default=None)
@click.confirmation_option(prompt="Delete this run?")
def delete_run(run_id, base_dir):
    """Delete a recorded run."""
    base  = Path(base_dir) if base_dir else _DEFAULT_DIR
    store = EventStore(run_id, base, read_only=True)
    store.delete()
    console.print(f"[red]Deleted run '{run_id}'[/red]")