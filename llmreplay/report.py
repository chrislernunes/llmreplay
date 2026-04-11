"""Export a shareable, human-readable bug report from a recorded run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llmreplay.core.event import EventKind
from llmreplay.core.store import EventStore


def export_report(
    run_id: str,
    dest: Path | str,
    *,
    base_dir: Path | None = None,
    compress: bool = False,
) -> Path:
    """
    Write a structured JSON bug report for ``run_id`` to ``dest``.

    The report includes:
    - run metadata + seed
    - per-step event log
    - cost breakdown (total + per-step)
    - exception trace (if any)
    - full prompt/response pairs
    """
    store  = EventStore(run_id, base_dir, read_only=True)
    events = list(store.iter_from())

    metadata: dict = {}
    seed: int | None = None
    steps: list[dict] = []
    cost_by_step: dict[int, float] = {}
    total_cost: float = 0.0
    exceptions: list[dict] = []
    llm_pairs: list[dict] = []
    _pending_request: dict | None = None

    for ev in events:
        if ev.kind == EventKind.RANDOM_SEED:
            seed = ev.payload["seed"]
        elif ev.kind == EventKind.METADATA:
            metadata = ev.payload.get("metadata", {})
        elif ev.kind == EventKind.LLM_REQUEST:
            _pending_request = ev.payload
        elif ev.kind == EventKind.LLM_RESPONSE:
            cost = ev.payload.get("cost_usd", 0.0)
            cost_by_step[ev.step] = cost
            total_cost += cost
            if _pending_request:
                llm_pairs.append({
                    "step":     ev.step,
                    "request":  _pending_request,
                    "response": ev.payload.get("raw", {}),
                    "cost_usd": cost,
                })
                _pending_request = None
        elif ev.kind == EventKind.EXCEPTION:
            exceptions.append({"step": ev.step, **ev.payload})

        steps.append({
            "step": ev.step,
            "kind": ev.kind.value,
            "ts":   datetime.fromtimestamp(ev.ts, tz=timezone.utc).isoformat(),
            "payload": ev.payload,
        })

    report = {
        "run_id":       run_id,
        "seed":         seed,
        "metadata":     metadata,
        "total_steps":  len(steps),
        "total_cost_usd": round(total_cost, 6),
        "cost_by_step": cost_by_step,
        "exceptions":   exceptions,
        "llm_pairs":    llm_pairs,
        "events":       steps,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        import gzip
        dest = dest.with_suffix(".json.gz")
        with gzip.open(dest, "wt", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    else:
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return dest
