"""
Phase 3: Streamlit web timeline UI.
Run with: streamlit run llmreplay/web.py -- --run-id <run_id>
Or use: llmreplay web <run_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Guard — only import streamlit at runtime
def launch(run_id: str, base_dir: Path | None = None):
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        print("Install web extras: pip install llmreplay[web]")
        sys.exit(1)

    from llmreplay.core.event import EventKind
    from llmreplay.core.store import EventStore, _DEFAULT_DIR

    base  = base_dir or _DEFAULT_DIR
    store = EventStore(run_id, base, read_only=True)
    events = list(store.iter_from())

    # ── page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title=f"llmreplay · {run_id}",
        page_icon="⏺",
        layout="wide",
    )

    # ── header ───────────────────────────────────────────────────────────────
    st.title(f"⏺ llmreplay  ·  `{run_id}`")

    # ── sidebar filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        kinds = [e.kind.value for e in events]
        unique_kinds = sorted(set(kinds))
        selected = st.multiselect("Event kinds", unique_kinds, default=unique_kinds)
        step_range = st.slider("Step range", 0, max(len(events) - 1, 1), (0, len(events) - 1))
        show_payload = st.checkbox("Show full payloads", value=True)

    filtered = [
        e for e in events
        if e.kind.value in selected and step_range[0] <= e.step <= step_range[1]
    ]

    # ── cost heatmap ─────────────────────────────────────────────────────────
    cost_steps, cost_vals = [], []
    for e in events:
        if e.kind == EventKind.LLM_RESPONSE:
            cost_steps.append(e.step)
            cost_vals.append(e.payload.get("cost_usd", 0.0))

    if cost_vals:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total steps", len(events))
        col2.metric("Total cost", f"${sum(cost_vals):.4f}")
        col3.metric("LLM calls", len(cost_vals))

        fig = go.Figure(go.Bar(
            x=cost_steps, y=cost_vals,
            marker_color=cost_vals,
            marker_colorscale="Reds",
            name="Cost per step",
        ))
        fig.update_layout(
            title="💰 Cost heatmap per LLM call",
            xaxis_title="Step",
            yaxis_title="USD",
            height=250,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── timeline ─────────────────────────────────────────────────────────────
    st.subheader(f"📋 Event timeline  ({len(filtered)} events)")

    KIND_EMOJI = {
        EventKind.LLM_REQUEST:  "🤖",
        EventKind.LLM_RESPONSE: "✅",
        EventKind.TOOL_CALL:    "🔧",
        EventKind.TOOL_RESULT:  "📦",
        EventKind.MEMORY_READ:  "🧠",
        EventKind.MEMORY_WRITE: "💾",
        EventKind.RANDOM_SEED:  "🎲",
        EventKind.EXCEPTION:    "🔥",
        EventKind.METADATA:     "📎",
        EventKind.BRANCH:       "🌿",
    }

    for ev in filtered:
        emoji = KIND_EMOJI.get(ev.kind, "•")
        label = f"{emoji} **[{ev.step:04d}]** `{ev.kind.value}`"
        if ev.kind == EventKind.EXCEPTION:
            with st.expander(label, expanded=True):
                st.error(f"{ev.payload.get('exc_type')}: {ev.payload.get('message')}")
        else:
            with st.expander(label, expanded=False):
                if show_payload:
                    st.code(json.dumps(ev.payload, indent=2), language="json")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--dir", default=None)
    args = p.parse_args()
    launch(args.run_id, Path(args.dir) if args.dir else None)
