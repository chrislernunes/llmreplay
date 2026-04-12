"""xAI Grok hook.

Grok uses an OpenAI-compatible API, so interception is handled directly
inside the OpenAI hook (llmreplay/hooks/openai.py) via base_url detection.

This module exists for:
- Direct import compatibility (``from llmreplay.hooks import grok``)
- Exposing the Grok cost table and estimate function independently
- Idempotent install/uninstall stubs (the real work is in openai.py)
"""
from __future__ import annotations

# ── cost table (prompt / completion per 1k tokens) ────────────────────────────
# Prices as of 2026-04 from https://x.ai/api
# Also present in openai.py._COST_TABLE — kept here for direct import.
_COST_TABLE: dict[str, tuple[float, float]] = {
    "grok-3":           (0.003,  0.015),
    "grok-3-fast":      (0.005,  0.025),
    "grok-3-mini":      (0.0003, 0.0005),
    "grok-3-mini-fast": (0.0006, 0.004),
    "grok-2":           (0.002,  0.010),
    "grok-2-mini":      (0.0002, 0.0004),
}

_XAI_BASE = "api.x.ai"


def _estimate_cost(model: str, usage: dict) -> float:
    """Estimate USD cost for a Grok response."""
    key = next((k for k in _COST_TABLE if model.startswith(k)), None)
    if key is None or not usage:
        return 0.0
    prompt_k   = usage.get("prompt_tokens", 0)     / 1000
    complete_k = usage.get("completion_tokens", 0) / 1000
    p, c = _COST_TABLE[key]
    return round(prompt_k * p + complete_k * c, 6)


def install() -> None:
    """No-op: Grok interception is handled by the OpenAI hook via base_url detection."""


def uninstall() -> None:
    """No-op: Grok interception is handled by the OpenAI hook via base_url detection."""
