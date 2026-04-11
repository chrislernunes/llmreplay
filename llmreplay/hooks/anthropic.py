"""Transparent patch for the Anthropic SDK (v0.20+)."""
from __future__ import annotations

from typing import Any

_original_create:  Any = None
_original_acreate: Any = None
_patched = False

_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4":    (0.015, 0.075),
    "claude-sonnet-4":  (0.003, 0.015),
    "claude-haiku-4":   (0.00025, 0.00125),
    "claude-3-5-sonnet":(0.003, 0.015),
    "claude-3-opus":    (0.015, 0.075),
    "claude-3-haiku":   (0.00025, 0.00125),
}


def _estimate_cost(model: str, usage: dict | None) -> float:
    if not usage:
        return 0.0
    key = next((k for k in _COST_TABLE if model.startswith(k)), None)
    if key is None:
        return 0.0
    p, c = _COST_TABLE[key]
    return round(
        usage.get("input_tokens", 0)  / 1000 * p +
        usage.get("output_tokens", 0) / 1000 * c,
        6,
    )


def _make_patched(original_fn, is_async: bool):
    from llmreplay.core.context import current_session

    if is_async:
        async def _patched_acreate(self, **kwargs):
            session = current_session()
            if session is None:
                return await original_fn(self, **kwargs)
            if kwargs.get("stream"):
                raise RuntimeError(
                    "llmreplay does not support streaming. "
                    "Use stream=False inside a record() context."
                )
            session.record_llm_request("anthropic", kwargs.get("model", ""),
                                       {k: v for k, v in kwargs.items() if k != "messages"},
                                       kwargs.get("messages", []))
            resp = await original_fn(self, **kwargs)
            raw  = resp.model_dump()
            cost = _estimate_cost(kwargs.get("model", ""), raw.get("usage"))
            session.record_llm_response(raw, cost)
            return resp
        return _patched_acreate
    else:
        def _patched_create(self, **kwargs):
            session = current_session()
            if session is None:
                return original_fn(self, **kwargs)
            if kwargs.get("stream"):
                raise RuntimeError(
                    "llmreplay does not support streaming. "
                    "Use stream=False inside a record() context."
                )
            session.record_llm_request("anthropic", kwargs.get("model", ""),
                                       {k: v for k, v in kwargs.items() if k != "messages"},
                                       kwargs.get("messages", []))
            resp = original_fn(self, **kwargs)
            raw  = resp.model_dump()
            cost = _estimate_cost(kwargs.get("model", ""), raw.get("usage"))
            session.record_llm_response(raw, cost)
            return resp
        return _patched_create


def install() -> None:
    global _original_create, _original_acreate, _patched
    if _patched:
        return
    try:
        import anthropic
        msgs = anthropic.resources.messages.Messages
        _original_create = msgs.create
        msgs.create = _make_patched(_original_create, is_async=False)

        amsgs = anthropic.resources.messages.AsyncMessages
        _original_acreate = amsgs.create
        amsgs.create = _make_patched(_original_acreate, is_async=True)

        _patched = True
    except ImportError:
        pass


def uninstall() -> None:
    global _patched
    if not _patched:
        return
    try:
        import anthropic
        if _original_create:
            anthropic.resources.messages.Messages.create = _original_create
        if _original_acreate:
            anthropic.resources.messages.AsyncMessages.create = _original_acreate
        _patched = False
    except ImportError:
        pass