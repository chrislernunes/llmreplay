"""Transparent patch for the OpenAI SDK (v1+)."""
from __future__ import annotations

from typing import Any

_original_create:       Any = None
_original_acreate:      Any = None
_patched = False

# ── cost table (prompt / completion per 1k tokens) ────────────────────────────
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o":               (0.005,  0.015),
    "gpt-4o-mini":          (0.00015, 0.0006),
    "gpt-4-turbo":          (0.01,   0.03),
    "gpt-3.5-turbo":        (0.0005, 0.0015),
    "o1":                   (0.015,  0.06),
    "o3-mini":              (0.0011, 0.0044),
}


def _estimate_cost(model: str, usage: dict) -> float:
    key = next((k for k in _COST_TABLE if model.startswith(k)), None)
    if key is None or not usage:
        return 0.0
    prompt_k   = usage.get("prompt_tokens", 0)     / 1000
    complete_k = usage.get("completion_tokens", 0) / 1000
    p, c = _COST_TABLE[key]
    return round(prompt_k * p + complete_k * c, 6)


def _make_patched_create(original_fn, is_async: bool):
    from llmreplay.core.context import current_session

    if is_async:
        async def _acreate(self, **kwargs):
            session = current_session()
            if session is None:
                return await original_fn(self, **kwargs)

            if kwargs.get("stream"):
                raise RuntimeError(
                    "llmreplay does not support streaming. "
                    "Use stream=False inside a record() context."
                )

            _sanitize(kwargs)
            session.record_llm_request("openai", kwargs.get("model", ""), _params(kwargs), kwargs.get("messages", []))
            response = await original_fn(self, **kwargs)
            raw  = response.model_dump()
            cost = _estimate_cost(kwargs.get("model", ""), raw.get("usage") or {})
            session.record_llm_response(raw, cost)
            return response
        return _acreate
    else:
        def _create(self, **kwargs):
            session = current_session()
            if session is None:
                return original_fn(self, **kwargs)

            if kwargs.get("stream"):
                raise RuntimeError(
                    "llmreplay does not support streaming. "
                    "Use stream=False inside a record() context."
                )

            _sanitize(kwargs)
            session.record_llm_request("openai", kwargs.get("model", ""), _params(kwargs), kwargs.get("messages", []))
            response = original_fn(self, **kwargs)
            raw  = response.model_dump()
            cost = _estimate_cost(kwargs.get("model", ""), raw.get("usage") or {})
            session.record_llm_response(raw, cost)
            return response
        return _create


def _params(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in ("messages", "stream")}


def _sanitize(kwargs: dict) -> None:
    """Inject a fixed seed when inside a recording session so replays are identical."""
    from llmreplay.core.context import current_session
    s = current_session()
    if s and "seed" not in kwargs:
        kwargs["seed"] = s.seed


def install() -> None:
    global _original_create, _original_acreate, _patched
    if _patched:
        return
    try:
        import openai
        completions = openai.resources.chat.completions.Completions
        _original_create  = completions.create
        _original_acreate = completions.create  # async version is .acreate on AsyncCompletions
        completions.create = _make_patched_create(_original_create, is_async=False)

        async_completions = openai.resources.chat.completions.AsyncCompletions
        _original_acreate = async_completions.create
        async_completions.create = _make_patched_create(_original_acreate, is_async=True)

        _patched = True
    except ImportError:
        pass


def uninstall() -> None:
    global _patched
    if not _patched:
        return
    try:
        import openai
        if _original_create:
            openai.resources.chat.completions.Completions.create = _original_create
        if _original_acreate:
            openai.resources.chat.completions.AsyncCompletions.create = _original_acreate
        _patched = False
    except ImportError:
        pass