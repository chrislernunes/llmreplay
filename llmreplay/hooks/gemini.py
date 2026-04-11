"""Transparent patch for Google Gemini SDK (google-genai).

Patches ``google.genai.models.Models.generate_content`` and its async
counterpart ``google.genai.models.AsyncModels.generate_content``.

Usage::

    import google.genai as genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with record("gemini_run"):
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents="Explain quantum entanglement.",
        )
        print(response.text)
"""
from __future__ import annotations

from typing import Any

_original_generate:  Any = None
_original_agenerate: Any = None
_patched = False

# ── cost table (input / output per 1k tokens) ─────────────────────────────────
# Prices as of 2026-04: https://ai.google.dev/pricing
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro":        (0.00125, 0.010),
    "gemini-2.5-flash":      (0.000075, 0.0003),
    "gemini-2.0-flash":      (0.0001,  0.0004),
    "gemini-2.0-flash-lite": (0.000075, 0.0003),
    "gemini-1.5-pro":        (0.00125, 0.005),
    "gemini-1.5-flash":      (0.000075, 0.0003),
    "gemini-1.5-flash-8b":   (0.0000375, 0.00015),
}


def _estimate_cost(model: str, usage_metadata) -> float:
    # Strip optional "models/" prefix the SDK may add
    model = model.removeprefix("models/")
    key = next((k for k in _COST_TABLE if model.startswith(k)), None)
    if key is None or usage_metadata is None:
        return 0.0
    try:
        prompt_k   = (getattr(usage_metadata, "prompt_token_count",     0) or 0) / 1000
        complete_k = (getattr(usage_metadata, "candidates_token_count", 0) or 0) / 1000
        p, c = _COST_TABLE[key]
        return round(prompt_k * p + complete_k * c, 6)
    except Exception:  # pragma: no cover
        return 0.0


def _extract_messages(contents) -> list:
    """Normalise genai contents into a plain list for event storage."""
    try:
        if isinstance(contents, str):
            return [{"role": "user", "content": contents}]
        if isinstance(contents, list):
            out = []
            for item in contents:
                if isinstance(item, str):
                    out.append({"role": "user", "content": item})
                elif hasattr(item, "role") and hasattr(item, "parts"):
                    text = " ".join(getattr(p, "text", str(p)) for p in item.parts)
                    out.append({"role": item.role, "content": text})
                else:  # pragma: no cover
                    out.append({"role": "user", "content": str(item)})
            return out
    except Exception:  # pragma: no cover
        pass
    return [{"role": "user", "content": str(contents)}]  # pragma: no cover


def _response_to_dict(response) -> dict:
    try:
        text = response.text
    except Exception:  # pragma: no cover
        text = ""
    try:
        um = response.usage_metadata
        usage = {
            "prompt_token_count":     getattr(um, "prompt_token_count",     0),
            "candidates_token_count": getattr(um, "candidates_token_count", 0),
        }
    except Exception:  # pragma: no cover
        usage = {}
    return {"text": text, "usage_metadata": usage}


def _make_patched_generate(original_fn, is_async: bool):
    from llmreplay.core.context import current_session

    if is_async:
        async def _agenerate(self, *, model: str, contents, **kwargs):
            session = current_session()
            if session is None:  # pragma: no cover
                return await original_fn(self, model=model, contents=contents, **kwargs)
            messages = _extract_messages(contents)
            session.record_llm_request("gemini", model, kwargs, messages)
            response = await original_fn(self, model=model, contents=contents, **kwargs)
            raw  = _response_to_dict(response)
            cost = _estimate_cost(model, getattr(response, "usage_metadata", None))
            session.record_llm_response(raw, cost)
            return response
        return _agenerate
    else:
        def _generate(self, *, model: str, contents, **kwargs):
            session = current_session()
            if session is None:  # pragma: no cover
                return original_fn(self, model=model, contents=contents, **kwargs)
            messages = _extract_messages(contents)
            session.record_llm_request("gemini", model, kwargs, messages)
            response = original_fn(self, model=model, contents=contents, **kwargs)
            raw  = _response_to_dict(response)
            cost = _estimate_cost(model, getattr(response, "usage_metadata", None))
            session.record_llm_response(raw, cost)
            return response
        return _generate


def install() -> None:
    global _original_generate, _original_agenerate, _patched
    if _patched:
        return
    try:
        import google.genai.models as _mod
        _original_generate  = _mod.Models.generate_content
        _original_agenerate = _mod.AsyncModels.generate_content
        _mod.Models.generate_content      = _make_patched_generate(_original_generate,  is_async=False)
        _mod.AsyncModels.generate_content = _make_patched_generate(_original_agenerate, is_async=True)
        _patched = True
    except ImportError:  # pragma: no cover
        pass


def uninstall() -> None:
    global _patched
    if not _patched:
        return
    try:
        import google.genai.models as _mod
        if _original_generate:
            _mod.Models.generate_content      = _original_generate
        if _original_agenerate:
            _mod.AsyncModels.generate_content = _original_agenerate
        _patched = False
    except ImportError:  # pragma: no cover
        pass
