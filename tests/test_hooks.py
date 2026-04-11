"""
Hook interception tests — proves that the monkey-patch actually intercepts
LLM calls and records them, without requiring real API keys.

Strategy: mock the *underlying* SDK method to return a fake response object,
then install the llmreplay hook on top. If the hook fires correctly, the
session will contain LLM_REQUEST and LLM_RESPONSE events.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmreplay.core.context import record
from llmreplay.core.event import EventKind
from llmreplay.core.store import EventStore


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


# ── Fake response objects ──────────────────────────────────────────────────────

def _fake_openai_response(content: str = "hello", model: str = "gpt-4o"):
    """Mimics openai.types.chat.ChatCompletion.model_dump() shape."""
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return resp


def _fake_anthropic_response(content: str = "hello", model: str = "claude-sonnet-4-test"):
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "msg-test",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI — Sync
# ═══════════════════════════════════════════════════════════════════════════════

def test_openai_sync_hook_intercepts(tmp_dir):
    """
    Prove the OpenAI sync patch fires:
    record() → fake network call → events contain LLM_REQUEST + LLM_RESPONSE
    """
    import llmreplay.hooks.openai as oai_hook

    fake_response = _fake_openai_response("test answer", "gpt-4o")

    # Patch the *real* underlying method (what the hook wraps)
    original_create = MagicMock(return_value=fake_response)

    with patch("openai.resources.chat.completions.Completions.create", original_create):
        oai_hook.install()
        try:
            with record("oai_sync_intercept", base_dir=tmp_dir, seed=1) as session:
                # Simulate what a user's code does — call the SDK normally
                import openai
                client = openai.OpenAI(api_key="fake-key")
                # We call the patched class method directly to bypass instance binding
                import openai.resources.chat.completions as _mod
                _mod.Completions.create(
                    client.chat.completions,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hello"}],
                )
        finally:
            oai_hook.uninstall()

    store  = EventStore("oai_sync_intercept", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]

    assert EventKind.LLM_REQUEST  in kinds, "LLM_REQUEST not recorded"
    assert EventKind.LLM_RESPONSE in kinds, "LLM_RESPONSE not recorded"

    req = next(e for e in events if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["provider"] == "openai"
    assert req.payload["model"]    == "gpt-4o"

    resp = next(e for e in events if e.kind == EventKind.LLM_RESPONSE)
    assert resp.payload["raw"]["choices"][0]["message"]["content"] == "test answer"


def test_openai_sync_no_session_passthrough(tmp_dir):
    """When outside a record() context, the patch must pass through transparently."""
    import llmreplay.hooks.openai as oai_hook

    fake_response = _fake_openai_response("passthrough")
    original_create = MagicMock(return_value=fake_response)

    with patch("openai.resources.chat.completions.Completions.create", original_create):
        oai_hook.install()
        try:
            import openai.resources.chat.completions as _mod
            import openai
            client = openai.OpenAI(api_key="fake")
            result = _mod.Completions.create(
                client.chat.completions,
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            oai_hook.uninstall()

    # original must have been called exactly once
    original_create.assert_called_once()
    assert result is fake_response


def test_openai_stream_raises_inside_record(tmp_dir):
    """stream=True inside a record() context must raise immediately."""
    import llmreplay.hooks.openai as oai_hook
    original_create = MagicMock()

    with patch("openai.resources.chat.completions.Completions.create", original_create):
        oai_hook.install()
        try:
            with pytest.raises(RuntimeError, match="streaming"):
                with record("oai_stream_guard", base_dir=tmp_dir, seed=0):
                    import openai
                    import openai.resources.chat.completions as _mod
                    client = openai.OpenAI(api_key="fake")
                    _mod.Completions.create(
                        client.chat.completions,
                        model="gpt-4o",
                        messages=[],
                        stream=True,
                    )
        finally:
            oai_hook.uninstall()

    # Underlying must NOT have been called when streaming is blocked
    original_create.assert_not_called()


def test_openai_seed_injected(tmp_dir):
    """The hook must inject the session seed into kwargs so OpenAI sampling is pinned."""
    import llmreplay.hooks.openai as oai_hook

    captured_kwargs: dict = {}

    def capturing_create(self, **kwargs):
        captured_kwargs.update(kwargs)
        return _fake_openai_response()

    with patch("openai.resources.chat.completions.Completions.create", capturing_create):
        oai_hook.install()
        try:
            with record("oai_seed_inject", base_dir=tmp_dir, seed=12345) as session:
                import openai
                import openai.resources.chat.completions as _mod
                client = openai.OpenAI(api_key="fake")
                _mod.Completions.create(
                    client.chat.completions,
                    model="gpt-4o-mini",
                    messages=[],
                )
        finally:
            oai_hook.uninstall()

    assert "seed" in captured_kwargs, "seed not injected"
    assert captured_kwargs["seed"] == 12345


def test_openai_cost_estimated(tmp_dir):
    """LLM_RESPONSE must contain a non-zero cost estimate for gpt-4o."""
    import llmreplay.hooks.openai as oai_hook

    fake_response = _fake_openai_response("answer", "gpt-4o")
    with patch("openai.resources.chat.completions.Completions.create",
               MagicMock(return_value=fake_response)):
        oai_hook.install()
        try:
            with record("oai_cost", base_dir=tmp_dir, seed=0) as session:
                import openai
                import openai.resources.chat.completions as _mod
                client = openai.OpenAI(api_key="fake")
                _mod.Completions.create(
                    client.chat.completions,
                    model="gpt-4o",
                    messages=[],
                )
        finally:
            oai_hook.uninstall()

    store = EventStore("oai_cost", tmp_dir, read_only=True)
    resp  = next(e for e in store.iter_from() if e.kind == EventKind.LLM_RESPONSE)
    assert resp.payload["cost_usd"] > 0, "cost_usd should be positive for gpt-4o"


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI — Async
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_openai_async_hook_intercepts(tmp_dir):
    """Async path: AsyncCompletions.create must be intercepted."""
    import llmreplay.hooks.openai as oai_hook

    fake_response = _fake_openai_response("async answer", "gpt-4o")
    async_mock     = AsyncMock(return_value=fake_response)

    with patch("openai.resources.chat.completions.AsyncCompletions.create", async_mock):
        oai_hook.install()
        try:
            with record("oai_async_intercept", base_dir=tmp_dir, seed=2) as session:
                import openai
                import openai.resources.chat.completions as _mod
                client = openai.AsyncOpenAI(api_key="fake")
                await _mod.AsyncCompletions.create(
                    client.chat.completions,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "async hello"}],
                )
        finally:
            oai_hook.uninstall()

    store  = EventStore("oai_async_intercept", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]
    assert EventKind.LLM_REQUEST  in kinds
    assert EventKind.LLM_RESPONSE in kinds


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic
# ═══════════════════════════════════════════════════════════════════════════════

def test_anthropic_sync_hook_intercepts(tmp_dir):
    import llmreplay.hooks.anthropic as ant_hook

    fake_response = _fake_anthropic_response("claude says hi", "claude-sonnet-4-test")
    original_create = MagicMock(return_value=fake_response)

    with patch("anthropic.resources.messages.Messages.create", original_create):
        ant_hook.install()
        try:
            with record("ant_sync_intercept", base_dir=tmp_dir, seed=3) as session:
                import anthropic
                import anthropic.resources.messages as _mod
                client = anthropic.Anthropic(api_key="fake")
                _mod.Messages.create(
                    client.messages,
                    model="claude-sonnet-4-test",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "hi claude"}],
                )
        finally:
            ant_hook.uninstall()

    store  = EventStore("ant_sync_intercept", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]
    assert EventKind.LLM_REQUEST  in kinds
    assert EventKind.LLM_RESPONSE in kinds

    req = next(e for e in events if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["provider"] == "anthropic"


def test_anthropic_stream_raises_inside_record(tmp_dir):
    import llmreplay.hooks.anthropic as ant_hook

    with patch("anthropic.resources.messages.Messages.create", MagicMock()):
        ant_hook.install()
        try:
            with pytest.raises(RuntimeError, match="streaming"):
                with record("ant_stream_guard", base_dir=tmp_dir, seed=0):
                    import anthropic
                    import anthropic.resources.messages as _mod
                    client = anthropic.Anthropic(api_key="fake")
                    _mod.Messages.create(
                        client.messages,
                        model="claude-sonnet-4-test",
                        max_tokens=100,
                        messages=[],
                        stream=True,
                    )
        finally:
            ant_hook.uninstall()


@pytest.mark.asyncio
async def test_anthropic_async_hook_intercepts(tmp_dir):
    import llmreplay.hooks.anthropic as ant_hook

    fake_response = _fake_anthropic_response("async claude", "claude-sonnet-4-test")
    async_mock     = AsyncMock(return_value=fake_response)

    with patch("anthropic.resources.messages.AsyncMessages.create", async_mock):
        ant_hook.install()
        try:
            with record("ant_async_intercept", base_dir=tmp_dir, seed=4) as session:
                import anthropic
                import anthropic.resources.messages as _mod
                client = anthropic.AsyncAnthropic(api_key="fake")
                await _mod.AsyncMessages.create(
                    client.messages,
                    model="claude-sonnet-4-test",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "async hi"}],
                )
        finally:
            ant_hook.uninstall()

    store = EventStore("ant_async_intercept", tmp_dir, read_only=True)
    kinds = [e.kind for e in store.iter_from()]
    assert EventKind.LLM_REQUEST  in kinds
    assert EventKind.LLM_RESPONSE in kinds


# ═══════════════════════════════════════════════════════════════════════════════
# LangChain
# ═══════════════════════════════════════════════════════════════════════════════

def test_langchain_handler_llm_start(tmp_dir):
    """Handler.on_llm_start fires and records LLM_REQUEST."""
    from llmreplay.hooks.langchain import get_handler

    with record("lc_llm_start", base_dir=tmp_dir, seed=5) as session:
        handler = get_handler()
        if handler is None:
            pytest.skip("langchain not installed")

        handler.on_llm_start(
            serialized={"name": "ChatOpenAI", "id": ["ChatOpenAI"]},
            prompts=["What is the capital of France?"],
        )

    store  = EventStore("lc_llm_start", tmp_dir, read_only=True)
    events = list(store.iter_from())
    req    = next((e for e in events if e.kind == EventKind.LLM_REQUEST), None)
    assert req is not None
    assert req.payload["provider"] == "langchain"
    assert req.payload["model"]    == "ChatOpenAI"
    assert req.payload["messages"][0]["content"] == "What is the capital of France?"


def test_langchain_handler_llm_end(tmp_dir):
    """Handler.on_llm_end records LLM_RESPONSE."""
    from llmreplay.hooks.langchain import get_handler

    with record("lc_llm_end", base_dir=tmp_dir, seed=6) as session:
        handler = get_handler()
        if handler is None:
            pytest.skip("langchain not installed")

        handler.on_llm_start({"name": "ChatOpenAI", "id": ["ChatOpenAI"]}, ["Hello"])

        mock_response = MagicMock()
        mock_response.dict.return_value = {
            "generations": [[{"text": "Paris"}]],
        }
        mock_response.llm_output = {"token_usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        handler.on_llm_end(mock_response)

    store  = EventStore("lc_llm_end", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]
    assert EventKind.LLM_REQUEST  in kinds
    assert EventKind.LLM_RESPONSE in kinds


def test_langchain_handler_tool_roundtrip(tmp_dir):
    """Tool call + result both recorded via the callback handler."""
    from llmreplay.hooks.langchain import get_handler

    with record("lc_tool_roundtrip", base_dir=tmp_dir, seed=7) as session:
        handler = get_handler()
        if handler is None:
            pytest.skip("langchain not installed")

        handler.on_tool_start({"name": "web_search"}, '{"query": "nifty 50"}')
        handler.on_tool_end('{"results": ["PE: 22.4"]}')

    store  = EventStore("lc_tool_roundtrip", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]
    assert EventKind.TOOL_CALL   in kinds
    assert EventKind.TOOL_RESULT in kinds


def test_langchain_handler_exception(tmp_dir):
    """Chain error recorded as EXCEPTION event."""
    from llmreplay.hooks.langchain import get_handler

    with record("lc_exception", base_dir=tmp_dir, seed=8) as session:
        handler = get_handler()
        if handler is None:
            pytest.skip("langchain not installed")

        handler.on_chain_error(ValueError("chain broke"))

    store  = EventStore("lc_exception", tmp_dir, read_only=True)
    events = list(store.iter_from())
    exc    = next((e for e in events if e.kind == EventKind.EXCEPTION), None)
    assert exc is not None
    assert exc.payload["exc_type"] == "ValueError"


def test_langchain_handler_noop_outside_session():
    """Handler must silently do nothing when no record() session is active."""
    from llmreplay.hooks.langchain import get_handler
    handler = get_handler()
    if handler is None:
        pytest.skip("langchain not installed")
    # These must not raise even without an active session
    handler.on_llm_start({"name": "X", "id": ["X"]}, ["hi"])
    mock_resp = MagicMock()
    mock_resp.dict.return_value = {}
    mock_resp.llm_output = {}
    handler.on_llm_end(mock_resp)
    handler.on_tool_start({"name": "t"}, "input")
    handler.on_tool_end("output")


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_anthropic_no_session_passthrough(tmp_dir):
    """Outside record() the Anthropic hook must pass through transparently."""
    import llmreplay.hooks.anthropic as ant_hook
    fake_response = _fake_anthropic_response("passthrough")
    original_create = MagicMock(return_value=fake_response)

    with patch("anthropic.resources.messages.Messages.create", original_create):
        ant_hook.install()
        try:
            import anthropic.resources.messages as _mod
            import anthropic
            client = anthropic.Anthropic(api_key="fake")
            result = _mod.Messages.create(
                client.messages,
                model="claude-sonnet-4-test",
                max_tokens=100,
                messages=[],
            )
        finally:
            ant_hook.uninstall()

    original_create.assert_called_once()
    assert result is fake_response


def test_anthropic_cost_estimated(tmp_dir):
    """LLM_RESPONSE for a known Anthropic model has positive cost_usd."""
    import llmreplay.hooks.anthropic as ant_hook

    fake_response = _fake_anthropic_response("cost check", "claude-3-5-sonnet-20241022")
    original_create = MagicMock(return_value=fake_response)

    with patch("anthropic.resources.messages.Messages.create", original_create):
        ant_hook.install()
        try:
            with record("ant_cost", base_dir=tmp_dir, seed=0) as session:
                import anthropic.resources.messages as _mod
                import anthropic
                client = anthropic.Anthropic(api_key="fake")
                _mod.Messages.create(
                    client.messages,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=100,
                    messages=[],
                )
        finally:
            ant_hook.uninstall()

    store = EventStore("ant_cost", tmp_dir, read_only=True)
    resp  = next(e for e in store.iter_from() if e.kind == EventKind.LLM_RESPONSE)
    assert resp.payload["cost_usd"] > 0


def test_anthropic_unknown_model_zero_cost():
    """Unknown Anthropic model → zero cost, no crash."""
    from llmreplay.hooks.anthropic import _estimate_cost
    assert _estimate_cost("claude-99-future", {"input_tokens": 100, "output_tokens": 50}) == 0.0


def test_anthropic_none_usage_zero_cost():
    """None usage → zero cost."""
    from llmreplay.hooks.anthropic import _estimate_cost
    assert _estimate_cost("claude-3-5-sonnet-20241022", None) == 0.0
