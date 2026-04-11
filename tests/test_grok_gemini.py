"""
Grok (xAI) and Gemini hook tests.

Grok: xAI is OpenAI-compatible. Detection is inside the OpenAI hook via
base_url. Tests exercise the openai hook with an xAI-pointed client.

Gemini: patches google.generativeai.GenerativeModel.generate_content.
No real API keys required — all SDK methods are mocked at the class level.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmreplay.core.context import record
from llmreplay.core.event import EventKind
from llmreplay.core.replay import replay, ReplayValidator
from llmreplay.core.store import EventStore


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fake_openai_response(content: str = "grok says hi", model: str = "grok-3"):
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "chatcmpl-grok-test",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }
    return resp


def _fake_gemini_response(text: str = "gemini says hi"):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 12
    usage.candidates_token_count = 6
    resp.usage_metadata = usage
    return resp


def _xai_client():
    """Real openai.OpenAI pointed at api.x.ai — triggers Grok detection."""
    import openai
    return openai.OpenAI(api_key="fake-xai-key", base_url="https://api.x.ai/v1")


def _xai_async_client():
    import openai
    return openai.AsyncOpenAI(api_key="fake-xai-key", base_url="https://api.x.ai/v1")


def _openai_client():
    """Standard OpenAI client — must NOT be labelled as grok."""
    import openai
    return openai.OpenAI(api_key="fake-openai-key")


# ══════════════════════════════════════════════════════════════════════════════
# GROK — architecture
# ══════════════════════════════════════════════════════════════════════════════

def test_grok_shim_install_uninstall_are_noop():
    """grok.install/uninstall are no-ops — Grok is handled by the OpenAI hook."""
    import llmreplay.hooks.grok as grok_hook
    import openai
    before = openai.resources.chat.completions.Completions.create
    grok_hook.install()
    after_install = openai.resources.chat.completions.Completions.create
    grok_hook.uninstall()
    after_uninstall = openai.resources.chat.completions.Completions.create
    assert before is after_install is after_uninstall


def test_grok_cost_table_positive():
    """Every entry in the Grok cost table produces positive cost."""
    from llmreplay.hooks.grok import _COST_TABLE, _estimate_cost
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    for model in _COST_TABLE:
        assert _estimate_cost(model, usage) > 0, f"zero cost for {model}"


def test_grok_unknown_model_zero_cost():
    """Unknown model → zero cost, no crash."""
    from llmreplay.hooks.grok import _estimate_cost
    assert _estimate_cost("grok-99-future", {"prompt_tokens": 100, "completion_tokens": 50}) == 0.0


def test_grok_empty_usage_zero_cost():
    """Empty usage dict → zero cost."""
    from llmreplay.hooks.grok import _estimate_cost
    assert _estimate_cost("grok-3", {}) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# GROK via OpenAI hook — Sync
# ══════════════════════════════════════════════════════════════════════════════

def test_grok_sync_provider_label(tmp_dir):
    """xAI client → LLM_REQUEST recorded with provider='grok'."""
    import llmreplay.hooks.openai as oai_hook
    client    = _xai_client()
    fake_resp = _fake_openai_response("42", "grok-3")

    def fake_create(self, **kwargs):
        return fake_resp

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        oai_hook.install()
        try:
            with record("grok_provider_label", base_dir=tmp_dir, seed=10):
                import openai.resources.chat.completions as _mod
                _mod.Completions.create(
                    client.chat.completions,
                    model="grok-3",
                    messages=[{"role": "user", "content": "meaning of life?"}],
                )
        finally:
            oai_hook.uninstall()

    store  = EventStore("grok_provider_label", tmp_dir, read_only=True)
    events = list(store.iter_from())
    req    = next(e for e in events if e.kind == EventKind.LLM_REQUEST)
    resp   = next(e for e in events if e.kind == EventKind.LLM_RESPONSE)

    assert req.payload["provider"] == "grok"
    assert req.payload["model"]    == "grok-3"
    assert resp.payload["raw"]["choices"][0]["message"]["content"] == "42"


def test_openai_client_still_labelled_openai(tmp_dir):
    """Standard OpenAI client must still be labelled 'openai', not 'grok'."""
    import llmreplay.hooks.openai as oai_hook
    client    = _openai_client()
    fake_resp = _fake_openai_response("hello", "gpt-4o")

    def fake_create(self, **kwargs):
        return fake_resp

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        oai_hook.install()
        try:
            with record("openai_label_check", base_dir=tmp_dir, seed=11):
                import openai.resources.chat.completions as _mod
                _mod.Completions.create(
                    client.chat.completions,
                    model="gpt-4o",
                    messages=[],
                )
        finally:
            oai_hook.uninstall()

    store = EventStore("openai_label_check", tmp_dir, read_only=True)
    req   = next(e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["provider"] == "openai"


def test_grok_cost_estimated_via_openai_hook(tmp_dir):
    """grok-3 cost is non-zero when recorded through the OpenAI hook."""
    import llmreplay.hooks.openai as oai_hook
    client    = _xai_client()
    fake_resp = _fake_openai_response("answer", "grok-3")

    def fake_create(self, **kwargs):
        return fake_resp

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        oai_hook.install()
        try:
            with record("grok_cost_check", base_dir=tmp_dir, seed=0):
                import openai.resources.chat.completions as _mod
                _mod.Completions.create(
                    client.chat.completions, model="grok-3", messages=[]
                )
        finally:
            oai_hook.uninstall()

    store = EventStore("grok_cost_check", tmp_dir, read_only=True)
    resp  = next(e for e in store.iter_from() if e.kind == EventKind.LLM_RESPONSE)
    assert resp.payload["cost_usd"] > 0


def test_grok_stream_raises(tmp_dir):
    """stream=True inside record() raises RuntimeError regardless of provider."""
    import llmreplay.hooks.openai as oai_hook
    client = _xai_client()

    with patch("openai.resources.chat.completions.Completions.create", MagicMock()):
        oai_hook.install()
        try:
            with pytest.raises(RuntimeError, match="streaming"):
                with record("grok_stream_guard", base_dir=tmp_dir, seed=0):
                    import openai.resources.chat.completions as _mod
                    _mod.Completions.create(
                        client.chat.completions, model="grok-3", messages=[], stream=True
                    )
        finally:
            oai_hook.uninstall()


def test_grok_parallel_sessions_no_bleed(tmp_dir):
    """Two sequential sessions have isolated event stores."""
    with record("grok_par_a", base_dir=tmp_dir, seed=30) as s:
        s.record_llm_request("grok", "grok-3", {}, [])
        s.record_llm_response({}, cost_usd=0.003)

    with record("grok_par_b", base_dir=tmp_dir, seed=31) as s:
        s.record_llm_request("grok", "grok-3-mini", {}, [])
        s.record_llm_response({}, cost_usd=0.0003)
        s.record_llm_request("grok", "grok-3-mini", {}, [])
        s.record_llm_response({}, cost_usd=0.0003)

    reqs_a = [e for e in EventStore("grok_par_a", tmp_dir, read_only=True).iter_from()
              if e.kind == EventKind.LLM_REQUEST]
    reqs_b = [e for e in EventStore("grok_par_b", tmp_dir, read_only=True).iter_from()
              if e.kind == EventKind.LLM_REQUEST]

    assert len(reqs_a) == 1
    assert len(reqs_b) == 2


def test_grok_replay_validator(tmp_dir):
    """ReplayValidator correctly sequences Grok events."""
    with record("grok_validator", base_dir=tmp_dir, seed=20) as s:
        s.record_llm_request("grok", "grok-3", {}, [{"role": "user", "content": "hi"}])
        s.record_llm_response({"raw": "ok"}, cost_usd=0.003)

    store = EventStore("grok_validator", tmp_dir, read_only=True)
    v = ReplayValidator(store, start=2)
    v.expect(EventKind.LLM_REQUEST, provider="grok", model="grok-3")
    v.expect(EventKind.LLM_RESPONSE)
    v.assert_done()


# ══════════════════════════════════════════════════════════════════════════════
# GROK — Async
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_grok_async_provider_label(tmp_dir):
    """Async xAI client → LLM_REQUEST provider='grok'."""
    import llmreplay.hooks.openai as oai_hook
    client    = _xai_async_client()
    fake_resp = _fake_openai_response("async grok", "grok-3-fast")
    async_mock = AsyncMock(return_value=fake_resp)

    with patch("openai.resources.chat.completions.AsyncCompletions.create", async_mock):
        oai_hook.install()
        try:
            with record("grok_async_label", base_dir=tmp_dir, seed=12):
                import openai.resources.chat.completions as _mod
                await _mod.AsyncCompletions.create(
                    client.chat.completions,
                    model="grok-3-fast",
                    messages=[{"role": "user", "content": "async hello grok"}],
                )
        finally:
            oai_hook.uninstall()

    store = EventStore("grok_async_label", tmp_dir, read_only=True)
    req   = next(e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["provider"] == "grok"
    assert req.payload["model"]    == "grok-3-fast"


@pytest.mark.asyncio
async def test_grok_async_stream_raises(tmp_dir):
    """Async stream=True inside record() raises RuntimeError."""
    import llmreplay.hooks.openai as oai_hook
    client = _xai_async_client()

    with patch("openai.resources.chat.completions.AsyncCompletions.create", AsyncMock()):
        oai_hook.install()
        try:
            with pytest.raises(RuntimeError, match="streaming"):
                with record("grok_async_stream", base_dir=tmp_dir, seed=0):
                    import openai.resources.chat.completions as _mod
                    await _mod.AsyncCompletions.create(
                        client.chat.completions, model="grok-3", messages=[], stream=True
                    )
        finally:
            oai_hook.uninstall()


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI — helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fake_genai_response(text: str = "gemini says hi"):
    """Fake google.genai GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count     = 12
    usage.candidates_token_count = 6
    resp.usage_metadata = usage
    return resp


def _genai_client():
    """Real google.genai.Client (no real API key needed — methods are patched)."""
    import google.genai as genai
    return genai.Client(api_key="fake-key")


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI — Sync
# ══════════════════════════════════════════════════════════════════════════════

def test_gemini_sync_intercepts(tmp_dir):
    """Gemini sync: generate_content patched → LLM_REQUEST + LLM_RESPONSE."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp = _fake_genai_response("quantum is weird")

    def fake_generate(self, *, model, contents, **kwargs):
        return fake_resp

    with patch("google.genai.models.Models.generate_content", fake_generate):
        gem_hook.install()
        try:
            with record("gemini_sync", base_dir=tmp_dir, seed=40):
                client = _genai_client()
                client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents="Explain quantum entanglement.",
                )
        finally:
            gem_hook.uninstall()

    store  = EventStore("gemini_sync", tmp_dir, read_only=True)
    events = list(store.iter_from())
    kinds  = [e.kind for e in events]
    assert EventKind.LLM_REQUEST  in kinds
    assert EventKind.LLM_RESPONSE in kinds

    req = next(e for e in events if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["provider"] == "gemini"
    assert req.payload["model"]    == "gemini-2.5-pro"
    assert req.payload["messages"][0]["content"] == "Explain quantum entanglement."

    resp = next(e for e in events if e.kind == EventKind.LLM_RESPONSE)
    assert resp.payload["raw"]["text"] == "quantum is weird"


def test_gemini_no_session_passthrough(tmp_dir):
    """Outside record() Gemini hook is transparent."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp  = _fake_genai_response("passthrough")
    underlying = MagicMock(return_value=fake_resp)

    with patch("google.genai.models.Models.generate_content", underlying):
        gem_hook.install()
        try:
            client = _genai_client()
            result = client.models.generate_content(
                model="gemini-2.5-flash", contents="hello"
            )
        finally:
            gem_hook.uninstall()

    underlying.assert_called_once()
    assert result is fake_resp


def test_gemini_cost_estimated(tmp_dir):
    """gemini-2.5-pro response carries positive cost_usd."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp = _fake_genai_response("cost test")

    def fake_generate(self, *, model, contents, **kwargs):
        return fake_resp

    with patch("google.genai.models.Models.generate_content", fake_generate):
        gem_hook.install()
        try:
            with record("gemini_cost", base_dir=tmp_dir, seed=0):
                client = _genai_client()
                client.models.generate_content(
                    model="gemini-2.5-pro", contents="test prompt"
                )
        finally:
            gem_hook.uninstall()

    store = EventStore("gemini_cost", tmp_dir, read_only=True)
    resp  = next(e for e in store.iter_from() if e.kind == EventKind.LLM_RESPONSE)
    assert resp.payload["cost_usd"] > 0


def test_gemini_cost_table_positive():
    """Every Gemini model in the cost table produces positive cost."""
    from llmreplay.hooks.gemini import _COST_TABLE, _estimate_cost
    usage = MagicMock()
    usage.prompt_token_count     = 1000
    usage.candidates_token_count = 500
    for model in _COST_TABLE:
        assert _estimate_cost(model, usage) > 0, f"zero cost for {model}"


def test_gemini_models_prefix_stripped():
    """Cost lookup works when model name has 'models/' prefix."""
    from llmreplay.hooks.gemini import _estimate_cost
    usage = MagicMock()
    usage.prompt_token_count     = 1000
    usage.candidates_token_count = 500
    assert _estimate_cost("models/gemini-2.5-pro", usage) == _estimate_cost("gemini-2.5-pro", usage) > 0


def test_gemini_unknown_model_zero_cost():
    """Unrecognised model → zero cost, no crash."""
    from llmreplay.hooks.gemini import _estimate_cost
    usage = MagicMock()
    usage.prompt_token_count = usage.candidates_token_count = 100
    assert _estimate_cost("gemini-99-ultra", usage) == 0.0


def test_gemini_none_usage_zero_cost():
    """None usage_metadata → zero cost gracefully."""
    from llmreplay.hooks.gemini import _estimate_cost
    assert _estimate_cost("gemini-2.5-pro", None) == 0.0


def test_gemini_string_contents_normalised(tmp_dir):
    """Plain string prompt → messages = [{role: user, content: ...}]."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp = _fake_genai_response("ok")

    def fake_generate(self, *, model, contents, **kwargs):
        return fake_resp

    with patch("google.genai.models.Models.generate_content", fake_generate):
        gem_hook.install()
        try:
            with record("gemini_str_contents", base_dir=tmp_dir, seed=41):
                _genai_client().models.generate_content(
                    model="gemini-2.5-flash", contents="a plain string"
                )
        finally:
            gem_hook.uninstall()

    store = EventStore("gemini_str_contents", tmp_dir, read_only=True)
    req   = next(e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["messages"][0] == {"role": "user", "content": "a plain string"}


def test_gemini_list_contents_normalised(tmp_dir):
    """List of strings → one message per item."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp = _fake_genai_response("ok")

    def fake_generate(self, *, model, contents, **kwargs):
        return fake_resp

    with patch("google.genai.models.Models.generate_content", fake_generate):
        gem_hook.install()
        try:
            with record("gemini_list_contents", base_dir=tmp_dir, seed=42):
                _genai_client().models.generate_content(
                    model="gemini-2.0-flash", contents=["part one", "part two"]
                )
        finally:
            gem_hook.uninstall()

    store = EventStore("gemini_list_contents", tmp_dir, read_only=True)
    req   = next(e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST)
    msgs  = req.payload["messages"]
    assert len(msgs) == 2
    assert msgs[0]["content"] == "part one"
    assert msgs[1]["content"] == "part two"


def test_gemini_content_object_normalised(tmp_dir):
    """Content objects with .role and .parts are normalised correctly."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp = _fake_genai_response("ok")

    part = MagicMock()
    part.text = "structured prompt"
    content_obj = MagicMock()
    content_obj.role  = "user"
    content_obj.parts = [part]

    def fake_generate(self, *, model, contents, **kwargs):
        return fake_resp

    with patch("google.genai.models.Models.generate_content", fake_generate):
        gem_hook.install()
        try:
            with record("gemini_content_obj", base_dir=tmp_dir, seed=60):
                _genai_client().models.generate_content(
                    model="gemini-2.5-pro", contents=[content_obj]
                )
        finally:
            gem_hook.uninstall()

    store = EventStore("gemini_content_obj", tmp_dir, read_only=True)
    req   = next(e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["messages"][0] == {"role": "user", "content": "structured prompt"}


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI — Async
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_gemini_async_intercepts(tmp_dir):
    """Async: AsyncModels.generate_content patched and events recorded."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp  = _fake_genai_response("async gemini")
    async_mock = AsyncMock(return_value=fake_resp)

    with patch("google.genai.models.AsyncModels.generate_content", async_mock):
        gem_hook.install()
        try:
            with record("gemini_async", base_dir=tmp_dir, seed=43):
                client = _genai_client()
                await client.aio.models.generate_content(
                    model="gemini-2.5-pro", contents="async prompt"
                )
        finally:
            gem_hook.uninstall()

    store = EventStore("gemini_async", tmp_dir, read_only=True)
    kinds = [e.kind for e in store.iter_from()]
    assert EventKind.LLM_REQUEST  in kinds
    assert EventKind.LLM_RESPONSE in kinds

    req = next(e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST)
    assert req.payload["provider"] == "gemini"
    assert req.payload["model"]    == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_gemini_async_no_session_passthrough():
    """Async outside record() → transparent passthrough."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp  = _fake_genai_response("passthrough")
    async_mock = AsyncMock(return_value=fake_resp)

    with patch("google.genai.models.AsyncModels.generate_content", async_mock):
        gem_hook.install()
        try:
            result = await _genai_client().aio.models.generate_content(
                model="gemini-2.5-flash", contents="hi"
            )
        finally:
            gem_hook.uninstall()

    async_mock.assert_called_once()
    assert result is fake_resp


@pytest.mark.asyncio
async def test_gemini_async_cost_estimated(tmp_dir):
    """Async Gemini response carries positive cost_usd."""
    import llmreplay.hooks.gemini as gem_hook
    fake_resp  = _fake_genai_response("async cost")
    async_mock = AsyncMock(return_value=fake_resp)

    with patch("google.genai.models.AsyncModels.generate_content", async_mock):
        gem_hook.install()
        try:
            with record("gemini_async_cost", base_dir=tmp_dir, seed=0):
                await _genai_client().aio.models.generate_content(
                    model="gemini-2.5-pro", contents="test"
                )
        finally:
            gem_hook.uninstall()

    store = EventStore("gemini_async_cost", tmp_dir, read_only=True)
    resp  = next(e for e in store.iter_from() if e.kind == EventKind.LLM_RESPONSE)
    assert resp.payload["cost_usd"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI — Replay determinism
# ══════════════════════════════════════════════════════════════════════════════

def test_gemini_replay_determinism(tmp_dir):
    """Replay 10× → identical event payloads every time."""
    with record("gemini_det", base_dir=tmp_dir, seed=50) as s:
        s.record_llm_request("gemini", "gemini-2.5-pro", {}, [{"role": "user", "content": "q"}])
        s.record_llm_response({"text": "answer"}, cost_usd=0.00125)

    store    = EventStore("gemini_det", tmp_dir, read_only=True)
    baseline = [(e.kind, e.payload) for e in store.iter_from()]
    for _ in range(10):
        rs      = replay("gemini_det", base_dir=tmp_dir)
        current = [(e.kind, e.payload) for e in rs.events()]
        assert current == baseline


def test_gemini_replay_validator(tmp_dir):
    """ReplayValidator sequences Gemini events correctly."""
    with record("gemini_val", base_dir=tmp_dir, seed=51) as s:
        s.record_llm_request("gemini", "gemini-2.5-flash", {}, [])
        s.record_llm_response({"text": "ok"}, cost_usd=0.0001)

    v = ReplayValidator(EventStore("gemini_val", tmp_dir, read_only=True), start=2)
    v.expect(EventKind.LLM_REQUEST, provider="gemini", model="gemini-2.5-flash")
    v.expect(EventKind.LLM_RESPONSE)
    v.assert_done()


def test_gemini_multi_turn(tmp_dir):
    """Multi-turn: 3 request/response pairs, correct step count and total cost."""
    with record("gemini_multi", base_dir=tmp_dir, seed=52) as s:
        for i in range(3):
            s.record_llm_request("gemini", "gemini-2.5-pro", {}, [{"role": "user", "content": f"turn {i}"}])
            s.record_llm_response({"text": f"resp {i}"}, cost_usd=0.001)

    rs   = replay("gemini_multi", base_dir=tmp_dir)
    reqs = [e for e in rs.events() if e.kind == EventKind.LLM_REQUEST]
    assert len(reqs) == 3
    assert rs.total_cost() == pytest.approx(0.003)


# ── OpenAI hook — _estimate_cost edge cases ───────────────────────────────────

def test_openai_hook_unknown_model_zero_cost():
    """_estimate_cost returns 0.0 for unrecognised model."""
    from llmreplay.hooks.openai import _estimate_cost
    assert _estimate_cost("gpt-99-ultra", {"prompt_tokens": 100, "completion_tokens": 50}) == 0.0


def test_openai_hook_empty_usage_zero_cost():
    """_estimate_cost returns 0.0 for empty usage dict."""
    from llmreplay.hooks.openai import _estimate_cost
    assert _estimate_cost("gpt-4o", {}) == 0.0


# ── CLI truncation branch ──────────────────────────────────────────────────────

def test_cli_view_truncates_large_payload(tmp_dir):
    """CLI view truncates payloads larger than 2000 chars."""
    from click.testing import CliRunner
    from llmreplay.cli import cli
    from llmreplay.core.context import record as _record

    big_payload = {"data": "x" * 3000}
    with _record("cli_trunc", base_dir=tmp_dir, seed=1) as s:
        s.emit(EventKind.METADATA, big_payload)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "cli_trunc", "--dir", str(tmp_dir)])
    assert result.exit_code == 0
    assert "truncated" in result.output


def test_cli_view_empty_run(tmp_dir):
    """CLI view on nonexistent run prints not found."""
    from click.testing import CliRunner
    from llmreplay.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "run_that_does_not_exist_abc", "--dir", str(tmp_dir)])
    assert result.exit_code == 0
    assert "not found" in result.output.lower()
