"""
Ground-truth proof tests — close the three validation gaps identified in the audit.

These tests prove:

1. test_real_api_interception_roundtrip
   The full SDK → hook → store → stub_client path works end-to-end.
   The stub client returns the *exact* recorded response with zero network calls,
   even when the underlying SDK method is completely removed after recording.

2. test_mutation_resistance_full_roundtrip
   Recording captures the actual execution state at record-time.
   Mutating tool logic, LLM prompts, *and* environment variables after recording
   has zero effect on what replay returns.

3. test_parallel_hook_interception_no_bleed
   Two async tasks each making patched SDK calls simultaneously produce
   fully isolated, correctly ordered event logs — no cross-session contamination
   at the hook level.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmreplay.core.context import record
from llmreplay.core.event import EventKind
from llmreplay.core.replay import replay, ReplayValidator, ReplayMismatchError
from llmreplay.core.store import EventStore
from llmreplay.mocking.tools import record_tool, ToolMocker


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Real API interception roundtrip
#    Proves: hook intercepts → store written → stub_client replays → zero network
# ═══════════════════════════════════════════════════════════════════════════════

def test_real_api_interception_roundtrip_openai(tmp_dir):
    """
    Full path: SDK call → hook intercepts → events stored → replay stub returns
    recorded response, even after the underlying SDK method is replaced with a
    bomb that raises if called.

    This is the ground-truth test that proves capture and replay are connected.
    """
    import llmreplay.hooks.openai as oai_hook

    EXPECTED_CONTENT = "The Nifty 50 closed at 22,500."
    fake_response = MagicMock()
    fake_response.model_dump.return_value = {
        "id": "chatcmpl-proof",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": EXPECTED_CONTENT}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }

    # ── Phase 1: Record via patched SDK ───────────────────────────────────────
    with patch("openai.resources.chat.completions.Completions.create",
               MagicMock(return_value=fake_response)):
        oai_hook.install()
        try:
            with record("proof_openai", base_dir=tmp_dir, seed=7) as session:
                import openai.resources.chat.completions as _mod
                import openai
                client = openai.OpenAI(api_key="fake")
                _mod.Completions.create(
                    client.chat.completions,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Nifty 50 close today?"}],
                )
        finally:
            oai_hook.uninstall()

    # ── Phase 2: Destroy the SDK — verify it raises if called directly ────────
    def _network_bomb(self, **kwargs):
        raise RuntimeError("Network call made during replay — this must never fire")

    with patch("openai.resources.chat.completions.Completions.create", _network_bomb):
        # ── Phase 3: Replay using stub_client only ────────────────────────────
        rs = replay("proof_openai", base_dir=tmp_dir)
        replayed = rs.stub_client.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "COMPLETELY DIFFERENT QUESTION"}],
        )

    assert replayed["choices"][0]["message"]["content"] == EXPECTED_CONTENT, (
        "stub_client must return the recorded response, not re-execute"
    )

    # ── Phase 4: Validate the stored event sequence ───────────────────────────
    store = EventStore("proof_openai", tmp_dir, read_only=True)
    v = ReplayValidator(store, start=2)  # skip seed + metadata
    req = v.expect(EventKind.LLM_REQUEST, provider="openai", model="gpt-4o")
    assert req.payload["messages"][0]["content"] == "Nifty 50 close today?"

    resp = v.expect(EventKind.LLM_RESPONSE)
    assert resp.payload["raw"]["choices"][0]["message"]["content"] == EXPECTED_CONTENT
    assert resp.payload["cost_usd"] > 0, "cost must be estimated for gpt-4o"
    v.assert_done()


def test_real_api_interception_roundtrip_anthropic(tmp_dir):
    """
    Same ground-truth test for the Anthropic hook path.
    """
    import llmreplay.hooks.anthropic as ant_hook

    EXPECTED_TEXT = "Dispersion is rich vs index vol."
    fake_response = MagicMock()
    fake_response.model_dump.return_value = {
        "id": "msg-proof",
        "model": "claude-sonnet-4-test",
        "content": [{"type": "text", "text": EXPECTED_TEXT}],
        "usage": {"input_tokens": 15, "output_tokens": 7},
    }

    with patch("anthropic.resources.messages.Messages.create",
               MagicMock(return_value=fake_response)):
        ant_hook.install()
        try:
            with record("proof_anthropic", base_dir=tmp_dir, seed=8) as session:
                import anthropic.resources.messages as _mod
                import anthropic
                client = anthropic.Anthropic(api_key="fake")
                _mod.Messages.create(
                    client.messages,
                    model="claude-sonnet-4-test",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "Is BankNifty dispersion rich?"}],
                )
        finally:
            ant_hook.uninstall()

    # Kill the network, verify replay still works
    def _bomb(self, **kwargs):
        raise RuntimeError("Network must not be called during replay")

    with patch("anthropic.resources.messages.Messages.create", _bomb):
        rs = replay("proof_anthropic", base_dir=tmp_dir)
        replayed = rs.stub_client.create(
            model="claude-sonnet-4-test",
            messages=[{"role": "user", "content": "DIFFERENT QUESTION"}],
        )

    assert replayed["content"][0]["text"] == EXPECTED_TEXT

    store = EventStore("proof_anthropic", tmp_dir, read_only=True)
    v = ReplayValidator(store, start=2)
    v.expect(EventKind.LLM_REQUEST, provider="anthropic")
    v.expect(EventKind.LLM_RESPONSE)
    v.assert_done()


@pytest.mark.asyncio
async def test_real_api_interception_roundtrip_async(tmp_dir):
    """
    Same proof for the async SDK path — AsyncCompletions.create.
    """
    import llmreplay.hooks.openai as oai_hook

    EXPECTED_CONTENT = "Async replay confirmed."
    fake_response = MagicMock()
    fake_response.model_dump.return_value = {
        "id": "chatcmpl-async-proof",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": EXPECTED_CONTENT}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    async_mock = AsyncMock(return_value=fake_response)

    with patch("openai.resources.chat.completions.AsyncCompletions.create", async_mock):
        oai_hook.install()
        try:
            with record("proof_async", base_dir=tmp_dir, seed=9) as session:
                import openai.resources.chat.completions as _mod
                import openai
                client = openai.AsyncOpenAI(api_key="fake")
                await _mod.AsyncCompletions.create(
                    client.chat.completions,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "async question"}],
                )
        finally:
            oai_hook.uninstall()

    # Replace async SDK with bomb
    async def _async_bomb(self, **kwargs):
        raise RuntimeError("Async network must not fire during replay")

    with patch("openai.resources.chat.completions.AsyncCompletions.create", _async_bomb):
        rs = replay("proof_async", base_dir=tmp_dir)
        # Async stub_client
        replayed = await rs.stub_client.acreate(model="gpt-4o", messages=[])

    assert replayed["choices"][0]["message"]["content"] == EXPECTED_CONTENT


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mutation resistance — full roundtrip
#    Proves: recorded state is immutable against tool + prompt + env mutations
# ═══════════════════════════════════════════════════════════════════════════════

def test_mutation_resistance_tool_plus_llm_plus_env(tmp_dir):
    """
    Record an agent run with:
      - a tool that returns a specific value
      - an LLM that returns a specific response
      - an env var baked into the run

    Then mutate ALL THREE and replay. Every replayed output must match the
    original recording — not the mutated values.

    This is a stronger mutation test than test_code_drift_* because it covers
    the full stack simultaneously.
    """
    ORIGINAL_PRICE   = 2847.30
    ORIGINAL_SIGNAL  = "BUY"
    ORIGINAL_ENV_VAL = "prod"

    # ── Record ────────────────────────────────────────────────────────────────
    os.environ["STRATEGY_ENV"] = ORIGINAL_ENV_VAL

    def fetch_price_v1(ticker: str) -> dict:
        return {"price": ORIGINAL_PRICE, "env": os.environ.get("STRATEGY_ENV")}

    with record("mutation_full", base_dir=tmp_dir, seed=42) as session:
        # Tool
        session.record_tool_call("fetch_price", {"ticker": "NIFTY"})
        result = fetch_price_v1("NIFTY")
        session.record_tool_result("fetch_price", result)

        # LLM response
        session.record_llm_request(
            "openai", "gpt-4o", {},
            [{"role": "user", "content": f"Price is {result['price']}. Signal?"}]
        )
        session.record_llm_response(
            {"choices": [{"message": {"content": ORIGINAL_SIGNAL}}]}, 0.001
        )

    # ── Mutate everything ─────────────────────────────────────────────────────
    os.environ["STRATEGY_ENV"] = "MUTATED_staging"  # mutate env

    def fetch_price_v2(ticker: str) -> dict:  # mutate tool — wrong data
        return {"price": 0.0, "env": "MUTATED", "source": "BROKEN"}

    # ── Replay ────────────────────────────────────────────────────────────────
    store  = EventStore("mutation_full", tmp_dir, read_only=True)
    mocker = ToolMocker()
    mocker.load(store)

    @mocker.mock(name="fetch_price")
    def fetch_price_replay(ticker: str) -> dict:
        return fetch_price_v2(ticker)  # body never runs

    replayed_tool = fetch_price_replay("NIFTY")
    assert replayed_tool["price"] == ORIGINAL_PRICE, (
        f"Tool replay must return recorded {ORIGINAL_PRICE}, got {replayed_tool['price']}"
    )
    assert replayed_tool.get("env") == ORIGINAL_ENV_VAL, (
        "Recorded env value must survive env mutation"
    )

    # Stub client also unaffected by env mutation
    rs = replay("mutation_full", base_dir=tmp_dir)
    replayed_llm = rs.stub_client.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "MUTATED PROMPT with price 0.0"}],
    )
    assert replayed_llm["choices"][0]["message"]["content"] == ORIGINAL_SIGNAL, (
        "LLM replay must return recorded SIGNAL, not re-execute against mutated prompt"
    )

    # Validate full event sequence
    v = ReplayValidator(store, start=2)
    v.expect(EventKind.TOOL_CALL,   name="fetch_price")
    v.expect(EventKind.TOOL_RESULT, name="fetch_price")
    v.expect(EventKind.LLM_REQUEST, provider="openai", model="gpt-4o")
    v.expect(EventKind.LLM_RESPONSE)
    v.assert_done()

    # Restore env
    os.environ["STRATEGY_ENV"] = ORIGINAL_ENV_VAL


def test_mutation_resistance_overwrite_guard(tmp_dir):
    """
    Attempting to re-record to the same run_id without overwrite=True must
    not corrupt the original store — old events are still readable.
    """
    with record("immutable_run", base_dir=tmp_dir, seed=1) as session:
        session.record_llm_response({"answer": "original"}, 0.001)

    original_count = EventStore("immutable_run", tmp_dir, read_only=True).count()

    # Re-record WITHOUT overwrite — should append to same store (or raise),
    # but either way the original events must survive.
    try:
        with record("immutable_run", base_dir=tmp_dir, seed=2) as session:
            session.record_llm_response({"answer": "intruder"}, 0.001)
    except Exception:
        pass  # append or conflict — we only care original survives

    store = EventStore("immutable_run", tmp_dir, read_only=True)
    first_resp = next(
        e for e in store.iter_from()
        if e.kind == EventKind.LLM_RESPONSE
    )
    assert first_resp.payload["raw"]["answer"] == "original", (
        "Original event must not be overwritten"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parallel hook interception — no bleed at the SDK patch level
#    Proves: concurrent sessions with patched SDK calls don't cross-contaminate
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_parallel_hook_interception_no_bleed(tmp_dir):
    """
    Two async tasks both make patched SDK calls simultaneously.
    Each task's hook must route events only to its own session/store.

    This is different from test_concurrent_sessions_isolated because here
    the events pass through the actual monkey-patched SDK path, not just
    direct session.record_* calls.
    """
    import llmreplay.hooks.openai as oai_hook

    def _make_response(content: str, task_id: str):
        r = MagicMock()
        r.model_dump.return_value = {
            "id": f"chatcmpl-{task_id}",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        return r

    responses = {
        "task_alpha": _make_response("ALPHA says: index vol cheap", "alpha"),
        "task_beta":  _make_response("BETA says: skew steep",       "beta"),
    }

    # Track which task's SDK call was served — ensures no cross-routing
    call_log: list[str] = []

    def _routing_create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        task_id = "task_alpha" if "ALPHA" in content else "task_beta"
        call_log.append(task_id)
        return responses[task_id]

    with patch("openai.resources.chat.completions.Completions.create", _routing_create):
        oai_hook.install()
        try:
            async def run_alpha():
                with record("hook_bleed_alpha", base_dir=tmp_dir, seed=10) as session:
                    await asyncio.sleep(0.01)
                    import openai.resources.chat.completions as _mod
                    import openai
                    client = openai.OpenAI(api_key="fake")
                    _mod.Completions.create(
                        client.chat.completions,
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "ALPHA question"}],
                    )
                    await asyncio.sleep(0.01)

            async def run_beta():
                await asyncio.sleep(0.005)  # interleave with alpha
                with record("hook_bleed_beta", base_dir=tmp_dir, seed=20) as session:
                    import openai.resources.chat.completions as _mod
                    import openai
                    client = openai.OpenAI(api_key="fake")
                    _mod.Completions.create(
                        client.chat.completions,
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "BETA question"}],
                    )
                    await asyncio.sleep(0.02)

            await asyncio.gather(run_alpha(), run_beta())
        finally:
            oai_hook.uninstall()

    # ── Verify store isolation ─────────────────────────────────────────────────
    store_a = EventStore("hook_bleed_alpha", tmp_dir, read_only=True)
    store_b = EventStore("hook_bleed_beta",  tmp_dir, read_only=True)

    resps_a = [e for e in store_a.iter_from() if e.kind == EventKind.LLM_RESPONSE]
    resps_b = [e for e in store_b.iter_from() if e.kind == EventKind.LLM_RESPONSE]

    assert len(resps_a) == 1, f"alpha store must have exactly 1 LLM_RESPONSE, got {len(resps_a)}"
    assert len(resps_b) == 1, f"beta store must have exactly 1 LLM_RESPONSE, got {len(resps_b)}"

    content_a = resps_a[0].payload["raw"]["choices"][0]["message"]["content"]
    content_b = resps_b[0].payload["raw"]["choices"][0]["message"]["content"]

    assert "ALPHA" in content_a, f"alpha store contaminated with beta response: {content_a!r}"
    assert "BETA"  in content_b, f"beta store contaminated with alpha response: {content_b!r}"

    # ── Verify stub clients are also isolated ──────────────────────────────────
    rs_a = replay("hook_bleed_alpha", base_dir=tmp_dir)
    rs_b = replay("hook_bleed_beta",  base_dir=tmp_dir)

    replayed_a = rs_a.stub_client.create(model="gpt-4o-mini", messages=[])
    replayed_b = rs_b.stub_client.create(model="gpt-4o-mini", messages=[])

    assert "ALPHA" in replayed_a["choices"][0]["message"]["content"]
    assert "BETA"  in replayed_b["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_parallel_hook_interception_ordering_stable(tmp_dir):
    """
    A single session making N async-interleaved patched SDK calls must record
    them in call-dispatch order, not in completion order.

    Simulates a realistic multi-call agent loop where calls are dispatched
    sequentially but context switches occur between them.
    """
    import llmreplay.hooks.openai as oai_hook

    call_counter = 0

    def _ordered_create(self, **kwargs):
        nonlocal call_counter
        idx = call_counter
        call_counter += 1
        r = MagicMock()
        r.model_dump.return_value = {
            "id": f"chatcmpl-{idx}",
            "model": "gpt-4o",
            "choices": [{"message": {"content": f"response_{idx}"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3},
        }
        return r

    with patch("openai.resources.chat.completions.Completions.create", _ordered_create):
        oai_hook.install()
        try:
            with record("hook_ordering", base_dir=tmp_dir, seed=55) as session:
                import openai.resources.chat.completions as _mod
                import openai
                client = openai.OpenAI(api_key="fake")
                for i in range(6):
                    _mod.Completions.create(
                        client.chat.completions,
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"question_{i}"}],
                    )
                    await asyncio.sleep(0)  # yield between each call
        finally:
            oai_hook.uninstall()

    store = EventStore("hook_ordering", tmp_dir, read_only=True)
    reqs  = [e for e in store.iter_from() if e.kind == EventKind.LLM_REQUEST]
    steps = [e.step for e in reqs]

    assert steps == sorted(steps), "LLM_REQUEST steps must be monotonically increasing"
    assert len(reqs) == 6, f"must have 6 recorded requests, got {len(reqs)}"

    contents = [e.payload["messages"][0]["content"] for e in reqs]
    assert contents == [f"question_{i}" for i in range(6)], (
        f"Request order corrupted: {contents}"
    )

    # Stub client must return responses in recorded order
    rs = replay("hook_ordering", base_dir=tmp_dir)
    for i in range(6):
        r = rs.stub_client.create(model="gpt-4o", messages=[])
        assert r["choices"][0]["message"]["content"] == f"response_{i}", (
            f"Stub client response order wrong at position {i}"
        )
