"""
Tool side-effect mocking.

Usage (recording)::

    @record_tool
    def send_email(to: str, body: str) -> dict:
        return smtp_send(to, body)   # real call — result recorded

Usage (replay)::

    @mock_tool("send_email")
    def send_email(to: str, body: str) -> dict:
        ...   # body never executed; recorded result returned instead
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from llmreplay.core.context import current_session
from llmreplay.core.event import EventKind


def record_tool(fn: Callable | None = None, *, name: str | None = None):
    """Decorator: records inputs + output of a tool call into the active session."""
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            session = current_session()
            inputs = {"args": args, "kwargs": kwargs}
            if session:
                session.record_tool_call(tool_name, inputs)
            result = await func(*args, **kwargs)
            if session:
                session.record_tool_result(tool_name, result)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            session = current_session()
            inputs = {"args": args, "kwargs": kwargs}
            if session:
                session.record_tool_call(tool_name, inputs)
            result = func(*args, **kwargs)
            if session:
                session.record_tool_result(tool_name, result)
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator(fn) if fn is not None else decorator


class ToolMocker:
    """
    Replay-time tool mocker.
    Call .load(store) to preload all TOOL_RESULT events from a recorded run,
    then decorate functions with .mock(name) to return those results.
    """

    def __init__(self):
        from collections import defaultdict, deque
        self._queues: dict[str, Any] = defaultdict(deque)

    def load(self, store) -> None:
        from llmreplay.core.event import EventKind
        for ev in store.iter_from():
            if ev.kind == EventKind.TOOL_RESULT:
                self._queues[ev.payload["name"]].append(ev.payload["result"])

    def mock(self, fn: Callable | None = None, *, name: str | None = None):
        """Decorator: replaces fn body with the next queued recorded result."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return self._pop(tool_name)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._pop(tool_name)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator(fn) if fn is not None else decorator

    def remaining(self, name: str) -> int:
        """How many recorded results are left for a given tool name."""
        return len(self._queues.get(name, []))

    def assert_exhausted(self, name: str) -> None:
        """Assert that all recorded results were consumed — no leftovers."""
        n = self.remaining(name)
        if n:
            raise AssertionError(
                f"ToolMocker: {n} unconsumed result(s) for tool '{name}'. "
                "Your replay called it fewer times than the original run."
            )

    def _pop(self, name: str) -> Any:
        q = self._queues.get(name)
        if not q:
            raise RuntimeError(
                f"ToolMocker: no recorded result for tool '{name}'. "
                "Either the tool was never recorded or the queue is exhausted."
            )
        return q.popleft()