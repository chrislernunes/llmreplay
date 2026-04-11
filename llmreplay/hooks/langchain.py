"""LangChain callback handler — hooks into any LangChain/LangGraph/CrewAI run."""
from __future__ import annotations

from typing import Any
from uuid import UUID


def get_handler():
    """Return a LangChain BaseCallbackHandler that records into the active session."""
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except ImportError:  # pragma: no cover
        try:
            from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
        except ImportError:  # pragma: no cover
            return None

    from llmreplay.core.context import current_session

    class LLMReplayHandler(BaseCallbackHandler):
        """Attach to any LangChain chain via `callbacks=[get_handler()]`."""

        def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
            s = current_session()
            if s:
                s.record_llm_request(
                    provider="langchain",
                    model=serialized.get("name", ""),
                    params=kwargs,
                    messages=[{"role": "user", "content": p} for p in prompts],
                )

        def on_llm_end(self, response: Any, **kwargs) -> None:
            s = current_session()
            if s:
                try:
                    raw = response.dict()
                except Exception:  # pragma: no cover
                    raw = {"generations": str(response)}
                s.record_llm_response(raw)

        def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
            s = current_session()
            if s:
                s.record_tool_call(serialized.get("name", "unknown"), input_str)

        def on_tool_end(self, output: str, **kwargs) -> None:
            s = current_session()
            if s:
                s.record_tool_result("unknown", output)

        def on_chain_error(self, error: Exception, **kwargs) -> None:
            s = current_session()
            if s:
                s.record_exception(type(error).__name__, str(error))

    return LLMReplayHandler()


def install() -> None:
    pass   # LangChain uses callbacks; no monkey-patching needed


def uninstall() -> None:
    pass
