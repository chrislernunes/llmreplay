"""Central registry — install/uninstall all provider hooks."""
from __future__ import annotations

from llmreplay.hooks import openai as _openai
from llmreplay.hooks import anthropic as _anthropic
from llmreplay.hooks import langchain as _langchain
from llmreplay.hooks import grok as _grok
from llmreplay.hooks import gemini as _gemini


_HOOKS = [_openai, _anthropic, _langchain, _grok, _gemini]


def install_all() -> None:
    for h in _HOOKS:
        h.install()


def uninstall_all() -> None:
    for h in _HOOKS:
        h.uninstall()


# Re-export the LangChain handler for direct use
from llmreplay.hooks.langchain import get_handler as langchain_handler  # noqa: E402
