"""
llmreplay — deterministic replay debugger for LLM agents.

    from llmreplay import record, replay

    with record("my_run"):
        result = await my_agent.run(query)

    session = replay("my_run", step=42)
"""
from llmreplay.core.context import record, current_session
from llmreplay.core.replay  import replay, fork
from llmreplay.report       import export_report
from llmreplay.mocking.tools import record_tool, ToolMocker
from llmreplay.regression   import RegressionSuite, export_finetune_dataset

try:
    from llmreplay.hooks.langchain import get_handler as langchain_handler
except ImportError:  # pragma: no cover
    langchain_handler = None  # type: ignore  # pragma: no cover

__version__ = "0.1.0"

__all__ = [
    "record",
    "replay",
    "fork",
    "current_session",
    "export_report",
    "record_tool",
    "ToolMocker",
    "RegressionSuite",
    "export_finetune_dataset",
    "langchain_handler",
]
