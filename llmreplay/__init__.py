"""
llmreplay — deterministic replay debugger for LLM agents.

Record a run with automatic SDK interception::

    from llmreplay import record, replay

    with record("my_run") as session:
        result = my_agent.run(query)      # sync or async agent code

Read back the recorded trace::

    session = replay("my_run")
    for event in session.events():
        print(event.step, event.kind, event.payload)

    print(f"Total cost: ${session.total_cost():.4f}")
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

__version__ = "0.1.2"

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
