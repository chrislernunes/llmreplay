# llmreplay

Deterministic replay debugger for LLM agents. Records LLM calls and tool executions to SQLite, replays from the log with no network calls.

```python
from llmreplay import record, replay

# Record
with record("my_run", seed=42):
    response = client.chat.completions.create(...)

# Replay — zero network calls
session = replay("my_run")
for event in session.events():
    print(event.step, event.kind, event.payload)
```

## Install

```bash
pip install llmreplay
```

Requirements: Python >= 3.10

## What gets recorded

- LLM requests/responses (OpenAI, Anthropic, Grok/xAI, Gemini)
- Tool calls/results (via `@record_tool` decorator)
- Random seeds (Python `random`, numpy)
- Exceptions

Events are stored in `~/.llmreplay/<run_id>.db`.

## CLI

```bash
llmreplay list                    # List recorded runs
llmreplay view my_run             # Show all events
llmreplay view my_run --step 42   # Jump to step
llmreplay cost my_run             # Cost breakdown
llmreplay export my_run --json    # Export bug report
llmreplay web my_run              # Launch timeline UI
```

## Features

**Auto-instrumentation** — OpenAI, Anthropic, Grok, Gemini, LangChain hooks install automatically within `record()` context.

**Tool mocking** — Record tool I/O with `@record_tool`, replay with `ToolMocker`:

```python
from llmreplay import ToolMocker, EventStore

mocker = ToolMocker()
mocker.load(EventStore("my_run"))

@mocker.mock(name="fetch_price")
def fetch_price(ticker: str) -> dict: ...  # returns recorded result
```

**Regression testing** — Run recorded traces against updated code:

```python
from llmreplay import RegressionSuite

suite = RegressionSuite()

@suite.case("run_001")
def check(original, session):
    return session.total_cost() <= original["total_cost_usd"] * 1.1

suite.run()
```

**Fork/branch** — Copy a trace up to a step for counterfactual debugging:

```python
from llmreplay import fork
new_store = fork("broken_run", "fixed_run", at_step=50)
```

**Fine-tuning export** — Export prompt/response pairs:

```python
from llmreplay import export_finetune_dataset
export_finetune_dataset(["run_001", "run_002"], "data.jsonl")
```

## License

MIT
