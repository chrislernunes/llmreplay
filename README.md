# llmreplay

Deterministic replay layer for LLM-driven systems.

---

## Overview

LLMReplay is a lightweight framework for capturing, replaying, and testing LLM interactions.

It converts non-deterministic LLM behavior into reproducible system behavior, enabling reliable debugging and testing.

---

## Problem

LLM applications are difficult to test because they are:

- Non-deterministic by design  
- Dependent on external APIs  
- Hard to reproduce across runs  
- Fragile in CI environments  
- Difficult to debug historically  

This leads to unreliable regression testing and unstable evaluation pipelines.

---

## Solution

LLMReplay introduces a replay abstraction layer for LLM systems.

It enables you to:

- Capture real LLM executions
- Store structured interaction traces
- Replay executions deterministically
- Remove dependency on live model calls during tests

---

## Features

- Request/response capture layer  
- Deterministic replay engine  
- Tool-call mocking support  
- Snapshot-based testing workflow  
- CI-safe execution mode  
- Minimal integration overhead  

---

## Architecture

LLMReplay operates in two primary modes:

### Record Mode

Captures live execution traces from your LLM application, including:

- Inputs
- Outputs
- Tool calls (if applicable)
- Execution metadata

These traces are persisted for later reuse.

---

### Replay Mode

Replays stored traces without invoking external LLM APIs.

This ensures:

- Deterministic outputs
- Fast execution
- No network dependency
- Stable CI behavior

---

## Core Workflow

1. Run your application in **record mode**
2. Generate and store interaction traces
3. Run the same application in **replay mode**
4. Validate outputs against recorded snapshots

---

## Use Cases

- LLM application testing  
- Agent workflow debugging  
- Prompt regression testing  
- Evaluation pipelines  
- CI/CD validation for LLM systems  
- Tool-using agent simulation  

---

## Installation

```bash
pip install llmreplay
````

---

## Quick Start

```python
from llmreplay import ReplayClient

client = ReplayClient()

# Record mode
client.record()
run_your_llm_app()

# Replay mode
client.replay()
run_your_llm_app()
```

---

## Design Principle

> If it cannot be replayed, it cannot be tested.

---

## Roadmap

* Structured trace DAG visualization
* Multi-model replay support
* Latency and stochasticity simulation layer
* Distributed trace collection
* Web-based replay inspector
* Plugin system for tool mocking

---

## License

MIT
