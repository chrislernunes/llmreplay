"""Immutable event schema — every recorded fact lives here."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    LLM_REQUEST   = "llm_request"
    LLM_RESPONSE  = "llm_response"
    TOOL_CALL     = "tool_call"
    TOOL_RESULT   = "tool_result"
    MEMORY_READ   = "memory_read"
    MEMORY_WRITE  = "memory_write"
    RANDOM_SEED   = "random_seed"
    EXCEPTION     = "exception"
    METADATA      = "metadata"
    # BRANCH is emitted by fork() to mark the divergence point in a forked run
    BRANCH        = "branch"


@dataclass
class Event:
    run_id:    str
    step:      int
    kind:      EventKind
    payload:   dict[str, Any]
    ts:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            run_id=d["run_id"],
            step=d["step"],
            kind=EventKind(d["kind"]),
            payload=d["payload"],
            ts=d["ts"],
        )
