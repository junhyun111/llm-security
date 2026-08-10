from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class SourceSpan:
    file: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0


@dataclass(slots=True)
class CallSite:
    call_id: str
    callee: str
    arguments: list[str]
    assigned_to: str | None
    span: SourceSpan
    text: str


@dataclass(slots=True)
class Assignment:
    assignment_id: str
    target: str
    expression: str
    defs: set[str]
    uses: set[str]
    span: SourceSpan
    text: str


@dataclass(slots=True)
class Condition:
    condition_id: str
    expression: str
    symbols: set[str]
    operator: str | None
    span: SourceSpan
    text: str


@dataclass(slots=True)
class MemoryAccess:
    access_id: str
    base: str
    index: str | None
    kind: str
    span: SourceSpan
    text: str


@dataclass(slots=True)
class FunctionIR:
    name: str
    file: str
    line_start: int
    line_end: int
    parameters: list[str]
    calls: list[CallSite]
    assignments: list[Assignment]
    conditions: list[Condition]
    memory_accesses: list[MemoryAccess]
    returns: list[SourceSpan]
    code: str


@dataclass(slots=True)
class ProgramIR:
    functions: dict[str, FunctionIR] = field(default_factory=dict)
    callers: dict[str, set[str]] = field(default_factory=dict)
    callees: dict[str, set[str]] = field(default_factory=dict)
