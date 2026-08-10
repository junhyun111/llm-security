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
class ExpressionInfo:
    text: str
    symbols: set[str]
    operators: set[str]
    cast_types: list[str]
    has_sizeof: bool = False


@dataclass(slots=True)
class CallSite:
    call_id: str
    callee: str
    arguments: list[str]
    assigned_to: str | None
    span: SourceSpan
    text: str
    argument_symbols: list[set[str]] = field(default_factory=list)
    argument_info: list[ExpressionInfo] = field(default_factory=list)


@dataclass(slots=True)
class Assignment:
    assignment_id: str
    target: str
    expression: str
    defs: set[str]
    uses: set[str]
    span: SourceSpan
    text: str
    expression_info: ExpressionInfo | None = None


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
    base_symbols: set[str] = field(default_factory=set)
    index_symbols: set[str] = field(default_factory=set)


@dataclass(slots=True)
class StatementIR:
    statement_id: str
    kind: str
    span: SourceSpan
    text: str
    defs: set[str]
    uses: set[str]
    assignment_ids: list[str] = field(default_factory=list)
    call_ids: list[str] = field(default_factory=list)
    parent_control_id: str | None = None
    control_branch: str | None = None


@dataclass(slots=True)
class ControlRegion:
    control_id: str
    kind: str
    condition_id: str
    span: SourceSpan
    body_span: SourceSpan
    alternative_span: SourceSpan | None = None
    parent_control_id: str | None = None
    control_branch: str | None = None
    initializer: StatementIR | None = None
    update: StatementIR | None = None


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
    statements: list[StatementIR] = field(default_factory=list)
    controls: list[ControlRegion] = field(default_factory=list)
    unsupported_constructs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProgramIR:
    functions: dict[str, FunctionIR] = field(default_factory=dict)
    callers: dict[str, set[str]] = field(default_factory=dict)
    callees: dict[str, set[str]] = field(default_factory=dict)
