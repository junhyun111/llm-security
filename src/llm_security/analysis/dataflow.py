from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .control_flow import ControlFlowGraph
from .ir import Assignment, CallSite, FunctionIR, StatementIR


@dataclass(slots=True, frozen=True)
class Definition:
    definition_id: str
    symbol: str
    node_id: str
    line: int | None
    kind: str


@dataclass(slots=True, frozen=True)
class DefUseEdge:
    symbol: str
    definition_id: str
    definition_node_id: str
    use_node_id: str
    definition_line: int | None
    use_line: int | None


@dataclass(slots=True)
class DataFlowGraph:
    definitions: dict[str, Definition]
    edges: list[DefUseEdge]
    reaching_in: dict[str, set[str]]
    reaching_out: dict[str, set[str]]
    by_symbol: dict[str, list[str]] = field(default_factory=dict)

    def definitions_reaching(
        self, node_id: str, symbol: str
    ) -> list[Definition]:
        return [
            self.definitions[definition_id]
            for definition_id in sorted(self.reaching_in.get(node_id, set()))
            if self.definitions[definition_id].symbol == symbol
        ]


@dataclass(slots=True, frozen=True)
class SliceStep:
    symbol: str
    from_node: str
    to_node: str
    depth: int


class ReachingDefinitionsAnalyzer:
    """Compute may-reaching definitions and intraprocedural def-use edges."""

    def analyze(
        self, function: FunctionIR, cfg: ControlFlowGraph
    ) -> DataFlowGraph:
        statements = _statement_index(function)
        calls = {call.call_id: call for call in function.calls}
        assignments = {
            assignment.assignment_id: assignment
            for assignment in function.assignments
        }
        definitions: dict[str, Definition] = {}
        generated: dict[str, set[str]] = {
            node_id: set() for node_id in cfg.nodes
        }

        for symbol in sorted(set(function.parameters)):
            definition = _definition(
                symbol=symbol,
                node_id=cfg.entry_id,
                line=function.line_start,
                kind="parameter",
            )
            definitions[definition.definition_id] = definition
            generated[cfg.entry_id].add(definition.definition_id)

        for node_id in sorted(cfg.nodes):
            if node_id in {cfg.entry_id, cfg.exit_id}:
                continue
            node = cfg.nodes[node_id]
            for symbol in sorted(node.defs):
                kind = _definition_kind(
                    node_id=node_id,
                    symbol=symbol,
                    statements=statements,
                    calls=calls,
                    assignments=assignments,
                )
                definition = _definition(
                    symbol=symbol,
                    node_id=node_id,
                    line=node.span.line_start if node.span is not None else None,
                    kind=kind,
                )
                definitions[definition.definition_id] = definition
                generated[node_id].add(definition.definition_id)

        by_symbol: dict[str, list[str]] = defaultdict(list)
        for definition_id, definition in definitions.items():
            by_symbol[definition.symbol].append(definition_id)
        normalized_by_symbol = {
            symbol: sorted(definition_ids)
            for symbol, definition_ids in sorted(by_symbol.items())
        }

        killed: dict[str, set[str]] = {}
        for node_id in cfg.nodes:
            generated_symbols = {
                definitions[definition_id].symbol
                for definition_id in generated[node_id]
            }
            killed[node_id] = {
                definition_id
                for symbol in generated_symbols
                for definition_id in normalized_by_symbol.get(symbol, [])
                if definition_id not in generated[node_id]
            }

        reaching_in = {node_id: set() for node_id in cfg.nodes}
        reaching_out = {
            node_id: set(generated[node_id]) for node_id in cfg.nodes
        }
        changed = True
        while changed:
            changed = False
            for node_id in sorted(cfg.nodes):
                incoming = {
                    definition_id
                    for predecessor in sorted(cfg.predecessors.get(node_id, set()))
                    for definition_id in reaching_out[predecessor]
                }
                outgoing = generated[node_id] | (incoming - killed[node_id])
                if incoming != reaching_in[node_id] or outgoing != reaching_out[node_id]:
                    reaching_in[node_id] = incoming
                    reaching_out[node_id] = outgoing
                    changed = True

        edges = [
            DefUseEdge(
                symbol=symbol,
                definition_id=definition_id,
                definition_node_id=definitions[definition_id].node_id,
                use_node_id=node_id,
                definition_line=definitions[definition_id].line,
                use_line=(
                    cfg.nodes[node_id].span.line_start
                    if cfg.nodes[node_id].span is not None
                    else None
                ),
            )
            for node_id in sorted(cfg.nodes)
            for symbol in sorted(cfg.nodes[node_id].uses)
            for definition_id in sorted(reaching_in[node_id])
            if definitions[definition_id].symbol == symbol
        ]
        edges.sort(
            key=lambda edge: (
                edge.use_node_id,
                edge.symbol,
                edge.definition_id,
            )
        )
        return DataFlowGraph(
            definitions=dict(sorted(definitions.items())),
            edges=edges,
            reaching_in={
                node_id: set(reaching_in[node_id]) for node_id in sorted(cfg.nodes)
            },
            reaching_out={
                node_id: set(reaching_out[node_id]) for node_id in sorted(cfg.nodes)
            },
            by_symbol=normalized_by_symbol,
        )


def backward_slice(
    function: FunctionIR,
    cfg: ControlFlowGraph,
    dataflow: DataFlowGraph,
    *,
    sink_node_id: str,
    symbols: set[str],
    max_depth: int = 8,
) -> list[SliceStep]:
    """Follow reaching definitions and their inputs backwards from a sink."""
    del function  # The explicit argument keeps this API aligned with FunctionAnalysis.
    if sink_node_id not in cfg.nodes:
        raise KeyError(f"unknown CFG sink node: {sink_node_id}")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    queue = deque(
        (sink_node_id, symbol, 0) for symbol in sorted(set(symbols))
    )
    visited: set[tuple[str, str]] = set()
    steps: set[SliceStep] = set()
    while queue:
        use_node_id, symbol, depth = queue.popleft()
        state = (use_node_id, symbol)
        if state in visited:
            continue
        visited.add(state)
        for definition in dataflow.definitions_reaching(use_node_id, symbol):
            steps.add(
                SliceStep(
                    symbol=symbol,
                    from_node=definition.node_id,
                    to_node=use_node_id,
                    depth=depth,
                )
            )
            if depth >= max_depth:
                continue
            definition_node = cfg.nodes.get(definition.node_id)
            if definition_node is None:
                continue
            for dependency in sorted(definition_node.uses):
                queue.append((definition.node_id, dependency, depth + 1))
    return sorted(
        steps,
        key=lambda step: (
            step.depth,
            step.symbol,
            step.from_node,
            step.to_node,
        ),
    )


def _definition(
    *, symbol: str, node_id: str, line: int | None, kind: str
) -> Definition:
    digest = hashlib.sha1(
        f"{node_id}:{symbol}:{kind}".encode("utf-8")
    ).hexdigest()[:12]
    return Definition(
        definition_id=f"DEF-{digest}",
        symbol=symbol,
        node_id=node_id,
        line=line,
        kind=kind,
    )


def _statement_index(function: FunctionIR) -> dict[str, StatementIR]:
    statements = {item.statement_id: item for item in function.statements}
    for control in function.controls:
        if control.initializer is not None:
            statements[control.initializer.statement_id] = control.initializer
        if control.update is not None:
            statements[control.update.statement_id] = control.update
    return statements


def _definition_kind(
    *,
    node_id: str,
    symbol: str,
    statements: dict[str, StatementIR],
    calls: dict[str, CallSite],
    assignments: dict[str, Assignment],
) -> str:
    statement = statements.get(node_id)
    if statement is None:
        return "assignment"
    for call_id in statement.call_ids:
        call = calls.get(call_id)
        assigned_to = call.assigned_to if call is not None else None
        if assigned_to and _last_identifier(assigned_to) == symbol:
            return "call_result"
    if statement.kind == "declaration" and not any(
        symbol in assignments[assignment_id].defs
        for assignment_id in statement.assignment_ids
        if assignment_id in assignments
    ):
        return "uninitialized"
    return "assignment"


def _last_identifier(value: str) -> str | None:
    matches = re.findall(r"[A-Za-z_]\w*", value)
    return matches[-1] if matches else None
