from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypeAlias

from .ir import Condition, ControlRegion, FunctionIR, SourceSpan, StatementIR


@dataclass(slots=True)
class CFGNode:
    node_id: str
    kind: str
    span: SourceSpan | None
    text: str
    defs: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ControlFlowGraph:
    function: str
    entry_id: str
    exit_id: str
    nodes: dict[str, CFGNode]
    successors: dict[str, set[str]]
    predecessors: dict[str, set[str]]
    edge_kinds: dict[tuple[str, str], str]
    warnings: list[str] = field(default_factory=list)


_FlowItem: TypeAlias = StatementIR | ControlRegion
_PendingEdge: TypeAlias = tuple[str, str]


class ControlFlowBuilder:
    """Build a structured intraprocedural CFG without consulting the raw AST."""

    def build(self, function: FunctionIR) -> ControlFlowGraph:
        self._function = function
        self._conditions = {
            condition.condition_id: condition for condition in function.conditions
        }
        self._children: dict[tuple[str | None, str | None], list[_FlowItem]] = (
            defaultdict(list)
        )
        for statement in function.statements:
            self._children[
                (statement.parent_control_id, statement.control_branch)
            ].append(statement)
        for control in function.controls:
            self._children[
                (control.parent_control_id, control.control_branch)
            ].append(control)
        for items in self._children.values():
            items.sort(key=_flow_item_sort_key)

        entry_id = _synthetic_id("ENTRY", function)
        exit_id = _synthetic_id("EXIT", function)
        self._cfg = ControlFlowGraph(
            function=function.name,
            entry_id=entry_id,
            exit_id=exit_id,
            nodes={},
            successors=defaultdict(set),
            predecessors=defaultdict(set),
            edge_kinds={},
            warnings=[
                f"unsupported control-flow construct: {construct}"
                for construct in sorted(set(function.unsupported_constructs))
            ],
        )
        self._add_node(
            CFGNode(
                node_id=entry_id,
                kind="entry",
                span=None,
                text="ENTRY",
                defs=set(function.parameters),
            )
        )
        self._add_node(
            CFGNode(
                node_id=exit_id,
                kind="exit",
                span=None,
                text="EXIT",
            )
        )

        pending = self._build_sequence(
            self._items(None, None), [(entry_id, "normal")]
        )
        for source, edge_kind in pending:
            self._add_edge(source, exit_id, edge_kind)
        self._normalize_maps()
        return self._cfg

    def _items(self, parent: str | None, branch: str | None) -> list[_FlowItem]:
        return self._children.get((parent, branch), [])

    def _build_sequence(
        self, items: list[_FlowItem], pending: list[_PendingEdge]
    ) -> list[_PendingEdge]:
        current = list(pending)
        for item in items:
            if not current:
                self._cfg.warnings.append(
                    f"unreachable {type(item).__name__}: {_flow_item_id(item)}"
                )
                continue
            if isinstance(item, StatementIR):
                current = self._build_statement(item, current)
            else:
                current = self._build_control(item, current)
        return current

    def _build_statement(
        self, statement: StatementIR, pending: list[_PendingEdge]
    ) -> list[_PendingEdge]:
        self._add_statement_node(statement)
        for source, edge_kind in pending:
            self._add_edge(source, statement.statement_id, edge_kind)
        if statement.kind == "return":
            self._add_edge(statement.statement_id, self._cfg.exit_id, "return")
            return []
        return [(statement.statement_id, "normal")]

    def _build_control(
        self, control: ControlRegion, pending: list[_PendingEdge]
    ) -> list[_PendingEdge]:
        if control.kind == "for" and control.initializer is not None:
            pending = self._build_statement(control.initializer, pending)

        self._add_condition_node(control)
        for source, edge_kind in pending:
            self._add_edge(source, control.condition_id, edge_kind)

        if control.kind == "if":
            then_pending = self._build_sequence(
                self._items(control.control_id, "then"),
                [(control.condition_id, "true")],
            )
            else_items = self._items(control.control_id, "else")
            else_pending = self._build_sequence(
                else_items, [(control.condition_id, "false")]
            )
            return then_pending + else_pending

        body_pending = self._build_sequence(
            self._items(control.control_id, "body"),
            [(control.condition_id, "true")],
        )
        if control.kind == "for" and control.update is not None and body_pending:
            update = control.update
            self._add_statement_node(update, kind="loop_update")
            for source, edge_kind in body_pending:
                self._add_edge(source, update.statement_id, edge_kind)
            self._add_edge(update.statement_id, control.condition_id, "loop_back")
        else:
            for source, _ in body_pending:
                self._add_edge(source, control.condition_id, "loop_back")
        condition = self._conditions.get(control.condition_id)
        if (
            control.kind == "for"
            and condition is not None
            and condition.expression == "true"
            and not condition.symbols
        ):
            return []
        return [(control.condition_id, "false")]

    def _add_statement_node(
        self, statement: StatementIR, *, kind: str = "statement"
    ) -> None:
        self._add_node(
            CFGNode(
                node_id=statement.statement_id,
                kind=kind,
                span=statement.span,
                text=statement.text,
                defs=set(statement.defs),
                uses=set(statement.uses),
            )
        )

    def _add_condition_node(self, control: ControlRegion) -> None:
        condition = self._conditions.get(control.condition_id)
        if condition is None:
            self._cfg.warnings.append(
                f"missing condition IR: {control.condition_id}"
            )
            condition = Condition(
                condition_id=control.condition_id,
                expression="<unknown>",
                symbols=set(),
                operator=None,
                span=control.span,
                text="<unknown condition>",
            )
        self._add_node(
            CFGNode(
                node_id=condition.condition_id,
                kind="condition",
                span=condition.span,
                text=condition.expression,
                uses=set(condition.symbols),
            )
        )

    def _add_node(self, node: CFGNode) -> None:
        self._cfg.nodes[node.node_id] = node
        self._cfg.successors.setdefault(node.node_id, set())
        self._cfg.predecessors.setdefault(node.node_id, set())

    def _add_edge(self, source: str, target: str, kind: str) -> None:
        self._cfg.successors[source].add(target)
        self._cfg.predecessors[target].add(source)
        self._cfg.edge_kinds.setdefault((source, target), kind)

    def _normalize_maps(self) -> None:
        self._cfg.successors = {
            node_id: set(self._cfg.successors.get(node_id, set()))
            for node_id in sorted(self._cfg.nodes)
        }
        self._cfg.predecessors = {
            node_id: set(self._cfg.predecessors.get(node_id, set()))
            for node_id in sorted(self._cfg.nodes)
        }
        self._cfg.edge_kinds = dict(sorted(self._cfg.edge_kinds.items()))
        self._cfg.warnings = list(dict.fromkeys(self._cfg.warnings))


class DominatorAnalysis:
    def compute(self, cfg: ControlFlowGraph) -> dict[str, set[str]]:
        return compute_dominators(cfg)


def compute_dominators(cfg: ControlFlowGraph) -> dict[str, set[str]]:
    node_ids = set(cfg.nodes)
    dominators = {
        node_id: ({node_id} if node_id == cfg.entry_id else set(node_ids))
        for node_id in cfg.nodes
    }
    changed = True
    while changed:
        changed = False
        for node_id in sorted(cfg.nodes):
            if node_id == cfg.entry_id:
                continue
            predecessors = sorted(cfg.predecessors.get(node_id, set()))
            if not predecessors:
                updated = {node_id}
            else:
                common = set(dominators[predecessors[0]])
                for predecessor in predecessors[1:]:
                    common.intersection_update(dominators[predecessor])
                updated = {node_id} | common
            if updated != dominators[node_id]:
                dominators[node_id] = updated
                changed = True
    return dominators


def dominates(
    dominators: dict[str, set[str]], dominator: str, node: str
) -> bool:
    return dominator in dominators.get(node, set())


def _synthetic_id(prefix: str, function: FunctionIR) -> str:
    digest = hashlib.sha1(
        f"{function.file}:{function.name}:{function.line_start}:{prefix}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _flow_item_sort_key(item: _FlowItem) -> tuple[int, int, int, int, str]:
    span = item.span
    return (
        span.line_start,
        span.column_start,
        span.line_end,
        span.column_end,
        _flow_item_id(item),
    )


def _flow_item_id(item: _FlowItem) -> str:
    return (
        item.statement_id if isinstance(item, StatementIR) else item.control_id
    )
