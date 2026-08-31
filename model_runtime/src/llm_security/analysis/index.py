from __future__ import annotations

from dataclasses import dataclass

from .analyzer import FunctionAnalysis
from .ir import (
    Assignment,
    CallSite,
    Condition,
    MemoryAccess,
    SourceSpan,
    StatementIR,
)


@dataclass(slots=True)
class FunctionAnalysisIndex:
    statement_by_id: dict[str, StatementIR]
    call_by_id: dict[str, CallSite]
    assignment_by_id: dict[str, Assignment]
    condition_by_id: dict[str, Condition]
    access_by_id: dict[str, MemoryAccess]
    call_to_node: dict[str, str]
    assignment_to_node: dict[str, str]
    access_to_node: dict[str, str]

    @classmethod
    def build(cls, analysis: FunctionAnalysis) -> FunctionAnalysisIndex:
        function = analysis.function
        statements = {item.statement_id: item for item in function.statements}
        for control in function.controls:
            if control.initializer is not None:
                statements[control.initializer.statement_id] = control.initializer
            if control.update is not None:
                statements[control.update.statement_id] = control.update

        calls = {item.call_id: item for item in function.calls}
        assignments = {
            item.assignment_id: item for item in function.assignments
        }
        conditions = {
            item.condition_id: item for item in function.conditions
        }
        accesses = {item.access_id: item for item in function.memory_accesses}
        call_to_node: dict[str, str] = {}
        assignment_to_node: dict[str, str] = {}
        for statement in statements.values():
            if statement.statement_id not in analysis.cfg.nodes:
                continue
            for call_id in statement.call_ids:
                call_to_node[call_id] = statement.statement_id
            for assignment_id in statement.assignment_ids:
                assignment_to_node[assignment_id] = statement.statement_id

        containers = [
            (statement.statement_id, statement.span)
            for statement in statements.values()
            if statement.statement_id in analysis.cfg.nodes
        ] + [
            (condition.condition_id, condition.span)
            for condition in conditions.values()
            if condition.condition_id in analysis.cfg.nodes
        ]
        for call in calls.values():
            call_to_node.setdefault(
                call.call_id, _smallest_container(call.span, containers) or ""
            )
        for assignment in assignments.values():
            assignment_to_node.setdefault(
                assignment.assignment_id,
                _smallest_container(assignment.span, containers) or "",
            )
        access_to_node = {
            access.access_id: node_id
            for access in accesses.values()
            for node_id in [_smallest_container(access.span, containers)]
            if node_id is not None
        }
        return cls(
            statement_by_id=dict(sorted(statements.items())),
            call_by_id=dict(sorted(calls.items())),
            assignment_by_id=dict(sorted(assignments.items())),
            condition_by_id=dict(sorted(conditions.items())),
            access_by_id=dict(sorted(accesses.items())),
            call_to_node={
                key: value
                for key, value in sorted(call_to_node.items())
                if value
            },
            assignment_to_node={
                key: value
                for key, value in sorted(assignment_to_node.items())
                if value
            },
            access_to_node=dict(sorted(access_to_node.items())),
        )

    def node_for_call(self, call_id: str) -> str | None:
        return self.call_to_node.get(call_id)

    def node_for_access(self, access_id: str) -> str | None:
        return self.access_to_node.get(access_id)


def _smallest_container(
    item: SourceSpan, containers: list[tuple[str, SourceSpan]]
) -> str | None:
    matches = [
        (node_id, span)
        for node_id, span in containers
        if _span_contains(span, item)
    ]
    if not matches:
        return None
    return min(matches, key=lambda value: (_span_size(value[1]), value[0]))[0]


def _span_contains(container: SourceSpan, item: SourceSpan) -> bool:
    if container.file != item.file:
        return False
    container_start = (container.line_start, container.column_start)
    container_end = (container.line_end, container.column_end)
    item_start = (item.line_start, item.column_start)
    item_end = (item.line_end, item.column_end)
    return container_start <= item_start and item_end <= container_end


def _span_size(span: SourceSpan) -> tuple[int, int]:
    return (
        span.line_end - span.line_start,
        span.column_end - span.column_start
        if span.line_end == span.line_start
        else span.column_end,
    )
