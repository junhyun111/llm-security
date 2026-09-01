from __future__ import annotations

from dataclasses import dataclass

from .analyzer import FunctionAnalysis
from .control_flow import dominates
from .path_queries import branch_path, branch_reachability


@dataclass(slots=True)
class GuardRelation:
    condition_id: str
    sink_node_id: str
    sensitive_symbols: set[str]
    sink_branch: str
    guard_kind: str
    effective_operator: str | None
    path_restricting: bool
    semantically_protective: bool
    path: list[str]


_NEGATED_RELATION = {
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
    "==": "!=",
    "!=": "==",
}
_REVERSED_RELATION = {
    "<": ">",
    "<=": ">=",
    ">": "<",
    ">=": "<=",
    "==": "==",
    "!=": "!=",
}
_NULL_VALUES = {"0", "NULL", "nullptr"}


def find_guard_relations(
    analysis: FunctionAnalysis,
    *,
    sink_node_id: str,
    sensitive_symbols: set[str],
    required_guard_kind: str | None = None,
) -> list[GuardRelation]:
    relations: list[GuardRelation] = []
    for condition in sorted(
        analysis.function.conditions,
        key=lambda item: item.condition_id,
    ):
        if condition.condition_id not in analysis.cfg.nodes:
            continue
        overlap = condition.symbols & sensitive_symbols
        if not overlap:
            continue
        if not dominates(
            analysis.dominators, condition.condition_id, sink_node_id
        ):
            continue
        reachability = branch_reachability(
            analysis.cfg, condition.condition_id, sink_node_id
        )
        reachable_branches = [
            branch
            for branch in ("true", "false")
            if reachability[branch]
        ]
        if len(reachable_branches) != 1:
            continue
        branch = reachable_branches[0]
        guard_kind, effective_operator = _classify_guard(
            condition, overlap, branch
        )
        path = branch_path(
            analysis.cfg, condition.condition_id, sink_node_id, branch
        )
        if path is None:
            continue
        relations.append(
            GuardRelation(
                condition_id=condition.condition_id,
                sink_node_id=sink_node_id,
                sensitive_symbols=set(overlap),
                sink_branch=branch,
                guard_kind=guard_kind,
                effective_operator=effective_operator,
                path_restricting=True,
                semantically_protective=(
                    required_guard_kind is not None
                    and guard_kind == required_guard_kind
                ),
                path=path,
            )
        )
    return relations


def _classify_guard(condition, sensitive: set[str], branch: str) -> tuple[str, str | None]:
    operator = condition.operator
    if operator is not None:
        effective = operator if branch == "true" else _NEGATED_RELATION.get(operator)
        left_symbols = condition.left_info.symbols if condition.left_info else set()
        right_symbols = condition.right_info.symbols if condition.right_info else set()
        other_text = ""
        if sensitive & right_symbols and not sensitive & left_symbols:
            effective = _REVERSED_RELATION.get(effective or "")
            other_text = condition.left_info.text if condition.left_info else ""
        else:
            other_text = condition.right_info.text if condition.right_info else ""
        if other_text in _NULL_VALUES and effective == "!=":
            return "non_null", effective
        if effective in {"<", "<="}:
            return "upper_bound", effective
        if effective in {">", ">="}:
            return "lower_bound", effective
        return "equality", effective

    if condition.left_info and sensitive & condition.left_info.symbols:
        negated = condition.unary_operator == "!"
        non_null = (branch == "false") if negated else (branch == "true")
        if non_null:
            return "non_null", "!="
        return "equality", "=="
    return "generic", None
