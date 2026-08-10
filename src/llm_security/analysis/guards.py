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
    path_restricting: bool
    path: list[str]


def find_guard_relations(
    analysis: FunctionAnalysis,
    *,
    sink_node_id: str,
    sensitive_symbols: set[str],
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
                path_restricting=True,
                path=path,
            )
        )
    return relations
