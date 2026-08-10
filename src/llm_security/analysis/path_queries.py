from __future__ import annotations

from collections import deque

from .control_flow import ControlFlowGraph


def is_reachable(cfg: ControlFlowGraph, source: str, target: str) -> bool:
    return shortest_path(cfg, source, target) is not None


def shortest_path(
    cfg: ControlFlowGraph, source: str, target: str
) -> list[str] | None:
    return _shortest_path(cfg, source, target, blocked_symbol=None)


def reachable_without_definition(
    cfg: ControlFlowGraph,
    source: str,
    target: str,
    symbol: str,
) -> bool:
    return shortest_path_without_definition(cfg, source, target, symbol) is not None


def shortest_path_without_definition(
    cfg: ControlFlowGraph,
    source: str,
    target: str,
    symbol: str,
) -> list[str] | None:
    return _shortest_path(cfg, source, target, blocked_symbol=symbol)


def branch_reachability(
    cfg: ControlFlowGraph, condition_id: str, target_id: str
) -> dict[str, bool]:
    result = {"true": False, "false": False}
    for successor in sorted(cfg.successors.get(condition_id, set())):
        edge_kind = cfg.edge_kinds.get((condition_id, successor))
        if edge_kind in result and is_reachable(cfg, successor, target_id):
            result[edge_kind] = True
    return result


def branch_path(
    cfg: ControlFlowGraph,
    condition_id: str,
    target_id: str,
    branch: str,
) -> list[str] | None:
    candidates = []
    for successor in sorted(cfg.successors.get(condition_id, set())):
        if cfg.edge_kinds.get((condition_id, successor)) != branch:
            continue
        path = shortest_path(cfg, successor, target_id)
        if path is not None:
            candidates.append([condition_id, *path])
    return min(candidates, key=lambda path: (len(path), path)) if candidates else None


def _shortest_path(
    cfg: ControlFlowGraph,
    source: str,
    target: str,
    *,
    blocked_symbol: str | None,
) -> list[str] | None:
    if source not in cfg.nodes or target not in cfg.nodes:
        return None
    queue = deque([(source, [source])])
    visited = {source}
    while queue:
        node_id, path = queue.popleft()
        if node_id == target:
            return path
        for successor in sorted(cfg.successors.get(node_id, set())):
            if successor in visited:
                continue
            if (
                blocked_symbol is not None
                and successor != target
                and blocked_symbol in cfg.nodes[successor].defs
            ):
                continue
            visited.add(successor)
            queue.append((successor, [*path, successor]))
    return None
