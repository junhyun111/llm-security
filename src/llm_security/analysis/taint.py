from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from .analyzer import FunctionAnalysis
from .api_semantics import ApiCatalog
from .index import FunctionAnalysisIndex
from .path_queries import shortest_path


@dataclass(slots=True, frozen=True)
class TaintPath:
    origin_id: str
    source_call_id: str
    sink_call_id: str
    source_node_id: str
    sink_node_id: str
    source_symbol: str
    sink_symbol: str
    path: list[str]
    sanitizer_found: bool


@dataclass(slots=True, frozen=True)
class _TaintOrigin:
    origin_id: str
    call_id: str
    node_id: str
    symbol: str


class TaintAnalyzer:
    """Intraprocedural forward may-taint analysis over the Phase 2B CFG."""

    def analyze(
        self,
        function_analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        catalog: ApiCatalog,
    ) -> list[TaintPath]:
        cfg = function_analysis.cfg
        calls_by_node: dict[str, list[str]] = defaultdict(list)
        assignments_by_node: dict[str, list[str]] = defaultdict(list)
        for call_id, node_id in index.call_to_node.items():
            calls_by_node[node_id].append(call_id)
        for assignment_id, node_id in index.assignment_to_node.items():
            assignments_by_node[node_id].append(assignment_id)
        for values in calls_by_node.values():
            values.sort()
        for values in assignments_by_node.values():
            values.sort()

        origins = self._origins(
            function_analysis, index, catalog, calls_by_node
        )
        origins_by_call_symbol = {
            (origin.call_id, origin.symbol): origin for origin in origins.values()
        }
        incoming: dict[str, dict[str, set[str]]] = {
            node_id: {} for node_id in cfg.nodes
        }
        outgoing: dict[str, dict[str, set[str]]] = {
            node_id: {} for node_id in cfg.nodes
        }
        changed = True
        while changed:
            changed = False
            for node_id in sorted(cfg.nodes):
                merged = _merge_states(
                    outgoing[predecessor]
                    for predecessor in sorted(cfg.predecessors.get(node_id, set()))
                )
                transferred = _copy_state(merged)
                for assignment_id in assignments_by_node.get(node_id, []):
                    assignment = index.assignment_by_id[assignment_id]
                    propagated = {
                        origin_id
                        for symbol in assignment.uses
                        for origin_id in transferred.get(symbol, set())
                    }
                    for target in sorted(assignment.defs):
                        if propagated:
                            transferred[target] = set(propagated)
                        else:
                            transferred.pop(target, None)

                for call_id in calls_by_node.get(node_id, []):
                    call = index.call_by_id[call_id]
                    name = catalog.canonical_name(call.callee)
                    source_spec = catalog.taint_sources.get(name)
                    if source_spec is not None:
                        source_symbols = {
                            symbol
                            for argument_index in source_spec.output_args
                            if argument_index < len(call.argument_symbols)
                            for symbol in call.argument_symbols[argument_index]
                        }
                        if source_spec.tainted_return:
                            source_symbols.update(
                                _call_result_symbols(function_analysis, node_id)
                            )
                        for symbol in sorted(source_symbols):
                            origin = origins_by_call_symbol.get((call_id, symbol))
                            if origin is not None:
                                transferred.setdefault(symbol, set()).add(
                                    origin.origin_id
                                )

                    sanitizer = catalog.sanitizers.get(name)
                    if sanitizer is not None:
                        killed = {
                            symbol
                            for argument_index in sanitizer.arguments
                            if argument_index < len(call.argument_symbols)
                            for symbol in call.argument_symbols[argument_index]
                        }
                        if sanitizer.result:
                            killed.update(
                                _call_result_symbols(function_analysis, node_id)
                            )
                        for symbol in killed:
                            transferred.pop(symbol, None)

                if merged != incoming[node_id] or transferred != outgoing[node_id]:
                    incoming[node_id] = merged
                    outgoing[node_id] = transferred
                    changed = True

        paths: list[TaintPath] = []
        for node_id in sorted(cfg.nodes):
            for call_id in calls_by_node.get(node_id, []):
                call = index.call_by_id[call_id]
                sink_spec = catalog.taint_sinks.get(
                    catalog.canonical_name(call.callee)
                )
                if sink_spec is None:
                    continue
                for argument_index in sink_spec.input_args:
                    if argument_index >= len(call.argument_symbols):
                        continue
                    for sink_symbol in sorted(call.argument_symbols[argument_index]):
                        for origin_id in sorted(
                            incoming[node_id].get(sink_symbol, set())
                        ):
                            origin = origins[origin_id]
                            path = shortest_path(cfg, origin.node_id, node_id)
                            if path is None:
                                continue
                            paths.append(
                                TaintPath(
                                    origin_id=origin_id,
                                    source_call_id=origin.call_id,
                                    sink_call_id=call_id,
                                    source_node_id=origin.node_id,
                                    sink_node_id=node_id,
                                    source_symbol=origin.symbol,
                                    sink_symbol=sink_symbol,
                                    path=path,
                                    sanitizer_found=False,
                                )
                            )
        unique = {
            (
                path.origin_id,
                path.sink_call_id,
                path.sink_symbol,
            ): path
            for path in paths
        }
        return sorted(
            unique.values(),
            key=lambda path: (
                path.source_node_id,
                path.sink_node_id,
                path.origin_id,
                path.sink_symbol,
            ),
        )

    def _origins(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        catalog: ApiCatalog,
        calls_by_node: dict[str, list[str]],
    ) -> dict[str, _TaintOrigin]:
        origins: dict[str, _TaintOrigin] = {}
        for node_id in sorted(calls_by_node):
            for call_id in calls_by_node[node_id]:
                call = index.call_by_id[call_id]
                spec = catalog.taint_sources.get(
                    catalog.canonical_name(call.callee)
                )
                if spec is None:
                    continue
                symbols = {
                    symbol
                    for argument_index in spec.output_args
                    if argument_index < len(call.argument_symbols)
                    for symbol in call.argument_symbols[argument_index]
                }
                if spec.tainted_return:
                    symbols.update(_call_result_symbols(analysis, node_id))
                for symbol in sorted(symbols):
                    digest = hashlib.sha1(
                        f"{call_id}:{node_id}:{symbol}".encode("utf-8")
                    ).hexdigest()[:12]
                    origin = _TaintOrigin(
                        origin_id=f"TAINT-{digest}",
                        call_id=call_id,
                        node_id=node_id,
                        symbol=symbol,
                    )
                    origins[origin.origin_id] = origin
        return dict(sorted(origins.items()))


def _call_result_symbols(
    analysis: FunctionAnalysis, node_id: str
) -> set[str]:
    return {
        definition.symbol
        for definition in analysis.dataflow.definitions.values()
        if definition.node_id == node_id and definition.kind == "call_result"
    }


def _merge_states(
    states,
) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for state in states:
        for symbol, origins in state.items():
            merged.setdefault(symbol, set()).update(origins)
    return merged


def _copy_state(state: dict[str, set[str]]) -> dict[str, set[str]]:
    return {symbol: set(origins) for symbol, origins in state.items()}
