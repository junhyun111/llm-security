from __future__ import annotations

import re
from dataclasses import dataclass

from .analyzer import FunctionAnalysis, ProgramAnalysis, StructuralAnalyzer
from .api_semantics import AllocationSpec, ApiCatalog, MemoryCopySpec, StatusReturnSpec
from .dataflow import backward_slice
from .control_flow import dominates
from .guards import GuardRelation, find_guard_relations
from .index import FunctionAnalysisIndex
from .ir import CallSite
from .path_queries import (
    shortest_path,
    shortest_path_without_definition,
)
from .semantic_facts import (
    SemanticFact,
    SemanticFactKind,
    make_semantic_fact,
    sort_facts,
)
from .taint import TaintAnalyzer, TaintPath


@dataclass(slots=True)
class SemanticFunctionAnalysis:
    structural: FunctionAnalysis
    facts: list[SemanticFact]
    taint_paths: list[TaintPath]


@dataclass(slots=True)
class SemanticProgramAnalysis:
    structural: ProgramAnalysis
    functions: dict[str, SemanticFunctionAnalysis]


class SemanticAnalyzer:
    def __init__(self, catalog: ApiCatalog | None = None) -> None:
        self.catalog = catalog or ApiCatalog.default()

    def analyze(self, structural: ProgramAnalysis) -> SemanticProgramAnalysis:
        functions = {
            key: self._analyze_function(structural.functions[key])
            for key in sorted(structural.functions)
        }
        return SemanticProgramAnalysis(structural=structural, functions=functions)

    def analyze_sources(
        self, source_files: dict[str, str]
    ) -> SemanticProgramAnalysis:
        return self.analyze(StructuralAnalyzer().analyze(source_files))

    def _analyze_function(
        self, analysis: FunctionAnalysis
    ) -> SemanticFunctionAnalysis:
        index = FunctionAnalysisIndex.build(analysis)
        facts: list[SemanticFact] = []
        reliable = not analysis.cfg.warnings
        calls = sorted(
            analysis.function.calls,
            key=lambda call: (
                call.span.line_start,
                call.span.column_start,
                call.call_id,
            ),
        )

        allocation_calls: list[tuple[CallSite, str, AllocationSpec]] = []
        release_calls: list[tuple[CallSite, str, set[str]]] = []
        check_calls: list[tuple[CallSite, str, set[str]]] = []
        use_calls: list[tuple[CallSite, str, set[str]]] = []
        for call in calls:
            node_id = index.node_for_call(call.call_id)
            if node_id is None:
                continue
            name = self.catalog.canonical_name(call.callee)
            facts.extend(
                self._numeric_conversion_facts(analysis, call, node_id, name)
            )
            status_spec = self.catalog.status_returns.get(name)
            if status_spec is not None:
                unchecked = self._unchecked_call_result_fact(
                    analysis, index, call, node_id, name, status_spec
                )
                if unchecked is not None:
                    facts.append(unchecked)
            if name in self.catalog.allocations:
                spec = self.catalog.allocations[name]
                allocation_calls.append((call, node_id, spec))
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.ALLOCATION,
                        subject=",".join(sorted(_call_result_symbols(analysis, node_id)))
                        or None,
                        object=name,
                        source_node_id=node_id,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=_symbols_at_arguments(call, spec.size_args),
                        attributes={
                            "callee": name,
                            "size_args": list(spec.size_args),
                            "size_expressions": _arguments(call, spec.size_args),
                            "nullable_return": spec.nullable_return,
                        },
                    )
                )
                facts.extend(
                    self._size_flow_facts(
                        analysis,
                        index,
                        call,
                        node_id,
                        spec.size_args,
                        SemanticFactKind.ARITHMETIC_TO_ALLOCATION,
                        "allocation",
                    )
                )

            release_spec = self.catalog.releases.get(name)
            if release_spec is not None:
                pointer_symbols = _symbols_at_arguments(
                    call, (release_spec.pointer_arg,)
                )
                release_calls.append((call, node_id, pointer_symbols))
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.RELEASE,
                        subject=",".join(sorted(pointer_symbols)) or None,
                        object=name,
                        source_node_id=node_id,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=pointer_symbols,
                        attributes={
                            "callee": name,
                            "pointer_arg": release_spec.pointer_arg,
                        },
                    )
                )

            copy_spec = self.catalog.memory_copy.get(name)
            if copy_spec is not None:
                facts.extend(
                    self._memory_copy_facts(
                        analysis, index, call, node_id, name, copy_spec
                    )
                )

            source_spec = self.catalog.taint_sources.get(name)
            if source_spec is not None:
                source_symbols = _symbols_at_arguments(
                    call, source_spec.output_args
                )
                if source_spec.tainted_return:
                    source_symbols.update(_call_result_symbols(analysis, node_id))
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.TAINT_SOURCE,
                        subject=",".join(sorted(source_symbols)) or None,
                        object=name,
                        source_node_id=node_id,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=source_symbols,
                        attributes={
                            "callee": name,
                            "output_args": list(source_spec.output_args),
                            "tainted_return": source_spec.tainted_return,
                        },
                    )
                )

            sink_spec = self.catalog.taint_sinks.get(name)
            if sink_spec is not None:
                sink_symbols = _symbols_at_arguments(call, sink_spec.input_args)
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.TAINT_SINK,
                        subject=",".join(sorted(sink_symbols)) or None,
                        object=name,
                        source_node_id=node_id,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=sink_symbols,
                        attributes={
                            "callee": name,
                            "input_args": list(sink_spec.input_args),
                        },
                    )
                )

            if name in self.catalog.thread_spawn:
                facts.append(self._local_api_fact(analysis, call, node_id, name, SemanticFactKind.THREAD_SPAWN))
            if name in self.catalog.lock_acquire:
                facts.append(self._local_api_fact(analysis, call, node_id, name, SemanticFactKind.LOCK_ACQUIRE))
            if name in self.catalog.lock_release:
                facts.append(self._local_api_fact(analysis, call, node_id, name, SemanticFactKind.LOCK_RELEASE))

            if name in self.catalog.toctou_checks:
                check_calls.append(
                    (
                        call,
                        node_id,
                        _symbols_at_arguments(
                            call, self.catalog.toctou_checks[name]
                        ),
                    )
                )
            if name in self.catalog.toctou_uses:
                use_calls.append(
                    (
                        call,
                        node_id,
                        _symbols_at_arguments(call, self.catalog.toctou_uses[name]),
                    )
                )

        taint_paths = TaintAnalyzer().analyze(analysis, index, self.catalog)
        for taint_path in taint_paths:
            attributes = {
                "origin_id": taint_path.origin_id,
                "source_call_id": taint_path.source_call_id,
                "sink_call_id": taint_path.sink_call_id,
                "control_flow_reliable": reliable,
            }
            facts.append(
                self._fact(
                    analysis,
                    SemanticFactKind.SOURCE_TO_SINK,
                    subject=taint_path.source_symbol,
                    object=taint_path.sink_symbol,
                    source_node_id=taint_path.source_node_id,
                    sink_node_id=taint_path.sink_node_id,
                    path=taint_path.path,
                    symbols={taint_path.source_symbol, taint_path.sink_symbol},
                    confidence=0.9 if reliable else 0.6,
                    attributes=attributes,
                )
            )
            if reliable and not taint_path.sanitizer_found:
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.UNSANITIZED_SOURCE_TO_SINK,
                        subject=taint_path.source_symbol,
                        object=taint_path.sink_symbol,
                        source_node_id=taint_path.source_node_id,
                        sink_node_id=taint_path.sink_node_id,
                        path=taint_path.path,
                        symbols={taint_path.source_symbol, taint_path.sink_symbol},
                        confidence=0.85,
                        attributes=attributes,
                    )
                )

        facts.extend(self._lifetime_facts(analysis, index, release_calls))
        facts.extend(self._expression_semantic_facts(analysis, index))
        facts.extend(self._status_check_facts(analysis, index, calls))
        facts.extend(self._uninitialized_facts(analysis, index))
        facts.extend(
            self._nullable_dereference_facts(
                analysis, index, allocation_calls
            )
        )
        facts.extend(self._index_facts(analysis, index))
        facts.extend(self._toctou_facts(analysis, check_calls, use_calls))
        return SemanticFunctionAnalysis(
            structural=analysis,
            facts=sort_facts(facts),
            taint_paths=taint_paths,
        )

    def _expression_semantic_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
    ) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        arithmetic_operators = {"+", "-", "*", "/", "%", "<<", ">>"}
        for assignment in analysis.function.assignments:
            node_id = index.assignment_to_node.get(assignment.assignment_id)
            info = assignment.expression_info
            if node_id is None or info is None:
                continue
            operators = sorted(info.operators & arithmetic_operators)
            destination_type = _assignment_destination_type(
                analysis, assignment.defs
            )
            source_types = _symbol_types_for(analysis, assignment.uses)
            if operators:
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.INTEGER_ARITHMETIC,
                        subject=assignment.target,
                        object="assignment",
                        source_node_id=node_id,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=assignment.defs | assignment.uses,
                        confidence=0.60 if not analysis.cfg.warnings else 0.45,
                        attributes={
                            "operators": operators,
                            "expression": assignment.expression,
                            "role": "assignment",
                            "destination_type": destination_type,
                            "source_types": source_types,
                        },
                    )
                )
            numeric_types = sorted(
                cast_type
                for cast_type in info.cast_types
                if _is_numeric_cast_type(cast_type)
            )
            implicit_conversion = _implicit_integer_conversion(
                destination_type, source_types
            )
            if numeric_types or implicit_conversion is not None:
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.NUMERIC_CONVERSION,
                        subject=assignment.target,
                        object="assignment",
                        source_node_id=node_id,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=assignment.defs | assignment.uses,
                        confidence=(
                            0.78
                            if implicit_conversion is not None
                            else 0.75
                        ) if not analysis.cfg.warnings else 0.50,
                        attributes={
                            "cast_types": numeric_types,
                            "expression": assignment.expression,
                            "role": "assignment",
                            "destination_type": destination_type,
                            "source_types": source_types,
                            **(implicit_conversion or {}),
                        },
                    )
                )
            target = assignment.target.rsplit("->", 1)[-1].rsplit(".", 1)[-1]
            if re.search(
                r"(?:state|status|phase|initialized|authenticated|mode|error)",
                target,
                re.IGNORECASE,
            ):
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.STATE_TRANSITION,
                        subject=assignment.target,
                        object=assignment.expression,
                        source_node_id=node_id,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=assignment.defs | assignment.uses,
                        confidence=0.55,
                        attributes={
                            "target": assignment.target,
                            "expression": assignment.expression,
                        },
                    )
                )

        for condition in analysis.function.conditions:
            infos = [
                info
                for info in (condition.left_info, condition.right_info)
                if info is not None
            ]
            operators = sorted(
                {
                    operator
                    for info in infos
                    for operator in info.operators
                    if operator in arithmetic_operators
                }
            )
            if operators:
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.INTEGER_ARITHMETIC,
                        subject=",".join(sorted(condition.symbols)) or None,
                        object="condition",
                        source_node_id=condition.condition_id,
                        sink_node_id=condition.condition_id,
                        path=[condition.condition_id],
                        symbols=condition.symbols,
                        confidence=0.55 if not analysis.cfg.warnings else 0.40,
                        attributes={
                            "operators": operators,
                            "expression": condition.expression,
                            "role": "condition",
                        },
                    )
                )
            numeric_types = sorted(
                {
                    cast_type
                    for info in infos
                    for cast_type in info.cast_types
                    if _is_numeric_cast_type(cast_type)
                }
            )
            if numeric_types:
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.NUMERIC_CONVERSION,
                        subject=",".join(sorted(condition.symbols)) or None,
                        object="condition",
                        source_node_id=condition.condition_id,
                        sink_node_id=condition.condition_id,
                        path=[condition.condition_id],
                        symbols=condition.symbols,
                        confidence=0.70 if not analysis.cfg.warnings else 0.45,
                        attributes={
                            "cast_types": numeric_types,
                            "expression": condition.expression,
                            "role": "condition",
                        },
                    )
                )
        return facts

    def _status_check_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        calls: list[CallSite],
    ) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        for call in calls:
            name = self.catalog.canonical_name(call.callee)
            if name not in self.catalog.status_returns:
                continue
            node_id = index.node_for_call(call.call_id)
            if node_id is None:
                continue
            result_symbols = _call_result_symbols(analysis, node_id)
            checking_nodes = sorted(
                condition.condition_id
                for condition in index.condition_by_id.values()
                if result_symbols & condition.symbols
                and shortest_path(analysis.cfg, node_id, condition.condition_id)
            )
            if call.result_usage == "condition":
                checking_nodes = [node_id]
            for check_node in checking_nodes[:1]:
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.ERROR_PATH,
                        subject=",".join(sorted(result_symbols)) or name,
                        object=name,
                        source_node_id=node_id,
                        sink_node_id=check_node,
                        path=shortest_path(analysis.cfg, node_id, check_node)
                        or [check_node],
                        symbols=result_symbols,
                        confidence=0.80 if not analysis.cfg.warnings else 0.55,
                        attributes={
                            "callee": name,
                            "checked": True,
                            "check_node_id": check_node,
                            "failure_semantics": self.catalog.status_returns[
                                name
                            ].failure_semantics,
                        },
                    )
                )
        return facts

    def _numeric_conversion_facts(
        self,
        analysis: FunctionAnalysis,
        call: CallSite,
        node_id: str,
        name: str,
    ) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        for argument_index, info in enumerate(call.argument_info):
            numeric_types = sorted(
                cast_type
                for cast_type in info.cast_types
                if _is_numeric_cast_type(cast_type)
            )
            if not numeric_types:
                continue
            facts.append(
                self._fact(
                    analysis,
                    SemanticFactKind.NUMERIC_CONVERSION,
                    subject=",".join(sorted(info.symbols)) or None,
                    object=name,
                    source_node_id=node_id,
                    sink_node_id=node_id,
                    path=[node_id],
                    symbols=info.symbols,
                    confidence=0.85 if not analysis.cfg.warnings else 0.6,
                    attributes={
                        "callee": name,
                        "argument_index": argument_index,
                        "cast_types": numeric_types,
                        "expression": info.text,
                    },
                )
            )
        return facts

    def _unchecked_call_result_fact(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        call: CallSite,
        node_id: str,
        name: str,
        spec: StatusReturnSpec,
    ) -> SemanticFact | None:
        if call.result_usage in {"condition", "returned", "argument", "expression"}:
            return None
        result_symbols = _call_result_symbols(analysis, node_id)
        checking_nodes = sorted(
            condition.condition_id
            for condition in index.condition_by_id.values()
            if result_symbols & condition.symbols
            and shortest_path(analysis.cfg, node_id, condition.condition_id)
        )
        if checking_nodes:
            return None
        usage = "assigned_without_check" if result_symbols else "discarded"
        return self._fact(
            analysis,
            SemanticFactKind.UNCHECKED_CALL_RESULT,
            subject=",".join(sorted(result_symbols)) or None,
            object=name,
            source_node_id=node_id,
            sink_node_id=node_id,
            path=[node_id],
            symbols=result_symbols,
            confidence=0.85 if usage == "discarded" else 0.75,
            attributes={
                "callee": name,
                "result_usage": usage,
                "failure_semantics": spec.failure_semantics,
                "checked": False,
            },
        )

    def _memory_copy_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        call: CallSite,
        node_id: str,
        name: str,
        spec: MemoryCopySpec,
    ) -> list[SemanticFact]:
        length_indices = () if spec.length_arg is None else (spec.length_arg,)
        sensitive = _symbols_at_arguments(call, length_indices)
        attributes = {
            "callee": name,
            "dst_arg": spec.dst_arg,
            "src_arg": spec.src_arg,
            "length_arg": spec.length_arg,
            "dst": _argument(call, spec.dst_arg),
            "src": _argument(call, spec.src_arg),
            "length": _argument(call, spec.length_arg),
            "unbounded": spec.length_arg is None,
        }
        facts = [
            self._fact(
                analysis,
                SemanticFactKind.MEMORY_COPY,
                subject=attributes["dst"],
                object=attributes["src"],
                source_node_id=node_id,
                sink_node_id=node_id,
                path=[node_id],
                symbols=sensitive,
                attributes=attributes,
            )
        ]
        guards = find_guard_relations(
            analysis,
            sink_node_id=node_id,
            sensitive_symbols=sensitive,
            required_guard_kind="upper_bound",
        )
        protective_guards = [guard for guard in guards if guard.semantically_protective]
        facts.extend(self._guard_facts(analysis, protective_guards, name))
        if not protective_guards and not analysis.cfg.warnings:
            facts.append(
                self._fact(
                    analysis,
                    SemanticFactKind.MEMORY_COPY_WITHOUT_GUARD,
                    subject=attributes["length"],
                    object=name,
                    sink_node_id=node_id,
                    path=[node_id],
                    symbols=sensitive,
                    confidence=0.8,
                    attributes={**attributes, "control_flow_reliable": True},
                )
            )
        if spec.length_arg is not None:
            facts.extend(
                self._size_flow_facts(
                    analysis,
                    index,
                    call,
                    node_id,
                    (spec.length_arg,),
                    SemanticFactKind.ARITHMETIC_TO_MEMORY_SINK,
                    "memory_copy_length",
                )
            )
        return facts

    def _size_flow_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        call: CallSite,
        node_id: str,
        argument_indices: tuple[int, ...],
        arithmetic_kind: SemanticFactKind,
        sink_role: str,
    ) -> list[SemanticFact]:
        arithmetic_operators: set[str] = set()
        cast_types: set[str] = set()
        has_sizeof = False
        source_nodes: set[str] = set()
        sensitive_symbols = _symbols_at_arguments(call, argument_indices)
        for argument_index in argument_indices:
            if argument_index >= len(call.argument_info):
                continue
            info = call.argument_info[argument_index]
            if info.operators:
                arithmetic_operators.update(info.operators)
                source_nodes.add(node_id)
            if info.cast_types:
                cast_types.update(info.cast_types)
                source_nodes.add(node_id)
            has_sizeof = has_sizeof or info.has_sizeof

        for symbol in sorted(sensitive_symbols):
            for step in backward_slice(
                analysis.function,
                analysis.cfg,
                analysis.dataflow,
                sink_node_id=node_id,
                symbols={symbol},
            ):
                statement = index.statement_by_id.get(step.from_node)
                if statement is None:
                    continue
                for assignment_id in statement.assignment_ids:
                    assignment = index.assignment_by_id.get(assignment_id)
                    info = assignment.expression_info if assignment is not None else None
                    if info is None:
                        continue
                    if info.operators:
                        arithmetic_operators.update(info.operators)
                        source_nodes.add(step.from_node)
                    if info.cast_types:
                        cast_types.update(info.cast_types)
                        source_nodes.add(step.from_node)
                    has_sizeof = has_sizeof or info.has_sizeof

        facts: list[SemanticFact] = []
        source_node = sorted(source_nodes)[0] if source_nodes else node_id
        path = shortest_path(analysis.cfg, source_node, node_id) or [node_id]
        common_attributes = {
            "callee": self.catalog.canonical_name(call.callee),
            "argument_indices": list(argument_indices),
            "sink_role": sink_role,
            "source_nodes": sorted(source_nodes),
            "control_flow_reliable": not analysis.cfg.warnings,
        }
        if arithmetic_operators:
            facts.append(
                self._fact(
                    analysis,
                    arithmetic_kind,
                    subject=",".join(sorted(sensitive_symbols)) or None,
                    object=sink_role,
                    source_node_id=source_node,
                    sink_node_id=node_id,
                    path=path,
                    symbols=sensitive_symbols,
                    confidence=0.9 if not analysis.cfg.warnings else 0.6,
                    attributes={
                        **common_attributes,
                        "operators": sorted(arithmetic_operators),
                        "has_sizeof": has_sizeof,
                    },
                )
            )
        if cast_types:
            facts.append(
                self._fact(
                    analysis,
                    SemanticFactKind.CAST_TO_SIZE_SINK,
                    subject=",".join(sorted(sensitive_symbols)) or None,
                    object=sink_role,
                    source_node_id=source_node,
                    sink_node_id=node_id,
                    path=path,
                    symbols=sensitive_symbols,
                    confidence=0.85 if not analysis.cfg.warnings else 0.55,
                    attributes={
                        **common_attributes,
                        "cast_types": sorted(cast_types),
                    },
                )
            )
        return facts

    def _lifetime_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        releases: list[tuple[CallSite, str, set[str]]],
    ) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        reliable = not analysis.cfg.warnings
        for call, release_node, pointer_symbols in releases:
            for access in analysis.function.memory_accesses:
                access_node = index.node_for_access(access.access_id)
                if access_node is None:
                    continue
                for symbol in sorted(pointer_symbols & access.base_symbols):
                    path = shortest_path_without_definition(
                        analysis.cfg, release_node, access_node, symbol
                    )
                    if path is None or release_node == access_node:
                        continue
                    facts.append(
                        self._fact(
                            analysis,
                            SemanticFactKind.USE_AFTER_RELEASE,
                            subject=symbol,
                            object=access.text,
                            source_node_id=release_node,
                            sink_node_id=access_node,
                            path=path,
                            symbols={symbol},
                            confidence=0.9 if reliable else 0.6,
                            attributes={
                                "release_call_id": call.call_id,
                                "access_id": access.access_id,
                                "access_kind": access.kind,
                                "control_flow_reliable": reliable,
                            },
                        )
                    )
        for index_a, (first_call, first_node, first_symbols) in enumerate(releases):
            for second_call, second_node, second_symbols in releases[index_a + 1 :]:
                for symbol in sorted(first_symbols & second_symbols):
                    path = shortest_path_without_definition(
                        analysis.cfg, first_node, second_node, symbol
                    )
                    if path is None or first_node == second_node:
                        continue
                    facts.append(
                        self._fact(
                            analysis,
                            SemanticFactKind.DOUBLE_RELEASE,
                            subject=symbol,
                            object=self.catalog.canonical_name(second_call.callee),
                            source_node_id=first_node,
                            sink_node_id=second_node,
                            path=path,
                            symbols={symbol},
                            confidence=0.9 if reliable else 0.6,
                            attributes={
                                "first_call_id": first_call.call_id,
                                "second_call_id": second_call.call_id,
                                "control_flow_reliable": reliable,
                            },
                        )
                    )
        return facts

    def _uninitialized_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
    ) -> list[SemanticFact]:
        facts = []
        reliable = not analysis.cfg.warnings
        output_writers: dict[str, set[str]] = {}
        for call_id, node_id in index.call_to_node.items():
            call = index.call_by_id[call_id]
            source_spec = self.catalog.taint_sources.get(
                self.catalog.canonical_name(call.callee)
            )
            if source_spec is None:
                continue
            for symbol in _symbols_at_arguments(call, source_spec.output_args):
                output_writers.setdefault(symbol, set()).add(node_id)
        for edge in analysis.dataflow.edges:
            definition = analysis.dataflow.definitions[edge.definition_id]
            if definition.kind != "uninitialized":
                continue
            if any(
                writer == edge.use_node_id
                or dominates(analysis.dominators, writer, edge.use_node_id)
                for writer in output_writers.get(edge.symbol, set())
            ):
                continue
            path = shortest_path(
                analysis.cfg, edge.definition_node_id, edge.use_node_id
            )
            if path is None:
                continue
            facts.append(
                self._fact(
                    analysis,
                    SemanticFactKind.UNINITIALIZED_USE,
                    subject=edge.symbol,
                    source_node_id=edge.definition_node_id,
                    sink_node_id=edge.use_node_id,
                    path=path,
                    symbols={edge.symbol},
                    confidence=0.9 if reliable else 0.6,
                    attributes={
                        "definition_id": edge.definition_id,
                        "control_flow_reliable": reliable,
                    },
                )
            )
        return facts

    def _nullable_dereference_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
        allocations: list[tuple[CallSite, str, AllocationSpec]],
    ) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        for call, allocation_node, spec in allocations:
            if not spec.nullable_return:
                continue
            for symbol in sorted(_call_result_symbols(analysis, allocation_node)):
                for access in analysis.function.memory_accesses:
                    if symbol not in access.base_symbols:
                        continue
                    access_node = index.node_for_access(access.access_id)
                    if access_node is None or access_node == allocation_node:
                        continue
                    path = shortest_path_without_definition(
                        analysis.cfg, allocation_node, access_node, symbol
                    )
                    if path is None:
                        continue
                    guards = find_guard_relations(
                        analysis,
                        sink_node_id=access_node,
                        sensitive_symbols={symbol},
                        required_guard_kind="non_null",
                    )
                    protective_guards = [
                        guard for guard in guards if guard.semantically_protective
                    ]
                    facts.extend(
                        self._guard_facts(
                            analysis,
                            protective_guards,
                            self.catalog.canonical_name(call.callee),
                        )
                    )
                    if not protective_guards and not analysis.cfg.warnings:
                        facts.append(
                            self._fact(
                                analysis,
                                SemanticFactKind.UNCHECKED_NULLABLE_DEREFERENCE,
                                subject=symbol,
                                object=access.text,
                                source_node_id=allocation_node,
                                sink_node_id=access_node,
                                path=path,
                                symbols={symbol},
                                confidence=0.8,
                                attributes={
                                    "allocator_call_id": call.call_id,
                                    "access_id": access.access_id,
                                    "control_flow_reliable": True,
                                },
                            )
                        )
        return facts

    def _index_facts(
        self,
        analysis: FunctionAnalysis,
        index: FunctionAnalysisIndex,
    ) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        for access in analysis.function.memory_accesses:
            if not access.kind.startswith("array_") or not access.index_symbols:
                continue
            node_id = index.node_for_access(access.access_id)
            if node_id is None:
                continue
            guards = find_guard_relations(
                analysis,
                sink_node_id=node_id,
                sensitive_symbols=access.index_symbols,
                required_guard_kind="upper_bound",
            )
            protective_guards = [guard for guard in guards if guard.semantically_protective]
            facts.extend(self._guard_facts(analysis, protective_guards, access.text))
            if not protective_guards and not analysis.cfg.warnings:
                facts.append(
                    self._fact(
                        analysis,
                        SemanticFactKind.UNCHECKED_INDEX,
                        subject=access.index,
                        object=access.base,
                        sink_node_id=node_id,
                        path=[node_id],
                        symbols=access.index_symbols,
                        confidence=0.65,
                        attributes={
                            "access_id": access.access_id,
                            "access_kind": access.kind,
                            "control_flow_reliable": True,
                        },
                    )
                )
        return facts

    def _toctou_facts(
        self,
        analysis: FunctionAnalysis,
        checks: list[tuple[CallSite, str, set[str]]],
        uses: list[tuple[CallSite, str, set[str]]],
    ) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        reliable = not analysis.cfg.warnings
        for check_call, check_node, check_symbols in checks:
            for use_call, use_node, use_symbols in uses:
                for symbol in sorted(check_symbols & use_symbols):
                    path = shortest_path_without_definition(
                        analysis.cfg, check_node, use_node, symbol
                    )
                    if path is None or check_node == use_node:
                        continue
                    facts.append(
                        self._fact(
                            analysis,
                            SemanticFactKind.TOCTOU_CHECK_USE,
                            subject=symbol,
                            object=self.catalog.canonical_name(use_call.callee),
                            source_node_id=check_node,
                            sink_node_id=use_node,
                            path=path,
                            symbols={symbol},
                            confidence=0.85 if reliable else 0.55,
                            attributes={
                                "check_api": self.catalog.canonical_name(check_call.callee),
                                "use_api": self.catalog.canonical_name(use_call.callee),
                                "check_call_id": check_call.call_id,
                                "use_call_id": use_call.call_id,
                                "control_flow_reliable": reliable,
                            },
                        )
                    )
        return facts

    def _guard_facts(
        self,
        analysis: FunctionAnalysis,
        guards: list[GuardRelation],
        protected_object: str,
    ) -> list[SemanticFact]:
        return [
            self._fact(
                analysis,
                SemanticFactKind.GUARD_PROTECTS_SINK,
                subject=",".join(sorted(guard.sensitive_symbols)),
                object=protected_object,
                source_node_id=guard.condition_id,
                sink_node_id=guard.sink_node_id,
                path=guard.path,
                symbols=guard.sensitive_symbols,
                confidence=0.9 if not analysis.cfg.warnings else 0.6,
                attributes={
                    "condition_id": guard.condition_id,
                    "sink_branch": guard.sink_branch,
                    "guard_kind": guard.guard_kind,
                    "effective_operator": guard.effective_operator,
                    "path_restricting": guard.path_restricting,
                    "semantically_protective": guard.semantically_protective,
                    "control_flow_reliable": not analysis.cfg.warnings,
                },
            )
            for guard in guards
        ]

    def _local_api_fact(
        self,
        analysis: FunctionAnalysis,
        call: CallSite,
        node_id: str,
        name: str,
        kind: SemanticFactKind,
    ) -> SemanticFact:
        symbols = {
            symbol for argument in call.argument_symbols for symbol in argument
        }
        return self._fact(
            analysis,
            kind,
            subject=",".join(sorted(symbols)) or None,
            object=name,
            source_node_id=node_id,
            sink_node_id=node_id,
            path=[node_id],
            symbols=symbols,
            attributes={"callee": name},
        )

    @staticmethod
    def _fact(
        analysis: FunctionAnalysis,
        kind: SemanticFactKind,
        **kwargs,
    ) -> SemanticFact:
        return make_semantic_fact(
            kind=kind,
            function=analysis.function.name,
            file=analysis.function.file,
            **kwargs,
        )


def _symbols_at_arguments(
    call: CallSite, indices: tuple[int, ...]
) -> set[str]:
    return {
        symbol
        for index in indices
        if index < len(call.argument_symbols)
        for symbol in call.argument_symbols[index]
    }


def _arguments(call: CallSite, indices: tuple[int, ...]) -> list[str]:
    return [call.arguments[index] for index in indices if index < len(call.arguments)]


def _argument(call: CallSite, index: int | None) -> str | None:
    if index is None or index >= len(call.arguments):
        return None
    return call.arguments[index]


def _call_result_symbols(
    analysis: FunctionAnalysis, node_id: str
) -> set[str]:
    return {
        definition.symbol
        for definition in analysis.dataflow.definitions.values()
        if definition.node_id == node_id and definition.kind == "call_result"
    }


def _is_numeric_cast_type(cast_type: str) -> bool:
    normalized = " ".join(cast_type.lower().replace("*", " * ").split())
    if "*" in normalized:
        return False
    tokens = set(normalized.split())
    return bool(
        tokens
        & {
            "char",
            "short",
            "int",
            "long",
            "float",
            "double",
            "signed",
            "unsigned",
            "size_t",
            "ssize_t",
        }
    )


def _assignment_destination_type(
    analysis: FunctionAnalysis, symbols: set[str]
) -> str | None:
    return next(
        (
            analysis.function.symbol_types[symbol]
            for symbol in sorted(symbols)
            if symbol in analysis.function.symbol_types
        ),
        None,
    )


def _symbol_types_for(
    analysis: FunctionAnalysis, symbols: set[str]
) -> dict[str, str]:
    return {
        symbol: analysis.function.symbol_types[symbol]
        for symbol in sorted(symbols)
        if symbol in analysis.function.symbol_types
    }


def _implicit_integer_conversion(
    destination_type: str | None,
    source_types: dict[str, str],
) -> dict[str, bool] | None:
    destination = _integer_type_properties(destination_type)
    sources = [
        value
        for raw in source_types.values()
        for value in [_integer_type_properties(raw)]
        if value is not None
    ]
    if destination is None or not sources:
        return None
    destination_signed, destination_rank = destination
    signedness_change = any(
        source_signed != destination_signed
        for source_signed, _ in sources
    )
    narrowing = any(
        source_rank > destination_rank for _, source_rank in sources
    )
    if not signedness_change and not narrowing:
        return None
    return {
        "implicit_conversion": True,
        "signedness_change": signedness_change,
        "narrowing": narrowing,
    }


def _integer_type_properties(raw_type: str | None) -> tuple[bool, int] | None:
    if not raw_type or "*" in raw_type:
        return None
    normalized = " ".join(raw_type.lower().split())
    tokens = set(normalized.split())
    if tokens & {"float", "double", "void", "struct", "union", "enum"}:
        return None
    if "size_t" in tokens:
        return True, 4
    if "ssize_t" in tokens:
        return False, 4
    if not tokens & {"char", "short", "int", "long", "signed", "unsigned"}:
        return None
    unsigned = "unsigned" in tokens
    if "char" in tokens:
        rank = 1
    elif "short" in tokens:
        rank = 2
    elif "long" in tokens:
        rank = 5 if normalized.count("long") >= 2 else 4
    else:
        rank = 3
    return unsigned, rank
