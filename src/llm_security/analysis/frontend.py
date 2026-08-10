from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

from .ir import (
    Assignment,
    CallSite,
    Condition,
    FunctionIR,
    MemoryAccess,
    ProgramIR,
    SourceSpan,
)


_C_LANGUAGE = Language(tree_sitter_c.language())
_CPP_LANGUAGE = Language(tree_sitter_cpp.language())
_IDENTIFIER_TYPES = {
    "identifier",
    "field_identifier",
    "namespace_identifier",
    "type_identifier",
}
_COMPARISON_OPERATORS = {"<", "<=", ">", ">=", "==", "!="}
_FUNCTION_NAME_RE = re.compile(
    r"(?P<name>(?:[A-Za-z_~]\w*::)*[A-Za-z_~]\w*|operator\s*[^\s(]+)\s*\("
)


class TreeSitterFrontend:
    """Convert C/C++ Tree-sitter ASTs into a stable, reusable local IR."""

    def parse_project(self, source_files: dict[str, str]) -> ProgramIR:
        parsed: list[FunctionIR] = []
        for file_name, source in sorted(source_files.items()):
            parsed.extend(self.parse_file(file_name, source))
        functions: dict[str, FunctionIR] = {}
        keys_by_name: dict[str, list[str]] = defaultdict(list)
        for function in sorted(
            parsed, key=lambda item: (item.file, item.line_start, item.name)
        ):
            key = function.name
            if key in functions:
                key = f"{function.file}:{function.name}:{function.line_start}"
            functions[key] = function
            keys_by_name[function.name].append(key)

        callers = {key: set() for key in functions}
        callees = {key: set() for key in functions}
        for caller_key, function in functions.items():
            for call in function.calls:
                targets = _direct_call_targets(call.callee, keys_by_name)
                if len(targets) != 1:
                    continue
                target = targets[0]
                callees[caller_key].add(target)
                callers[target].add(caller_key)
        return ProgramIR(functions=functions, callers=callers, callees=callees)

    def parse_file(self, file: str, source: str) -> list[FunctionIR]:
        encoded = source.encode("utf-8")
        parser = Parser(_language_for(file))
        tree = parser.parse(encoded)
        function_nodes = sorted(
            (
                node
                for node in _walk(tree.root_node)
                if node.type == "function_definition"
            ),
            key=lambda node: (node.start_byte, node.end_byte),
        )
        return [
            self._function_ir(node, file=file, source=encoded)
            for node in function_nodes
        ]

    def _function_ir(self, node: Node, *, file: str, source: bytes) -> FunctionIR:
        declarator = node.child_by_field_name("declarator")
        name = _function_name(declarator, source)
        function_code = _text(node, source)
        nodes = list(_walk(node))

        parameters = [
            parameter
            for parameter_node in nodes
            if parameter_node.type == "parameter_declaration"
            for parameter in [_defined_symbol(parameter_node, source)]
            if parameter
        ]

        assignments: list[Assignment] = []
        assignment_nodes = [
            item
            for item in nodes
            if item.type in {"init_declarator", "assignment_expression"}
        ]
        for item in assignment_nodes:
            assignment = _assignment(item, file=file, function=name, source=source)
            if assignment is not None:
                assignments.append(assignment)

        calls = [
            _call_site(item, file=file, function=name, source=source)
            for item in nodes
            if item.type == "call_expression"
        ]
        conditions = [
            condition
            for item in nodes
            if item.type in {"if_statement", "for_statement", "while_statement"}
            for condition in [
                _condition(item, file=file, function=name, source=source)
            ]
            if condition is not None
        ]
        memory_accesses = [
            access
            for item in nodes
            for access in [
                _memory_access(item, file=file, function=name, source=source)
            ]
            if access is not None
        ]
        returns = [
            _span(item, file)
            for item in nodes
            if item.type == "return_statement"
        ]
        return FunctionIR(
            name=name,
            file=file,
            line_start=node.start_point.row + 1,
            line_end=node.end_point.row + 1,
            parameters=list(dict.fromkeys(parameters)),
            calls=sorted(calls, key=lambda item: item.span.line_start),
            assignments=sorted(
                assignments, key=lambda item: item.span.line_start
            ),
            conditions=sorted(conditions, key=lambda item: item.span.line_start),
            memory_accesses=sorted(
                memory_accesses,
                key=lambda item: (item.span.line_start, item.span.column_start, item.kind),
            ),
            returns=sorted(returns, key=lambda item: item.line_start),
            code=function_code,
        )


def _language_for(file: str) -> Language:
    return _C_LANGUAGE if Path(file).suffix.lower() == ".c" else _CPP_LANGUAGE


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _text(node: Node | None, source: bytes) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _span(node: Node, file: str) -> SourceSpan:
    return SourceSpan(
        file=file,
        line_start=node.start_point.row + 1,
        line_end=node.end_point.row + 1,
        column_start=node.start_point.column,
        column_end=node.end_point.column,
    )


def _stable_id(prefix: str, file: str, function: str, node: Node) -> str:
    digest = hashlib.sha1(
        f"{file}:{function}:{node.type}:{node.start_byte}:{node.end_byte}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _function_name(declarator: Node | None, source: bytes) -> str:
    text = _text(declarator, source)
    matches = list(_FUNCTION_NAME_RE.finditer(text))
    if matches:
        return matches[-1].group("name").strip()
    identifiers = _identifier_texts(declarator, source) if declarator else []
    return identifiers[-1] if identifiers else "<anonymous>"


def _defined_symbol(node: Node, source: bytes) -> str | None:
    declarator = node.child_by_field_name("declarator")
    identifiers = _identifier_texts(declarator or node, source)
    return identifiers[-1] if identifiers else None


def _identifier_texts(node: Node, source: bytes) -> list[str]:
    return [
        _text(item, source)
        for item in _walk(node)
        if item.type in _IDENTIFIER_TYPES
    ]


def _symbols(node: Node | None, source: bytes) -> set[str]:
    if node is None:
        return set()
    return {
        _text(item, source)
        for item in _walk(node)
        if item.type in {"identifier", "field_identifier"}
    }


def _assignment(
    node: Node, *, file: str, function: str, source: bytes
) -> Assignment | None:
    if node.type == "init_declarator":
        left = node.child_by_field_name("declarator")
        right = node.child_by_field_name("value")
    else:
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
    if left is None or right is None:
        return None
    target = _text(left, source).strip()
    defined = _defined_symbol(left, source)
    return Assignment(
        assignment_id=_stable_id("AS", file, function, node),
        target=target,
        expression=_text(right, source).strip(),
        defs={defined} if defined else set(),
        uses=_symbols(right, source),
        span=_span(node, file),
        text=_text(node, source).strip(),
    )


def _call_site(
    node: Node, *, file: str, function: str, source: bytes
) -> CallSite:
    callee_node = node.child_by_field_name("function")
    arguments_node = node.child_by_field_name("arguments")
    arguments = (
        [_text(item, source).strip() for item in arguments_node.named_children]
        if arguments_node is not None
        else []
    )
    return CallSite(
        call_id=_stable_id("CALL", file, function, node),
        callee=_normalize_callee(_text(callee_node, source)),
        arguments=arguments,
        assigned_to=_assigned_target(node, source),
        span=_span(node, file),
        text=_text(node, source).strip(),
    )


def _assigned_target(node: Node, source: bytes) -> str | None:
    current = node.parent
    while current is not None and current.type not in {
        "expression_statement",
        "declaration",
        "return_statement",
        "compound_statement",
    }:
        if current.type == "assignment_expression":
            return _text(current.child_by_field_name("left"), source).strip() or None
        if current.type == "init_declarator":
            declarator = current.child_by_field_name("declarator")
            return _defined_symbol(declarator or current, source)
        current = current.parent
    return None


def _condition(
    node: Node, *, file: str, function: str, source: bytes
) -> Condition | None:
    condition_node = node.child_by_field_name("condition")
    if condition_node is None:
        return None
    expression = _text(condition_node, source).strip()
    if expression.startswith("(") and expression.endswith(")"):
        expression = expression[1:-1].strip()
    return Condition(
        condition_id=_stable_id("COND", file, function, condition_node),
        expression=expression,
        symbols=_symbols(condition_node, source),
        operator=_comparison_operator(condition_node),
        span=_span(condition_node, file),
        text=_text(node, source).strip(),
    )


def _comparison_operator(node: Node) -> str | None:
    current = node
    while current.type in {"parenthesized_expression", "condition_clause"}:
        named = current.named_children
        if len(named) != 1:
            break
        current = named[0]
    if current.type != "binary_expression":
        return None
    return next(
        (child.type for child in current.children if child.type in _COMPARISON_OPERATORS),
        None,
    )


def _memory_access(
    node: Node, *, file: str, function: str, source: bytes
) -> MemoryAccess | None:
    if node.type == "subscript_expression":
        base = node.child_by_field_name("argument") or (
            node.named_children[0] if node.named_children else None
        )
        index = node.child_by_field_name("index") or (
            node.named_children[1] if len(node.named_children) > 1 else None
        )
        kind = "array_write" if _is_assignment_target(node) else "array_read"
        index_text = _text(index, source).strip()
        if index_text.startswith("[") and index_text.endswith("]"):
            index_text = index_text[1:-1].strip()
        return MemoryAccess(
            access_id=_stable_id("MEM", file, function, node),
            base=_text(base, source).strip(),
            index=index_text or None,
            kind=kind,
            span=_span(node, file),
            text=_text(node, source).strip(),
        )
    if node.type == "pointer_expression" and _text(node, source).lstrip().startswith("*"):
        argument = node.child_by_field_name("argument") or (
            node.named_children[-1] if node.named_children else None
        )
        return MemoryAccess(
            access_id=_stable_id("MEM", file, function, node),
            base=_text(argument, source).strip(),
            index=None,
            kind="dereference",
            span=_span(node, file),
            text=_text(node, source).strip(),
        )
    if node.type == "field_expression" and "->" in _text(node, source):
        argument = node.child_by_field_name("argument") or (
            node.named_children[0] if node.named_children else None
        )
        return MemoryAccess(
            access_id=_stable_id("MEM", file, function, node),
            base=_text(argument, source).strip(),
            index=None,
            kind="dereference",
            span=_span(node, file),
            text=_text(node, source).strip(),
        )
    return None


def _is_assignment_target(node: Node) -> bool:
    current = node
    while current.parent is not None and current.parent.type in {
        "parenthesized_expression",
        "field_expression",
        "subscript_expression",
    }:
        current = current.parent
    parent = current.parent
    return bool(
        parent is not None
        and parent.type == "assignment_expression"
        and parent.child_by_field_name("left") == current
    )


def _normalize_callee(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).replace("->", ".")


def _direct_call_targets(
    callee: str, keys_by_name: dict[str, list[str]]
) -> list[str]:
    candidates = [callee]
    if "." in callee:
        candidates.append(callee.rsplit(".", 1)[-1])
    if "::" in callee:
        candidates.append(callee.rsplit("::", 1)[-1])
    matches: list[str] = []
    for candidate in candidates:
        matches.extend(keys_by_name.get(candidate, []))
        matches.extend(
            key
            for name, keys in keys_by_name.items()
            if name.endswith("::" + candidate)
            for key in keys
        )
    return list(dict.fromkeys(matches))
