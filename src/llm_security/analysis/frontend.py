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
    ControlRegion,
    ExpressionInfo,
    FunctionIR,
    MemoryAccess,
    ProgramIR,
    SourceSpan,
    StatementIR,
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
_ARITHMETIC_OPERATORS = {"+", "-", "*", "/", "%", "<<", ">>"}
_CONTROL_TYPES = {"if_statement", "while_statement", "for_statement"}
_STATEMENT_TYPES = {"declaration", "expression_statement", "return_statement"}
_UNSUPPORTED_CONTROL_TYPES = {
    "break_statement": "break",
    "continue_statement": "continue",
    "goto_statement": "goto",
    "switch_statement": "switch",
}
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
        conditions: list[Condition] = []
        condition_by_control: dict[tuple[int, int], Condition] = {}
        for item in nodes:
            if item.type not in _CONTROL_TYPES:
                continue
            condition = _condition(item, file=file, function=name, source=source)
            if condition is None:
                condition = _implicit_condition(
                    item, file=file, function=name, source=source
                )
            conditions.append(condition)
            condition_by_control[(item.start_byte, item.end_byte)] = condition

        assignment_by_id = {item.assignment_id: item for item in assignments}
        call_by_id = {item.call_id: item for item in calls}
        controls = [
            _control_region(
                item,
                condition=condition_by_control[(item.start_byte, item.end_byte)],
                file=file,
                function=name,
                source=source,
                assignments=assignment_by_id,
                calls=call_by_id,
            )
            for item in nodes
            if item.type in _CONTROL_TYPES
        ]
        statements = [
            _statement_ir(
                item,
                file=file,
                function=name,
                source=source,
                assignments=assignment_by_id,
                calls=call_by_id,
            )
            for item in nodes
            if item.type in _STATEMENT_TYPES and not _is_for_header_component(item)
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
            statements=sorted(statements, key=_statement_sort_key),
            controls=sorted(controls, key=_control_sort_key),
            calls=sorted(calls, key=lambda item: _span_sort_key(item.span)),
            assignments=sorted(
                assignments, key=lambda item: _span_sort_key(item.span)
            ),
            conditions=sorted(conditions, key=lambda item: _span_sort_key(item.span)),
            memory_accesses=sorted(
                memory_accesses,
                key=lambda item: (item.span.line_start, item.span.column_start, item.kind),
            ),
            returns=sorted(returns, key=lambda item: item.line_start),
            code=function_code,
            unsupported_constructs=sorted(
                {
                    value
                    for item in nodes
                    for node_type, value in _UNSUPPORTED_CONTROL_TYPES.items()
                    if item.type == node_type
                }
            ),
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


def _expression_symbols(node: Node | None, source: bytes) -> set[str]:
    """Return data symbols while excluding the name part of a call expression."""
    if node is None:
        return set()
    if node.type == "call_expression":
        arguments = node.child_by_field_name("arguments")
        if arguments is None:
            return set()
        return {
            symbol
            for argument in arguments.named_children
            for symbol in _expression_symbols(argument, source)
        }
    if node.type in {"identifier", "field_identifier"}:
        return {_text(node, source)}
    return {
        symbol
        for child in node.named_children
        for symbol in _expression_symbols(child, source)
    }


def _expression_info(node: Node | None, source: bytes) -> ExpressionInfo:
    if node is None:
        return ExpressionInfo("", set(), set(), [], False)
    cast_types = [
        _text(type_node, source).strip()
        for item in _walk(node)
        if item.type == "cast_expression"
        for type_node in [item.child_by_field_name("type")]
        if type_node is not None
    ]
    operators = {
        child.type
        for item in _walk(node)
        for child in item.children
        if not child.is_named and child.type in _ARITHMETIC_OPERATORS
    }
    return ExpressionInfo(
        text=_text(node, source).strip(),
        symbols=_expression_symbols(node, source),
        operators=operators,
        cast_types=cast_types,
        has_sizeof=any(item.type == "sizeof_expression" for item in _walk(node)),
    )


def _assignment_defined_symbol(left: Node, source: bytes) -> str | None:
    current = left
    while current.type == "parenthesized_expression" and current.named_children:
        current = current.named_children[0]
    if current.type in {
        "subscript_expression",
        "field_expression",
        "pointer_expression",
    }:
        return None
    return _defined_symbol(current, source)


def _assignment_operator(node: Node) -> str | None:
    if node.type != "assignment_expression":
        return None
    return next(
        (
            child.type
            for child in node.children
            if not child.is_named and child.type.endswith("=")
        ),
        None,
    )


def _statement_ir(
    node: Node,
    *,
    file: str,
    function: str,
    source: bytes,
    assignments: dict[str, Assignment],
    calls: dict[str, CallSite],
    context_override: tuple[str | None, str | None] | None = None,
) -> StatementIR:
    assignment_ids = [
        assignment_id
        for item in _walk(node)
        if item.type in {"init_declarator", "assignment_expression"}
        for assignment_id in [_stable_id("AS", file, function, item)]
        if assignment_id in assignments
    ]
    call_ids = [
        call_id
        for item in _walk(node)
        if item.type == "call_expression"
        for call_id in [_stable_id("CALL", file, function, item)]
        if call_id in calls
    ]
    defs = {
        symbol
        for assignment_id in assignment_ids
        for symbol in assignments[assignment_id].defs
    }
    uses = {
        symbol
        for assignment_id in assignment_ids
        for symbol in assignments[assignment_id].uses
    }
    uses.update(
        symbol
        for call_id in call_ids
        for argument in calls[call_id].argument_symbols
        for symbol in argument
    )

    if node.type == "declaration":
        kind = "declaration"
        defs.update(_declaration_symbols(node, source))
    elif node.type == "return_statement":
        kind = "return"
    elif assignment_ids:
        kind = "assignment"
    elif call_ids:
        kind = "call"
    else:
        kind = "expression"

    if not assignment_ids and not call_ids:
        uses.update(_expression_symbols(node, source))
    update_symbols = _update_expression_symbols(node, source)
    if update_symbols:
        defs.update(update_symbols)
        uses.update(update_symbols)

    parent_control_id, control_branch = (
        context_override
        if context_override is not None
        else _control_context(node, file=file, function=function)
    )
    return StatementIR(
        statement_id=_stable_id("STMT", file, function, node),
        kind=kind,
        span=_span(node, file),
        text=_text(node, source).strip(),
        defs=defs,
        uses=uses,
        assignment_ids=assignment_ids,
        call_ids=call_ids,
        parent_control_id=parent_control_id,
        control_branch=control_branch,
    )


def _declaration_symbols(node: Node, source: bytes) -> set[str]:
    symbols: set[str] = set()
    for child in node.named_children:
        if child.type == "init_declarator":
            declarator = child.child_by_field_name("declarator")
            symbol = _defined_symbol(declarator or child, source)
        elif child.type == "identifier" or "declarator" in child.type:
            symbol = _defined_symbol(child, source)
        else:
            symbol = None
        if symbol:
            symbols.add(symbol)
    return symbols


def _update_expression_symbols(node: Node, source: bytes) -> set[str]:
    return {
        symbol
        for item in _walk(node)
        if item.type == "update_expression"
        for symbol in _expression_symbols(item, source)
    }


def _control_region(
    node: Node,
    *,
    condition: Condition,
    file: str,
    function: str,
    source: bytes,
    assignments: dict[str, Assignment],
    calls: dict[str, CallSite],
) -> ControlRegion:
    control_id = _stable_id("CTRL", file, function, node)
    parent_control_id, control_branch = _control_context(
        node, file=file, function=function
    )
    body = (
        node.child_by_field_name("consequence")
        if node.type == "if_statement"
        else node.child_by_field_name("body")
    )
    alternative = (
        node.child_by_field_name("alternative")
        if node.type == "if_statement"
        else None
    )
    initializer_node = (
        node.child_by_field_name("initializer")
        if node.type == "for_statement"
        else None
    )
    update_node = (
        node.child_by_field_name("update")
        if node.type == "for_statement"
        else None
    )
    context = (parent_control_id, control_branch)
    initializer = (
        _statement_ir(
            initializer_node,
            file=file,
            function=function,
            source=source,
            assignments=assignments,
            calls=calls,
            context_override=context,
        )
        if initializer_node is not None
        else None
    )
    update = (
        _statement_ir(
            update_node,
            file=file,
            function=function,
            source=source,
            assignments=assignments,
            calls=calls,
            context_override=(control_id, "body"),
        )
        if update_node is not None
        else None
    )
    return ControlRegion(
        control_id=control_id,
        kind=node.type.removesuffix("_statement"),
        condition_id=condition.condition_id,
        span=_span(node, file),
        body_span=_span(body or node, file),
        alternative_span=_span(alternative, file) if alternative is not None else None,
        parent_control_id=parent_control_id,
        control_branch=control_branch,
        initializer=initializer,
        update=update,
    )


def _control_context(
    node: Node, *, file: str, function: str
) -> tuple[str | None, str | None]:
    current = node.parent
    while current is not None:
        if current.type in _CONTROL_TYPES:
            control_id = _stable_id("CTRL", file, function, current)
            if current.type == "if_statement":
                alternative = current.child_by_field_name("alternative")
                if alternative is not None and _contains(alternative, node):
                    return control_id, "else"
                consequence = current.child_by_field_name("consequence")
                if consequence is not None and _contains(consequence, node):
                    return control_id, "then"
            else:
                body = current.child_by_field_name("body")
                if body is not None and _contains(body, node):
                    return control_id, "body"
        current = current.parent
    return None, None


def _contains(container: Node, item: Node) -> bool:
    return (
        container.start_byte <= item.start_byte
        and item.end_byte <= container.end_byte
    )


def _is_for_header_component(node: Node) -> bool:
    current = node.parent
    while current is not None:
        if current.type == "for_statement":
            for field in ("initializer", "update"):
                component = current.child_by_field_name(field)
                if component is not None and _contains(component, node):
                    return True
        current = current.parent
    return False


def _span_sort_key(span: SourceSpan) -> tuple[int, int, int, int]:
    return (
        span.line_start,
        span.column_start,
        span.line_end,
        span.column_end,
    )


def _statement_sort_key(item: StatementIR) -> tuple[int, int, int, int, str]:
    return (*_span_sort_key(item.span), item.statement_id)


def _control_sort_key(item: ControlRegion) -> tuple[int, int, int, int, str]:
    return (*_span_sort_key(item.span), item.control_id)


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
    defined = _assignment_defined_symbol(left, source)
    uses = _expression_symbols(right, source)
    if defined is None:
        uses.update(_expression_symbols(left, source))
    if node.type == "assignment_expression" and _assignment_operator(node) != "=":
        uses.update(_expression_symbols(left, source))
    return Assignment(
        assignment_id=_stable_id("AS", file, function, node),
        target=target,
        expression=_text(right, source).strip(),
        defs={defined} if defined else set(),
        uses=uses,
        span=_span(node, file),
        text=_text(node, source).strip(),
        expression_info=_expression_info(right, source),
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
    argument_symbols = (
        [_expression_symbols(item, source) for item in arguments_node.named_children]
        if arguments_node is not None
        else []
    )
    argument_info = (
        [_expression_info(item, source) for item in arguments_node.named_children]
        if arguments_node is not None
        else []
    )
    return CallSite(
        call_id=_stable_id("CALL", file, function, node),
        callee=_normalize_callee(_text(callee_node, source)),
        arguments=arguments,
        argument_symbols=argument_symbols,
        argument_info=argument_info,
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
    left_info, right_info, unary_operator = _condition_expression_info(
        condition_node, source
    )
    return Condition(
        condition_id=_stable_id("COND", file, function, condition_node),
        expression=expression,
        symbols=_expression_symbols(condition_node, source),
        operator=_comparison_operator(condition_node),
        span=_span(condition_node, file),
        text=_text(node, source).strip(),
        left_info=left_info,
        right_info=right_info,
        unary_operator=unary_operator,
    )


def _implicit_condition(
    node: Node, *, file: str, function: str, source: bytes
) -> Condition:
    return Condition(
        condition_id=_stable_id("COND", file, function, node),
        expression="true",
        symbols=set(),
        operator=None,
        span=_span(node, file),
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


def _condition_expression_info(
    node: Node, source: bytes
) -> tuple[ExpressionInfo | None, ExpressionInfo | None, str | None]:
    current = node
    while current.type in {"parenthesized_expression", "condition_clause"}:
        named = current.named_children
        if len(named) != 1:
            break
        current = named[0]
    if current.type == "binary_expression":
        left = current.child_by_field_name("left")
        right = current.child_by_field_name("right")
        return _expression_info(left, source), _expression_info(right, source), None
    if current.type in {"unary_expression", "pointer_expression"}:
        argument = current.child_by_field_name("argument") or (
            current.named_children[-1] if current.named_children else None
        )
        operator = next(
            (
                child.type
                for child in current.children
                if not child.is_named and child.type in {"!", "~", "+", "-"}
            ),
            None,
        )
        return _expression_info(argument, source), None, operator
    return _expression_info(current, source), None, None


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
            base_symbols=_expression_symbols(base, source),
            index_symbols=_expression_symbols(index, source),
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
            base_symbols=_expression_symbols(argument, source),
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
            base_symbols=_expression_symbols(argument, source),
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
