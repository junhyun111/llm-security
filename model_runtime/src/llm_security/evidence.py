from __future__ import annotations

from dataclasses import dataclass

from .cwe import expert_for_cwe
from .knowledge import KnowledgeRetriever, format_knowledge
from .models import Candidate, ExpertFamily


@dataclass(slots=True)
class ExpertContext:
    candidate_id: str
    expert: ExpertFamily
    code: str
    evidence_text: str
    cwe_hypotheses_text: str
    comments_untrusted: str = ""
    knowledge_text: str = "No retrieved security knowledge."
    code_slice: str = "No evidence-local code slice."
    related_functions_text: str = "No direct caller/callee summaries."
    type_information_text: str = "No explicit type or conversion information."
    evidence_graph_text: str = "No evidence graph edges."


class ContextBuilder:
    def __init__(
        self,
        max_characters: int = 20_000,
        *,
        knowledge_retriever: KnowledgeRetriever | None = None,
        max_knowledge_characters: int = 6_000,
        slice_context_lines: int = 4,
        max_related_context_characters: int = 8_000,
    ) -> None:
        self.max_characters = max_characters
        self.knowledge_retriever = knowledge_retriever
        self.max_knowledge_characters = max_knowledge_characters
        self.slice_context_lines = slice_context_lines
        self.max_related_context_characters = max_related_context_characters

    def build(self, candidate: Candidate, expert: ExpertFamily) -> ExpertContext:
        relevant_kinds = _EXPERT_EVIDENCE_KINDS[expert]
        selected = [
            item
            for item in candidate.evidence
            if item.kind in relevant_kinds or item.kind in {"guard", "error_path"}
        ]
        evidence_lines = [
            f"[{item.evidence_id}] {item.kind} {item.file}:{item.line}: {item.expression}"
            for item in selected
        ]
        cwe_lines = [
            (
                f"[{hypothesis.cwe}] confidence={hypothesis.confidence:.3f} "
                f"scope={'selected-expert' if expert_for_cwe(hypothesis.cwe) == expert else 'cross-family'} "
                f"evidence={','.join(hypothesis.evidence_ids) or '(none)'}; "
                f"reasons={'; '.join(hypothesis.reasons) or '(none)'}"
            )
            for hypothesis in candidate.cwe_hypotheses
        ]
        raw_code = candidate.code[: self.max_characters]
        code, comments = _separate_cpp_comments(raw_code)
        code_slice, slice_comments = _vulnerability_slice(
            candidate,
            selected,
            context_lines=self.slice_context_lines,
        )
        related_functions, related_comments = _format_related_functions(
            candidate,
            expert,
            max_characters=self.max_related_context_characters,
        )
        comments = "\n".join(
            value for value in (comments, slice_comments, related_comments) if value
        )
        knowledge = (
            self.knowledge_retriever.retrieve(candidate, expert)
            if self.knowledge_retriever is not None
            else []
        )
        return ExpertContext(
            candidate_id=candidate.candidate_id,
            expert=expert,
            code=code,
            evidence_text="\n".join(evidence_lines) or "No matching static evidence.",
            cwe_hypotheses_text=(
                "\n".join(cwe_lines) or "No static CWE hypothesis."
            ),
            comments_untrusted=comments,
            knowledge_text=format_knowledge(
                knowledge, max_characters=self.max_knowledge_characters
            ),
            code_slice=code_slice,
            related_functions_text=related_functions,
            type_information_text=_format_type_information(candidate, selected),
            evidence_graph_text=_format_evidence_graph(selected),
        )


def _vulnerability_slice(
    candidate: Candidate,
    evidence,
    *,
    context_lines: int,
) -> tuple[str, str]:
    source_lines = candidate.code.splitlines()
    if not source_lines:
        return "No evidence-local code slice.", ""
    evidence_lines = [
        item.line
        for item in evidence
        if candidate.line_start <= item.line <= candidate.line_end
    ]
    if not evidence_lines:
        evidence_lines = [candidate.line_start]
    indexes: set[int] = set()
    for line in evidence_lines:
        local = line - candidate.line_start
        indexes.update(
            range(
                max(0, local - context_lines),
                min(len(source_lines), local + context_lines + 1),
            )
        )
    chunks: list[str] = []
    previous = -2
    for index in sorted(indexes):
        if index > previous + 1:
            chunks.append("...")
        chunks.append(f"{candidate.line_start + index}: {source_lines[index]}")
        previous = index
    return _separate_cpp_comments("\n".join(chunks))


def _format_related_functions(
    candidate: Candidate,
    expert: ExpertFamily,
    *,
    max_characters: int,
) -> tuple[str, str]:
    relevant = _EXPERT_EVIDENCE_KINDS[expert]
    prioritized = sorted(
        candidate.related_functions,
        key=lambda item: (
            not bool(set(item.semantic_facts) & relevant),
            item.relation,
            item.file,
            item.line_start,
        ),
    )
    sections: list[str] = []
    comments: list[str] = []
    for item in prioritized:
        normalized, untrusted = _separate_cpp_comments(item.code)
        section = (
            f"[{item.relation}] {item.file}:{item.line_start}-{item.line_end} "
            f"function={item.function}; parameters={item.parameters or ['(none)']}; "
            f"calls={item.calls or ['(none)']}; "
            f"types={item.symbol_types or {'(none)': '(unknown)'}}; "
            f"semantic_effects={item.semantic_facts or ['(none)']}\n{normalized}"
        )
        if sum(len(value) for value in sections) + len(section) > max_characters:
            break
        sections.append(section)
        if untrusted:
            comments.append(
                f"related function {item.function}:\n{untrusted}"
            )
    return (
        "\n\n".join(sections) or "No direct caller/callee summaries.",
        "\n".join(comments),
    )


def _format_type_information(candidate: Candidate, evidence) -> str:
    lines: list[str] = [
        f"{symbol}: {declared_type}"
        for symbol, declared_type in sorted(candidate.symbol_types.items())
    ]
    for item in evidence:
        cast_types = item.facts.get("cast_types", [])
        if isinstance(cast_types, str):
            cast_types = [cast_types]
        if cast_types:
            lines.append(
                f"[{item.evidence_id}] symbols={item.subject or '(unknown)'}; "
                f"cast_types={','.join(str(value) for value in cast_types)}"
            )
        destination_type = item.facts.get("destination_type")
        source_types = item.facts.get("source_types", {})
        if destination_type or source_types:
            lines.append(
                f"[{item.evidence_id}] source_types={source_types or '(unknown)'}; "
                f"destination_type={destination_type or '(unknown)'}; "
                f"signedness_change={item.facts.get('signedness_change', False)}; "
                f"narrowing={item.facts.get('narrowing', False)}"
            )
    return "\n".join(lines) or "No explicit type or conversion information."


def _format_evidence_graph(evidence) -> str:
    lines: list[str] = []
    for item in evidence:
        path = item.facts.get("path", [])
        path_text = " -> ".join(str(value) for value in path) if path else "(local)"
        lines.append(
            f"[{item.evidence_id}] {item.kind}: "
            f"{item.subject or '(unknown)'} -> {item.object or '(unknown)'}; "
            f"path={path_text}"
        )
    return "\n".join(lines) or "No evidence graph edges."


def _separate_cpp_comments(source: str) -> tuple[str, str]:
    """Return a line-preserving code view and an isolated untrusted comment view."""
    normalized: list[str] = []
    comments: list[str] = []
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == "/" and next_char == "/":
                state = "line_comment"
                normalized.extend((" ", " "))
                comments.extend(("/", "/"))
                index += 2
                continue
            if char == "/" and next_char == "*":
                state = "block_comment"
                normalized.extend((" ", " "))
                comments.extend(("/", "*"))
                index += 2
                continue
            if char in {'"', "'"}:
                state = "quoted"
                quote = char
            normalized.append(char)
            index += 1
            continue
        if state == "quoted":
            normalized.append(char)
            if char == "\\" and index + 1 < len(source):
                normalized.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                state = "code"
            index += 1
            continue
        comments.append(char)
        if char == "\n":
            normalized.append("\n")
            if state == "line_comment":
                state = "code"
        else:
            normalized.append(" ")
        if state == "block_comment" and char == "*" and next_char == "/":
            comments.append("/")
            normalized.append(" ")
            index += 2
            state = "code"
            continue
        index += 1
    return "".join(normalized), "".join(comments).strip()


_EXPERT_EVIDENCE_KINDS: dict[ExpertFamily, set[str]] = {
    ExpertFamily.MEMORY_BOUNDS: {
        "memory_sink", "memory_access", "allocation", "memory_copy",
        "memory_copy_without_guard", "unchecked_index", "guard_protects_sink",
        "release", "use_after_release", "double_release",
        "unchecked_nullable_dereference",
    },
    ExpertFamily.LIFETIME_RESOURCE: {
        "allocation", "release", "memory_access", "use_after_release",
        "double_release", "unchecked_nullable_dereference",
    },
    ExpertFamily.INTEGER_SIZE_TYPE: {
        "integer_arithmetic",
        "type_conversion",
        "numeric_conversion",
        "allocation",
        "memory_sink",
        "arithmetic_to_allocation",
        "arithmetic_to_memory_sink",
        "cast_to_size_sink",
    },
    ExpertFamily.TAINT_API_CONTRACT: {
        "taint_source", "taint_sink", "memory_sink", "source_to_sink",
        "unsanitized_source_to_sink",
    },
    ExpertFamily.CONTROL_STATE_ERROR: {
        "state", "error_path", "guard", "uninitialized_use",
        "unchecked_nullable_dereference", "unchecked_call_result",
        "guard_protects_sink",
    },
    ExpertFamily.CONCURRENCY_TOCTOU: {
        "concurrency", "synchronization", "toctou", "thread_spawn",
        "lock_acquire", "lock_release", "toctou_check_use",
    },
}
