from __future__ import annotations

from dataclasses import dataclass

from .knowledge import KnowledgeRetriever, format_knowledge
from .models import Candidate, ExpertFamily


@dataclass(slots=True)
class ExpertContext:
    candidate_id: str
    expert: ExpertFamily
    code: str
    evidence_text: str
    comments_untrusted: str = ""
    knowledge_text: str = "No retrieved security knowledge."


class ContextBuilder:
    def __init__(
        self,
        max_characters: int = 20_000,
        *,
        knowledge_retriever: KnowledgeRetriever | None = None,
        max_knowledge_characters: int = 6_000,
    ) -> None:
        self.max_characters = max_characters
        self.knowledge_retriever = knowledge_retriever
        self.max_knowledge_characters = max_knowledge_characters

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
        raw_code = candidate.code[: self.max_characters]
        code, comments = _separate_cpp_comments(raw_code)
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
            comments_untrusted=comments,
            knowledge_text=format_knowledge(
                knowledge, max_characters=self.max_knowledge_characters
            ),
        )


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
        "unchecked_nullable_dereference", "guard_protects_sink",
    },
    ExpertFamily.CONCURRENCY_TOCTOU: {
        "concurrency", "synchronization", "toctou", "thread_spawn",
        "lock_acquire", "lock_release", "toctou_check_use",
    },
}
