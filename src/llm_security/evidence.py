from __future__ import annotations

from dataclasses import dataclass

from .models import Candidate, ExpertFamily


@dataclass(slots=True)
class ExpertContext:
    candidate_id: str
    expert: ExpertFamily
    code: str
    evidence_text: str


class ContextBuilder:
    def __init__(self, max_characters: int = 20_000) -> None:
        self.max_characters = max_characters

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
        code = candidate.code[: self.max_characters]
        return ExpertContext(
            candidate_id=candidate.candidate_id,
            expert=expert,
            code=code,
            evidence_text="\n".join(evidence_lines) or "No matching static evidence.",
        )


_EXPERT_EVIDENCE_KINDS: dict[ExpertFamily, set[str]] = {
    ExpertFamily.MEMORY_BOUNDS: {
        "memory_sink", "memory_access", "allocation", "memory_copy",
        "memory_copy_without_guard", "unchecked_index", "guard_protects_sink",
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
