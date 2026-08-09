from __future__ import annotations

from typing import Any

from .evidence import ExpertContext
from .models import Candidate, ExpertFamily, Finding


EXPERT_PROMPTS: dict[ExpertFamily, str] = {
    ExpertFamily.MEMORY_BOUNDS: (
        "Analyze spatial memory safety only: buffer capacity, index, copy length, "
        "pointer arithmetic, and bounds guards. Do not report speculative issues."
    ),
    ExpertFamily.LIFETIME_RESOURCE: (
        "Analyze allocation, ownership, release, aliases, use-after-free, double free, "
        "null lifetime, and cleanup paths."
    ),
    ExpertFamily.INTEGER_SIZE_TYPE: (
        "Analyze overflow, underflow, truncation, signedness, casts, and size arithmetic "
        "that can affect allocation, indexing, or copy operations."
    ),
    ExpertFamily.TAINT_API_CONTRACT: (
        "Analyze attacker-controlled sources, propagation, validation, sanitizers, sinks, "
        "and violated API preconditions."
    ),
    ExpertFamily.CONTROL_STATE_ERROR: (
        "Analyze error paths, unchecked results, state-machine invariants, guards, and "
        "pre/postconditions."
    ),
    ExpertFamily.CONCURRENCY_TOCTOU: (
        "Analyze shared accesses, locks, atomics, ordering, races, and check/use gaps."
    ),
}


def expert_messages(candidate: Candidate, context: ExpertContext) -> list[dict[str, str]]:
    system = (
        "You are a C/C++ security reviewer. "
        + EXPERT_PROMPTS[context.expert]
        + " Every factual claim must cite one of the supplied evidence IDs. "
        "Return an empty findings array when evidence is insufficient."
    )
    user = (
        f"Candidate: {candidate.candidate_id}\n"
        f"Location: {candidate.file}:{candidate.line_start}-{candidate.line_end} "
        f"function {candidate.function}\n\n"
        f"Static evidence:\n{context.evidence_text}\n\n"
        f"Code:\n{context.code}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def findings_schema() -> dict[str, Any]:
    finding = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "root_cause": {"type": "string"},
            "consequence": {"type": "string"},
            "file": {"type": "string"},
            "function": {"type": "string"},
            "line_start": {"type": "integer"},
            "line_end": {"type": "integer"},
            "cwes": {"type": "array", "items": {"type": "string"}},
            "source": {"type": ["string", "null"]},
            "sink": {"type": ["string", "null"]},
            "missing_guard": {"type": ["string", "null"]},
            "trigger_path": {"type": "array", "items": {"type": "string"}},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "title",
            "root_cause",
            "consequence",
            "file",
            "function",
            "line_start",
            "line_end",
            "cwes",
            "source",
            "sink",
            "missing_guard",
            "trigger_path",
            "evidence_ids",
            "confidence",
        ],
        "additionalProperties": False,
    }
    return {
        "name": "security_findings",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"findings": {"type": "array", "items": finding}},
            "required": ["findings"],
            "additionalProperties": False,
        },
    }


def finding_from_payload(
    payload: dict[str, Any],
    *,
    index: int,
    candidate: Candidate,
    expert: ExpertFamily,
) -> Finding:
    return Finding(
        finding_id=f"F-{candidate.candidate_id}-{expert.value}-{index}",
        candidate_id=candidate.candidate_id,
        expert=expert,
        title=str(payload["title"]),
        root_cause=str(payload["root_cause"]),
        consequence=str(payload["consequence"]),
        file=str(payload["file"]),
        function=str(payload["function"]),
        line_start=int(payload["line_start"]),
        line_end=int(payload["line_end"]),
        cwes=[str(item) for item in payload["cwes"]],
        source=None if payload["source"] is None else str(payload["source"]),
        sink=None if payload["sink"] is None else str(payload["sink"]),
        missing_guard=(
            None if payload["missing_guard"] is None else str(payload["missing_guard"])
        ),
        trigger_path=[str(item) for item in payload["trigger_path"]],
        evidence_ids=[str(item) for item in payload["evidence_ids"]],
        confidence=max(0.0, min(1.0, float(payload["confidence"]))),
    )

