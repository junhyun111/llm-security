from __future__ import annotations

from collections import defaultdict

from ..cwe import normalize_cwe
from ..models import CweHypothesis, Evidence


_DIRECT_RULES: dict[str, tuple[str, float, str]] = {
    "integer_arithmetic": (
        "CWE-190",
        0.55,
        "integer arithmetic may wrap before a security-sensitive use",
    ),
    "memory_copy_without_guard": (
        "CWE-120",
        0.95,
        "bounded copy protection is not established",
    ),
    "use_after_release": (
        "CWE-416",
        1.00,
        "reachable memory access follows release",
    ),
    "double_release": (
        "CWE-415",
        1.00,
        "the same resource reaches two release operations",
    ),
    "unchecked_nullable_dereference": (
        "CWE-476",
        0.95,
        "nullable result reaches a dereference without a protective guard",
    ),
    "arithmetic_to_allocation": (
        "CWE-190",
        0.95,
        "integer arithmetic reaches an allocation size",
    ),
    "arithmetic_to_memory_sink": (
        "CWE-190",
        0.95,
        "integer arithmetic reaches a memory operation size",
    ),
    "cast_to_size_sink": (
        "CWE-681",
        0.95,
        "numeric conversion reaches a size-sensitive operation",
    ),
    "numeric_conversion": (
        "CWE-681",
        0.90,
        "an explicit numeric conversion can narrow or alter the represented value",
    ),
    "uninitialized_use": (
        "CWE-457",
        1.00,
        "a use is reached by an uninitialized definition",
    ),
    "unchecked_call_result": (
        "CWE-252",
        0.90,
        "a security-relevant return value is not checked",
    ),
    "toctou_check_use": (
        "CWE-367",
        1.00,
        "a resource check and use are separated by a mutable interval",
    ),
}

_TAINT_SINK_CWES = {
    "system": "CWE-78",
    "popen": "CWE-78",
    "open": "CWE-22",
    "fopen": "CWE-22",
    "printf": "CWE-134",
    "fprintf": "CWE-134",
}


class StaticCweHypothesisEngine:
    """Translate semantic evidence into fallible, evidence-backed CWE leads."""

    def __init__(self, *, max_hypotheses: int = 8) -> None:
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses must be positive")
        self.max_hypotheses = max_hypotheses

    def infer(self, evidence: list[Evidence]) -> list[CweHypothesis]:
        support: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        for item in evidence:
            rule = _DIRECT_RULES.get(item.kind)
            if rule is not None:
                cwe, weight, reason = rule
                self._add(support, cwe, item, weight, reason)
            if item.kind == "unchecked_index":
                access_kind = str(item.facts.get("access_kind", ""))
                if access_kind == "array_write":
                    cwe, reason = "CWE-787", "unchecked index reaches an array write"
                elif access_kind == "array_read":
                    cwe, reason = "CWE-125", "unchecked index reaches an array read"
                else:
                    cwe, reason = "CWE-129", "array index lacks a proven upper bound"
                self._add(support, cwe, item, 0.95, reason)

        unsanitized = [
            item for item in evidence if item.kind == "unsanitized_source_to_sink"
        ]
        if unsanitized:
            for path in unsanitized:
                self._add(
                    support,
                    "CWE-20",
                    path,
                    0.80,
                    "external input reaches a sink without proven sanitization",
                )
            for sink in (item for item in evidence if item.kind == "taint_sink"):
                callee = str(sink.facts.get("callee", sink.object or "")).lower()
                cwe = _TAINT_SINK_CWES.get(callee)
                if cwe is None:
                    continue
                for path in unsanitized:
                    self._add(
                        support,
                        cwe,
                        path,
                        1.00,
                        f"unsanitized input reaches {callee}",
                    )
                    self._add(
                        support,
                        cwe,
                        sink,
                        0.90,
                        f"{callee} is the security-sensitive sink",
                    )

        hypotheses = []
        for cwe, rows in support.items():
            confidence = max(row[0] for row in rows)
            hypotheses.append(
                CweHypothesis(
                    cwe=normalize_cwe(cwe),
                    confidence=confidence,
                    evidence_ids=sorted({row[1] for row in rows}),
                    reasons=sorted({row[2] for row in rows}),
                )
            )
        return sorted(
            hypotheses,
            key=lambda item: (-item.confidence, item.cwe),
        )[: self.max_hypotheses]

    @staticmethod
    def _add(
        support: dict[str, list[tuple[float, str, str]]],
        cwe: str,
        evidence: Evidence,
        weight: float,
        reason: str,
    ) -> None:
        fact_confidence = float(evidence.facts.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, fact_confidence * weight))
        support[normalize_cwe(cwe)].append(
            (confidence, evidence.evidence_id, reason)
        )
