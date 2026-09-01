from __future__ import annotations

import re

from .models import Evidence, ExpertFamily


_CWE_CATEGORIES: dict[str, str] = {}


def _register(category: str, *cwes: int) -> None:
    for cwe in cwes:
        _CWE_CATEGORIES[str(cwe)] = category


_register(
    "memory_spatial",
    119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 131, 170, 680, 787, 788,
)
_register("memory_temporal", 401, 415, 416, 457, 476, 562, 590, 761, 762, 763)
_register("integer", 190, 191, 192, 194, 195, 196, 197, 369, 681, 682)
_register(
    "taint_api",
    15, 20, 22, 23, 36, 73, 74, 77, 78, 79, 89, 90, 91, 94, 114, 134,
    247, 426, 427, 601,
)
_register("control_state", 252, 253, 273, 394, 457, 478, 670, 703, 754, 755)
_register("concurrency", 362, 366, 367, 667, 764, 765, 821, 832, 833)


def normalize_cwe(value: str) -> str:
    match = re.search(r"(?:CWE[-_: ]*)?(\d+)", str(value), re.I)
    return f"CWE-{int(match.group(1))}" if match else ""


def cwe_number(value: str) -> str:
    normalized = normalize_cwe(value)
    return normalized.removeprefix("CWE-") if normalized else ""


def cwe_category(value: str) -> str | None:
    return _CWE_CATEGORIES.get(cwe_number(value))


def cwe_categories(cwes: list[str]) -> set[str]:
    return {
        category
        for value in cwes
        for category in [cwe_category(value)]
        if category is not None
    }


def expert_for_cwe(cwe: str) -> ExpertFamily | None:
    return {
        "memory_spatial": ExpertFamily.MEMORY_SAFETY,
        "memory_temporal": ExpertFamily.MEMORY_SAFETY,
        "integer": ExpertFamily.INTEGER_SIZE_TYPE,
        "taint_api": ExpertFamily.TAINT_API_CONTRACT,
        "control_state": ExpertFamily.CONTROL_STATE_ERROR,
        "concurrency": ExpertFamily.CONCURRENCY_TOCTOU,
    }.get(cwe_category(cwe))


_EVIDENCE_CATEGORIES = {
    "memory_sink": "memory_spatial",
    "memory_access": "memory_spatial",
    "memory_copy": "memory_spatial",
    "memory_copy_without_guard": "memory_spatial",
    "unchecked_index": "memory_spatial",
    "release": "memory_temporal",
    "use_after_release": "memory_temporal",
    "double_release": "memory_temporal",
    "unchecked_nullable_dereference": "memory_temporal",
    "integer_arithmetic": "integer",
    "type_conversion": "integer",
    "numeric_conversion": "integer",
    "arithmetic_to_allocation": "integer",
    "arithmetic_to_memory_sink": "integer",
    "cast_to_size_sink": "integer",
    "taint_source": "taint_api",
    "taint_sink": "taint_api",
    "source_to_sink": "taint_api",
    "unsanitized_source_to_sink": "taint_api",
    "state": "control_state",
    "error_path": "control_state",
    "uninitialized_use": "control_state",
    "unchecked_call_result": "control_state",
    "concurrency": "concurrency",
    "synchronization": "concurrency",
    "toctou": "concurrency",
    "thread_spawn": "concurrency",
    "toctou_check_use": "concurrency",
}


def cwes_supported_by_evidence(cwes: list[str], evidence: list[Evidence]) -> bool:
    claimed = cwe_categories(cwes)
    cited = {
        category
        for item in evidence
        for category in [_EVIDENCE_CATEGORIES.get(item.kind)]
        if category is not None
    }
    if not claimed:
        return False
    integer_to_memory = any(
        item.kind in {
            "arithmetic_to_allocation",
            "arithmetic_to_memory_sink",
            "cast_to_size_sink",
        }
        for item in evidence
    )
    # Every reported CWE family must be supported. Accept the explicit causal
    # bridge where integer corruption reaches an allocation/copy/index size.
    return all(
        category in cited
        or (
            category == "memory_spatial"
            and "integer" in cited
            and integer_to_memory
        )
        for category in claimed
    )
