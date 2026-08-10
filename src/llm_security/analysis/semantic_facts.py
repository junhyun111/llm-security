from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SemanticFactKind(str, Enum):
    ALLOCATION = "allocation"
    RELEASE = "release"
    MEMORY_COPY = "memory_copy"
    MEMORY_COPY_WITHOUT_GUARD = "memory_copy_without_guard"
    UNCHECKED_INDEX = "unchecked_index"
    GUARD_PROTECTS_SINK = "guard_protects_sink"
    ARITHMETIC_TO_ALLOCATION = "arithmetic_to_allocation"
    ARITHMETIC_TO_MEMORY_SINK = "arithmetic_to_memory_sink"
    CAST_TO_SIZE_SINK = "cast_to_size_sink"
    TAINT_SOURCE = "taint_source"
    TAINT_SINK = "taint_sink"
    SOURCE_TO_SINK = "source_to_sink"
    UNSANITIZED_SOURCE_TO_SINK = "unsanitized_source_to_sink"
    USE_AFTER_RELEASE = "use_after_release"
    DOUBLE_RELEASE = "double_release"
    UNINITIALIZED_USE = "uninitialized_use"
    UNCHECKED_NULLABLE_DEREFERENCE = "unchecked_nullable_dereference"
    UNCHECKED_CALL_RESULT = "unchecked_call_result"
    THREAD_SPAWN = "thread_spawn"
    LOCK_ACQUIRE = "lock_acquire"
    LOCK_RELEASE = "lock_release"
    TOCTOU_CHECK_USE = "toctou_check_use"


@dataclass(slots=True)
class SemanticFact:
    fact_id: str
    kind: SemanticFactKind
    function: str
    file: str
    subject: str | None
    object: str | None
    source_node_id: str | None
    sink_node_id: str | None
    path: list[str]
    symbols: list[str]
    confidence: float
    attributes: dict[str, Any]


def make_semantic_fact(
    *,
    kind: SemanticFactKind,
    function: str,
    file: str,
    subject: str | None = None,
    object: str | None = None,
    source_node_id: str | None = None,
    sink_node_id: str | None = None,
    path: list[str] | None = None,
    symbols: set[str] | list[str] | tuple[str, ...] = (),
    confidence: float = 1.0,
    attributes: dict[str, Any] | None = None,
) -> SemanticFact:
    normalized_path = list(path or [])
    normalized_symbols = sorted(set(symbols))
    normalized_attributes = attributes or {}
    identity = {
        "kind": kind.value,
        "function": function,
        "file": file,
        "subject": subject,
        "object": object,
        "source_node_id": source_node_id,
        "sink_node_id": sink_node_id,
        "path": normalized_path,
        "symbols": normalized_symbols,
        "attributes": normalized_attributes,
    }
    digest = hashlib.sha1(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return SemanticFact(
        fact_id=f"FACT-{digest}",
        kind=kind,
        function=function,
        file=file,
        subject=subject,
        object=object,
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        path=normalized_path,
        symbols=normalized_symbols,
        confidence=confidence,
        attributes=normalized_attributes,
    )


def sort_facts(facts: list[SemanticFact]) -> list[SemanticFact]:
    unique = {fact.fact_id: fact for fact in facts}
    return sorted(
        unique.values(),
        key=lambda fact: (
            fact.kind.value,
            fact.source_node_id or "",
            fact.sink_node_id or "",
            fact.fact_id,
        ),
    )
