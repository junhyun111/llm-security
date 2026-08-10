from __future__ import annotations

import math

from .semantic_analyzer import SemanticFunctionAnalysis
from .semantic_facts import SemanticFactKind


FEATURE_SCHEMA_SEMANTIC_V1 = (
    "function_length",
    "statement_count",
    "branch_count",
    "loop_count",
    "caller_count",
    "callee_count",
    "cfg_warning_count",
    "control_flow_reliable",
    "allocation_count",
    "memory_copy_count",
    "unbounded_memory_copy_count",
    "memory_copy_without_guard_count",
    "unchecked_index_count",
    "guard_protects_sink_count",
    "array_access_count",
    "dereference_count",
    "release_count",
    "use_after_release_count",
    "double_release_count",
    "arithmetic_to_allocation_count",
    "arithmetic_to_memory_sink_count",
    "cast_to_size_sink_count",
    "taint_source_count",
    "taint_sink_count",
    "source_to_sink_count",
    "unsanitized_source_to_sink_count",
    "uninitialized_use_count",
    "unchecked_nullable_dereference_count",
    "thread_spawn_count",
    "lock_acquire_count",
    "lock_release_count",
    "toctou_check_use_count",
    "semantic_fact_count",
    "mean_fact_confidence",
    "max_fact_confidence",
)


FACT_FEATURE_MAP = {
    SemanticFactKind.ALLOCATION: "allocation_count",
    SemanticFactKind.MEMORY_COPY: "memory_copy_count",
    SemanticFactKind.MEMORY_COPY_WITHOUT_GUARD: "memory_copy_without_guard_count",
    SemanticFactKind.UNCHECKED_INDEX: "unchecked_index_count",
    SemanticFactKind.GUARD_PROTECTS_SINK: "guard_protects_sink_count",
    SemanticFactKind.RELEASE: "release_count",
    SemanticFactKind.USE_AFTER_RELEASE: "use_after_release_count",
    SemanticFactKind.DOUBLE_RELEASE: "double_release_count",
    SemanticFactKind.ARITHMETIC_TO_ALLOCATION: "arithmetic_to_allocation_count",
    SemanticFactKind.ARITHMETIC_TO_MEMORY_SINK: "arithmetic_to_memory_sink_count",
    SemanticFactKind.CAST_TO_SIZE_SINK: "cast_to_size_sink_count",
    SemanticFactKind.TAINT_SOURCE: "taint_source_count",
    SemanticFactKind.TAINT_SINK: "taint_sink_count",
    SemanticFactKind.SOURCE_TO_SINK: "source_to_sink_count",
    SemanticFactKind.UNSANITIZED_SOURCE_TO_SINK: "unsanitized_source_to_sink_count",
    SemanticFactKind.UNINITIALIZED_USE: "uninitialized_use_count",
    SemanticFactKind.UNCHECKED_NULLABLE_DEREFERENCE: "unchecked_nullable_dereference_count",
    SemanticFactKind.THREAD_SPAWN: "thread_spawn_count",
    SemanticFactKind.LOCK_ACQUIRE: "lock_acquire_count",
    SemanticFactKind.LOCK_RELEASE: "lock_release_count",
    SemanticFactKind.TOCTOU_CHECK_USE: "toctou_check_use_count",
}


class SemanticFeatureExtractor:
    schema_version = "semantic-v1"

    def extract(
        self,
        semantic_analysis: SemanticFunctionAnalysis,
        *,
        caller_count: int,
        callee_count: int,
    ) -> dict[str, float]:
        features = {name: 0.0 for name in FEATURE_SCHEMA_SEMANTIC_V1}
        structural = semantic_analysis.structural
        function = structural.function
        features.update(
            {
                "function_length": float(function.line_end - function.line_start + 1),
                "statement_count": float(len(function.statements)),
                "branch_count": float(sum(control.kind == "if" for control in function.controls)),
                "loop_count": float(sum(control.kind in {"while", "for"} for control in function.controls)),
                "caller_count": float(caller_count),
                "callee_count": float(callee_count),
                "cfg_warning_count": float(len(structural.cfg.warnings)),
                "control_flow_reliable": float(not structural.cfg.warnings),
                "array_access_count": float(sum(access.kind.startswith("array_") for access in function.memory_accesses)),
                "dereference_count": float(sum(access.kind == "dereference" for access in function.memory_accesses)),
            }
        )
        for fact in semantic_analysis.facts:
            feature_name = FACT_FEATURE_MAP.get(fact.kind)
            if feature_name is not None:
                features[feature_name] += 1.0
            if fact.kind == SemanticFactKind.MEMORY_COPY and fact.attributes.get("unbounded") is True:
                features["unbounded_memory_copy_count"] += 1.0
        confidences = [float(fact.confidence) for fact in semantic_analysis.facts]
        features["semantic_fact_count"] = float(len(confidences))
        if confidences:
            features["mean_fact_confidence"] = sum(confidences) / len(confidences)
            features["max_fact_confidence"] = max(confidences)
        if any(not math.isfinite(value) for value in features.values()):
            raise ValueError("Semantic features must be finite")
        return features
