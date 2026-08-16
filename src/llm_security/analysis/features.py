from __future__ import annotations

import math

from ..cwe import cwe_number, expert_for_cwe
from ..models import CweHypothesis, ExpertFamily
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
    "numeric_conversion_count",
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

CWE_SCORE_NUMBERS = (
    "20", "22", "78", "120", "125", "129", "134", "190", "252", "367",
    "415", "416", "457", "476", "681", "787",
)

FEATURE_SCHEMA_SEMANTIC_CWE_V2 = (
    *FEATURE_SCHEMA_SEMANTIC_V1,
    "cwe_hypothesis_count",
    "cwe_top1_confidence",
    "cwe_top1_top2_margin",
    "cwe_cross_family_count",
    "cwe_memory_score",
    "cwe_integer_score",
    "cwe_taint_score",
    "cwe_control_score",
    "cwe_concurrency_score",
    *(f"cwe_{number}_score" for number in CWE_SCORE_NUMBERS),
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
    SemanticFactKind.NUMERIC_CONVERSION: "numeric_conversion_count",
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
    schema_version = "semantic-cwe-v2"

    def extract(
        self,
        semantic_analysis: SemanticFunctionAnalysis,
        *,
        caller_count: int,
        callee_count: int,
        cwe_hypotheses: list[CweHypothesis] | None = None,
    ) -> dict[str, float]:
        features = {name: 0.0 for name in FEATURE_SCHEMA_SEMANTIC_CWE_V2}
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
        hypotheses = sorted(
            cwe_hypotheses or [], key=lambda item: (-item.confidence, item.cwe)
        )
        features["cwe_hypothesis_count"] = float(len(hypotheses))
        if hypotheses:
            features["cwe_top1_confidence"] = hypotheses[0].confidence
            second = hypotheses[1].confidence if len(hypotheses) > 1 else 0.0
            features["cwe_top1_top2_margin"] = hypotheses[0].confidence - second
        families = {
            family
            for hypothesis in hypotheses
            for family in [expert_for_cwe(hypothesis.cwe)]
            if family is not None
        }
        features["cwe_cross_family_count"] = float(len(families))
        group_features = {
            ExpertFamily.MEMORY_SAFETY: "cwe_memory_score",
            ExpertFamily.INTEGER_SIZE_TYPE: "cwe_integer_score",
            ExpertFamily.TAINT_API_CONTRACT: "cwe_taint_score",
            ExpertFamily.CONTROL_STATE_ERROR: "cwe_control_score",
            ExpertFamily.CONCURRENCY_TOCTOU: "cwe_concurrency_score",
        }
        for hypothesis in hypotheses:
            family = expert_for_cwe(hypothesis.cwe)
            group_feature = group_features.get(family)
            if group_feature is not None:
                features[group_feature] = max(
                    features[group_feature], hypothesis.confidence
                )
            number = cwe_number(hypothesis.cwe)
            score_feature = f"cwe_{number}_score"
            if score_feature in features:
                features[score_feature] = max(
                    features[score_feature], hypothesis.confidence
                )
        if any(not math.isfinite(value) for value in features.values()):
            raise ValueError("Semantic features must be finite")
        return features
