from __future__ import annotations

import hashlib
from pathlib import Path

from llm_security.analysis import LearnedCandidateRanker
from llm_security.models import ACTIVE_UTILITY_EXPERTS
from llm_security.routing import BudgetedUtilityRouter

from .paths import RuntimePaths


def inspect_artifacts(paths: RuntimePaths) -> dict[str, object]:
    """Load trusted local artifacts and return deployment-safe metadata."""

    paths.require_artifacts()
    router = BudgetedUtilityRouter.load(paths.router_artifact)
    ranker = LearnedCandidateRanker.load(paths.candidate_ranker_artifact)

    if router.feature_schema_version != ranker.feature_schema_version:
        raise ValueError(
            "Router and Candidate Ranker feature schemas differ: "
            f"{router.feature_schema_version} != {ranker.feature_schema_version}"
        )
    available = {assignment.expert for assignment in router.assignments.values()}
    missing = set(ACTIVE_UTILITY_EXPERTS) - available
    if missing:
        raise ValueError(
            "Router is missing required Experts: "
            + ", ".join(sorted(expert.value for expert in missing))
        )

    model_ids = sorted(
        {assignment.model_id for assignment in router.assignments.values()}
    )
    training_summary = getattr(router.model, "training_summary", {})
    return {
        "status": "ready",
        "router": {
            "path": str(paths.router_artifact),
            "sha256": _sha256(paths.router_artifact),
            "artifact_version": getattr(router, "_artifact_version", None),
            "backend": training_summary.get(
                "backend", type(router.model).__name__
            ),
            "model_class": (
                f"{type(router.model).__module__}.{type(router.model).__name__}"
            ),
            "feature_schema": router.feature_schema_version,
            "assignment_count": len(router.assignments),
            "expert_model_ids": model_ids,
            "normal_top_k": router.policy.normal_top_k,
            "full_expert_count": router.policy.full_expert_count,
            "escalation_gate": (
                type(router.escalation_gate).__name__
                if router.escalation_gate is not None
                else None
            ),
        },
        "candidate_ranker": {
            "path": str(paths.candidate_ranker_artifact),
            "sha256": _sha256(paths.candidate_ranker_artifact),
            "backend": ranker.backend,
            "feature_schema": ranker.feature_schema_version,
        },
    }


def assert_model_compatible(metadata: dict[str, object], model_id: str) -> None:
    router = metadata["router"]
    if not isinstance(router, dict):
        raise TypeError("Invalid runtime Router metadata")
    model_ids = router.get("expert_model_ids", [])
    if model_id not in model_ids:
        raise ValueError(
            f"OPENROUTER_EXPERT_MODEL={model_id!r} is incompatible with router.pkl. "
            "Expected one of: " + ", ".join(str(item) for item in model_ids)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

