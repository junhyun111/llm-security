from __future__ import annotations

import sys
from pathlib import Path

from ..paths import PROJECT_ROOT


PARENT_SOURCE = (PROJECT_ROOT / "src").resolve()


def activate_parent_package() -> Path:
    value = str(PARENT_SOURCE)
    if value not in sys.path:
        sys.path.insert(0, value)
    return PARENT_SOURCE


def frontend(*, max_source_bytes: int | None, parse_timeout_ms: int):
    activate_parent_package()
    from llm_security.analysis.frontend import TreeSitterFrontend

    return TreeSitterFrontend(
        max_source_bytes=max_source_bytes,
        parse_timeout_ms=parse_timeout_ms,
    )


def semantic_analyzer(*, max_source_bytes: int | None, parse_timeout_ms: int):
    activate_parent_package()
    from llm_security.analysis.semantic_static import SemanticStaticAnalyzer

    return SemanticStaticAnalyzer(
        max_source_bytes=max_source_bytes,
        parse_timeout_ms=parse_timeout_ms,
    )


def case_from_dict(raw: dict):
    activate_parent_package()
    from llm_security.datasets import case_from_dict as restore

    return restore(raw)


def candidate_to_dict(candidate) -> dict:
    activate_parent_package()
    from llm_security.models import to_dict

    return to_dict(candidate)


def candidate_from_dict(raw: dict):
    activate_parent_package()
    from llm_security.datasets import candidate_from_dict as restore

    return restore(raw)


def load_cases(path: str | Path):
    activate_parent_package()
    from llm_security.datasets import iter_cases_jsonl

    return iter_cases_jsonl(path)


def load_outcomes(path: str | Path):
    activate_parent_package()
    from llm_security.datasets import load_utility_samples_jsonl

    return load_utility_samples_jsonl(path)


def app_config(env_file: str | Path):
    activate_parent_package()
    from llm_security.config import AppConfig

    return AppConfig.from_env(env_file)


def active_experts():
    activate_parent_package()
    from llm_security.models import ACTIVE_UTILITY_EXPERTS

    return ACTIVE_UTILITY_EXPERTS


def expert_assignments(
    model_ids: list[str] | tuple[str, ...],
    *,
    prompt_version: str = "batched-expert-v4-ko-cwe-hypothesis",
):
    activate_parent_package()
    from llm_security.models import ACTIVE_UTILITY_EXPERTS, ExpertAssignment

    models = tuple(dict.fromkeys(item.strip() for item in model_ids if item.strip()))
    if not models:
        raise ValueError("At least one canonical model ID is required")
    return [
        ExpertAssignment(
            expert=expert,
            model_id=model,
            prompt_version=prompt_version,
        )
        for expert in ACTIVE_UTILITY_EXPERTS
        for model in models
    ]
