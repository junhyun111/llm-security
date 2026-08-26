from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

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

    values: dict[str, str] = {}
    source = Path(env_file)
    if source.is_file():
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    sweep = next(
        (
            item.strip()
            for item in values.get("OPENROUTER_SWEEP_MODELS", "").split(",")
            if item.strip()
        ),
        None,
    )
    fallback = (
        values.get("OPENROUTER_EXPERT_MODEL", "").strip()
        or sweep
        or "deepseek/deepseek-v4-flash-0731"
    )
    defaults = {
        key: fallback
        for key in (
            "OPENROUTER_EXPERT_MODEL",
            "OPENROUTER_VALIDATOR_MODEL",
            "OPENROUTER_PATCH_MODEL",
        )
        if not values.get(key, "").strip()
    }
    with patch.dict(os.environ, defaults, clear=False):
        return AppConfig.from_env(env_file)


def evaluation_api_config(env_file: str | Path):
    """Return the API configuration used only by this batch benchmark.

    A batch response has to contain a complete JSON result for every Expert
    task. Some reasoning providers spend the complete output budget on their
    hidden reasoning channel and return ``content: null``. Disable that mode
    for the evaluation calls so a notebook run needs only the API key and
    reliably receives the schema-constrained answer.
    """
    config = app_config(env_file)
    config.model.reasoning_enabled = False
    config.model.reasoning_effort = None
    return config


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
