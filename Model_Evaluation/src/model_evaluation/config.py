from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .paths import EVALUATION_ROOT, require_input_directory, resolve_evaluation_path


ACTIVE_EXPERTS = (
    "memory_bounds",
    "integer_size_type",
    "taint_api_contract",
    "control_state_error",
    "concurrency_toctou",
)


@dataclass(frozen=True, slots=True)
class EvaluationPaths:
    mapping: Path
    index: Path
    index_summary: Path
    split_manifest: Path
    pilot_directory: Path
    candidate_directory: Path
    analyzer_report: Path


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    schema_version: str
    dataset_root: Path
    seed: int
    pilot_per_expert: int
    max_packages: int
    parse_timeout_ms: int
    max_source_bytes: int | None
    index_batch_size: int
    train_fraction: float
    dev_fraction: float
    test_fraction: float
    paths: EvaluationPaths
    config_hash: str


@dataclass(frozen=True, slots=True)
class ExpertMapping:
    schema_version: str
    experts: dict[str, str]
    cwe_to_expert: dict[str, str]
    mapping_hash: str

    def expert_for(self, cwe: str) -> str | None:
        digits = "".join(character for character in cwe if character.isdigit())
        return self.cwe_to_expert.get(str(int(digits))) if digits else None


def load_config(path: str | Path) -> EvaluationConfig:
    source = Path(path)
    if not source.is_absolute():
        candidate = (Path.cwd() / source).resolve()
        source = candidate if candidate.is_file() else (EVALUATION_ROOT / source).resolve()
    if not source.is_file():
        raise ValueError(f"Evaluation config does not exist: {source}")
    raw_bytes = source.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    split = raw["splits"]
    fractions = (
        float(split["train"]),
        float(split["dev"]),
        float(split["test"]),
    )
    if any(value <= 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must be positive and sum to 1.0")
    pilot_per_expert = int(raw["pilot_per_expert"])
    if pilot_per_expert < 1:
        raise ValueError("pilot_per_expert must be positive")
    max_source_bytes = int(raw.get("max_source_bytes", 0))
    configured_paths = raw["paths"]
    paths = EvaluationPaths(
        mapping=resolve_evaluation_path(configured_paths["mapping"]),
        index=resolve_evaluation_path(configured_paths["index"]),
        index_summary=resolve_evaluation_path(configured_paths["index_summary"]),
        split_manifest=resolve_evaluation_path(configured_paths["split_manifest"]),
        pilot_directory=resolve_evaluation_path(configured_paths["pilot_directory"]),
        candidate_directory=resolve_evaluation_path(configured_paths["candidate_directory"]),
        analyzer_report=resolve_evaluation_path(configured_paths["analyzer_report"]),
    )
    return EvaluationConfig(
        schema_version=str(raw["schema_version"]),
        dataset_root=require_input_directory(raw["dataset_root"]),
        seed=int(raw["seed"]),
        pilot_per_expert=pilot_per_expert,
        max_packages=int(raw.get("max_packages", 0)),
        parse_timeout_ms=int(raw.get("parse_timeout_ms", 30_000)),
        max_source_bytes=max_source_bytes or None,
        index_batch_size=max(1, int(raw.get("index_batch_size", 500))),
        train_fraction=fractions[0],
        dev_fraction=fractions[1],
        test_fraction=fractions[2],
        paths=paths,
        config_hash=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_mapping(path: str | Path) -> ExpertMapping:
    source = resolve_evaluation_path(path)
    raw_bytes = source.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    experts = {str(key): str(value) for key, value in raw["experts"].items()}
    if tuple(experts) != ACTIVE_EXPERTS:
        raise ValueError(
            "Expert mapping must declare exactly the active E1/E3/E4/E5/E6 order"
        )
    cwe_to_expert = {
        str(int(key)): str(value) for key, value in raw["cwe"].items()
    }
    unknown = sorted(set(cwe_to_expert.values()) - set(ACTIVE_EXPERTS))
    if unknown:
        raise ValueError("CWE mapping contains unknown Experts: " + ", ".join(unknown))
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return ExpertMapping(
        schema_version=str(raw["schema_version"]),
        experts=experts,
        cwe_to_expert=cwe_to_expert,
        mapping_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )

