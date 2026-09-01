from __future__ import annotations

import json
from pathlib import Path

from llm_security.config import AppConfig
from llm_security.factory import build_batched_web_pipeline
from llm_security.models import ProjectCase, to_dict
from llm_security.routing import BudgetedUtilityRouter

from .paths import RuntimePaths


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "vendor",
}


def analyze_source(
    source: str | Path,
    output: str | Path,
    *,
    config: AppConfig,
    paths: RuntimePaths,
) -> dict[str, object]:
    """Run the deployment pipeline and persist its JSON result."""

    source_path = Path(source).expanduser().resolve()
    source_files = _read_sources(source_path)
    router = BudgetedUtilityRouter.load(paths.router_artifact)
    case = ProjectCase(
        case_id="runtime-analysis",
        project_id=source_path.stem if source_path.is_file() else source_path.name,
        source_files=source_files,
        split="unlabeled",
        metadata={"source": str(source_path)},
    )
    result = build_batched_web_pipeline(
        config,
        router,
        max_batch_characters=120_000,
        max_batch_tasks=24,
    ).run(case)
    payload = to_dict(result)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "output": str(destination),
        "candidate_count": len(result.candidates),
        "finding_count": len(result.findings),
        "validated_finding_count": len(result.validated_findings),
        "request_count": len(result.usage),
    }


def _read_sources(path: Path) -> dict[str, str]:
    if path.is_file():
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            raise ValueError(f"Unsupported source suffix: {path.suffix}")
        return {path.name: path.read_text(encoding="utf-8", errors="replace")}
    if not path.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {path}")

    sources: dict[str, str] = {}
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = file_path.relative_to(path)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts[:-1]):
            continue
        sources[relative.as_posix()] = file_path.read_text(
            encoding="utf-8", errors="replace"
        )
    if not sources:
        raise ValueError(f"No C/C++ source files found under {path}")
    return sources

