from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EVALUATION_ROOT.parent


def resolve_evaluation_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = EVALUATION_ROOT / path
    return require_within(path, EVALUATION_ROOT)


def require_within(path: str | Path, root: str | Path) -> Path:
    resolved = Path(path).resolve()
    boundary = Path(root).resolve()
    try:
        resolved.relative_to(boundary)
    except ValueError as error:
        raise ValueError(
            f"Refusing output outside isolated evaluation root {boundary}: {resolved}"
        ) from error
    return resolved


def require_input_directory(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"Input dataset directory does not exist: {resolved}")
    return resolved


def write_json(path: str | Path, payload: Any) -> Path:
    destination = resolve_evaluation_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination

