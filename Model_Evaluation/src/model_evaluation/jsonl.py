from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .paths import EVALUATION_ROOT, require_within


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL {source}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {source}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    destination = require_within(path, EVALUATION_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def append_jsonl(path: str | Path, row: dict) -> None:
    destination = require_within(path, EVALUATION_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
