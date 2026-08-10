from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..models import to_dict


def write_json(value: Any, path: str | Path) -> None:
    """Write stable JSON suitable for diffing and regression tests."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            to_dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_confusion_csv(
    labels: list[str], matrix: list[list[int]], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def sorted_jsonl(items: Iterable[Any], path: str | Path, *, key) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(items, key=key):
            handle.write(
                json.dumps(to_dict(item), ensure_ascii=False, sort_keys=True) + "\n"
            )
