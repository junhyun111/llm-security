from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Candidate, Evidence, ExpertFamily, GroundTruth, ProjectCase, to_dict


@dataclass(slots=True)
class RouterSample:
    candidate: Candidate
    labels: list[ExpertFamily]


def write_cases_jsonl(cases: Iterable[ProjectCase], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(to_dict(case), ensure_ascii=False) + "\n")


def load_cases_jsonl(path: str | Path) -> list[ProjectCase]:
    cases: list[ProjectCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                truths = [
                    GroundTruth(
                        truth_id=item["truth_id"],
                        file=item["file"],
                        function=item["function"],
                        line_start=int(item["line_start"]),
                        line_end=int(item["line_end"]),
                        experts=[ExpertFamily(value) for value in item.get("experts", [])],
                        cwes=[str(value) for value in item.get("cwes", [])],
                    )
                    for item in raw.get("ground_truth", [])
                ]
                cases.append(
                    ProjectCase(
                        case_id=raw["case_id"],
                        project_id=raw["project_id"],
                        source_files={
                            str(name): str(content)
                            for name, content in raw["source_files"].items()
                        },
                        split=raw.get("split", "dev"),
                        vulnerable_revision=raw.get("vulnerable_revision"),
                        fixed_revision=raw.get("fixed_revision"),
                        ground_truth=truths,
                        metadata=dict(raw.get("metadata", {})),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid case JSONL at line {line_number}: {error}") from error
    return cases


def write_router_samples_jsonl(samples: Iterable[RouterSample], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {
                        "candidate": to_dict(sample.candidate),
                        "labels": [label.value for label in sample.labels],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_router_samples_jsonl(path: str | Path) -> list[RouterSample]:
    samples: list[RouterSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                item = raw["candidate"]
                candidate = Candidate(
                    candidate_id=item["candidate_id"],
                    project_id=item["project_id"],
                    file=item["file"],
                    function=item["function"],
                    line_start=int(item["line_start"]),
                    line_end=int(item["line_end"]),
                    code=item.get("code", ""),
                    evidence=[
                        Evidence(
                            evidence_id=evidence["evidence_id"],
                            kind=evidence["kind"],
                            file=evidence["file"],
                            line=int(evidence["line"]),
                            expression=evidence["expression"],
                            function=str(evidence.get("function", item["function"])),
                            subject=evidence.get("subject"),
                            object=evidence.get("object"),
                            facts=dict(evidence.get("facts", {})),
                        )
                        for evidence in item.get("evidence", [])
                    ],
                    features={str(key): float(value) for key, value in item["features"].items()},
                    suspicion_score=float(
                        item.get("suspicion_score", item.get("static_score", 0.0))
                    ),
                    callers=[str(value) for value in item.get("callers", [])],
                    callees=[str(value) for value in item.get("callees", [])],
                )
                samples.append(
                    RouterSample(
                        candidate=candidate,
                        labels=[ExpertFamily(value) for value in raw["labels"]],
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid router sample JSONL at line {line_number}: {error}"
                ) from error
    return samples
