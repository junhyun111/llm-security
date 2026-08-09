from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluation import AggregateMetrics, CaseMetrics, aggregate_metrics, evaluate_case
from .models import PipelineResult, ProjectCase, to_dict
from .pipeline import VulnerabilityPipeline


@dataclass(slots=True)
class ExperimentOutput:
    results: list[PipelineResult]
    case_metrics: list[CaseMetrics]
    aggregate: AggregateMetrics


class ExperimentRunner:
    def __init__(self, pipeline: VulnerabilityPipeline) -> None:
        self.pipeline = pipeline

    def run(self, cases: list[ProjectCase]) -> ExperimentOutput:
        results = [self.pipeline.run(case) for case in cases]
        metrics = [
            evaluate_case(case, result)
            for case, result in zip(cases, results, strict=True)
        ]
        return ExperimentOutput(
            results=results,
            case_metrics=metrics,
            aggregate=aggregate_metrics(metrics, results),
        )

    @staticmethod
    def save(output: ExperimentOutput, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "aggregate": to_dict(output.aggregate),
            "cases": [to_dict(item) for item in output.case_metrics],
            "results": [to_dict(item) for item in output.results],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

