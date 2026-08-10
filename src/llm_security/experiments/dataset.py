from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..datasets import RouterSample
from ..models import Candidate, ExpertFamily, ProjectCase
from .analyzer_eval import candidate_matches_truth
from .io import sorted_jsonl


def router_samples_from_cached_candidates(
    cases: list[ProjectCase],
    candidates_by_case: Mapping[str, list[Candidate]],
) -> list[RouterSample]:
    """Build labels without re-running an analyzer or dropping benchmark cases."""
    samples: list[RouterSample] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        candidates = candidates_by_case.get(case.case_id, [])
        by_candidate: dict[str, tuple[Candidate, set[ExpertFamily]]] = {}
        for truth in case.ground_truth:
            matching = [
                candidate
                for candidate in candidates
                if candidate_matches_truth(candidate, truth)
            ]
            if not matching:
                continue
            candidate = min(
                matching,
                key=lambda item: (
                    item.line_end - item.line_start,
                    -item.suspicion_score,
                    item.candidate_id,
                ),
            )
            stored, labels = by_candidate.setdefault(
                candidate.candidate_id, (candidate, set())
            )
            labels.update(truth.experts)
            by_candidate[candidate.candidate_id] = stored, labels
        samples.extend(
            RouterSample(
                candidate=candidate,
                labels=sorted(labels, key=lambda family: family.value),
            )
            for candidate, labels in sorted(by_candidate.values(), key=lambda item: item[0].candidate_id)
        )
    return samples


def single_label_samples(samples: list[RouterSample]) -> list[RouterSample]:
    """The current Softmax Router is deliberately a single-label classifier."""
    return [sample for sample in samples if len(sample.labels) == 1]


def write_backend_router_datasets(
    samples_by_split: Mapping[str, list[RouterSample]], output_directory: str | Path
) -> None:
    destination = Path(output_directory)
    for split_name in ("train", "dev", "test"):
        sorted_jsonl(
            samples_by_split[split_name],
            destination / f"router_{split_name}.jsonl",
            key=lambda sample: sample.candidate.candidate_id,
        )
