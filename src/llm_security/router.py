from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .datasets import RouterSample
from .models import Candidate, ExpertFamily, RouteDecision


class Router(Protocol):
    kind: str

    def route(self, candidate: Candidate) -> RouteDecision: ...


@dataclass(slots=True)
class RouterMetrics:
    micro_f1: float
    macro_f1: float
    coverage_at_1: float
    coverage_at_2: float
    sample_count: int


class BaseRouter:
    kind = "base"

    def __init__(self, threshold: float = 0.25, max_experts: int = 3) -> None:
        self.threshold = threshold
        self.max_experts = max_experts

    def _decision(
        self, candidate: Candidate, scores: dict[ExpertFamily, float]
    ) -> RouteDecision:
        ranked = sorted(scores, key=lambda family: scores[family], reverse=True)
        selected = [family for family in ranked if scores[family] >= self.threshold]
        if not selected:
            selected = ranked[:1]
        selected = selected[: self.max_experts]
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            scores=scores,
            selected=selected,
            router_kind=self.kind,
        )


class RuleRouter(BaseRouter):
    kind = "rule"

    def route(self, candidate: Candidate) -> RouteDecision:
        f = candidate.features
        raw = {
            ExpertFamily.MEMORY_BOUNDS: (
                1.8 * f.get("memory_api_count", 0.0)
                + 1.2 * f.get("array_access_count", 0.0)
                + 0.5 * f.get("allocation_count", 0.0)
            ),
            ExpertFamily.LIFETIME_RESOURCE: (
                1.3 * f.get("free_count", 0.0)
                + 2.5 * f.get("released_then_accessed", 0.0)
                + 0.5 * f.get("allocation_count", 0.0)
            ),
            ExpertFamily.INTEGER_SIZE_TYPE: (
                2.0 * f.get("integer_size_arithmetic_count", 0.0)
                + f.get("cast_count", 0.0)
            ),
            ExpertFamily.TAINT_API_CONTRACT: (
                f.get("input_source_count", 0.0)
                + f.get("dangerous_sink_count", 0.0)
                + 2.0 * f.get("external_input_to_sink", 0.0)
            ),
            ExpertFamily.CONTROL_STATE_ERROR: (
                f.get("error_signal_count", 0.0)
                + 1.5 * f.get("state_signal_count", 0.0)
            ),
            ExpertFamily.CONCURRENCY_TOCTOU: (
                f.get("thread_signal_count", 0.0)
                + f.get("lock_signal_count", 0.0)
                + 2.5 * f.get("toctou_pair", 0.0)
            ),
        }
        maximum = max(max(raw.values()), 1.0)
        scores = {family: min(0.99, score / maximum) for family, score in raw.items()}
        return self._decision(candidate, scores)


class LearnedRouter(BaseRouter):
    kind = "learned"

    def __init__(
        self,
        threshold: float = 0.25,
        max_experts: int = 3,
        seed: int = 2026,
    ) -> None:
        super().__init__(threshold=threshold, max_experts=max_experts)
        self.seed = seed
        self.vectorizer = DictVectorizer(sparse=True)
        self.labeler = MultiLabelBinarizer()
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                class_weight="balanced",
                max_iter=10_000,
                random_state=seed,
                solver="lbfgs",
            )
        )
        self._fitted = False

    def fit(self, samples: Iterable[RouterSample]) -> "LearnedRouter":
        materialized = list(samples)
        if not materialized:
            raise ValueError("At least one router training sample is required")
        features = [sample.candidate.features for sample in materialized]
        labels = [[family.value for family in sample.labels] for sample in materialized]
        x = self.vectorizer.fit_transform(features)
        y = self.labeler.fit_transform(labels)
        self.classifier.fit(x, y)
        self._fitted = True
        return self

    def predict_scores(self, candidate: Candidate) -> dict[ExpertFamily, float]:
        if not self._fitted:
            raise RuntimeError("LearnedRouter.fit must be called before inference")
        x = self.vectorizer.transform([candidate.features])
        probabilities = np.asarray(self.classifier.predict_proba(x))[0]
        scores = {family: 0.0 for family in ExpertFamily}
        scores.update({
            ExpertFamily(label): float(probability)
            for label, probability in zip(self.labeler.classes_, probabilities, strict=True)
        })
        return scores

    def route(self, candidate: Candidate) -> RouteDecision:
        return self._decision(candidate, self.predict_scores(candidate))

    def evaluate(self, samples: Iterable[RouterSample]) -> RouterMetrics:
        materialized = list(samples)
        if not materialized:
            raise ValueError("At least one router evaluation sample is required")
        known_labels = set(self.labeler.classes_)
        unseen_labels = {
            family.value
            for sample in materialized
            for family in sample.labels
            if family.value not in known_labels
        }
        if unseen_labels:
            raise ValueError(
                "Evaluation contains Expert families absent from training: "
                + ", ".join(sorted(unseen_labels))
            )
        expected = self.labeler.transform(
            [[family.value for family in sample.labels] for sample in materialized]
        )
        score_rows = [self.predict_scores(sample.candidate) for sample in materialized]
        decisions = [
            self._decision(sample.candidate, scores)
            for sample, scores in zip(materialized, score_rows, strict=True)
        ]
        predicted = np.asarray(
            [
                [
                    int(ExpertFamily(label) in decision.selected)
                    for label in self.labeler.classes_
                ]
                for decision in decisions
            ]
        )
        coverage_1 = 0
        coverage_2 = 0
        for sample, scores in zip(materialized, score_rows, strict=True):
            ranked = sorted(scores, key=scores.get, reverse=True)
            truth = set(sample.labels)
            coverage_1 += int(bool(truth.intersection(ranked[:1])))
            coverage_2 += int(bool(truth.intersection(ranked[:2])))
        present_columns = np.flatnonzero(expected.sum(axis=0) > 0)
        macro_f1 = (
            f1_score(
                expected[:, present_columns],
                predicted[:, present_columns],
                average="macro",
                zero_division=0,
            )
            if len(present_columns)
            else 0.0
        )
        return RouterMetrics(
            micro_f1=float(f1_score(expected, predicted, average="micro", zero_division=0)),
            macro_f1=float(macro_f1),
            coverage_at_1=coverage_1 / len(materialized),
            coverage_at_2=coverage_2 / len(materialized),
            sample_count=len(materialized),
        )

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted router")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRouter":
        with Path(path).open("rb") as handle:
            router = pickle.load(handle)  # noqa: S301 - trusted local experiment artifact
        if not isinstance(router, cls):
            raise TypeError("The artifact does not contain a LearnedRouter")
        return router


def train_and_evaluate_router(
    train_samples: Iterable[RouterSample],
    test_samples: Iterable[RouterSample],
    *,
    threshold: float = 0.25,
    max_experts: int = 3,
    seed: int = 2026,
) -> tuple[LearnedRouter, RouterMetrics]:
    router = LearnedRouter(
        threshold=threshold,
        max_experts=max_experts,
        seed=seed,
    ).fit(train_samples)
    return router, router.evaluate(test_samples)
