from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..models import ExpertFamily


class ExpertRoutingModel(Protocol):
    @property
    def available_families(self) -> tuple[ExpertFamily, ...]: ...

    def fit(
        self,
        features: Sequence[dict[str, float]],
        labels: Sequence[ExpertFamily],
    ) -> "ExpertRoutingModel": ...

    def predict_proba(
        self, features: dict[str, float]
    ) -> dict[ExpertFamily, float]: ...


class SoftmaxRoutingModel:
    """Multiclass probability model for P(Expert | features, candidate)."""

    def __init__(self, *, seed: int = 2026) -> None:
        self.seed = seed
        self.vectorizer = DictVectorizer(sparse=True)
        self.scaler = StandardScaler(with_mean=False)
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
            solver="lbfgs",
        )
        self._available_families: tuple[ExpertFamily, ...] = ()
        self._fitted = False

    @property
    def available_families(self) -> tuple[ExpertFamily, ...]:
        return self._available_families

    def fit(
        self,
        features: Sequence[dict[str, float]],
        labels: Sequence[ExpertFamily],
    ) -> "SoftmaxRoutingModel":
        if not features:
            raise ValueError("At least one routing training sample is required")
        if len(features) != len(labels):
            raise ValueError("Routing features and labels must have the same length")
        unique = tuple(sorted(set(labels), key=lambda family: family.value))
        if len(unique) < 2:
            raise ValueError("Softmax routing requires at least two Expert families")
        matrix = self.vectorizer.fit_transform(features)
        matrix = self.scaler.fit_transform(matrix)
        self.classifier.fit(matrix, [family.value for family in labels])
        self._available_families = tuple(
            ExpertFamily(value) for value in self.classifier.classes_
        )
        self._fitted = True
        return self

    def predict_proba(
        self, features: dict[str, float]
    ) -> dict[ExpertFamily, float]:
        if not self._fitted:
            raise RuntimeError("SoftmaxRoutingModel.fit must be called before inference")
        matrix = self.vectorizer.transform([features])
        matrix = self.scaler.transform(matrix)
        probabilities = np.asarray(self.classifier.predict_proba(matrix))[0]
        return {
            ExpertFamily(label): float(probability)
            for label, probability in zip(
                self.classifier.classes_, probabilities, strict=True
            )
        }
