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
        matrix = _ensure_int32_sparse(self.vectorizer.fit_transform(features))
        matrix = _ensure_int32_sparse(self.scaler.fit_transform(matrix))
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
        matrix = _ensure_int32_sparse(self.vectorizer.transform([features]))
        matrix = _ensure_int32_sparse(self.scaler.transform(matrix))
        probabilities = np.asarray(self.classifier.predict_proba(matrix))[0]
        return {
            ExpertFamily(label): float(probability)
            for label, probability in zip(
                self.classifier.classes_, probabilities, strict=True
            )
        }


class BinaryRoutingModel:
    """Calibrated binary estimator with safe handling for one-class data."""

    def __init__(self, *, seed: int = 2026) -> None:
        self.seed = seed
        self.vectorizer = DictVectorizer(sparse=True)
        self.scaler = StandardScaler(with_mean=False)
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
            solver="liblinear",
        )
        self.constant_probability: float | None = None
        self._fitted = False

    def fit(
        self,
        features: Sequence[dict[str, float]],
        labels: Sequence[bool | int],
    ) -> "BinaryRoutingModel":
        if not features:
            raise ValueError("At least one binary routing sample is required")
        if len(features) != len(labels):
            raise ValueError("Binary routing features and labels must have the same length")
        normalized = np.asarray([int(bool(label)) for label in labels], dtype=np.int32)
        unique = np.unique(normalized)
        self.vectorizer.fit(features)
        if len(unique) == 1:
            self.constant_probability = float(unique[0])
            self._fitted = True
            return self
        matrix = _ensure_int32_sparse(self.vectorizer.transform(features))
        matrix = _ensure_int32_sparse(self.scaler.fit_transform(matrix))
        self.classifier.fit(matrix, normalized)
        self.constant_probability = None
        self._fitted = True
        return self

    def predict_proba(self, features: dict[str, float]) -> float:
        if not self._fitted:
            raise RuntimeError("BinaryRoutingModel.fit must be called before inference")
        if self.constant_probability is not None:
            return self.constant_probability
        matrix = _ensure_int32_sparse(self.vectorizer.transform([features]))
        matrix = _ensure_int32_sparse(self.scaler.transform(matrix))
        return float(self.classifier.predict_proba(matrix)[0, 1])


class UtilityRoutingModel:
    """Independent estimators for P(Expert x model succeeds | candidate)."""

    def __init__(self, *, seed: int = 2026) -> None:
        self.seed = seed
        self.models: dict[str, BinaryRoutingModel] = {}

    @property
    def available_assignments(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))

    def fit(
        self,
        features: Sequence[dict[str, float]],
        assignment_ids: Sequence[str],
        labels: Sequence[bool | int],
    ) -> "UtilityRoutingModel":
        if not features:
            raise ValueError("At least one utility routing sample is required")
        if not (len(features) == len(assignment_ids) == len(labels)):
            raise ValueError("Utility features, assignments, and labels must align")
        grouped: dict[str, tuple[list[dict[str, float]], list[bool | int]]] = {}
        for row, assignment_id, label in zip(
            features, assignment_ids, labels, strict=True
        ):
            rows, targets = grouped.setdefault(assignment_id, ([], []))
            rows.append(row)
            targets.append(label)
        self.models = {
            assignment_id: BinaryRoutingModel(seed=self.seed).fit(rows, targets)
            for assignment_id, (rows, targets) in grouped.items()
        }
        return self

    def predict_proba(
        self, features: dict[str, float]
    ) -> dict[str, float]:
        if not self.models:
            raise RuntimeError("UtilityRoutingModel.fit must be called before inference")
        return {
            assignment_id: model.predict_proba(features)
            for assignment_id, model in self.models.items()
        }


def _ensure_int32_sparse(matrix):
    """Keep scipy index arrays compatible with all supported sklearn solvers."""
    if hasattr(matrix, "indices") and matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32)
    if hasattr(matrix, "indptr") and matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix
