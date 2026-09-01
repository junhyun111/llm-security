from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..models import Candidate
from .features import FEATURE_SCHEMA_SEMANTIC_CWE_V3


class LearnedCandidateRanker:
    """Estimate whether a candidate overlaps a true vulnerability.

    The score is used only to order candidates before the Top-K cut.  Expert
    routing remains a separate model and consumes the unchanged feature map.
    """

    artifact_version = 1
    required_feature_schema_version = "semantic-cwe-v3"
    supported_backends = ("logistic_regression", "gradient_boosting", "small_mlp")

    def __init__(self, *, backend: str, estimator, seed: int = 2026) -> None:
        if backend not in self.supported_backends:
            raise ValueError(f"Unsupported Candidate Ranker backend: {backend}")
        self.backend = backend
        self.estimator = estimator
        self.seed = seed
        self.feature_names = tuple(FEATURE_SCHEMA_SEMANTIC_CWE_V3)
        self.feature_schema_version = self.required_feature_schema_version
        self._artifact_version = self.artifact_version

    @classmethod
    def fit(
        cls,
        candidates: Iterable[Candidate],
        labels: Iterable[bool],
        *,
        backend: str = "logistic_regression",
        seed: int = 2026,
    ) -> "LearnedCandidateRanker":
        rows = list(candidates)
        targets = np.asarray([int(value) for value in labels], dtype=np.int64)
        if len(rows) != len(targets) or not rows:
            raise ValueError("Candidate Ranker requires aligned non-empty rows and labels")
        if len(set(targets.tolist())) != 2:
            raise ValueError("Candidate Ranker training requires positive and negative labels")
        matrix = _feature_matrix(rows)
        if backend == "logistic_regression":
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                ),
            )
            estimator.fit(matrix, targets)
        elif backend == "gradient_boosting":
            estimator = GradientBoostingClassifier(random_state=seed)
            weights = _balanced_sample_weights(targets)
            estimator.fit(matrix, targets, sample_weight=weights)
        elif backend == "small_mlp":
            estimator = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=250,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=seed,
                ),
            )
            estimator.fit(matrix, targets)
        else:
            raise ValueError(f"Unsupported Candidate Ranker backend: {backend}")
        return cls(backend=backend, estimator=estimator, seed=seed)

    def score(self, candidate: Candidate) -> float:
        self._validate_candidate(candidate)
        probability = self.estimator.predict_proba(_feature_matrix([candidate]))[0, 1]
        return float(probability)

    def rank(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        rows = list(candidates)
        if not rows:
            return []
        for candidate, probability in zip(
            rows,
            self.estimator.predict_proba(_feature_matrix(rows))[:, 1],
            strict=True,
        ):
            self._validate_candidate(candidate)
            candidate.suspicion_score = float(probability)
        return sorted(
            rows,
            key=lambda candidate: (
                -candidate.suspicion_score,
                candidate.file,
                candidate.line_start,
                candidate.candidate_id,
            ),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(self, handle)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "LearnedCandidateRanker":
        with Path(path).open("rb") as handle:
            ranker = pickle.load(handle)
        if not isinstance(ranker, cls):
            raise ValueError("Artifact is not a LearnedCandidateRanker")
        if getattr(ranker, "_artifact_version", None) != cls.artifact_version:
            raise ValueError("Candidate Ranker artifact version is incompatible")
        if ranker.feature_schema_version != cls.required_feature_schema_version:
            raise ValueError("Candidate Ranker feature schema is incompatible")
        return ranker

    def _validate_candidate(self, candidate: Candidate) -> None:
        if candidate.feature_schema_version != self.feature_schema_version:
            raise ValueError(
                f"Candidate Ranker expects {self.feature_schema_version} but candidate "
                f"uses {candidate.feature_schema_version}"
            )
        missing = set(self.feature_names) - set(candidate.features)
        unexpected = set(candidate.features) - set(self.feature_names)
        if missing or unexpected:
            raise ValueError(
                "Candidate Ranker feature mismatch; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )


def _feature_matrix(candidates: list[Candidate]) -> np.ndarray:
    for candidate in candidates:
        if candidate.feature_schema_version != "semantic-cwe-v3":
            raise ValueError(
                "Candidate Ranker requires semantic-cwe-v3 candidates; got "
                + candidate.feature_schema_version
            )
        if set(candidate.features) != set(FEATURE_SCHEMA_SEMANTIC_CWE_V3):
            raise ValueError("Candidate Ranker received an incompatible feature map")
    return np.asarray(
        [
            [float(candidate.features[name]) for name in FEATURE_SCHEMA_SEMANTIC_CWE_V3]
            for candidate in candidates
        ],
        dtype=np.float64,
    )


def _balanced_sample_weights(targets: np.ndarray) -> np.ndarray:
    positive = max(1, int(targets.sum()))
    negative = max(1, int(len(targets) - targets.sum()))
    return np.asarray(
        [len(targets) / (2 * positive) if value else len(targets) / (2 * negative) for value in targets],
        dtype=np.float64,
    )
