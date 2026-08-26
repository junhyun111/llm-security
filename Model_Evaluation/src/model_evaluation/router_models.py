from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised by the friendly runtime error
    torch = None
    nn = None
    DataLoader = TensorDataset = None


class GradientBoostedUtilityRoutingModel:
    """Independent nonlinear tabular baselines for each Expert assignment."""

    def __init__(self, *, seed: int = 2026) -> None:
        self.seed = seed
        self.vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
        self.models: dict[str, HistGradientBoostingClassifier] = {}
        self.constants: dict[str, float] = {}
        self.training_summary: dict[str, object] = {}

    @property
    def available_assignments(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.models) | set(self.constants)))

    def fit(
        self,
        features: Sequence[dict[str, float]],
        assignment_ids: Sequence[str],
        labels: Sequence[bool | int],
    ) -> "GradientBoostedUtilityRoutingModel":
        if not features or not (
            len(features) == len(assignment_ids) == len(labels)
        ):
            raise ValueError("GBDT utility features, assignments, and labels must align")
        matrix = np.asarray(self.vectorizer.fit_transform(features), dtype=np.float32)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, assignment_id in enumerate(assignment_ids):
            grouped[str(assignment_id)].append(index)
        self.models = {}
        self.constants = {}
        positives: dict[str, int] = {}
        for assignment_id, indices in sorted(grouped.items()):
            target = np.asarray([int(bool(labels[index])) for index in indices])
            positives[assignment_id] = int(target.sum())
            unique = np.unique(target)
            if len(unique) == 1:
                self.constants[assignment_id] = float(unique[0])
                continue
            classifier = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=self.seed,
            )
            classifier.fit(matrix[indices], target)
            self.models[assignment_id] = classifier
        self.training_summary = {
            "backend": "gradient_boosting",
            "feature_count": len(self.vectorizer.feature_names_),
            "outcome_count": len(labels),
            "positive_outcomes": positives,
            "constant_heads": dict(self.constants),
        }
        return self

    def predict_proba(self, features: dict[str, float]) -> dict[str, float]:
        if not self.available_assignments:
            raise RuntimeError("GradientBoostedUtilityRoutingModel is not fitted")
        matrix = np.asarray(self.vectorizer.transform([features]), dtype=np.float32)
        output = dict(self.constants)
        output.update(
            {
                assignment_id: float(model.predict_proba(matrix)[0, 1])
                for assignment_id, model in self.models.items()
            }
        )
        return output


if nn is not None:
    class _SharedExpertMLP(nn.Module):
        def __init__(self, input_size: int, output_size: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, output_size),
            )

        def forward(self, features):
            return self.layers(features)
else:
    class _SharedExpertMLP:  # pragma: no cover
        pass


class MultiTaskMLPUtilityRoutingModel:
    """Shared multi-task MLP for five Expert success probabilities."""

    def __init__(
        self,
        *,
        seed: int = 2026,
        batch_size: int = 128,
        max_epochs: int = 300,
        patience: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.seed = seed
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.assignment_ids: tuple[str, ...] = ()
        self.network = None
        self.temperatures: np.ndarray | None = None
        self.constant_probabilities: dict[int, float] = {}
        self.training_summary: dict[str, object] = {}

    @property
    def available_assignments(self) -> tuple[str, ...]:
        return self.assignment_ids

    def fit_samples(
        self,
        samples: Iterable,
        *,
        case_weights: dict[str, float] | None = None,
    ) -> "MultiTaskMLPUtilityRoutingModel":
        if torch is None:
            raise RuntimeError(
                "Multi-task MLP requires torch. Install Model_Evaluation/requirements.txt."
            )
        rows = list(samples)
        if not rows:
            raise ValueError("Multi-task MLP requires outcome samples")
        assignment_map = {
            row.assignment.assignment_id: row.assignment for row in rows
        }
        self.assignment_ids = tuple(
            sorted(
                assignment_map,
                key=lambda item: (
                    assignment_map[item].expert.value,
                    assignment_map[item].model_id,
                    item,
                ),
            )
        )
        assignment_index = {
            assignment_id: index
            for index, assignment_id in enumerate(self.assignment_ids)
        }
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for row in rows:
            grouped[(row.case_id, row.candidate.candidate_id)].append(row)
        records = []
        expected = set(self.assignment_ids)
        for key, group in sorted(grouped.items()):
            present = {row.assignment.assignment_id for row in group}
            if present != expected:
                raise ValueError(
                    f"Incomplete multi-task labels for {key}: "
                    f"missing={sorted(expected - present)}"
                )
            labels = np.zeros(len(self.assignment_ids), dtype=np.float32)
            for row in group:
                labels[assignment_index[row.assignment.assignment_id]] = float(
                    row.success
                )
            first = group[0]
            records.append(
                (
                    first.candidate.features,
                    labels,
                    first.candidate.project_id,
                    first.case_id,
                    float((case_weights or {}).get(first.case_id, 1.0)),
                )
            )
        features = [item[0] for item in records]
        matrix = np.asarray(self.vectorizer.fit_transform(features), dtype=np.float32)
        self.feature_mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale[self.feature_scale < 1e-6] = 1.0
        matrix = (matrix - self.feature_mean) / self.feature_scale
        labels = np.stack([item[1] for item in records])
        weights = np.asarray([item[4] for item in records], dtype=np.float32)
        train_indices, validation_indices = _project_holdout(
            [item[2] for item in records], self.seed
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.network = _SharedExpertMLP(matrix.shape[1], labels.shape[1])
        positive = labels[train_indices].sum(axis=0)
        negative = len(train_indices) - positive
        pos_weight = np.divide(
            negative,
            np.maximum(positive, 1.0),
            dtype=np.float32,
        )
        pos_weight = np.clip(pos_weight, 1.0, 50.0)
        self.constant_probabilities = {
            index: float(positive[index] > 0)
            for index in range(labels.shape[1])
            if positive[index] == 0 or negative[index] == 0
        }
        active = np.ones(labels.shape[1], dtype=np.float32)
        for index in self.constant_probabilities:
            active[index] = 0.0
        if not active.any():
            raise ValueError("Every MLP output head is constant")

        dataset = TensorDataset(
            torch.from_numpy(matrix[train_indices]),
            torch.from_numpy(labels[train_indices]),
            torch.from_numpy(weights[train_indices]),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.from_numpy(pos_weight)
        )
        active_tensor = torch.from_numpy(active)
        validation_x = torch.from_numpy(matrix[validation_indices])
        validation_y = torch.from_numpy(labels[validation_indices])
        validation_w = torch.from_numpy(weights[validation_indices])
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            for batch_x, batch_y, batch_weight in loader:
                optimizer.zero_grad()
                element_loss = criterion(self.network(batch_x), batch_y)
                loss = (
                    element_loss
                    * active_tensor
                    * batch_weight[:, None]
                ).sum() / (
                    active_tensor.sum() * batch_weight.sum().clamp_min(1e-8)
                )
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                validation_loss = (
                    criterion(self.network(validation_x), validation_y)
                    * active_tensor
                    * validation_w[:, None]
                ).sum() / (
                    active_tensor.sum() * validation_w.sum().clamp_min(1e-8)
                )
                value = float(validation_loss.item())
            if value < best_loss - 1e-6:
                best_loss = value
                best_epoch = epoch
                best_state = copy.deepcopy(self.network.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.temperatures = self._fit_temperatures(validation_x, validation_y, active)
        self.network.eval()
        self.training_summary = {
            "backend": "multitask_mlp",
            "architecture": [matrix.shape[1], 128, 64, labels.shape[1]],
            "feature_count": matrix.shape[1],
            "candidate_count": len(records),
            "train_candidates": len(train_indices),
            "validation_candidates": len(validation_indices),
            "positive_outcomes": {
                assignment_id: int(labels[:, index].sum())
                for index, assignment_id in enumerate(self.assignment_ids)
            },
            "pos_weight": {
                assignment_id: float(pos_weight[index])
                for index, assignment_id in enumerate(self.assignment_ids)
            },
            "constant_heads": {
                self.assignment_ids[index]: value
                for index, value in self.constant_probabilities.items()
            },
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "temperatures": {
                assignment_id: float(self.temperatures[index])
                for index, assignment_id in enumerate(self.assignment_ids)
            },
        }
        return self

    def predict_proba(self, features: dict[str, float]) -> dict[str, float]:
        if torch is None or self.network is None or self.feature_mean is None:
            raise RuntimeError("MultiTaskMLPUtilityRoutingModel is not fitted")
        matrix = np.asarray(self.vectorizer.transform([features]), dtype=np.float32)
        matrix = (matrix - self.feature_mean) / self.feature_scale
        with torch.no_grad():
            logits = self.network(torch.from_numpy(matrix)).numpy()[0]
        temperatures = self.temperatures if self.temperatures is not None else 1.0
        probabilities = 1.0 / (1.0 + np.exp(-logits / temperatures))
        for index, value in self.constant_probabilities.items():
            probabilities[index] = value
        return {
            assignment_id: float(np.clip(probabilities[index], 0.0, 1.0))
            for index, assignment_id in enumerate(self.assignment_ids)
        }

    def _fit_temperatures(self, features, labels, active: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            logits = self.network(features).detach()
        temperatures = np.ones(labels.shape[1], dtype=np.float32)
        for index in range(labels.shape[1]):
            if not active[index] or len(torch.unique(labels[:, index])) < 2:
                continue
            log_temperature = torch.zeros(1, requires_grad=True)
            optimizer = torch.optim.LBFGS(
                [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                temperature = log_temperature.exp().clamp(0.05, 20.0)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[:, index] / temperature,
                    labels[:, index],
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            temperatures[index] = float(
                log_temperature.detach().exp().clamp(0.05, 20.0).item()
            )
        return temperatures


def _project_holdout(project_ids: Sequence[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    projects = sorted(
        set(project_ids),
        key=lambda value: hashlib.sha256(
            f"{seed}:mlp-validation:{value}".encode("utf-8")
        ).hexdigest(),
    )
    if len(projects) >= 2:
        validation_count = min(len(projects) - 1, max(1, round(len(projects) * 0.15)))
        validation_projects = set(projects[:validation_count])
        validation = np.asarray(
            [index for index, value in enumerate(project_ids) if value in validation_projects],
            dtype=np.int64,
        )
        train = np.asarray(
            [index for index, value in enumerate(project_ids) if value not in validation_projects],
            dtype=np.int64,
        )
    else:
        ordered = sorted(
            range(len(project_ids)),
            key=lambda index: hashlib.sha256(
                f"{seed}:mlp-validation:{index}".encode("utf-8")
            ).hexdigest(),
        )
        cut = min(len(ordered) - 1, max(1, round(len(ordered) * 0.15)))
        validation = np.asarray(ordered[:cut], dtype=np.int64)
        train = np.asarray(ordered[cut:], dtype=np.int64)
    if not len(train) or not len(validation):
        raise ValueError("MLP needs at least two candidate groups for validation")
    return train, validation

