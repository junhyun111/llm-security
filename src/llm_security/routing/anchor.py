from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..datasets import RouterSample
from ..models import Candidate, ExpertFamily, RouteDecision
from .model import BinaryRoutingModel


DEFAULT_ANCHORS = (
    ExpertFamily.MEMORY_BOUNDS,
    ExpertFamily.CONTROL_STATE_ERROR,
)

_RARE_CWE_SCORE_FEATURES = {
    ExpertFamily.INTEGER_SIZE_TYPE: "cwe_integer_score",
    ExpertFamily.TAINT_API_CONTRACT: "cwe_taint_score",
    ExpertFamily.CONCURRENCY_TOCTOU: "cwe_concurrency_score",
}


@dataclass(slots=True)
class AnchorRareMetrics:
    sample_count: int
    expert_coverage: float
    exact_coverage: float
    rare_recall: float
    rare_precision: float
    rare_trigger_rate: float
    average_experts_per_candidate: float
    llm_calls_saved_vs_all_six: int
    rare_family_metrics: dict[str, dict[str, float | int]]


@dataclass(slots=True)
class RareThresholdCalibration:
    threshold: float
    target_recall: float
    achieved_recall: float
    rare_trigger_rate: float
    target_met: bool
    family_thresholds: dict[str, float]
    family_recall: dict[str, float]
    family_trigger_rate: dict[str, float]


class AnchorRareRouter:
    """Always run common Experts and learn only whether a rare Expert is needed.

    Probabilities are independent and intentionally do not sum to one. This keeps
    multi-cause vulnerabilities representable and avoids treating Expert families
    as mutually exclusive classes.
    """

    artifact_version = 3
    required_feature_schema_version = "semantic-cwe-v3"

    def __init__(
        self,
        models: dict[ExpertFamily, BinaryRoutingModel],
        *,
        anchors: tuple[ExpertFamily, ...] = DEFAULT_ANCHORS,
        threshold: float = 0.5,
        thresholds: dict[ExpertFamily, float] | None = None,
        feature_schema_version: str = "semantic-cwe-v3",
    ) -> None:
        if not anchors:
            raise ValueError("AnchorRareRouter requires at least one anchor Expert")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Rare trigger threshold must be between 0 and 1")
        self.models = dict(models)
        self.anchors = tuple(dict.fromkeys(anchors))
        self.threshold = threshold
        self.thresholds = {
            family: (thresholds or {}).get(family, threshold)
            for family in self.models
        }
        if any(not 0.0 <= value <= 1.0 for value in self.thresholds.values()):
            raise ValueError("Every rare trigger threshold must be between 0 and 1")
        self.feature_schema_version = feature_schema_version
        self._artifact_version = self.artifact_version

    @property
    def available_families(self) -> tuple[ExpertFamily, ...]:
        return tuple(dict.fromkeys((*self.anchors, *sorted(self.models, key=lambda x: x.value))))

    @property
    def rare_families(self) -> tuple[ExpertFamily, ...]:
        return tuple(sorted(self.models, key=lambda family: family.value))

    @classmethod
    def fit(
        cls,
        samples: Iterable[RouterSample],
        *,
        anchors: tuple[ExpertFamily, ...] = DEFAULT_ANCHORS,
        rare_families: Iterable[ExpertFamily] | None = None,
        threshold: float = 0.5,
        seed: int = 2026,
    ) -> "AnchorRareRouter":
        materialized = list(samples)
        if not materialized:
            raise ValueError("Anchor/Rare Router training requires samples")
        schemas = {sample.candidate.feature_schema_version for sample in materialized}
        if len(schemas) != 1:
            raise ValueError(
                "Router training samples must use exactly one feature schema; got: "
                + ", ".join(sorted(schemas))
            )
        if schemas != {cls.required_feature_schema_version}:
            raise ValueError(
                "Anchor/Rare Router v3 requires static CWE hypothesis features with "
                f"schema={cls.required_feature_schema_version}; rerun phase2e-prepare."
            )
        anchor_set = set(anchors)
        observed = {
            label for sample in materialized for label in sample.labels if label not in anchor_set
        }
        rare = tuple(
            sorted(
                set(rare_families) if rare_families is not None else observed,
                key=lambda family: family.value,
            )
        )
        if not rare:
            raise ValueError("No rare Expert labels are available for trigger training")
        rows = [sample.candidate.features for sample in materialized]
        models = {
            family: BinaryRoutingModel(seed=seed).fit(
                rows, [family in sample.labels for sample in materialized]
            )
            for family in rare
        }
        return cls(
            models,
            anchors=anchors,
            threshold=threshold,
            feature_schema_version=next(iter(schemas)),
        )

    def route(self, candidate: Candidate) -> RouteDecision:
        self._check_schema(candidate)
        learned_scores = {
            family: model.predict_proba(candidate.features)
            for family, model in self.models.items()
        }
        static_scores = self._static_cwe_scores(candidate)
        rare_scores = {
            family: max(learned_scores[family], static_scores.get(family, 0.0))
            for family in self.models
        }
        selected = list(self.anchors)
        triggered = [
            family
            for family in sorted(
                rare_scores, key=lambda item: (-rare_scores[item], item.value)
            )
            if rare_scores[family] >= self.thresholds.get(family, self.threshold)
        ]
        selected.extend(family for family in triggered if family not in selected)
        scores = {family: 1.0 for family in self.anchors}
        scores.update(rare_scores)
        ranked_probabilities = sorted(rare_scores.values(), reverse=True)
        top = ranked_probabilities[0] if ranked_probabilities else 0.0
        second = ranked_probabilities[1] if len(ranked_probabilities) > 1 else 0.0
        reasons = [
            "common-family anchor is always executed: " + family.value
            for family in self.anchors
        ]
        reasons.extend(
            f"static CWE evidence raised {family.value}: "
            f"{learned_scores[family]:.3f} -> {rare_scores[family]:.3f}"
            for family in self.rare_families
            if static_scores.get(family, 0.0) > learned_scores[family]
        )
        reasons.extend(
            f"rare trigger {family.value}={rare_scores[family]:.3f} >= "
            f"{self.thresholds.get(family, self.threshold):.3f}"
            for family in triggered
        )
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            scores=scores,
            selected=selected,
            top1_confidence=top,
            top1_top2_margin=top - second,
            policy="anchor_rare_trigger",
            reasons=reasons,
            available_families=list(self.available_families),
            learned_scores=learned_scores,
            trigger_scores={family: rare_scores[family] for family in triggered},
        )

    def calibrate_threshold(
        self,
        samples: Iterable[RouterSample],
        *,
        target_rare_recall: float = 0.95,
    ) -> RareThresholdCalibration:
        materialized = list(samples)
        if not materialized:
            raise ValueError("Rare trigger calibration requires dev samples")
        if not 0.0 <= target_rare_recall <= 1.0:
            raise ValueError("Target rare recall must be between 0 and 1")
        rows = [
            (
                sample,
                self._rare_scores(sample.candidate),
            )
            for sample in materialized
        ]
        family_results: dict[ExpertFamily, tuple[float, float, float, int, int]] = {}
        for family in self.rare_families:
            thresholds = sorted(
                {0.0, 1.0, *(scores[family] for _, scores in rows)},
                reverse=True,
            )
            evaluations = [
                self._family_threshold_result(rows, family, threshold)
                for threshold in thresholds
            ]
            feasible = [item for item in evaluations if item[1] >= target_rare_recall]
            chosen = (
                min(feasible, key=lambda item: (item[2], -item[0]))
                if feasible
                else max(evaluations, key=lambda item: (item[1], -item[2], item[0]))
            )
            family_results[family] = chosen

        self.thresholds = {
            family: result[0] for family, result in family_results.items()
        }
        # Retain the scalar for compatibility with older callers and reports. It
        # represents the most recall-conservative threshold; routing uses the map.
        self.threshold = min(self.thresholds.values(), default=self.threshold)
        positives = sum(result[3] for result in family_results.values())
        true_positives = sum(result[4] for result in family_results.values())
        recall = true_positives / positives if positives else 1.0
        trigger_rate = (
            sum(result[2] for result in family_results.values()) / len(family_results)
            if family_results
            else 0.0
        )
        return RareThresholdCalibration(
            threshold=self.threshold,
            target_recall=target_rare_recall,
            achieved_recall=recall,
            rare_trigger_rate=trigger_rate,
            target_met=recall >= target_rare_recall,
            family_thresholds={
                family.value: result[0] for family, result in family_results.items()
            },
            family_recall={
                family.value: result[1] for family, result in family_results.items()
            },
            family_trigger_rate={
                family.value: result[2] for family, result in family_results.items()
            },
        )

    def evaluate(self, samples: Iterable[RouterSample]) -> AnchorRareMetrics:
        materialized = list(samples)
        if not materialized:
            raise ValueError("Anchor/Rare Router evaluation requires samples")
        decisions = [self.route(sample.candidate) for sample in materialized]
        covered = exact = rare_tp = rare_fp = rare_positive = trigger_count = calls = 0
        rare_set = set(self.rare_families)
        family_counts = {
            family: {"positive": 0, "true_positive": 0, "false_positive": 0, "triggered": 0}
            for family in rare_set
        }
        for sample, decision in zip(materialized, decisions, strict=True):
            truth = set(sample.labels)
            selected = set(decision.selected)
            covered += int(bool(truth & selected))
            exact += int(truth.issubset(selected))
            calls += len(decision.selected)
            for family in rare_set:
                positive = family in truth
                triggered = family in selected
                rare_positive += int(positive)
                rare_tp += int(positive and triggered)
                rare_fp += int(not positive and triggered)
                trigger_count += int(triggered)
                family_counts[family]["positive"] += int(positive)
                family_counts[family]["true_positive"] += int(positive and triggered)
                family_counts[family]["false_positive"] += int(not positive and triggered)
                family_counts[family]["triggered"] += int(triggered)
        count = len(materialized)
        per_family = {}
        for family, values in sorted(family_counts.items(), key=lambda item: item[0].value):
            positive = values["positive"]
            true_positive = values["true_positive"]
            false_positive = values["false_positive"]
            triggered = values["triggered"]
            per_family[family.value] = {
                **values,
                "recall": true_positive / positive if positive else 1.0,
                "precision": (
                    true_positive / (true_positive + false_positive)
                    if true_positive + false_positive
                    else 1.0
                ),
                "trigger_rate": triggered / count,
                "threshold": self.thresholds.get(family, self.threshold),
            }
        return AnchorRareMetrics(
            sample_count=count,
            expert_coverage=covered / count,
            exact_coverage=exact / count,
            rare_recall=rare_tp / rare_positive if rare_positive else 1.0,
            rare_precision=rare_tp / (rare_tp + rare_fp) if rare_tp + rare_fp else 1.0,
            rare_trigger_rate=trigger_count / (count * len(rare_set)) if rare_set else 0.0,
            average_experts_per_candidate=calls / count,
            llm_calls_saved_vs_all_six=len(ExpertFamily) * count - calls,
            rare_family_metrics=per_family,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "AnchorRareRouter":
        with Path(path).open("rb") as handle:
            router = pickle.load(handle)  # noqa: S301 - trusted local artifact
        if not isinstance(router, cls) or getattr(router, "_artifact_version", None) != cls.artifact_version:
            raise TypeError("Incompatible Anchor/Rare Router artifact; retrain it")
        if not hasattr(router, "thresholds"):
            router.thresholds = {
                family: router.threshold for family in router.models
            }
        return router

    def _check_schema(self, candidate: Candidate) -> None:
        if candidate.feature_schema_version != self.feature_schema_version:
            raise ValueError(
                f"Router expects {self.feature_schema_version} but candidate uses "
                f"{candidate.feature_schema_version}. Train or load a matching Router."
            )

    def _static_cwe_scores(
        self,
        candidate: Candidate,
    ) -> dict[ExpertFamily, float]:
        return {
            family: max(
                0.0,
                min(1.0, float(candidate.features.get(feature_name, 0.0))),
            )
            for family, feature_name in _RARE_CWE_SCORE_FEATURES.items()
            if family in self.models
        }

    def _rare_scores(self, candidate: Candidate) -> dict[ExpertFamily, float]:
        learned = {
            family: model.predict_proba(candidate.features)
            for family, model in self.models.items()
        }
        static = self._static_cwe_scores(candidate)
        return {
            family: max(score, static.get(family, 0.0))
            for family, score in learned.items()
        }

    @staticmethod
    def _family_threshold_result(
        rows,
        family: ExpertFamily,
        threshold: float,
    ) -> tuple[float, float, float, int, int]:
        positives = true_positives = triggered = 0
        for sample, scores in rows:
            positive = family in sample.labels
            active = scores[family] >= threshold
            positives += int(positive)
            true_positives += int(positive and active)
            triggered += int(active)
        recall = true_positives / positives if positives else 1.0
        trigger_rate = triggered / len(rows) if rows else 0.0
        return threshold, recall, trigger_rate, positives, true_positives
