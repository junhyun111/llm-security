from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from model_evaluation.config import ACTIVE_EXPERTS
from model_evaluation.router_models import (
    GradientBoostedUtilityRoutingModel,
    MultiTaskMLPUtilityRoutingModel,
)
from model_evaluation.schemas import IndexedScenario
from model_evaluation.stages.select_cohort import (
    _diversity_rank,
    _weighted_fair_merge,
    load_cohort_config,
)


def _scenario(
    case_id: str, *, expert: str, cwe: str, group: str
) -> IndexedScenario:
    return IndexedScenario(
        case_id=case_id,
        package_id=case_id,
        package_path="package",
        cwe=cwe,
        expert=expert,
        language="c",
        state="supported",
        template_group=group,
        exact_hash=f"exact-{case_id}",
        canonical_hash=f"canonical-{group}",
        source_artifacts=[],
        positive_regions=[],
        leakage_group=group,
        split="train",
    )


def test_cohort_config_has_exact_targets() -> None:
    config = load_cohort_config("Model_Evaluation/configs/cohort_15837.toml")
    assert sum(config.splits["train"].quotas.values()) == 6_000
    assert sum(config.splits["dev"].quotas.values()) == 1_500
    assert config.splits["train"].quotas["concurrency_toctou"] == 72
    assert config.splits["dev"].quotas["concurrency_toctou"] == 18
    assert config.splits["test"].mode == "all"


def test_diversity_rank_covers_cwe_and_groups_before_variations() -> None:
    rows = [
        _scenario("a1", expert="memory_bounds", cwe="CWE-121", group="a"),
        _scenario("a2", expert="memory_bounds", cwe="CWE-121", group="a"),
        _scenario("b1", expert="memory_bounds", cwe="CWE-121", group="b"),
        _scenario("c1", expert="memory_bounds", cwe="CWE-122", group="c"),
        _scenario("c2", expert="memory_bounds", cwe="CWE-122", group="c"),
        _scenario("d1", expert="memory_bounds", cwe="CWE-122", group="d"),
    ]
    ranked = _diversity_rank(rows, 2026, "train", "memory_bounds")
    assert {item.cwe for item, _, _ in ranked[:2]} == {"CWE-121", "CWE-122"}
    phases = [phase for _, phase, _ in ranked]
    first_variation = phases.index("family_variation")
    assert first_variation == 4
    assert {item.leakage_group for item, _, _ in ranked[:first_variation]} == {
        "a", "b", "c", "d"
    }
    assert [item.case_id for item, _, _ in ranked] == [
        item.case_id
        for item, _, _ in _diversity_rank(rows, 2026, "train", "memory_bounds")
    ]


def test_weighted_fair_merge_does_not_start_with_one_expert_only() -> None:
    selected = {
        expert: [
            (_scenario(f"{expert}-{index}", expert=expert, cwe="CWE-1", group=f"g-{index}"), "new_cwe", index)
            for index in range(1, count + 1)
        ]
        for expert, count in zip(ACTIVE_EXPERTS, (8, 5, 6, 3, 1), strict=True)
    }
    merged = _weighted_fair_merge(selected)
    assert len(merged) == 23
    assert {item.expert for item, _, _ in merged[:5]} == set(ACTIVE_EXPERTS)


def test_gradient_boosting_exposes_all_assignment_probabilities() -> None:
    features = []
    assignments = []
    labels = []
    for expert_index, expert in enumerate(ACTIVE_EXPERTS):
        assignment = f"{expert}::model::prompt"
        for row in range(30):
            features.append({"x": float(row), "expert_signal": float(expert_index)})
            assignments.append(assignment)
            labels.append(row % (expert_index + 2) == 0)
    model = GradientBoostedUtilityRoutingModel(seed=7).fit(
        features, assignments, labels
    )
    probabilities = model.predict_proba({"x": 3.0, "expert_signal": 2.0})
    assert set(probabilities) == set(assignments)
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())


def test_multitask_mlp_produces_five_outputs() -> None:
    torch = pytest.importorskip("torch")
    assignments = [
        SimpleNamespace(
            assignment_id=f"{expert}::model::prompt",
            expert=SimpleNamespace(value=expert),
            model_id="model",
        )
        for expert in ACTIVE_EXPERTS
    ]
    samples = []
    for candidate_index in range(24):
        candidate = SimpleNamespace(
            candidate_id=f"candidate-{candidate_index}",
            project_id=f"project-{candidate_index // 2}",
            features={
                "bias": 1.0,
                "x": float(candidate_index),
                "parity": float(candidate_index % 2),
            },
        )
        for expert_index, assignment in enumerate(assignments):
            samples.append(
                SimpleNamespace(
                    case_id=f"case-{candidate_index}",
                    candidate=candidate,
                    assignment=assignment,
                    success=(candidate_index + expert_index) % 4 == 0,
                )
            )
    model = MultiTaskMLPUtilityRoutingModel(
        seed=7, max_epochs=4, patience=2, batch_size=16
    ).fit_samples(samples)
    probabilities = model.predict_proba(samples[0].candidate.features)
    assert len(probabilities) == 5
    assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities.values())
    assert model.training_summary["training_device"] == (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    assert next(model.network.parameters()).device.type == "cpu"


def test_mlp_backend_does_not_fit_hidden_logistic_regression(monkeypatch) -> None:
    pytest.importorskip("torch")
    from model_evaluation.adapters.llm_security import activate_parent_package
    from model_evaluation.workflow import _fit_router_backend

    activate_parent_package()
    from llm_security.datasets import UtilitySample
    from llm_security.models import (
        ACTIVE_UTILITY_EXPERTS,
        Candidate,
        ExpertAssignment,
    )
    from llm_security.routing import BudgetedUtilityRouter, UtilityPolicyConfig

    def forbidden_lr_fit(cls, *args, **kwargs):
        raise AssertionError("MLP backend must not call BudgetedUtilityRouter.fit")

    monkeypatch.setattr(BudgetedUtilityRouter, "fit", classmethod(forbidden_lr_fit))
    assignments = [
        ExpertAssignment(expert=expert, model_id="model")
        for expert in ACTIVE_UTILITY_EXPERTS
    ]
    rows = []
    for candidate_index in range(20):
        candidate = Candidate(
            candidate_id=f"candidate-{candidate_index}",
            project_id=f"project-{candidate_index // 2}",
            file="unit.c",
            function="f",
            line_start=1,
            line_end=2,
            code="int f(void) { return 0; }",
            evidence=[],
            features={"bias": 1.0, "x": float(candidate_index)},
            feature_schema_version="semantic-cwe-v3",
        )
        for expert_index, assignment in enumerate(assignments):
            rows.append(
                UtilitySample(
                    candidate=candidate,
                    assignment=assignment,
                    success=(candidate_index + expert_index) % 3 == 0,
                    truth_labels_available=True,
                    case_id=f"case-{candidate_index}",
                    label_version="semantic-causal-v2",
                )
            )

    router = _fit_router_backend(
        rows,
        backend="multitask_mlp",
        policy=UtilityPolicyConfig(),
        seed=7,
        case_weights={},
        mlp_device="cpu",
        mlp_batch_size=64,
        mlp_max_epochs=2,
        mlp_patience=2,
        progress=None,
    )
    assert router.model.training_summary["backend"] == "multitask_mlp"
    assert len(router.statistics) == 5
