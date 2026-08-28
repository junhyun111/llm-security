from pathlib import Path

import pytest

from llm_security.analysis import LearnedCandidateRanker, SemanticStaticAnalyzer
from llm_security.config import AppConfig
from llm_security.factory import build_candidate_analyzer
from llm_security.models import ProjectCase


def _candidates():
    case = ProjectCase(
        "ranker-case",
        "ranker-project",
        {
            "sample.c": """
void safe(int value) { consume(value); }
void arithmetic(int n) { int bytes = n * 4; consume(bytes); }
void unchecked(int fd, char *b) { read(fd, b, 8); consume(b); }
void memory(char *p) { free(p); p[0] = 1; }
"""
        },
    )
    return SemanticStaticAnalyzer().analyze(case)


def test_candidate_ranker_scores_orders_and_round_trips(tmp_path: Path) -> None:
    candidates = _candidates()
    assert len(candidates) == 3
    labels = [candidate.function in {"unchecked", "memory"} for candidate in candidates]
    ranker = LearnedCandidateRanker.fit(
        candidates, labels, backend="logistic_regression", seed=7
    )

    ranked = ranker.rank(candidates)
    assert ranked[0].suspicion_score >= ranked[-1].suspicion_score
    artifact = ranker.save(tmp_path / "candidate_ranker.pkl")
    restored = LearnedCandidateRanker.load(artifact)
    assert restored.backend == "logistic_regression"
    assert restored.rank(candidates)[0].candidate_id == ranked[0].candidate_id


def test_candidate_ranker_rejects_old_feature_schema() -> None:
    candidates = _candidates()
    labels = [candidate.function == "memory" for candidate in candidates]
    ranker = LearnedCandidateRanker.fit(candidates, labels)
    candidates[0].feature_schema_version = "semantic-cwe-v2"

    with pytest.raises(ValueError, match="semantic-cwe-v3"):
        ranker.score(candidates[0])


def test_factory_loads_configured_candidate_ranker(tmp_path: Path) -> None:
    candidates = _candidates()
    labels = [candidate.function == "memory" for candidate in candidates]
    artifact = LearnedCandidateRanker.fit(candidates, labels).save(
        tmp_path / "candidate_ranker.pkl"
    )
    config = AppConfig()
    config.model.expert_model = "model/expert"
    config.model.validator_model = "model/validator"
    config.model.patch_model = "model/patch"
    config.analysis.candidate_ranker_path = str(artifact)
    config.analysis.candidate_ranker_required = True

    analyzer = build_candidate_analyzer(config)

    assert analyzer.candidate_ranker is not None
    assert analyzer.candidate_ranker.backend == "logistic_regression"


def test_factory_fails_closed_when_configured_ranker_is_missing(tmp_path: Path) -> None:
    config = AppConfig()
    config.model.expert_model = "model/expert"
    config.model.validator_model = "model/validator"
    config.model.patch_model = "model/patch"
    config.analysis.candidate_ranker_path = str(tmp_path / "missing.pkl")
    config.analysis.candidate_ranker_required = True

    with pytest.raises(ValueError, match="Cannot load configured Candidate Ranker"):
        build_candidate_analyzer(config)


def test_utility_path_requires_a_candidate_ranker() -> None:
    config = AppConfig()
    config.model.expert_model = "model/expert"
    config.model.validator_model = "model/validator"
    config.model.patch_model = "model/patch"

    with pytest.raises(ValueError, match="Candidate Ranker artifact is required"):
        build_candidate_analyzer(config, require_ranker=True)
