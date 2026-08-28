from pathlib import Path

import pytest

from llm_security.analysis import LearnedCandidateRanker, SemanticStaticAnalyzer
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
