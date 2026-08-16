from __future__ import annotations

from pathlib import Path

import pytest

from llm_security.analysis import SemanticStaticAnalyzer
from llm_security.cwe import expert_for_cwe
from llm_security.datasets import (
    RouterSample,
    load_router_samples_jsonl,
    write_router_samples_jsonl,
)
from llm_security.evidence import ContextBuilder
from llm_security.models import (
    ExpertFamily,
    Finding,
    GroundTruth,
    ProjectCase,
    ValidationVerdict,
)
from llm_security.prompts import expert_messages
from llm_security.routing import AnchorRareRouter
from llm_security.validation import EvidenceValidator


def _analyze(source: str, *, truths: list[GroundTruth] | None = None):
    case = ProjectCase(
        case_id="case",
        project_id="project",
        source_files={"sample.c": source},
        ground_truth=list(truths or []),
    )
    return SemanticStaticAnalyzer().analyze(case)[0]


def test_static_cwe_hypotheses_are_evidence_backed_router_features() -> None:
    candidate = _analyze(
        "void risky(char *p, int n) { free(p); p[n] = 1; }"
    )

    hypotheses = {item.cwe: item for item in candidate.cwe_hypotheses}
    evidence_ids = {item.evidence_id for item in candidate.evidence}

    assert candidate.feature_schema_version == "semantic-cwe-v2"
    assert {"CWE-416", "CWE-787"} <= set(hypotheses)
    assert all(set(item.evidence_ids) <= evidence_ids for item in hypotheses.values())
    assert candidate.features["cwe_memory_score"] > 0.0
    assert candidate.features["cwe_416_score"] > 0.0
    assert candidate.features["cwe_hypothesis_count"] == 2.0


def test_cwe_hypotheses_do_not_read_ground_truth() -> None:
    source = "void risky(char *p) { free(p); p[0] = 1; }"
    without_truth = _analyze(source)
    wrong_truth = _analyze(
        source,
        truths=[
            GroundTruth(
                "truth",
                "sample.c",
                "risky",
                1,
                1,
                [ExpertFamily.CONCURRENCY_TOCTOU],
                ["CWE-367"],
            )
        ],
    )

    assert [item.cwe for item in without_truth.cwe_hypotheses] == [
        item.cwe for item in wrong_truth.cwe_hypotheses
    ]
    assert "CWE-367" not in {item.cwe for item in wrong_truth.cwe_hypotheses}


def test_taint_sink_produces_specific_and_generic_cwe_hypotheses() -> None:
    candidate = _analyze(
        'void risky(void) { char *cmd = getenv("CMD"); system(cmd); }'
    )

    cwes = {item.cwe for item in candidate.cwe_hypotheses}

    assert {"CWE-20", "CWE-78"} <= cwes
    assert candidate.features["cwe_taint_score"] > 0.0
    assert candidate.features["cwe_78_score"] > 0.0


def test_router_jsonl_and_prompt_preserve_fallible_cwe_hypotheses(
    tmp_path: Path,
) -> None:
    candidate = _analyze("void risky(char *p) { free(p); p[0] = 1; }")
    path = tmp_path / "router.jsonl"
    write_router_samples_jsonl(
        [RouterSample(candidate, [ExpertFamily.MEMORY_SAFETY])], path
    )

    restored = load_router_samples_jsonl(path)[0].candidate
    context = ContextBuilder().build(restored, ExpertFamily.MEMORY_SAFETY)
    messages = expert_messages(restored, context)

    assert restored.cwe_hypotheses[0].cwe == "CWE-416"
    assert "CWE-416" in messages[-1]["content"]
    assert "verify; do not copy blindly" in messages[-1]["content"]


def test_validator_requires_cwe_semantics_to_match_cited_evidence() -> None:
    candidate = _analyze("void risky(char *p) { free(p); p[0] = 1; }")
    uaf = next(item for item in candidate.evidence if item.kind == "use_after_release")

    def finding(cwe: str) -> Finding:
        return Finding(
            finding_id=f"finding-{cwe}",
            candidate_id=candidate.candidate_id,
            expert=ExpertFamily.MEMORY_SAFETY,
            title="reachable use after free",
            root_cause="released pointer is dereferenced",
            consequence="memory corruption",
            file=candidate.file,
            function=candidate.function,
            line_start=uaf.line,
            line_end=uaf.line,
            cwes=[cwe],
            source=None,
            sink=None,
            missing_guard=None,
            trigger_path=[],
            evidence_ids=[uaf.evidence_id],
            confidence=0.9,
        )

    validator = EvidenceValidator()
    supported = validator.validate(finding("CWE-416"), candidate)
    unsupported = validator.validate(finding("CWE-78"), candidate)

    assert supported.verdict == ValidationVerdict.VALIDATED
    assert supported.checks["cwe_semantics_supported"] is True
    assert unsupported.verdict == ValidationVerdict.REJECTED
    assert unsupported.checks["cwe_semantics_supported"] is False


def test_anchor_router_rejects_pre_cwe_feature_schema() -> None:
    candidate = _analyze("void risky(char *p) { free(p); p[0] = 1; }")
    candidate.feature_schema_version = "semantic-v1"
    sample = RouterSample(candidate, [ExpertFamily.MEMORY_SAFETY])

    with pytest.raises(ValueError, match="semantic-cwe-v2"):
        AnchorRareRouter.fit([sample])


def test_unlock_without_lock_routes_to_concurrency_expert() -> None:
    assert expert_for_cwe("CWE-832") == ExpertFamily.CONCURRENCY_TOCTOU
