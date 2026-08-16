from __future__ import annotations

import hashlib

from ..models import Candidate, ProjectCase
from .api_semantics import ApiCatalog
from .cwe_hypotheses import StaticCweHypothesisEngine
from .evidence_normalizer import SemanticEvidenceNormalizer
from .features import SemanticFeatureExtractor
from .semantic_analyzer import SemanticProgramAnalysis
from .suspicion import SuspicionScorer


class SemanticCandidateBuilder:
    def __init__(
        self,
        *,
        catalog: ApiCatalog | None = None,
        feature_extractor: SemanticFeatureExtractor | None = None,
        evidence_normalizer: SemanticEvidenceNormalizer | None = None,
        suspicion_scorer: SuspicionScorer | None = None,
        cwe_hypothesis_engine: StaticCweHypothesisEngine | None = None,
    ) -> None:
        self.catalog = catalog or ApiCatalog.default()
        self.feature_extractor = feature_extractor or SemanticFeatureExtractor()
        self.evidence_normalizer = evidence_normalizer or SemanticEvidenceNormalizer()
        self.suspicion_scorer = suspicion_scorer or SuspicionScorer()
        self.cwe_hypothesis_engine = (
            cwe_hypothesis_engine or StaticCweHypothesisEngine()
        )

    def build(
        self,
        case: ProjectCase,
        analysis: SemanticProgramAnalysis,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        program = analysis.structural.program
        for function_key in sorted(analysis.functions):
            semantic_function = analysis.functions[function_key]
            function = semantic_function.structural.function
            if not self._is_candidate(semantic_function):
                continue
            callers = sorted(program.callers.get(function_key, set()))
            callees = sorted(program.callees.get(function_key, set()))
            evidence = self.evidence_normalizer.normalize(semantic_function)
            cwe_hypotheses = self.cwe_hypothesis_engine.infer(evidence)
            features = self.feature_extractor.extract(
                semantic_function,
                caller_count=len(callers),
                callee_count=len(callees),
                cwe_hypotheses=cwe_hypotheses,
            )
            digest = hashlib.sha1(
                f"{case.case_id}:{function.file}:{function.name}:{function.line_start}:{self.feature_extractor.schema_version}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            candidates.append(
                Candidate(
                    candidate_id=f"C-{digest}",
                    project_id=case.project_id,
                    file=function.file,
                    function=function.name,
                    line_start=function.line_start,
                    line_end=function.line_end,
                    code=function.code,
                    evidence=evidence,
                    features=features,
                    suspicion_score=self.suspicion_scorer.score(features),
                    callers=callers,
                    callees=callees,
                    feature_schema_version=self.feature_extractor.schema_version,
                    cwe_hypotheses=cwe_hypotheses,
                )
            )
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.file,
                candidate.line_start,
                candidate.function,
                candidate.candidate_id,
            ),
        )

    def _is_candidate(self, semantic_function) -> bool:
        function = semantic_function.structural.function
        if semantic_function.facts or function.memory_accesses:
            return True
        relevant = self._security_relevant_apis()
        return any(
            self.catalog.canonical_name(call.callee) in relevant
            for call in function.calls
        )

    def _security_relevant_apis(self) -> set[str]:
        return (
            set(self.catalog.memory_copy)
            | set(self.catalog.allocations)
            | set(self.catalog.releases)
            | set(self.catalog.taint_sources)
            | set(self.catalog.taint_sinks)
            | set(self.catalog.thread_spawn)
            | set(self.catalog.lock_acquire)
            | set(self.catalog.lock_release)
            | set(self.catalog.toctou_checks)
            | set(self.catalog.toctou_uses)
        )
