from __future__ import annotations

from ..models import Candidate, ProjectCase
from .analyzer import StructuralAnalyzer
from .api_semantics import ApiCatalog
from .candidate_builder import SemanticCandidateBuilder
from .candidate_ranker import LearnedCandidateRanker
from .frontend import TreeSitterFrontend
from .semantic_analyzer import SemanticAnalyzer


class SemanticStaticAnalyzer:
    """Production-style semantic Candidate analyzer; gating remains downstream."""

    analysis_version = "semantic-analyzer-v3.1-types-context"

    def __init__(
        self,
        *,
        structural_analyzer: StructuralAnalyzer | None = None,
        semantic_analyzer: SemanticAnalyzer | None = None,
        candidate_builder: SemanticCandidateBuilder | None = None,
        candidate_ranker: LearnedCandidateRanker | None = None,
        catalog: ApiCatalog | None = None,
        max_source_bytes: int | None = 2 * 1024 * 1024,
        parse_timeout_ms: int | None = 30_000,
    ) -> None:
        selected_catalog = catalog or (
            semantic_analyzer.catalog if semantic_analyzer is not None else ApiCatalog.default()
        )
        self.structural_analyzer = structural_analyzer or StructuralAnalyzer(
            frontend=TreeSitterFrontend(
                max_source_bytes=max_source_bytes,
                parse_timeout_ms=parse_timeout_ms,
            )
        )
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer(selected_catalog)
        self.candidate_builder = candidate_builder or SemanticCandidateBuilder(
            catalog=selected_catalog
        )
        self.candidate_ranker = candidate_ranker

    def analyze(self, case: ProjectCase) -> list[Candidate]:
        structural = self.structural_analyzer.analyze(case.source_files)
        semantic = self.semantic_analyzer.analyze(structural)
        candidates = self.candidate_builder.build(case, semantic)
        return (
            self.candidate_ranker.rank(candidates)
            if self.candidate_ranker is not None
            else candidates
        )
