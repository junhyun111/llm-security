from __future__ import annotations

from ..models import Candidate, ProjectCase
from .analyzer import StructuralAnalyzer
from .api_semantics import ApiCatalog
from .candidate_builder import SemanticCandidateBuilder
from .semantic_analyzer import SemanticAnalyzer


class SemanticStaticAnalyzer:
    """Production-style semantic Candidate analyzer; gating remains downstream."""

    def __init__(
        self,
        *,
        structural_analyzer: StructuralAnalyzer | None = None,
        semantic_analyzer: SemanticAnalyzer | None = None,
        candidate_builder: SemanticCandidateBuilder | None = None,
        catalog: ApiCatalog | None = None,
    ) -> None:
        selected_catalog = catalog or (
            semantic_analyzer.catalog if semantic_analyzer is not None else ApiCatalog.default()
        )
        self.structural_analyzer = structural_analyzer or StructuralAnalyzer()
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer(selected_catalog)
        self.candidate_builder = candidate_builder or SemanticCandidateBuilder(
            catalog=selected_catalog
        )

    def analyze(self, case: ProjectCase) -> list[Candidate]:
        structural = self.structural_analyzer.analyze(case.source_files)
        semantic = self.semantic_analyzer.analyze(structural)
        return self.candidate_builder.build(case, semantic)
