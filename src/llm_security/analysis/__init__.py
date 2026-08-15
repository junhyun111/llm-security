from .analyzer import FunctionAnalysis, ProgramAnalysis, StructuralAnalyzer
from .api_semantics import (
    AllocationSpec,
    ApiCatalog,
    MemoryCopySpec,
    ReleaseSpec,
    SanitizerSpec,
    SinkSpec,
    SourceSpec,
)
from .candidate_builder import SemanticCandidateBuilder
from .control_flow import (
    CFGNode,
    ControlFlowBuilder,
    ControlFlowGraph,
    DominatorAnalysis,
    compute_dominators,
    dominates,
    is_statically_true_condition,
)
from .dataflow import (
    DataFlowGraph,
    Definition,
    DefUseEdge,
    ReachingDefinitionsAnalyzer,
    SliceStep,
    backward_slice,
)
from .frontend import AnalysisLimitError, TreeSitterFrontend
from .evidence_normalizer import SemanticEvidenceNormalizer
from .features import (
    FACT_FEATURE_MAP,
    FEATURE_SCHEMA_SEMANTIC_V1,
    SemanticFeatureExtractor,
)
from .guards import GuardRelation, find_guard_relations
from .index import FunctionAnalysisIndex
from .ir import (
    Assignment,
    CallSite,
    Condition,
    ControlRegion,
    ExpressionInfo,
    FunctionIR,
    MemoryAccess,
    ProgramIR,
    SourceSpan,
    StatementIR,
)
from .legacy import LegacyRegexAnalyzer
from .path_queries import (
    branch_reachability,
    is_reachable,
    reachable_without_definition,
    shortest_path,
    shortest_path_without_definition,
)
from .protocols import CandidateAnalyzer
from .semantic_analyzer import (
    SemanticAnalyzer,
    SemanticFunctionAnalysis,
    SemanticProgramAnalysis,
)
from .semantic_facts import SemanticFact, SemanticFactKind
from .semantic_static import SemanticStaticAnalyzer
from .suspicion import RISK_WEIGHTS, SuspicionScorer
from .taint import TaintAnalyzer, TaintPath

__all__ = [
    "Assignment",
    "AllocationSpec",
    "AnalysisLimitError",
    "ApiCatalog",
    "CandidateAnalyzer",
    "CallSite",
    "CFGNode",
    "Condition",
    "ControlFlowBuilder",
    "ControlFlowGraph",
    "ControlRegion",
    "DataFlowGraph",
    "Definition",
    "DefUseEdge",
    "DominatorAnalysis",
    "ExpressionInfo",
    "FACT_FEATURE_MAP",
    "FEATURE_SCHEMA_SEMANTIC_V1",
    "FunctionIR",
    "FunctionAnalysis",
    "FunctionAnalysisIndex",
    "GuardRelation",
    "LegacyRegexAnalyzer",
    "MemoryAccess",
    "MemoryCopySpec",
    "ProgramIR",
    "ProgramAnalysis",
    "ReachingDefinitionsAnalyzer",
    "ReleaseSpec",
    "SanitizerSpec",
    "SemanticAnalyzer",
    "SemanticCandidateBuilder",
    "SemanticEvidenceNormalizer",
    "SemanticFact",
    "SemanticFactKind",
    "SemanticFunctionAnalysis",
    "SemanticFeatureExtractor",
    "SemanticProgramAnalysis",
    "SemanticStaticAnalyzer",
    "SinkSpec",
    "SliceStep",
    "SourceSpan",
    "StatementIR",
    "StructuralAnalyzer",
    "SourceSpec",
    "SuspicionScorer",
    "TaintAnalyzer",
    "TaintPath",
    "TreeSitterFrontend",
    "RISK_WEIGHTS",
    "backward_slice",
    "branch_reachability",
    "compute_dominators",
    "dominates",
    "find_guard_relations",
    "is_reachable",
    "is_statically_true_condition",
    "reachable_without_definition",
    "shortest_path",
    "shortest_path_without_definition",
]
