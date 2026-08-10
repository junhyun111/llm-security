from .analyzer import FunctionAnalysis, ProgramAnalysis, StructuralAnalyzer
from .control_flow import (
    CFGNode,
    ControlFlowBuilder,
    ControlFlowGraph,
    DominatorAnalysis,
    compute_dominators,
    dominates,
)
from .dataflow import (
    DataFlowGraph,
    Definition,
    DefUseEdge,
    ReachingDefinitionsAnalyzer,
    SliceStep,
    backward_slice,
)
from .frontend import TreeSitterFrontend
from .ir import (
    Assignment,
    CallSite,
    Condition,
    ControlRegion,
    FunctionIR,
    MemoryAccess,
    ProgramIR,
    SourceSpan,
    StatementIR,
)
from .legacy import LegacyRegexAnalyzer

__all__ = [
    "Assignment",
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
    "FunctionIR",
    "FunctionAnalysis",
    "LegacyRegexAnalyzer",
    "MemoryAccess",
    "ProgramIR",
    "ProgramAnalysis",
    "ReachingDefinitionsAnalyzer",
    "SliceStep",
    "SourceSpan",
    "StatementIR",
    "StructuralAnalyzer",
    "TreeSitterFrontend",
    "backward_slice",
    "compute_dominators",
    "dominates",
]
