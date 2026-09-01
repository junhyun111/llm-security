from __future__ import annotations

from dataclasses import dataclass

from .control_flow import (
    ControlFlowBuilder,
    ControlFlowGraph,
    compute_dominators,
)
from .dataflow import DataFlowGraph, ReachingDefinitionsAnalyzer
from .frontend import TreeSitterFrontend
from .ir import FunctionIR, ProgramIR


@dataclass(slots=True)
class FunctionAnalysis:
    function: FunctionIR
    cfg: ControlFlowGraph
    dominators: dict[str, set[str]]
    dataflow: DataFlowGraph


@dataclass(slots=True)
class ProgramAnalysis:
    program: ProgramIR
    functions: dict[str, FunctionAnalysis]


class StructuralAnalyzer:
    """Compose frontend, CFG, dominators, and data flow for experimentation."""

    def __init__(self, frontend: TreeSitterFrontend | None = None) -> None:
        self.frontend = frontend or TreeSitterFrontend()

    def analyze(self, source_files: dict[str, str]) -> ProgramAnalysis:
        program = self.frontend.parse_project(source_files)
        functions: dict[str, FunctionAnalysis] = {}
        for key, function in program.functions.items():
            cfg = ControlFlowBuilder().build(function)
            functions[key] = FunctionAnalysis(
                function=function,
                cfg=cfg,
                dominators=compute_dominators(cfg),
                dataflow=ReachingDefinitionsAnalyzer().analyze(function, cfg),
            )
        return ProgramAnalysis(program=program, functions=functions)
