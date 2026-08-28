from __future__ import annotations

from ..models import RelatedFunctionSummary
from .semantic_analyzer import SemanticProgramAnalysis


def build_related_function_summaries(
    analysis: SemanticProgramAnalysis,
    function_key: str,
    *,
    max_per_relation: int = 3,
    max_code_characters: int = 3_000,
) -> list[RelatedFunctionSummary]:
    """Create deterministic direct caller/callee summaries for Expert context.

    This is deliberately bounded: candidate generation remains local while the
    Expert gets the neighboring functions that explain inputs, returned values,
    ownership effects, status checks, and sinks.
    """
    program = analysis.structural.program
    related: list[RelatedFunctionSummary] = []
    for relation, keys in (
        ("caller", sorted(program.callers.get(function_key, set()))),
        ("callee", sorted(program.callees.get(function_key, set()))),
    ):
        for related_key in keys[:max_per_relation]:
            semantic = analysis.functions.get(related_key)
            if semantic is None:
                continue
            function = semantic.structural.function
            related.append(
                RelatedFunctionSummary(
                    relation=relation,
                    function_key=related_key,
                    file=function.file,
                    function=function.name,
                    line_start=function.line_start,
                    line_end=function.line_end,
                    parameters=list(function.parameters),
                    calls=sorted({call.callee for call in function.calls}),
                    semantic_facts=sorted({fact.kind.value for fact in semantic.facts}),
                    code=function.code[:max_code_characters],
                    symbol_types=dict(function.symbol_types),
                )
            )
    return related
