from llm_security.analysis import SemanticStaticAnalyzer
from llm_security.models import ProjectCase


def test_builder_includes_relevant_functions_and_excludes_plain_arithmetic() -> None:
    case = ProjectCase(
        "builder",
        "project",
        {
            "sample.c": """
int add(int a, int b) { return a + b; }
void risky(char *p) { free(p); p[0] = 1; }
"""
        },
    )

    candidates = SemanticStaticAnalyzer().analyze(case)

    assert [candidate.function for candidate in candidates] == ["risky"]
    assert candidates[0].feature_schema_version == "semantic-cwe-v2"


def test_candidate_id_and_direct_call_graph_are_deterministic() -> None:
    case = ProjectCase(
        "builder-graph",
        "project",
        {
            "sample.c": """
void helper(char *p) { free(p); }
void caller(char *p) { helper(p); p[0] = 1; }
"""
        },
    )
    analyzer = SemanticStaticAnalyzer()

    first = analyzer.analyze(case)
    second = analyzer.analyze(case)
    helper = next(candidate for candidate in first if candidate.function == "helper")
    caller = next(candidate for candidate in first if candidate.function == "caller")

    assert helper.callers == ["caller"]
    assert caller.callees == ["helper"]
    assert [candidate.candidate_id for candidate in first] == [
        candidate.candidate_id for candidate in second
    ]
