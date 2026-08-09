from llm_security.models import ProjectCase
from llm_security.static_analysis import LightweightStaticAnalyzer


def test_candidate_generator_finds_vulnerable_function() -> None:
    case = ProjectCase(
        case_id="memory-vulnerable",
        project_id="synthetic",
        source_files={
            "memory.c": """#include <string.h>
void copy_packet(const char *src, size_t len) {
    char destination[16];
    memcpy(destination, src, len);
}
"""
        },
    )
    candidate = LightweightStaticAnalyzer().analyze(case)[0]

    assert candidate.function == "copy_packet"
    assert candidate.features["memory_api_count"] == 1.0
    assert candidate.features["bounds_guard_count"] == 0.0


def test_candidate_generator_records_fixed_guard() -> None:
    case = ProjectCase(
        case_id="memory-fixed",
        project_id="synthetic",
        source_files={
            "memory_fixed.c": """#include <string.h>
void copy_packet_fixed(const char *src, size_t len) {
    char destination[16];
    if (len <= sizeof(destination)) {
        memcpy(destination, src, len);
    }
}
"""
        },
    )
    candidate = LightweightStaticAnalyzer().analyze(case)[0]

    assert candidate.features["memory_api_count"] == 1.0
    assert candidate.features["bounds_guard_count"] == 1.0


def test_tree_sitter_does_not_treat_control_flow_as_functions() -> None:
    case = ProjectCase(
        case_id="cpp-functions",
        project_id="parser-test",
        source_files={
            "parser.cpp": """
template <typename T>
int parse_value(
    T *data,
    int len
) {
    if (len > 0) {
        return data[0];
    } else if (data) {
        return data[1];
    }
    return -1;
}
"""
        },
    )

    candidates = LightweightStaticAnalyzer().analyze(case)

    assert [candidate.function for candidate in candidates] == ["parse_value"]
    assert all(candidate.function != "if" for candidate in candidates)
