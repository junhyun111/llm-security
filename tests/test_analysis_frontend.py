from llm_security.analysis import TreeSitterFrontend


def test_c_frontend_builds_structured_function_ir_and_call_graph() -> None:
    source = """\
int helper(int value) {
    return value + 1;
}

void parse(char *buf, size_t len) {
    size_t n = len * 4;
    char *dst = malloc(n);
    if (n > 16) return;
    memcpy(dst, buf, len);
    dst[0] = buf[1];
    helper(n);
}
"""
    frontend = TreeSitterFrontend()

    program = frontend.parse_project({"parser.c": source})
    function = program.functions["parse"]

    assert function.parameters == ["buf", "len"]
    assignments = {item.target: item for item in function.assignments}
    assert assignments["n"].expression == "len * 4"
    assert assignments["n"].defs == {"n"}
    assert assignments["n"].uses == {"len"}
    assert assignments["*dst"].expression == "malloc(n)"

    calls = {item.callee: item for item in function.calls}
    assert calls["malloc"].arguments == ["n"]
    assert calls["malloc"].assigned_to == "dst"
    assert calls["memcpy"].arguments == ["dst", "buf", "len"]
    assert calls["helper"].arguments == ["n"]

    assert function.conditions[0].expression == "n > 16"
    assert function.conditions[0].symbols == {"n"}
    assert function.conditions[0].operator == ">"
    accesses = {(item.base, item.index, item.kind) for item in function.memory_accesses}
    assert ("dst", "0", "array_write") in accesses
    assert ("buf", "1", "array_read") in accesses
    assert program.callees["parse"] == {"helper"}
    assert program.callers["helper"] == {"parse"}


def test_cpp_frontend_handles_qualified_method_and_scoped_calls() -> None:
    source = """\
#include <utility>
#include <vector>

namespace demo {
class Parser {
public:
    int parse(const std::vector<int>& values);
};

int helper(int value) { return value; }

int Parser::parse(const std::vector<int>& values) {
    int first = values[0];
    auto copy = std::move(first);
    return helper(copy);
}
}
"""
    frontend = TreeSitterFrontend()

    first = frontend.parse_project({"parser.cpp": source})
    second = frontend.parse_project({"parser.cpp": source})
    method_key = next(key for key in first.functions if key.endswith("Parser::parse"))
    method = first.functions[method_key]

    assert method.parameters == ["values"]
    assert any(call.callee == "std::move" for call in method.calls)
    assert any(call.callee == "helper" for call in method.calls)
    assert any(access.base == "values" and access.index == "0" for access in method.memory_accesses)
    assert first.callees[method_key] == {"helper"}
    assert [call.call_id for call in method.calls] == [
        call.call_id for call in second.functions[method_key].calls
    ]


def test_frontend_extracts_loop_conditions_and_pointer_dereference() -> None:
    source = """\
int consume(char *ptr, int length) {
    int total = 0;
    for (int i = 0; i < length; ++i) total += ptr[i];
    while (length > 0) --length;
    return *ptr + total;
}
"""

    function = TreeSitterFrontend().parse_file("loop.c", source)[0]

    assert [(item.operator, item.symbols) for item in function.conditions] == [
        ("<", {"i", "length"}),
        (">", {"length"}),
    ]
    assert any(
        access.kind == "dereference" and access.base == "ptr"
        for access in function.memory_accesses
    )
