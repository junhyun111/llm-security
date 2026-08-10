from llm_security.analysis import (
    StructuralAnalyzer,
    backward_slice,
)


def _analyze(source: str):
    result = StructuralAnalyzer().analyze({"sample.c": source})
    return next(iter(result.functions.values()))


def _node_with(analysis, text: str):
    matches = [
        node for node in analysis.cfg.nodes.values() if text in node.text
    ]
    assert len(matches) == 1, (text, [node.text for node in matches])
    return matches[0]


def test_parameter_definition_reaches_assignment_and_sink() -> None:
    analysis = _analyze(
        "void f(unsigned long len) { unsigned long n = len * 4; sink(n); }"
    )
    assignment = _node_with(analysis, "n = len * 4")
    sink = _node_with(analysis, "sink(n)")

    len_definitions = analysis.dataflow.definitions_reaching(
        assignment.node_id, "len"
    )
    n_definitions = analysis.dataflow.definitions_reaching(sink.node_id, "n")

    assert len(len_definitions) == 1
    assert len_definitions[0].kind == "parameter"
    assert len_definitions[0].node_id == analysis.cfg.entry_id
    assert len(n_definitions) == 1
    assert n_definitions[0].node_id == assignment.node_id


def test_both_branch_definitions_reach_merged_use() -> None:
    analysis = _analyze(
        "void f(int flag, int a, int b) { int x = 0; "
        "if (flag) x = a; else x = b; sink(x); }"
    )
    then_node = _node_with(analysis, "x = a")
    else_node = _node_with(analysis, "x = b")
    sink = _node_with(analysis, "sink(x)")

    reaching = analysis.dataflow.definitions_reaching(sink.node_id, "x")

    assert {definition.node_id for definition in reaching} == {
        then_node.node_id,
        else_node.node_id,
    }


def test_call_assignment_is_classified_as_call_result() -> None:
    analysis = _analyze(
        "void f(int n) { char *dst = malloc(n); consume(dst); }"
    )
    sink = _node_with(analysis, "consume(dst)")

    reaching = analysis.dataflow.definitions_reaching(sink.node_id, "dst")

    assert len(reaching) == 1
    assert reaching[0].kind == "call_result"


def test_backward_slice_follows_assignment_chain_to_parameter() -> None:
    analysis = _analyze(
        "void f(unsigned long len) { unsigned long raw = len; "
        "unsigned long n = raw + 1; unsigned long bytes = n * 4; "
        "sink(bytes); }"
    )
    sink = _node_with(analysis, "sink(bytes)")

    steps = backward_slice(
        analysis.function,
        analysis.cfg,
        analysis.dataflow,
        sink_node_id=sink.node_id,
        symbols={"bytes"},
    )

    assert [step.symbol for step in steps] == ["bytes", "n", "raw", "len"]
    assert steps[-1].from_node == analysis.cfg.entry_id
    assert [step.depth for step in steps] == [0, 1, 2, 3]


def test_loop_dataflow_converges_and_slice_terminates() -> None:
    analysis = _analyze(
        "void f(int i, int n) { while (i < n) i = i + 1; sink(i); }"
    )
    sink = _node_with(analysis, "sink(i)")

    steps = backward_slice(
        analysis.function,
        analysis.cfg,
        analysis.dataflow,
        sink_node_id=sink.node_id,
        symbols={"i"},
        max_depth=4,
    )

    assert steps
    assert len(steps) < 10
