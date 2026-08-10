from llm_security.analysis import (
    ControlFlowBuilder,
    TreeSitterFrontend,
    compute_dominators,
    dominates,
)


def _build(source: str):
    function = TreeSitterFrontend().parse_file("sample.c", source)[0]
    return function, ControlFlowBuilder().build(function)


def _node_with(cfg, text: str):
    matches = [node for node in cfg.nodes.values() if text in node.text]
    assert len(matches) == 1, (text, [node.text for node in matches])
    return matches[0]


def test_sequential_cfg_has_one_execution_path() -> None:
    _, cfg = _build(
        "int f(int x) { int a = x; int b = a; sink(b); return b; }"
    )
    a = _node_with(cfg, "int a = x")
    b = _node_with(cfg, "int b = a")
    sink = _node_with(cfg, "sink(b)")
    result = _node_with(cfg, "return b")

    assert cfg.successors[cfg.entry_id] == {a.node_id}
    assert cfg.successors[a.node_id] == {b.node_id}
    assert cfg.successors[b.node_id] == {sink.node_id}
    assert cfg.successors[sink.node_id] == {result.node_id}
    assert cfg.edge_kinds[(result.node_id, cfg.exit_id)] == "return"


def test_if_else_cfg_branches_and_merges() -> None:
    _, cfg = _build(
        "void f(int flag, int a, int b) { "
        "int x; if (flag) x = a; else x = b; sink(x); }"
    )
    condition = _node_with(cfg, "flag")
    then_node = _node_with(cfg, "x = a")
    else_node = _node_with(cfg, "x = b")
    sink = _node_with(cfg, "sink(x)")

    assert cfg.edge_kinds[(condition.node_id, then_node.node_id)] == "true"
    assert cfg.edge_kinds[(condition.node_id, else_node.node_id)] == "false"
    assert cfg.successors[then_node.node_id] == {sink.node_id}
    assert cfg.successors[else_node.node_id] == {sink.node_id}
    assert cfg.predecessors[sink.node_id] == {
        then_node.node_id,
        else_node.node_id,
    }


def test_early_return_does_not_fall_through_to_sink() -> None:
    _, cfg = _build(
        "void f(int len, int cap) { "
        "if (len > cap) return; sink(len); }"
    )
    condition = _node_with(cfg, "len > cap")
    result = _node_with(cfg, "return")
    sink = _node_with(cfg, "sink(len)")

    assert cfg.edge_kinds[(condition.node_id, result.node_id)] == "true"
    assert cfg.edge_kinds[(condition.node_id, sink.node_id)] == "false"
    assert cfg.edge_kinds[(result.node_id, cfg.exit_id)] == "return"
    assert sink.node_id not in cfg.successors[result.node_id]


def test_while_cfg_has_loop_back_and_false_exit() -> None:
    _, cfg = _build(
        "void f(int i, int n) { "
        "while (i < n) { i = i + 1; } sink(i); }"
    )
    condition = _node_with(cfg, "i < n")
    body = _node_with(cfg, "i = i + 1")
    sink = _node_with(cfg, "sink(i)")

    assert cfg.edge_kinds[(condition.node_id, body.node_id)] == "true"
    assert cfg.edge_kinds[(body.node_id, condition.node_id)] == "loop_back"
    assert cfg.edge_kinds[(condition.node_id, sink.node_id)] == "false"


def test_for_cfg_preserves_initializer_and_update() -> None:
    _, cfg = _build(
        "void f(int n) { for (int i = 0; i < n; ++i) use(i); done(); }"
    )
    initializer = _node_with(cfg, "int i = 0")
    condition = _node_with(cfg, "i < n")
    body = _node_with(cfg, "use(i)")
    update = _node_with(cfg, "++i")
    done = _node_with(cfg, "done()")

    assert cfg.successors[cfg.entry_id] == {initializer.node_id}
    assert cfg.successors[initializer.node_id] == {condition.node_id}
    assert cfg.successors[body.node_id] == {update.node_id}
    assert cfg.nodes[update.node_id].kind == "loop_update"
    assert cfg.nodes[update.node_id].defs == {"i"}
    assert cfg.nodes[update.node_id].uses == {"i"}
    assert cfg.edge_kinds[(update.node_id, condition.node_id)] == "loop_back"
    assert cfg.edge_kinds[(condition.node_id, done.node_id)] == "false"


def test_dominators_distinguish_top_level_and_nested_guard() -> None:
    _, guarded_cfg = _build(
        "void f(int len, int cap) { "
        "if (len > cap) return; memcpy(dst, src, len); }"
    )
    top_guard = _node_with(guarded_cfg, "len > cap")
    guarded_sink = _node_with(guarded_cfg, "memcpy")
    guarded_dominators = compute_dominators(guarded_cfg)
    assert dominates(guarded_dominators, top_guard.node_id, guarded_sink.node_id)

    _, nested_cfg = _build(
        "void g(int flag, int len, int cap) { if (flag) { "
        "if (len > cap) return; } memcpy(dst, src, len); }"
    )
    inner_guard = _node_with(nested_cfg, "len > cap")
    nested_sink = _node_with(nested_cfg, "memcpy")
    nested_dominators = compute_dominators(nested_cfg)
    assert not dominates(nested_dominators, inner_guard.node_id, nested_sink.node_id)


def test_unsupported_control_flow_is_reported() -> None:
    _, cfg = _build("void f(int x) { switch (x) { case 1: break; } }")

    assert "unsupported control-flow construct: switch" in cfg.warnings
    assert "unsupported control-flow construct: break" in cfg.warnings


def test_cfg_is_deterministic_and_edge_maps_are_symmetric() -> None:
    source = "void f(int x) { if (x) use(x); else use(0); done(); }"
    _, first = _build(source)
    _, second = _build(source)

    assert list(first.nodes) == list(second.nodes)
    assert first.edge_kinds == second.edge_kinds
    for source_id, targets in first.successors.items():
        for target_id in targets:
            assert source_id in first.predecessors[target_id]
    for target_id, sources in first.predecessors.items():
        for source_id in sources:
            assert target_id in first.successors[source_id]


def test_statement_after_unconditional_return_is_not_added_as_reachable() -> None:
    _, cfg = _build("void f(void) { return; unreachable_call(); }")

    assert not any("unreachable_call" in node.text for node in cfg.nodes.values())
    assert any(warning.startswith("unreachable StatementIR") for warning in cfg.warnings)
