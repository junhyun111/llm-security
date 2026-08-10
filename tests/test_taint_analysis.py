from llm_security.analysis import (
    ApiCatalog,
    SanitizerSpec,
    SemanticAnalyzer,
    SemanticFactKind,
)


def _analyze(source: str, catalog: ApiCatalog | None = None):
    analyzer = SemanticAnalyzer(catalog)
    result = analyzer.analyze_sources({"taint.c": source})
    return next(iter(result.functions.values()))


def _kinds(analysis) -> set[SemanticFactKind]:
    return {fact.kind for fact in analysis.facts}


def test_read_assignment_chain_reaches_system() -> None:
    analysis = _analyze(
        "void f(int fd) { char buf[100]; read(fd, buf, 100); "
        "char *cmd = buf; system(cmd); }"
    )

    assert SemanticFactKind.TAINT_SOURCE in _kinds(analysis)
    assert SemanticFactKind.TAINT_SINK in _kinds(analysis)
    assert SemanticFactKind.SOURCE_TO_SINK in _kinds(analysis)
    assert SemanticFactKind.UNSANITIZED_SOURCE_TO_SINK in _kinds(analysis)
    assert len(analysis.taint_paths) == 1
    assert analysis.taint_paths[0].source_symbol == "buf"
    assert analysis.taint_paths[0].sink_symbol == "cmd"
    assert SemanticFactKind.UNINITIALIZED_USE not in _kinds(analysis)


def test_unrelated_safe_value_does_not_create_source_to_sink() -> None:
    analysis = _analyze(
        'void f(int fd) { char buf[100]; read(fd, buf, 100); '
        'char *safe = "echo hello"; system(safe); }'
    )

    assert SemanticFactKind.SOURCE_TO_SINK not in _kinds(analysis)
    assert analysis.taint_paths == []


def test_tainted_return_propagates_to_sink() -> None:
    analysis = _analyze(
        'void f(void) { char *value = getenv("CMD"); system(value); }'
    )

    assert SemanticFactKind.SOURCE_TO_SINK in _kinds(analysis)


def test_injectable_sanitizer_kills_result_taint() -> None:
    catalog = ApiCatalog.default()
    catalog.sanitizers["sanitize"] = SanitizerSpec(result=True)
    analysis = _analyze(
        "void f(int fd) { char buf[100]; read(fd, buf, 100); "
        "char *cmd = sanitize(buf); system(cmd); }",
        catalog,
    )

    assert SemanticFactKind.SOURCE_TO_SINK not in _kinds(analysis)


def test_taint_fixed_point_terminates_on_loop() -> None:
    analysis = _analyze(
        "void f(int fd, int flag) { char buf[100]; read(fd, buf, 100); "
        "char *cmd = buf; while (flag) cmd = cmd; system(cmd); }"
    )

    assert SemanticFactKind.SOURCE_TO_SINK in _kinds(analysis)
    assert len(analysis.taint_paths) == 1
