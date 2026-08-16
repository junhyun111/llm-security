from llm_security.analysis import SemanticAnalyzer, SemanticFactKind


def _analyze(source: str):
    result = SemanticAnalyzer().analyze_sources({"sample.c": source})
    return next(iter(result.functions.values()))


def _kinds(analysis) -> list[SemanticFactKind]:
    return [fact.kind for fact in analysis.facts]


def test_allocation_and_memory_copy_local_facts() -> None:
    allocation = _analyze("void f(int n) { char *p = malloc(n); consume(p); }")
    memory_copy = _analyze(
        "void f(char *d, char *s, int n) { memcpy(d, s, n); }"
    )

    assert SemanticFactKind.ALLOCATION in _kinds(allocation)
    copy_fact = next(
        fact
        for fact in memory_copy.facts
        if fact.kind == SemanticFactKind.MEMORY_COPY
    )
    assert copy_fact.attributes["dst"] == "d"
    assert copy_fact.attributes["src"] == "s"
    assert copy_fact.attributes["length"] == "n"


def test_early_return_guard_protects_memory_copy() -> None:
    analysis = _analyze(
        "void f(char *d, char *s, int len, int cap) { "
        "if (len > cap) return; memcpy(d, s, len); }"
    )

    assert SemanticFactKind.GUARD_PROTECTS_SINK in _kinds(analysis)
    assert SemanticFactKind.MEMORY_COPY_WITHOUT_GUARD not in _kinds(analysis)
    guard = next(
        fact
        for fact in analysis.facts
        if fact.kind == SemanticFactKind.GUARD_PROTECTS_SINK
    )
    assert guard.attributes["sink_branch"] == "false"
    assert guard.attributes["path_restricting"] is True


def test_nested_non_dominating_guard_is_not_protection() -> None:
    analysis = _analyze(
        "void f(char *d, char *s, int flag, int len, int cap) { "
        "if (flag) { if (len > cap) return; } memcpy(d, s, len); }"
    )

    assert SemanticFactKind.GUARD_PROTECTS_SINK not in _kinds(analysis)
    assert SemanticFactKind.MEMORY_COPY_WITHOUT_GUARD in _kinds(analysis)


def test_lower_bound_path_restriction_is_not_memory_protection() -> None:
    analysis = _analyze(
        "void f(char *d, char *s, int len) { "
        "if (len > 0) memcpy(d, s, len); }"
    )

    assert SemanticFactKind.GUARD_PROTECTS_SINK not in _kinds(analysis)
    assert SemanticFactKind.MEMORY_COPY_WITHOUT_GUARD in _kinds(analysis)


def test_arithmetic_and_cast_flow_to_size_sinks() -> None:
    allocation = _analyze(
        "void f(int count) { char *p = malloc(count * sizeof(int)); use(p); }"
    )
    memory_copy = _analyze(
        "void f(char *d, char *s, int len) { "
        "int n = len * 4; memcpy(d, s, n); }"
    )
    cast_sink = _analyze(
        "void f(int count) { char *p = malloc((unsigned long)count); use(p); }"
    )

    arithmetic_allocation = next(
        fact
        for fact in allocation.facts
        if fact.kind == SemanticFactKind.ARITHMETIC_TO_ALLOCATION
    )
    assert arithmetic_allocation.attributes["operators"] == ["*"]
    assert arithmetic_allocation.attributes["has_sizeof"] is True
    assert SemanticFactKind.ARITHMETIC_TO_MEMORY_SINK in _kinds(memory_copy)
    assert SemanticFactKind.CAST_TO_SIZE_SINK in _kinds(cast_sink)


def test_use_after_release_requires_no_intervening_redefinition() -> None:
    vulnerable = _analyze("void f(char *p) { free(p); p->field = 1; }")
    redefined = _analyze(
        "void f(char *p, int n) { free(p); p = malloc(n); p->field = 1; }"
    )

    assert SemanticFactKind.USE_AFTER_RELEASE in _kinds(vulnerable)
    assert SemanticFactKind.USE_AFTER_RELEASE not in _kinds(redefined)


def test_double_release_fact() -> None:
    analysis = _analyze("void f(char *p) { free(p); free(p); }")

    assert SemanticFactKind.DOUBLE_RELEASE in _kinds(analysis)


def test_uninitialized_use_is_path_sensitive_to_branch_definitions() -> None:
    maybe_uninitialized = _analyze(
        "void f(int flag) { int value; if (flag) value = 10; sink(value); }"
    )
    fully_initialized = _analyze(
        "void f(int flag) { int value; "
        "if (flag) value = 1; else value = 2; sink(value); }"
    )

    assert SemanticFactKind.UNINITIALIZED_USE in _kinds(maybe_uninitialized)
    assert SemanticFactKind.UNINITIALIZED_USE not in _kinds(fully_initialized)
    definitions = maybe_uninitialized.structural.dataflow.definitions.values()
    assert any(definition.kind == "uninitialized" for definition in definitions)


def test_nullable_dereference_respects_dominating_guard() -> None:
    unchecked = _analyze(
        "void f(int n) { char *p = malloc(n); *p = 1; }"
    )
    checked = _analyze(
        "void f(int n) { char *p = malloc(n); if (!p) return; *p = 1; }"
    )

    assert SemanticFactKind.UNCHECKED_NULLABLE_DEREFERENCE in _kinds(unchecked)
    assert SemanticFactKind.UNCHECKED_NULLABLE_DEREFERENCE not in _kinds(checked)


def test_null_branch_direction_must_really_protect_dereference() -> None:
    wrong_branch = _analyze(
        "void f(int n) { char *p = malloc(n); if (p) return; *p = 1; }"
    )
    protected = _analyze(
        "void f(int n) { char *p = malloc(n); if (!p) return; *p = 1; }"
    )

    assert SemanticFactKind.UNCHECKED_NULLABLE_DEREFERENCE in _kinds(wrong_branch)
    assert SemanticFactKind.UNCHECKED_NULLABLE_DEREFERENCE not in _kinds(protected)


def test_toctou_check_and_use_share_reaching_path_symbol() -> None:
    analysis = _analyze(
        "void f(char *path) { if (access(path, 0) == 0) open(path, 0); }"
    )

    fact = next(
        fact
        for fact in analysis.facts
        if fact.kind == SemanticFactKind.TOCTOU_CHECK_USE
    )
    assert fact.symbols == ["path"]
    assert fact.attributes["check_api"] == "access"
    assert fact.attributes["use_api"] == "open"


def test_concurrency_api_level_facts() -> None:
    analysis = _analyze(
        "void f(void *lock) { pthread_create(0, 0, worker, 0); "
        "pthread_mutex_lock(lock); pthread_mutex_unlock(lock); }"
    )

    assert SemanticFactKind.THREAD_SPAWN in _kinds(analysis)
    assert SemanticFactKind.LOCK_ACQUIRE in _kinds(analysis)
    assert SemanticFactKind.LOCK_RELEASE in _kinds(analysis)


def test_juliet_thread_wrappers_produce_concurrency_facts() -> None:
    analysis = _analyze(
        "void f(void *lock) { stdThreadCreate(worker, 0, 0); "
        "stdThreadLockAcquire(lock); stdThreadLockRelease(lock); }"
    )

    assert SemanticFactKind.THREAD_SPAWN in _kinds(analysis)
    assert SemanticFactKind.LOCK_ACQUIRE in _kinds(analysis)
    assert SemanticFactKind.LOCK_RELEASE in _kinds(analysis)


def test_cfg_warning_suppresses_absence_based_facts() -> None:
    analysis = _analyze(
        "void f(int x, char *d, char *s, int n) { "
        "switch (x) { default: memcpy(d, s, n); } }"
    )

    assert analysis.structural.cfg.warnings
    assert SemanticFactKind.MEMORY_COPY in _kinds(analysis)
    assert SemanticFactKind.MEMORY_COPY_WITHOUT_GUARD not in _kinds(analysis)


def test_fact_ids_and_order_are_deterministic() -> None:
    source = (
        "void f(char *d, char *s, int n) { char *p = malloc(n); "
        "memcpy(d, s, n); free(p); }"
    )

    first = _analyze(source)
    second = _analyze(source)

    assert [(fact.fact_id, fact.kind) for fact in first.facts] == [
        (fact.fact_id, fact.kind) for fact in second.facts
    ]
