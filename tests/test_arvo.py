from llm_security import arvo
from llm_security.arvo import (
    ArvoRecord,
    build_case,
    classify_crash,
    first_old_hunk_line,
    normalize_frame_path,
    parse_git_patch,
    parse_github_repository,
    reverse_apply_patch,
    prepare_arvo_cases,
    split_cases_by_project,
)
from llm_security.models import ExpertFamily, ProjectCase


def test_arvo_helpers() -> None:
    assert parse_github_repository("https://github.com/curl/curl.git") == ("curl", "curl")
    assert first_old_hunk_line("@@ -120,4 +120,7 @@ int parse()") == 120
    assert normalize_frame_path("/src/curl/./lib/http.c", "curl") == "lib/http.c"


def test_crash_family_mapping() -> None:
    family, cwes = classify_crash("Heap-buffer-overflow WRITE 4")
    assert family is ExpertFamily.MEMORY_BOUNDS
    assert cwes == ["CWE-787"]

    family, cwes = classify_crash("Heap-use-after-free READ 8")
    assert family is ExpertFamily.LIFETIME_RESOURCE
    assert cwes == ["CWE-416"]


def test_ground_truth_comes_from_stack_and_patch() -> None:
    class FakeClient:
        def commit_patch(self, owner, repository, revision):
            return (
                "diff --git a/src/vuln.c b/src/vuln.c\n"
                "--- a/src/vuln.c\n"
                "+++ b/src/vuln.c\n"
                "@@ -1,3 +1,4 @@ int vulnerable(int *p)\n"
                " int vulnerable(int *p) {\n"
                "+  if (!p) return 0;\n"
                "   return p[1];\n"
                " }\n"
            )

        def raw_file(self, owner, repository, revision, file_path):
            return "int vulnerable(int *p) {\n  if (!p) return 0;\n  return p[1];\n}\n"

    record = ArvoRecord(
        local_id=1,
        project="demo",
        crash_type="Heap-buffer-overflow READ 4",
        crash_output=(
            "#0 0x1 in vulnerable /src/demo/src/vuln.c:2:10\n"
        ),
        severity="High",
        report="report",
        fix_commit="fixed",
        repo_addr="https://github.com/example/demo.git",
        patch_url="patch",
        sanitizer="asan",
        fuzz_target="target",
        fuzz_engine="libfuzzer",
        language="c",
    )

    case = build_case(record, client=FakeClient())

    assert case is not None
    assert case.ground_truth[0].file == "src/vuln.c"
    assert case.ground_truth[0].line_start == 2
    assert case.ground_truth[0].function == "vulnerable"
    assert case.metadata["ground_truth_source"] == "crash-stack+patch"
    assert case.source_files["src/vuln.c"] == (
        "int vulnerable(int *p) {\n  return p[1];\n}\n"
    )


def test_reverse_apply_patch_restores_removed_code() -> None:
    patch = (
        "diff --git a/a.c b/a.c\n--- a/a.c\n+++ b/a.c\n"
        "@@ -1,3 +1,3 @@\n int f(void) {\n-  return unsafe();\n"
        "+  return safe();\n }\n"
    )
    file_patch = parse_git_patch(patch)[0]

    vulnerable = reverse_apply_patch(
        "int f(void) {\n  return safe();\n}\n", file_patch
    )

    assert vulnerable == "int f(void) {\n  return unsafe();\n}\n"


def test_project_split_has_no_project_leakage() -> None:
    cases = [
        ProjectCase(
            case_id=f"case-{index}",
            project_id=f"project-{index // 2}",
            source_files={"a.c": "int f(void) { return 0; }"},
        )
        for index in range(12)
    ]

    splits = split_cases_by_project(cases, seed=2026)
    project_sets = {
        split: {case.project_id for case in split_cases}
        for split, split_cases in splits.items()
    }

    assert project_sets["train"].isdisjoint(project_sets["dev"])
    assert project_sets["train"].isdisjoint(project_sets["test"])
    assert project_sets["dev"].isdisjoint(project_sets["test"])


def test_all_mode_collects_every_available_record(monkeypatch, tmp_path) -> None:
    records = [
        ArvoRecord(
            local_id=index,
            project=f"demo-{index}",
            crash_type="Heap-buffer-overflow READ 4",
            crash_output="",
            severity="",
            report="",
            fix_commit="fixed",
            repo_addr="https://github.com/example/demo.git",
            patch_url="",
            sanitizer="",
            fuzz_target="",
            fuzz_engine="",
            language="c",
        )
        for index in (1, 2)
    ]

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def commit_patch(self, *_args):
            return (
                "diff --git a/a.c b/a.c\n--- a/a.c\n+++ b/a.c\n"
                "@@ -1,3 +1,4 @@\n int f(void) {\n+  return 0;\n"
                "   return unsafe();\n }\n"
            )

        def raw_file(self, *_args):
            return "int f(void) {\n  return 0;\n  return unsafe();\n}\n"

    monkeypatch.setattr(arvo, "load_arvo_records", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(arvo, "GitHubClient", FakeClient)
    destination = tmp_path / "cases.jsonl"

    cases = prepare_arvo_cases(
        "unused.db",
        destination,
        count=None,
        balanced=False,
        require_routable=False,
        allow_partial=True,
    )

    assert [case.case_id for case in cases] == ["arvo-1", "arvo-2"]
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 2
