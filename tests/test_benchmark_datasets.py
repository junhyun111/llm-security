from __future__ import annotations

from pathlib import Path

from llm_security.benchmarks import (
    merge_case_split_directories,
    prepare_juliet_dataset,
)
from llm_security.datasets import load_cases_jsonl, write_cases_jsonl
from llm_security.models import ExpertFamily, GroundTruth, ProjectCase


def _write_juliet_case(root: Path, cwe: int, name: str, expression: str) -> None:
    path = root / f"CWE{cwe}_{name}__demo_01.c"
    path.write_text(
        (
            f"void CWE{cwe}_{name}__demo_01_bad(void) {{\n"
            f"  /* FLAW: benchmark vulnerability */ {expression};\n"
            "}\n"
            f"void CWE{cwe}_{name}__demo_01_good(void) {{ return; }}\n"
        ),
        encoding="utf-8",
    )


def test_juliet_converter_builds_e3_e4_e6_template_disjoint_splits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "juliet"
    source.mkdir()
    for suffix in ("A", "B", "C"):
        _write_juliet_case(
            source,
            190,
            f"Integer_Overflow_{suffix}",
            "int x = 2147483647 + 1",
        )
        _write_juliet_case(
            source,
            78,
            f"OS_Command_Injection_{suffix}",
            'system("echo bad")',
        )
        _write_juliet_case(
            source, 362, f"Race_Condition_{suffix}", "shared++"
        )
    output = tmp_path / "converted"

    manifest = prepare_juliet_dataset(source, output, seed=2026)
    cases = load_cases_jsonl(output / "cases_all.jsonl")

    assert manifest["case_count"] == 9
    assert len(cases) == 9
    assert {truth.experts[0] for case in cases for truth in case.ground_truth} == {
        ExpertFamily.INTEGER_SIZE_TYPE,
        ExpertFamily.TAINT_API_CONTRACT,
        ExpertFamily.CONCURRENCY_TOCTOU,
    }
    assert all(truth.line_start == 2 for case in cases for truth in case.ground_truth)
    projects_by_split = {
        split: {
            case.project_id
            for case in load_cases_jsonl(output / f"cases_{split}.jsonl")
        }
        for split in ("train", "dev", "test")
    }
    assert projects_by_split["train"].isdisjoint(projects_by_split["dev"])
    assert projects_by_split["train"].isdisjoint(projects_by_split["test"])
    for split in ("train", "dev", "test"):
        assert {
            truth.experts[0]
            for case in load_cases_jsonl(output / f"cases_{split}.jsonl")
            for truth in case.ground_truth
        } == {
            ExpertFamily.INTEGER_SIZE_TYPE,
            ExpertFamily.TAINT_API_CONTRACT,
            ExpertFamily.CONCURRENCY_TOCTOU,
        }


def _case(case_id: str, project_id: str, split: str) -> ProjectCase:
    return ProjectCase(
        case_id=case_id,
        project_id=project_id,
        source_files={"x.c": "void f(void) {}"},
        split=split,
        ground_truth=[
            GroundTruth(
                f"truth-{case_id}",
                "x.c",
                "f",
                1,
                1,
                [ExpertFamily.MEMORY_SAFETY],
                ["CWE-787"],
            )
        ],
    )


def test_merge_case_splits_preserves_source_frozen_assignments(tmp_path: Path) -> None:
    inputs = []
    for source_index in range(2):
        source = tmp_path / f"source-{source_index}"
        inputs.append(source)
        for split in ("train", "dev", "test"):
            write_cases_jsonl(
                [_case(f"case-{source_index}-{split}", f"p-{source_index}-{split}", split)],
                source / f"cases_{split}.jsonl",
            )

    manifest = merge_case_split_directories(inputs, tmp_path / "merged")

    assert manifest["case_count"] == 6
    assert len(load_cases_jsonl(tmp_path / "merged" / "cases_all.jsonl")) == 6
