from pathlib import Path

import pytest

from model_evaluation.paths import EVALUATION_ROOT, require_within
from model_evaluation.juliet.sanitizer import (
    find_oracle_leaks,
    sanitize_source_files,
)


def test_output_guard_rejects_parent_project() -> None:
    with pytest.raises(ValueError, match="outside isolated evaluation root"):
        require_within(EVALUATION_ROOT.parent / "results" / "x.json", EVALUATION_ROOT)


def test_sanitizer_preserves_layout_and_removes_oracle_tokens() -> None:
    raw = {
        "src/testcases/CWE190_demo_bad.cpp": (
            "/* FLAW: CWE-190 */\n"
            "void CWE190_demo_bad() { puts(\"Calling bad()\"); }\n"
            "// FIX\n"
            "void goodB2G() { return; }\n"
        )
    }
    result = sanitize_source_files(raw, case_id="case-1")
    sanitized = next(iter(result.source_files.values()))

    assert len(sanitized) == len(next(iter(raw.values())))
    assert sanitized.count("\n") == next(iter(raw.values())).count("\n")
    assert find_oracle_leaks(result.source_files) == []
    assert all("CWE" not in path for path in result.source_files)

