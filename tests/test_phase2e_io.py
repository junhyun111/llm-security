from __future__ import annotations

from llm_security.experiments.io import write_json


def test_result_json_is_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"z": [3, 2, 1], "a": {"d": 2, "b": 1}}
    write_json(payload, first)
    write_json(payload, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text("utf-8").index('"a"') < first.read_text("utf-8").index('"z"')
