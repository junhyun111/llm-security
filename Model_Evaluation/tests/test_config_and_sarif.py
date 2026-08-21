import json
from pathlib import Path

from model_evaluation.config import load_mapping
from model_evaluation.paths import EVALUATION_ROOT
from model_evaluation.juliet.sarif_labels import parse_sarif_package


def test_mapping_matches_five_expert_taxonomy() -> None:
    mapping = load_mapping(EVALUATION_ROOT / "configs/cwe_expert_mapping.toml")

    assert mapping.expert_for("CWE-401") == "memory_bounds"
    assert mapping.expert_for("CWE-190") == "integer_size_type"
    assert mapping.expert_for("CWE-78") == "taint_api_contract"
    assert mapping.expert_for("CWE-252") == "control_state_error"
    assert mapping.expert_for("CWE-367") == "concurrency_toctou"
    assert mapping.expert_for("CWE-476") == "memory_bounds"


def test_sarif_parser_uses_result_location(tmp_path: Path) -> None:
    package = tmp_path / "100-v1.0.0"
    source = package / "src/testcases/CWE190_Demo/CWE190_Demo__x_01.c"
    source.parent.mkdir(parents=True)
    source.write_text("void bad(void) { int x = 1; }\n", encoding="utf-8")
    manifest = {
        "runs": [
            {
                "properties": {
                    "id": 100,
                    "language": "c",
                    "state": "mixed",
                },
                "artifacts": [
                    {
                        "location": {
                            "uri": "src/testcases/CWE190_Demo/CWE190_Demo__x_01.c"
                        }
                    }
                ],
                "results": [
                    {
                        "ruleId": "CWE-190",
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"index": 0},
                                    "region": {"startLine": 1},
                                }
                            }
                        ],
                    }
                ],
            }
        ]
    }
    (package / "manifest.sarif").write_text(json.dumps(manifest), encoding="utf-8")
    mapping = load_mapping(EVALUATION_ROOT / "configs/cwe_expert_mapping.toml")

    scenarios, warnings = parse_sarif_package(package, mapping)

    assert warnings == []
    assert len(scenarios) == 1
    assert scenarios[0].expert == "integer_size_type"
    assert scenarios[0].positive_regions[0].line_start == 1

