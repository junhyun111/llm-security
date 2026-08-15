from pathlib import Path

from llm_security.experiments import write_utility_tradeoff_report


def test_utility_tradeoff_report_writes_table_and_svg_figures(tmp_path: Path) -> None:
    policies = {
        "best_single": {
            "truth_recall": 0.70,
            "outcome_precision": 0.80,
            "outcome_f1": 0.75,
            "average_assignments": 1.0,
            "average_realized_cost": 0.01,
        },
        "adaptive_gate": {
            "truth_recall": 0.95,
            "outcome_precision": 0.85,
            "outcome_f1": 0.90,
            "average_assignments": 2.4,
            "average_realized_cost": 0.025,
        },
        "full5": {
            "truth_recall": 0.97,
            "outcome_precision": 0.75,
            "outcome_f1": 0.85,
            "average_assignments": 5.0,
            "average_realized_cost": 0.05,
        },
    }

    outputs = write_utility_tradeoff_report(policies, tmp_path)

    assert set(outputs) == {
        "table",
        "recall_vs_average_experts",
        "recall_vs_api_cost",
    }
    assert "best_single" in (tmp_path / "utility_policy_table.csv").read_text(
        encoding="utf-8"
    )
    assert "<svg" in (tmp_path / "recall_vs_average_experts.svg").read_text(
        encoding="utf-8"
    )
