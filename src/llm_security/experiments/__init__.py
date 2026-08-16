"""Reproducible architecture experiments and outcome collection."""

from .phase2e import (
    Phase2EConfig,
    prepare_phase2e_frozen_jsonl,
    prepare_phase2e_jsonl,
    run_phase2e,
    run_phase2e_jsonl,
)
from .utility_data import collect_expert_outcomes
from .utility_end_to_end import UtilityEndToEndMetrics, evaluate_utility_end_to_end
from .utility_reporting import write_utility_tradeoff_report

__all__ = [
    "Phase2EConfig",
    "prepare_phase2e_frozen_jsonl",
    "prepare_phase2e_jsonl",
    "run_phase2e",
    "run_phase2e_jsonl",
    "collect_expert_outcomes",
    "UtilityEndToEndMetrics",
    "evaluate_utility_end_to_end",
    "write_utility_tradeoff_report",
]
