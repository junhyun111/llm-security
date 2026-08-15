"""Reproducible, offline architecture experiments.

Nothing in this package invokes an LLM.  Phase 2E stops at expert selection.
"""

from .phase2e import (
    Phase2EConfig,
    prepare_phase2e_jsonl,
    run_phase2e,
    run_phase2e_jsonl,
)

__all__ = [
    "Phase2EConfig",
    "prepare_phase2e_jsonl",
    "run_phase2e",
    "run_phase2e_jsonl",
]
