"""Reproducible, offline architecture experiments.

Nothing in this package invokes an LLM.  Phase 2E stops at expert selection.
"""

from .phase2e import Phase2EConfig, run_phase2e

__all__ = ["Phase2EConfig", "run_phase2e"]
