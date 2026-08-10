from __future__ import annotations

from ..static_analysis import FunctionRegion, LightweightStaticAnalyzer


class LegacyRegexAnalyzer(LightweightStaticAnalyzer):
    """Phase 1 analyzer retained as the baseline for future ablations."""


__all__ = ["FunctionRegion", "LegacyRegexAnalyzer", "LightweightStaticAnalyzer"]
