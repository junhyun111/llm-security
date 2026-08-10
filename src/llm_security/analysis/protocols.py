from __future__ import annotations

from typing import Protocol

from ..models import Candidate, ProjectCase


class CandidateAnalyzer(Protocol):
    def analyze(self, case: ProjectCase) -> list[Candidate]: ...
