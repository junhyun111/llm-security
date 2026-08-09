from __future__ import annotations

from collections import defaultdict

from .models import Finding


class FindingAggregator:
    """Deterministic deduplication that preserves cross-family causal findings."""

    def aggregate(self, findings: list[Finding]) -> list[Finding]:
        groups: dict[tuple[str, str, str, int, int], list[Finding]] = defaultdict(list)
        for finding in findings:
            key = (
                finding.expert.value,
                finding.file,
                finding.function,
                finding.line_start,
                finding.line_end,
            )
            groups[key].append(finding)
        aggregated: list[Finding] = []
        for group in groups.values():
            primary = max(group, key=lambda item: item.confidence)
            primary.cwes = sorted({cwe for item in group for cwe in item.cwes})
            primary.evidence_ids = sorted(
                {evidence_id for item in group for evidence_id in item.evidence_ids}
            )
            primary.trigger_path = list(
                dict.fromkeys(node for item in group for node in item.trigger_path)
            )
            aggregated.append(primary)
        aggregated.sort(key=lambda item: (-item.confidence, item.file, item.line_start))
        return aggregated

