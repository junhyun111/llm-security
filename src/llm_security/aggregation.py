from __future__ import annotations

from collections import defaultdict

from .models import Finding


class FindingAggregator:
    """Fuse causal findings across Expert families and model providers."""

    def aggregate(self, findings: list[Finding]) -> list[Finding]:
        buckets: dict[tuple[str, str, str], list[Finding]] = defaultdict(list)
        for finding in findings:
            buckets[(finding.candidate_id, finding.file, finding.function)].append(finding)

        aggregated: list[Finding] = []
        for bucket in buckets.values():
            aggregated.extend(self._aggregate_bucket(bucket))
        aggregated.sort(key=lambda item: (-item.confidence, item.file, item.line_start))
        return aggregated

    def _aggregate_bucket(self, findings: list[Finding]) -> list[Finding]:
        parents = list(range(len(findings)))

        def root(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = root(left), root(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left in range(len(findings)):
            for right in range(left + 1, len(findings)):
                if _causally_related(findings[left], findings[right]):
                    union(left, right)

        groups: dict[int, list[Finding]] = defaultdict(list)
        for index, finding in enumerate(findings):
            groups[root(index)].append(finding)
        return [self._fuse(group) for group in groups.values()]

    @staticmethod
    def _fuse(group: list[Finding]) -> Finding:
        primary = max(group, key=lambda item: item.confidence)
        experts = {
            expert
            for item in group
            for expert in (item.supporting_experts or [item.expert])
        }
        models = {
            model
            for item in group
            for model in (
                item.supporting_models
                or ([item.model_id] if item.model_id else [])
            )
        }
        primary.cwes = sorted({cwe for item in group for cwe in item.cwes})
        primary.evidence_ids = sorted(
            {evidence_id for item in group for evidence_id in item.evidence_ids}
        )
        primary.evidence_for = list(
            dict.fromkeys(value for item in group for value in item.evidence_for)
        )
        primary.evidence_against = list(
            dict.fromkeys(value for item in group for value in item.evidence_against)
        )
        primary.preconditions = list(
            dict.fromkeys(value for item in group for value in item.preconditions)
        )
        primary.trigger_path = list(
            dict.fromkeys(node for item in group for node in item.trigger_path)
        )
        primary.supporting_experts = sorted(experts, key=lambda item: item.value)
        primary.supporting_models = sorted(models)
        agreement_bonus = 0.03 * max(0, len(experts) - 1)
        diversity_bonus = 0.02 * max(0, len(models) - 1)
        primary.confidence = min(
            1.0,
            max(item.confidence for item in group) + agreement_bonus + diversity_bonus,
        )
        primary.line_start = min(item.line_start for item in group)
        primary.line_end = max(item.line_end for item in group)
        return primary


def _causally_related(left: Finding, right: Finding) -> bool:
    overlap = (
        left.line_start <= right.line_end + 2
        and right.line_start <= left.line_end + 2
    )
    same_sink = bool(
        left.sink
        and right.sink
        and _normalize(left.sink) == _normalize(right.sink)
    )
    shared_evidence = bool(set(left.evidence_ids) & set(right.evidence_ids))
    shared_path = bool(set(left.trigger_path) & set(right.trigger_path))
    return overlap or same_sink or shared_evidence or shared_path


def _normalize(value: str) -> str:
    return "".join(
        character.lower()
        for character in value
        if character.isalnum() or character == "_"
    )
