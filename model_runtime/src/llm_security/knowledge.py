from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .models import Candidate, ExpertFamily


@dataclass(frozen=True, slots=True)
class SecurityKnowledge:
    knowledge_id: str
    title: str
    content: str
    families: tuple[ExpertFamily, ...] = ()
    tags: tuple[str, ...] = ()
    source: str | None = None


class KnowledgeRetriever(Protocol):
    def retrieve(
        self, candidate: Candidate, expert: ExpertFamily, *, top_k: int = 3
    ) -> list[SecurityKnowledge]: ...


class LocalSecurityKnowledgeRetriever:
    """Small deterministic retriever for CWE, patch, safe-example, and API notes."""

    def __init__(self, entries: Iterable[SecurityKnowledge]) -> None:
        self.entries = list(entries)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LocalSecurityKnowledgeRetriever":
        entries: list[SecurityKnowledge] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    entries.append(
                        SecurityKnowledge(
                            knowledge_id=str(raw["knowledge_id"]),
                            title=str(raw["title"]),
                            content=str(raw["content"]),
                            families=tuple(
                                ExpertFamily(value) for value in raw.get("families", [])
                            ),
                            tags=tuple(str(value).lower() for value in raw.get("tags", [])),
                            source=(None if raw.get("source") is None else str(raw["source"])),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid security knowledge JSONL at line {line_number}: {error}"
                    ) from error
        return cls(entries)

    def retrieve(
        self, candidate: Candidate, expert: ExpertFamily, *, top_k: int = 3
    ) -> list[SecurityKnowledge]:
        query = _candidate_terms(candidate) | set(expert.value.split("_"))
        ranked: list[tuple[float, str, SecurityKnowledge]] = []
        for entry in self.entries:
            terms = set(entry.tags) | _tokens(entry.title) | _tokens(entry.content)
            overlap = len(query & terms)
            family_match = expert in entry.families or (
                expert == ExpertFamily.MEMORY_SAFETY
                and ExpertFamily.LIFETIME_RESOURCE in entry.families
            )
            family_bonus = 4.0 if family_match else 0.0
            score = family_bonus + float(overlap)
            if score > 0.0:
                ranked.append((score, entry.knowledge_id, entry))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [entry for _, _, entry in ranked[:top_k]]


def format_knowledge(entries: Iterable[SecurityKnowledge], *, max_characters: int) -> str:
    blocks = []
    for entry in entries:
        source = f" source={entry.source}" if entry.source else ""
        blocks.append(
            f"[{entry.knowledge_id}] {entry.title}{source}\n{entry.content.strip()}"
        )
    return "\n\n".join(blocks)[:max_characters] or "No retrieved security knowledge."


def _candidate_terms(candidate: Candidate) -> set[str]:
    terms = set()
    for name, value in candidate.features.items():
        if value > 0.0:
            terms.update(name.lower().split("_"))
    for evidence in candidate.evidence:
        terms.update(evidence.kind.lower().split("_"))
        terms.update(_tokens(evidence.expression))
    return terms


def _tokens(value: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", value)}
