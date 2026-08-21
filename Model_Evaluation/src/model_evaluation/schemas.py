from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    raw_uri: str
    raw_path: str
    virtual_path: str
    language: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Region:
    file: str
    function: str
    line_start: int
    line_end: int
    label: str
    source: str


@dataclass(slots=True)
class IndexedScenario:
    case_id: str
    package_id: str
    package_path: str
    cwe: str
    expert: str
    language: str
    state: str
    template_group: str
    exact_hash: str
    canonical_hash: str
    source_artifacts: list[SourceArtifact]
    positive_regions: list[Region]
    warnings: list[str] = field(default_factory=list)
    duplicate_of: str = ""
    leakage_group: str = ""
    split: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IndexedScenario":
        return cls(
            case_id=str(raw["case_id"]),
            package_id=str(raw["package_id"]),
            package_path=str(raw["package_path"]),
            cwe=str(raw["cwe"]),
            expert=str(raw["expert"]),
            language=str(raw["language"]),
            state=str(raw["state"]),
            template_group=str(raw["template_group"]),
            exact_hash=str(raw["exact_hash"]),
            canonical_hash=str(raw["canonical_hash"]),
            source_artifacts=[SourceArtifact(**item) for item in raw["source_artifacts"]],
            positive_regions=[Region(**item) for item in raw["positive_regions"]],
            warnings=[str(value) for value in raw.get("warnings", [])],
            duplicate_of=str(raw.get("duplicate_of", "")),
            leakage_group=str(raw.get("leakage_group", "")),
            split=str(raw.get("split", "")),
        )

