from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from ..config import EvaluationConfig, ExpertMapping
from ..paths import EVALUATION_ROOT, require_within, write_json
from ..schemas import IndexedScenario, Region, SourceArtifact
from .sanitizer import canonicalize_for_hash
from .sarif_labels import ManifestScenario, parse_sarif_package, resolve_artifact


_FLOW_SUFFIX_RE = re.compile(
    r"_\d{2}(?:[a-z]|_(?:bad|good[a-z0-9_]*))?$", re.I
)


def build_index(
    config: EvaluationConfig,
    mapping: ExpertMapping,
    *,
    rebuild: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    destination = require_within(config.paths.index, EVALUATION_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not rebuild:
            return index_summary(destination)
        destination.unlink()

    connection = sqlite3.connect(destination)
    _create_schema(connection)
    stats: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    cwe_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    exact_seen: dict[tuple[str, str], str] = {}
    fingerprint = hashlib.sha256()
    try:
        directories = _package_directories(config.dataset_root, config.max_packages)
        for ordinal, package_directory in enumerate(directories, start=1):
            stats["packages_scanned"] += 1
            manifest = package_directory / "manifest.sarif"
            try:
                manifest_bytes = manifest.read_bytes()
                fingerprint.update(package_directory.name.encode("utf-8"))
                fingerprint.update(hashlib.sha256(manifest_bytes).digest())
                scenarios, warnings = parse_sarif_package(package_directory, mapping)
                if not scenarios:
                    exclusion_counts["no_supported_cwe_result"] += 1
                for item in scenarios:
                    indexed = _index_scenario(item, warnings)
                    duplicate_key = (indexed.cwe, indexed.exact_hash)
                    indexed.duplicate_of = exact_seen.get(duplicate_key, "")
                    exact_seen.setdefault(duplicate_key, indexed.case_id)
                    _insert_scenario(connection, indexed)
                    stats["scenario_count"] += 1
                    family_counts[indexed.expert] += 1
                    cwe_counts[indexed.cwe] += 1
                    stats["duplicate_count"] += int(bool(indexed.duplicate_of))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                exclusion_counts[type(error).__name__] += 1
                _insert_exclusion(connection, package_directory, str(error))
            if ordinal % config.index_batch_size == 0:
                connection.commit()
                if progress:
                    progress(
                        f"index: {ordinal} packages, {stats['scenario_count']} supported scenarios"
                    )
        connection.commit()
        metadata = {
            "schema_version": config.schema_version,
            "mapping_version": mapping.schema_version,
            "mapping_hash": mapping.mapping_hash,
            "config_hash": config.config_hash,
            "dataset_root": str(config.dataset_root),
            "dataset_fingerprint": fingerprint.hexdigest(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "packages_scanned": stats["packages_scanned"],
            "scenario_count": stats["scenario_count"],
            "duplicate_count": stats["duplicate_count"],
            "family_distribution": dict(sorted(family_counts.items())),
            "cwe_distribution": dict(sorted(cwe_counts.items())),
            "exclusion_distribution": dict(sorted(exclusion_counts.items())),
        }
        for key, value in metadata.items():
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
        connection.commit()
    finally:
        connection.close()
    write_json(config.paths.index_summary, metadata)
    return metadata


def iter_scenarios(
    database: str | Path, *, include_duplicates: bool = False
) -> Iterator[IndexedScenario]:
    connection = sqlite3.connect(Path(database))
    try:
        query = "SELECT scenario_json, duplicate_of, leakage_group, split FROM scenarios"
        if not include_duplicates:
            query += " WHERE duplicate_of = ''"
        query += " ORDER BY case_id"
        for raw, duplicate_of, leakage_group, split in connection.execute(query):
            scenario = IndexedScenario.from_dict(json.loads(raw))
            scenario.duplicate_of = duplicate_of
            scenario.leakage_group = leakage_group
            scenario.split = split
            yield scenario
    finally:
        connection.close()


def index_summary(database: str | Path) -> dict[str, object]:
    connection = sqlite3.connect(Path(database))
    try:
        return {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value FROM metadata")
        }
    finally:
        connection.close()


def _index_scenario(item: ManifestScenario, warnings: list[str]) -> IndexedScenario:
    sources: dict[str, str] = {}
    artifacts: list[SourceArtifact] = []
    for index, uri in enumerate(sorted(set(item.source_uris)), start=1):
        path = resolve_artifact(item.package_path, uri)
        source = path.read_text(encoding="utf-8", errors="replace")
        sources[uri] = source
        artifacts.append(
            SourceArtifact(
                raw_uri=uri,
                raw_path=str(path),
                virtual_path=f"unit_{index:03d}{path.suffix.lower()}",
                language=_language_for(path),
                sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )
        )
    primary_uri = item.positive_regions[0].uri
    template_group = _template_group(item.cwe, primary_uri)
    exact_hash = _combined_hash(sources, canonical=False)
    canonical_hash = _combined_hash(sources, canonical=True)
    region_key = ";".join(
        f"{region.uri}:{region.line_start}:{region.line_end}"
        for region in item.positive_regions
    )
    case_seed = f"{item.package_id}:{item.cwe}:{region_key}"
    case_id = "juliet-sard-" + hashlib.sha256(case_seed.encode("utf-8")).hexdigest()[:20]
    return IndexedScenario(
        case_id=case_id,
        package_id=item.package_id,
        package_path=str(item.package_path),
        cwe=item.cwe,
        expert=item.expert,
        language=item.language,
        state=item.state,
        template_group=template_group,
        exact_hash=exact_hash,
        canonical_hash=canonical_hash,
        source_artifacts=artifacts,
        positive_regions=[
            Region(
                file=region.uri,
                function="",
                line_start=region.line_start,
                line_end=region.line_end,
                label="vulnerable",
                source="sarif",
            )
            for region in item.positive_regions
        ],
        warnings=list(warnings),
    )


def _package_directories(root: Path, maximum: int) -> Iterator[Path]:
    entries = sorted(
        (
            Path(entry.path)
            for entry in os.scandir(root)
            if entry.is_dir(follow_symlinks=False)
            and (Path(entry.path) / "manifest.sarif").is_file()
        ),
        key=lambda path: path.name,
    )
    if maximum:
        entries = entries[:maximum]
    yield from entries


def _combined_hash(sources: dict[str, str], *, canonical: bool) -> str:
    digest = hashlib.sha256()
    for uri, source in sorted(sources.items()):
        content = canonicalize_for_hash(source) if canonical else source
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _template_group(cwe: str, uri: str) -> str:
    stem = Path(uri).stem
    normalized = _FLOW_SUFFIX_RE.sub("", stem)
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", normalized).strip("-").lower()
    return f"juliet-{cwe.lower()}-{normalized}"


def _language_for(path: Path) -> str:
    return "cplusplus" if path.suffix.lower() in {".cc", ".cpp", ".cxx"} else "c"


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE scenarios (
            case_id TEXT PRIMARY KEY,
            package_id TEXT NOT NULL,
            cwe TEXT NOT NULL,
            expert TEXT NOT NULL,
            template_group TEXT NOT NULL,
            exact_hash TEXT NOT NULL,
            canonical_hash TEXT NOT NULL,
            duplicate_of TEXT NOT NULL DEFAULT '',
            leakage_group TEXT NOT NULL DEFAULT '',
            split TEXT NOT NULL DEFAULT '',
            scenario_json TEXT NOT NULL
        );
        CREATE INDEX scenarios_expert_split ON scenarios(expert, split);
        CREATE INDEX scenarios_template ON scenarios(template_group);
        CREATE INDEX scenarios_hashes ON scenarios(exact_hash, canonical_hash);
        CREATE TABLE exclusions (
            package_path TEXT NOT NULL,
            reason TEXT NOT NULL
        );
        """
    )


def _insert_scenario(connection: sqlite3.Connection, scenario: IndexedScenario) -> None:
    connection.execute(
        """
        INSERT INTO scenarios(
            case_id, package_id, cwe, expert, template_group, exact_hash,
            canonical_hash, duplicate_of, leakage_group, split, scenario_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', ?)
        """,
        (
            scenario.case_id,
            scenario.package_id,
            scenario.cwe,
            scenario.expert,
            scenario.template_group,
            scenario.exact_hash,
            scenario.canonical_hash,
            scenario.duplicate_of,
            json.dumps(scenario.to_dict(), ensure_ascii=False, sort_keys=True),
        ),
    )


def _insert_exclusion(
    connection: sqlite3.Connection, package_directory: Path, reason: str
) -> None:
    connection.execute(
        "INSERT INTO exclusions(package_path, reason) VALUES (?, ?)",
        (str(package_directory), reason[:2_000]),
    )
