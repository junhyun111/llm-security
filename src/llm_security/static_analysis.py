from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Node, Parser, Query, QueryCursor

from .models import Candidate, Evidence, ProjectCase


@dataclass(slots=True)
class FunctionRegion:
    name: str
    start_line: int
    end_line: int
    code: str


_FUNCTION_HEADER = re.compile(
    r"^(?!\s*(?:if|for|while|switch|catch)\b)"
    r"\s*(?:[\w:<>~*&]+\s+)+(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
_CONTROL_NAMES = {"if", "for", "while", "switch", "catch", "else", "do"}
_C_LANGUAGE = Language(tree_sitter_c.language())
_CPP_LANGUAGE = Language(tree_sitter_cpp.language())
_C_FUNCTION_QUERY = Query(_C_LANGUAGE, "(function_definition) @function")
_CPP_FUNCTION_QUERY = Query(_CPP_LANGUAGE, "(function_definition) @function")

_SIGNALS: dict[str, tuple[str, re.Pattern[str]]] = {
    "memory_api": (
        "memory_sink",
        re.compile(r"\b(?:memcpy|memmove|strcpy|strcat|sprintf|gets)\s*\("),
    ),
    "array_access": ("memory_access", re.compile(r"\b[A-Za-z_]\w*\s*\[[^\]]+\]")),
    "bounds_guard": (
        "guard",
        re.compile(r"\bif\s*\([^)]*(?:sizeof|capacity|size|max|length|len|count)[^)]*[<>=]"),
    ),
    "allocation": (
        "allocation",
        re.compile(r"\b(?:malloc|calloc|realloc|new)\b"),
    ),
    "free": ("release", re.compile(r"\b(?:free\s*\(|delete\b)")),
    "integer_size_arithmetic": (
        "integer_arithmetic",
        re.compile(r"\b(?:size|len|length|count|bytes|capacity)\w*\s*(?:\*|\+|<<)"),
    ),
    "cast": (
        "type_conversion",
        re.compile(r"(?:\([us]?(?:char|short|int|long|size_t)\)|static_cast<)"),
    ),
    "input_source": (
        "taint_source",
        re.compile(r"\b(?:read|recv|fread|scanf|getenv|argv)\b"),
    ),
    "dangerous_sink": (
        "taint_sink",
        re.compile(r"\b(?:system|popen|execl|execv|printf|fprintf|open|fopen)\s*\("),
    ),
    "error_signal": (
        "error_path",
        re.compile(r"\b(?:errno|return\s+-1|goto\s+(?:fail|error)|cleanup|parse|decode)\b"),
    ),
    "state_signal": (
        "state",
        re.compile(r"\b(?:state|status|initialized|authenticated|phase)\w*\b", re.IGNORECASE),
    ),
    "thread_signal": (
        "concurrency",
        re.compile(r"\b(?:pthread_|std::thread|atomic|shared_|global_)\w*"),
    ),
    "lock_signal": (
        "synchronization",
        re.compile(r"\b(?:mutex|lock_guard|unique_lock|pthread_mutex)\w*"),
    ),
    "toctou_signal": (
        "toctou",
        re.compile(r"\b(?:access|stat|lstat)\s*\([^;]+;[\s\S]{0,300}\b(?:open|fopen|unlink|rename)\s*\("),
    ),
}


def _function_name(node: Node, source: bytes) -> str:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return ""
    text = source[declarator.start_byte : declarator.end_byte].decode(
        "utf-8", errors="replace"
    )
    matches = list(
        re.finditer(
            r"(?P<name>(?:[A-Za-z_~]\w*::)*[A-Za-z_~]\w*|operator\s*[^\s(]+)\s*\(",
            text,
        )
    )
    if not matches:
        return ""
    name = matches[-1].group("name").strip()
    return name.rsplit("::", 1)[-1]


class LightweightStaticAnalyzer:
    """Portable fallback analyzer used in tests and before Clang/CodeQL adapters.

    This intentionally produces candidate evidence rather than claiming semantic
    correctness. Production adapters can emit the same Candidate/Evidence schema.
    """

    def __init__(self, max_candidates: int = 50, context_lines: int = 18) -> None:
        self.max_candidates = max_candidates
        self.context_lines = context_lines

    def analyze(self, case: ProjectCase) -> list[Candidate]:
        candidates: list[Candidate] = []
        for file_name, source in sorted(case.source_files.items()):
            for region in self.extract_functions(source, file_name):
                candidate = self._candidate_from_region(case, file_name, region)
                if candidate is not None:
                    candidates.append(candidate)
        candidates.sort(key=lambda item: (-item.static_score, item.file, item.line_start))
        return candidates[: self.max_candidates]

    def extract_functions(self, source: str, file_name: str) -> list[FunctionRegion]:
        regions = self._tree_sitter_functions(source, file_name)
        if regions:
            return regions
        return self._fallback_functions(source)

    @staticmethod
    def _tree_sitter_functions(source: str, file_name: str) -> list[FunctionRegion]:
        suffix = Path(file_name).suffix.lower()
        language = _C_LANGUAGE if suffix == ".c" else _CPP_LANGUAGE
        query = _C_FUNCTION_QUERY if suffix == ".c" else _CPP_FUNCTION_QUERY
        encoded = source.encode("utf-8")
        parser = Parser(language)
        tree = parser.parse(encoded)
        regions: list[FunctionRegion] = []
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)
        for node in captures.get("function", []):
            name = _function_name(node, encoded)
            if name and name not in _CONTROL_NAMES:
                regions.append(
                    FunctionRegion(
                        name=name,
                        start_line=node.start_point.row + 1,
                        end_line=node.end_point.row + 1,
                        code=encoded[node.start_byte : node.end_byte].decode(
                            "utf-8", errors="replace"
                        ),
                    )
                )
        return sorted(regions, key=lambda region: (region.start_line, region.end_line))

    @staticmethod
    def _fallback_functions(source: str) -> list[FunctionRegion]:
        lines = source.splitlines()
        regions: list[FunctionRegion] = []
        index = 0
        while index < len(lines):
            match = _FUNCTION_HEADER.search(lines[index])
            if not match or match.group("name") in _CONTROL_NAMES:
                index += 1
                continue
            start = index
            depth = lines[index].count("{") - lines[index].count("}")
            index += 1
            while index < len(lines) and depth > 0:
                depth += lines[index].count("{") - lines[index].count("}")
                index += 1
            end = max(start, index - 1)
            regions.append(
                FunctionRegion(
                    name=match.group("name"),
                    start_line=start + 1,
                    end_line=end + 1,
                    code="\n".join(lines[start : end + 1]),
                )
            )
        if not regions and source.strip():
            regions.append(
                FunctionRegion(
                    name="<translation_unit>",
                    start_line=1,
                    end_line=max(1, len(lines)),
                    code=source,
                )
            )
        return regions

    def _candidate_from_region(
        self, case: ProjectCase, file_name: str, region: FunctionRegion
    ) -> Candidate | None:
        digest = hashlib.sha1(
            f"{case.case_id}:{file_name}:{region.name}:{region.start_line}".encode("utf-8")
        ).hexdigest()[:12]
        features: dict[str, float] = {}
        evidence: list[Evidence] = []
        for signal_name, (kind, pattern) in _SIGNALS.items():
            matches = list(pattern.finditer(region.code))
            features[f"{signal_name}_count"] = float(len(matches))
            for match_index, match in enumerate(matches):
                relative_line = region.code[: match.start()].count("\n")
                expression = region.code.splitlines()[relative_line].strip()
                evidence.append(
                    Evidence(
                        evidence_id=f"EV-{digest}-{len(evidence) + 1}",
                        kind=kind,
                        file=file_name,
                        line=region.start_line + relative_line,
                        expression=expression,
                        facts={"signal": signal_name, "match_index": match_index},
                    )
                )

        features["function_length"] = float(region.end_line - region.start_line + 1)
        features["external_input_to_sink"] = float(
            features.get("input_source_count", 0.0) > 0
            and features.get("dangerous_sink_count", 0.0) > 0
        )
        features["released_then_accessed"] = float(self._released_then_accessed(region.code))
        features["guard_density"] = features.get("bounds_guard_count", 0.0) / max(
            1.0,
            features.get("memory_api_count", 0.0)
            + features.get("array_access_count", 0.0),
        )
        features["toctou_pair"] = float(
            bool(re.search(r"\b(?:access|stat|lstat)\s*\(", region.code))
            and bool(re.search(r"\b(?:open|fopen|unlink|rename)\s*\(", region.code))
        )
        signal_count = sum(
            value for key, value in features.items() if key.endswith("_count")
        )
        if signal_count == 0:
            return None
        risk = (
            signal_count
            + 2.0 * features["external_input_to_sink"]
            + 2.0 * features["released_then_accessed"]
            + 2.0 * features["toctou_pair"]
            - features.get("bounds_guard_count", 0.0)
            - features.get("lock_signal_count", 0.0)
        )
        return Candidate(
            candidate_id=f"C-{digest}",
            project_id=case.project_id,
            file=file_name,
            function=region.name,
            line_start=region.start_line,
            line_end=region.end_line,
            code=region.code,
            evidence=evidence,
            features=features,
            static_score=max(0.0, risk),
        )

    @staticmethod
    def _released_then_accessed(code: str) -> bool:
        released_names: list[tuple[str, int]] = []
        for match in re.finditer(r"\bfree\s*\(\s*([A-Za-z_]\w*)\s*\)", code):
            released_names.append((match.group(1), match.end()))
        for name, position in released_names:
            tail = code[position:]
            if re.search(rf"\b{re.escape(name)}\s*(?:\[|->|\.)", tail):
                return True
            if re.search(rf"\bfree\s*\(\s*{re.escape(name)}\s*\)", tail):
                return True
        return False


def evidence_index(candidates: Iterable[Candidate]) -> dict[str, Evidence]:
    return {
        evidence.evidence_id: evidence
        for candidate in candidates
        for evidence in candidate.evidence
    }
