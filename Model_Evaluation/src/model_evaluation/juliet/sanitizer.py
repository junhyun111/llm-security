from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_ORACLE_IDENTIFIER_RE = re.compile(r"cwe\d+|bad|good|flaw|fix", re.I)
ORACLE_LEAK_RE = re.compile(r"cwe[_-]?\d+|bad|good|flaw|fix", re.I)


@dataclass(frozen=True, slots=True)
class SanitizedSources:
    source_files: dict[str, str]
    raw_to_virtual: dict[str, str]
    identifier_aliases: dict[str, str]

    def alias_text(self, value: str) -> str:
        return _IDENTIFIER_RE.sub(
            lambda match: self.identifier_aliases.get(match.group(0), match.group(0)),
            value,
        )


def sanitize_source_files(
    source_files: dict[str, str], *, case_id: str
) -> SanitizedSources:
    raw_to_virtual: dict[str, str] = {}
    aliases: dict[str, str] = {}
    sanitized: dict[str, str] = {}
    case_token = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    for index, raw_path in enumerate(sorted(source_files), start=1):
        suffix = _safe_suffix(raw_path)
        virtual = f"case_{case_token}/unit_{index:03d}{suffix}"
        raw_to_virtual[raw_path] = virtual
        without_comments = strip_comments_preserve_layout(source_files[raw_path])
        sanitized[virtual] = _neutralize_oracle_tokens(without_comments, aliases)
    return SanitizedSources(
        source_files=sanitized,
        raw_to_virtual=raw_to_virtual,
        identifier_aliases=aliases,
    )


def strip_comments_preserve_layout(source: str) -> str:
    """Blank C/C++ comments without changing source length or line numbers."""
    output = list(source)
    index = 0
    state = "normal"
    quote = ""
    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "normal":
            if current in {'"', "'"}:
                state = "quoted"
                quote = current
            elif current == "/" and following == "/":
                output[index] = output[index + 1] = " "
                index += 1
                state = "line_comment"
            elif current == "/" and following == "*":
                output[index] = output[index + 1] = " "
                index += 1
                state = "block_comment"
        elif state == "quoted":
            if current == "\\":
                index += 1
            elif current == quote:
                state = "normal"
        elif state == "line_comment":
            if current == "\n":
                state = "normal"
            else:
                output[index] = " "
        elif state == "block_comment":
            if current == "*" and following == "/":
                output[index] = output[index + 1] = " "
                index += 1
                state = "normal"
            elif current not in "\r\n":
                output[index] = " "
        index += 1
    return "".join(output)


def canonicalize_for_hash(source: str) -> str:
    without_comments = strip_comments_preserve_layout(source)
    aliases: dict[str, str] = {}
    neutral = _neutralize_oracle_tokens(without_comments, aliases)
    return re.sub(r"\s+", " ", neutral).strip()


def find_oracle_leaks(source_files: dict[str, str]) -> list[str]:
    leaks: list[str] = []
    for file, source in sorted(source_files.items()):
        path_match = ORACLE_LEAK_RE.search(file)
        if path_match:
            leaks.append(f"path:{file}:{path_match.group(0)}")
        for line_number, line in enumerate(source.splitlines(), start=1):
            match = ORACLE_LEAK_RE.search(line)
            if match:
                leaks.append(f"source:{file}:{line_number}:{match.group(0)}")
                if len(leaks) >= 20:
                    return leaks
    return leaks


def _neutralize_oracle_tokens(source: str, aliases: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if not _ORACLE_IDENTIFIER_RE.search(token):
            return token
        alias = aliases.get(token)
        if alias is None:
            digest = str(int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest(), "big"))
            material = "n" + digest * ((len(token) // len(digest)) + 1)
            alias = material[: len(token)]
            aliases[token] = alias
        return alias

    return _IDENTIFIER_RE.sub(replace, source)


def _safe_suffix(path: str) -> str:
    lower = path.lower()
    for suffix in (".cpp", ".cxx", ".cc", ".c"):
        if lower.endswith(suffix):
            return suffix
    return ".c"
