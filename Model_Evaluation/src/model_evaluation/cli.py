from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EvaluationConfig, load_config, load_mapping
from .juliet.indexer import build_index
from .juliet.splitter import assign_frozen_splits
from .stages.build_pilot import build_pilot
from .stages.evaluate_analyzer import evaluate_semantic_analyzer


DEFAULT_CONFIG = "Model_Evaluation/configs/pilot.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-security-eval",
        description="Isolated Juliet evaluation harness (no LLM calls in initial stages)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("index", "Index Juliet SARIF packages and freeze leakage-safe splits"),
        ("build-pilot", "Build and audit the balanced sanitized pilot"),
        ("evaluate-analyzer", "Measure Semantic Analyzer Candidate Recall"),
        ("run-initial", "Run index, pilot, and analyzer stages"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default=DEFAULT_CONFIG)
        if name in {"index", "run-initial"}:
            command.add_argument("--rebuild", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    mapping = load_mapping(config.paths.mapping)
    if args.command == "index":
        summary = _index(config, mapping, rebuild=args.rebuild)
        _print(summary)
        return 0
    if args.command == "build-pilot":
        _require_index(config)
        summary = build_pilot(config, mapping, progress=print)
        _print(summary)
        return 0
    if args.command == "evaluate-analyzer":
        report = evaluate_semantic_analyzer(config, mapping, progress=print)
        _print(report["overall"])
        return 0
    if args.command == "run-initial":
        index = _index(config, mapping, rebuild=args.rebuild)
        pilot = build_pilot(config, mapping, progress=print)
        analyzer = evaluate_semantic_analyzer(config, mapping, progress=print)
        _print(
            {
                "index": index,
                "pilot": pilot,
                "analyzer_overall": analyzer["overall"],
            }
        )
        return 0
    return 1


def _index(config: EvaluationConfig, mapping, *, rebuild: bool) -> dict[str, object]:
    summary = build_index(config, mapping, rebuild=rebuild, progress=print)
    if summary.get("mapping_hash") != mapping.mapping_hash:
        raise ValueError("Existing index uses a different CWE mapping; rerun with --rebuild")
    split = assign_frozen_splits(config, mapping)
    return {"index": summary, "split": split}


def _require_index(config: EvaluationConfig) -> None:
    if not config.paths.index.is_file() or not config.paths.split_manifest.is_file():
        raise ValueError("Juliet index/split is missing; run the index command first")


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

