from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arvo import prepare_arvo_cases, prepare_arvo_training_dataset
from .config import AppConfig
from .datasets import load_cases_jsonl, load_router_samples_jsonl, write_cases_jsonl
from .experiment import ExperimentRunner
from .experiments import Phase2EConfig, prepare_phase2e_jsonl, run_phase2e_jsonl
from .factory import build_pipeline
from .models import ProjectCase, to_dict
from .routing import AdaptiveExpertRouter, RoutingPolicyConfig


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-security",
        description="Conditional expert C/C++ vulnerability pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train-router", help="Train and evaluate a Router from Candidate/label JSONL files"
    )
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument(
        "--dev", required=True, help="Dev JSONL used only to calibrate routing policy"
    )
    train_parser.add_argument("--env-file", default=".env")
    train_parser.add_argument("--output", default="router.pkl")

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a C/C++ source file or directory through OpenRouter"
    )
    analyze_parser.add_argument("source")
    analyze_parser.add_argument("--env-file", default=".env")
    analyze_parser.add_argument("--router-artifact", default=None)
    analyze_parser.add_argument("--output", default="analysis.json")

    cases_parser = subparsers.add_parser(
        "run-cases", help="Run a JSONL benchmark through OpenRouter"
    )
    cases_parser.add_argument("cases")
    cases_parser.add_argument("--env-file", default=".env")
    cases_parser.add_argument("--router-artifact", default=None)
    cases_parser.add_argument("--output", default="experiment.json")

    export_parser = subparsers.add_parser(
        "export-case-template", help="Write a JSONL case template"
    )
    export_parser.add_argument("--output", default="case-template.jsonl")

    arvo_parser = subparsers.add_parser(
        "prepare-arvo", help="Prepare project-split Router training data from ARVO"
    )
    arvo_parser.add_argument("--db", default="data/arvo/arvo.db")
    arvo_parser.add_argument("--output", default="data/arvo/cases_all.jsonl")
    arvo_parser.add_argument("--dataset-dir", default="data/arvo")
    arvo_parser.add_argument("--count", type=int, default=30)
    arvo_parser.add_argument(
        "--all",
        action="store_true",
        help="Process every eligible ARVO C/C++ record; allows partial results and writes failures.",
    )
    arvo_parser.add_argument("--ids", nargs="*", type=int, default=[])
    arvo_parser.add_argument("--unique-projects", action="store_true")
    arvo_parser.add_argument("--no-balance", action="store_true")
    arvo_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cases already present in --output and continue remaining records.",
    )
    arvo_parser.add_argument(
        "--failure-log",
        default=None,
        help="JSONL path for records that could not be downloaded or reconstructed.",
    )
    arvo_parser.add_argument("--seed", type=int, default=2026)

    split_arvo_parser = subparsers.add_parser(
        "split-arvo",
        help="Build project-disjoint train/dev/test files from an existing ARVO cases JSONL",
    )
    split_arvo_parser.add_argument(
        "--cases", default="data/arvo/cases_all.jsonl"
    )
    split_arvo_parser.add_argument("--dataset-dir", default="data/arvo")
    split_arvo_parser.add_argument("--seed", type=int, default=2026)

    phase2e_parser = subparsers.add_parser(
        "phase2e",
        help="Run offline ARVO calibration and routing ablations (no LLM calls)",
    )
    phase2e_parser.add_argument(
        "--cases", default="data/arvo/cases_all.jsonl"
    )
    phase2e_parser.add_argument("--data-dir", default="data/phase2e")
    phase2e_parser.add_argument("--artifacts-dir", default="artifacts/phase2e")
    phase2e_parser.add_argument("--output", default="results/phase2e")
    phase2e_parser.add_argument("--seed", type=int, default=2026)
    phase2e_parser.add_argument("--target-gate-retention", type=float, default=0.98)
    phase2e_parser.add_argument("--target-routing-coverage", type=float, default=0.95)
    _add_semantic_safety_options(phase2e_parser)

    phase2e_prepare_parser = subparsers.add_parser(
        "phase2e-prepare",
        help="Stream ARVO into compact Phase 2E Router JSONL files without training",
    )
    phase2e_prepare_parser.add_argument(
        "--cases", default="data/arvo/cases_all.jsonl"
    )
    phase2e_prepare_parser.add_argument("--data-dir", default="data/phase2e")
    phase2e_prepare_parser.add_argument("--seed", type=int, default=2026)
    phase2e_prepare_parser.add_argument(
        "--backend",
        choices=("all", "legacy", "semantic"),
        default="all",
        help="Prepare both backends or rerun only one backend.",
    )
    _add_semantic_safety_options(phase2e_prepare_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train-router":
        config = AppConfig.from_env(args.env_file)
        router = AdaptiveExpertRouter.fit(
            load_router_samples_jsonl(args.train),
            policy_config=_routing_policy_config(config),
            seed=config.runtime.seed,
            use_rule_fallback=config.router.use_rule_fallback,
        )
        dev_samples = load_router_samples_jsonl(args.dev)
        calibration = router.calibrate_policy(
            dev_samples, target_coverage=config.router.target_coverage
        )
        metrics = router.evaluate(dev_samples)
        router.save(args.output)
        print(
            json.dumps(
                {"calibration": to_dict(calibration), "metrics": to_dict(metrics)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "analyze":
        config = AppConfig.from_env(args.env_file)
        router = _load_router(config, args.router_artifact)
        case = ProjectCase(
            case_id="ad-hoc-analysis",
            project_id=Path(args.source).resolve().name,
            source_files=_read_sources(Path(args.source)),
            split="unlabeled",
        )
        result = build_pipeline(config, router).run(case)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(result.validated_findings)} validated findings to {destination}")
        return 0
    if args.command == "run-cases":
        config = AppConfig.from_env(args.env_file)
        router = _load_router(config, args.router_artifact)
        output = ExperimentRunner(build_pipeline(config, router)).run(load_cases_jsonl(args.cases))
        ExperimentRunner.save(output, args.output)
        print(json.dumps(to_dict(output.aggregate), ensure_ascii=False, indent=2))
        return 0
    if args.command == "export-case-template":
        case = ProjectCase(
            case_id="replace-me",
            project_id="replace-me",
            source_files={"example.c": "int example(void) { return 0; }\n"},
            split="dev",
            metadata={"note": "Ground truth is used only by the evaluator."},
        )
        write_cases_jsonl([case], args.output)
        print(f"Wrote {args.output}")
        return 0
    if args.command == "prepare-arvo":
        if args.all and args.ids:
            raise ValueError("--all cannot be combined with --ids")
        if args.all and args.unique_projects:
            raise ValueError("--all cannot be combined with --unique-projects")
        failure_log = args.failure_log
        if args.all and failure_log is None:
            failure_log = str(Path(args.dataset_dir) / "arvo_failures.jsonl")
        cases = prepare_arvo_cases(
            args.db,
            args.output,
            count=None if args.all else args.count,
            case_ids=args.ids,
            unique_projects=False if args.all else args.unique_projects,
            balanced=False if args.all else not args.no_balance,
            seed=args.seed,
            # Benchmark cases must not disappear merely because the analyzer
            # missed their ground-truth location.
            require_routable=False,
            # Full collection is intentionally resumable by default so an
            # existing partial dataset is never discarded on a transient
            # GitHub failure.
            resume=args.resume or args.all,
            failure_log=failure_log,
            allow_partial=args.all,
        )
        manifest = prepare_arvo_training_dataset(
            cases,
            args.dataset_dir,
            seed=args.seed,
        )
        scope = "eligible ARVO records" if args.all else "ARVO cases"
        print(f"Prepared {len(cases)} {scope} in {args.output}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "split-arvo":
        cases = load_cases_jsonl(args.cases)
        if not cases:
            raise ValueError(f"No cases found in {args.cases}")
        manifest = prepare_arvo_training_dataset(
            cases,
            args.dataset_dir,
            seed=args.seed,
        )
        print(f"Split {len(cases)} ARVO cases into {args.dataset_dir}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "phase2e":
        result = run_phase2e_jsonl(
            args.cases,
            config=Phase2EConfig(
                seed=args.seed,
                target_gate_retention=args.target_gate_retention,
                target_routing_coverage=args.target_routing_coverage,
                semantic_max_source_bytes=_source_limit_bytes(args.max_source_mb),
                semantic_parse_timeout_ms=_parse_timeout_ms(args.parse_timeout_seconds),
                data_directory=args.data_dir,
                artifact_directory=args.artifacts_dir,
                output_directory=args.output,
            ),
            progress=print,
        )
        print(f"Phase 2E results saved to {args.output}")
        print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "phase2e-prepare":
        summary = prepare_phase2e_jsonl(
            args.cases,
            config=Phase2EConfig(
                seed=args.seed,
                semantic_max_source_bytes=_source_limit_bytes(args.max_source_mb),
                semantic_parse_timeout_ms=_parse_timeout_ms(args.parse_timeout_seconds),
                data_directory=args.data_dir,
            ),
            progress=print,
            backends=("legacy", "semantic")
            if args.backend == "all"
            else (args.backend,),
        )
        print(f"Phase 2E Router data saved to {args.data_dir}")
        print(json.dumps(summary["router_sample_counts"], ensure_ascii=False, indent=2))
        return 0
    return 1


def _load_router(config: AppConfig, artifact: str | None):
    if not artifact:
        raise ValueError(
            "Adaptive routing requires --router-artifact. "
            "Create one with 01_train_router.ipynb or train-router."
        )
    return AdaptiveExpertRouter.load(artifact)


def _routing_policy_config(config: AppConfig) -> RoutingPolicyConfig:
    return RoutingPolicyConfig(
        high_confidence=config.router.high_confidence,
        min_margin=config.router.min_margin,
        max_entropy=config.router.max_entropy,
        max_experts=config.router.max_experts,
    )


def _add_semantic_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-source-mb",
        type=float,
        default=2.0,
        help=(
            "Skip and log a case containing a source file larger than this "
            "value during semantic analysis (0 disables the limit)."
        ),
    )
    parser.add_argument(
        "--parse-timeout-seconds",
        type=int,
        default=30,
        help="Tree-sitter parse timeout per source file (0 disables the timeout).",
    )


def _source_limit_bytes(value: float) -> int | None:
    if value < 0:
        raise ValueError("--max-source-mb must be non-negative")
    if value == 0:
        return None
    return int(value * 1024 * 1024)


def _parse_timeout_ms(value: int) -> int | None:
    if value < 0:
        raise ValueError("--parse-timeout-seconds must be non-negative")
    if value == 0:
        return None
    return value * 1_000


def _read_sources(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in SOURCE_SUFFIXES:
            raise ValueError(f"Unsupported source suffix: {resolved.suffix}")
        return {resolved.name: resolved.read_text(encoding="utf-8", errors="replace")}
    if not resolved.is_dir():
        raise ValueError(f"Source path does not exist: {resolved}")
    sources: dict[str, str] = {}
    for file_path in sorted(resolved.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SOURCE_SUFFIXES:
            relative = file_path.relative_to(resolved).as_posix()
            sources[relative] = file_path.read_text(encoding="utf-8", errors="replace")
    if not sources:
        raise ValueError(f"No C/C++ source files found under {resolved}")
    return sources


if __name__ == "__main__":
    sys.exit(main())
