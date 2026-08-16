from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

from .arvo import prepare_arvo_cases, prepare_arvo_training_dataset
from .benchmarks import prepare_juliet_dataset, merge_case_split_directories
from .config import AppConfig
from .analysis import SemanticStaticAnalyzer
from .datasets import (
    load_cases_jsonl,
    iter_cases_jsonl,
    load_router_samples_jsonl,
    load_utility_samples_jsonl,
    write_cases_jsonl,
)
from .experts import ExpertRunner
from .experiment import ExperimentRunner
from .experiments import (
    Phase2EConfig,
    collect_expert_outcomes,
    evaluate_utility_end_to_end,
    prepare_phase2e_frozen_jsonl,
    prepare_phase2e_jsonl,
    run_phase2e_jsonl,
    write_utility_tradeoff_report,
)
from .factory import build_context_builder, build_openrouter_client, build_pipeline
from .models import (
    ACTIVE_UTILITY_EXPERTS,
    ExpertAssignment,
    ExpertFamily,
    ProjectCase,
    to_dict,
)
from .routing import (
    AdaptiveExpertRouter,
    AnchorRareRouter,
    BudgetedUtilityRouter,
    CandidateGate,
    RoutingPolicyConfig,
    UtilityPolicyConfig,
    assert_project_disjoint,
    split_gate_calibration_samples,
)


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

    anchor_parser = subparsers.add_parser(
        "train-anchor-router",
        help="Train common-family anchors plus high-recall rare Expert triggers",
    )
    anchor_parser.add_argument("--train", required=True)
    anchor_parser.add_argument("--dev", required=True)
    anchor_parser.add_argument("--env-file", default=".env")
    anchor_parser.add_argument("--target-rare-recall", type=float, default=0.95)
    anchor_parser.add_argument(
        "--output", default="artifacts/phase2e/router_anchor_rare_v2.pkl"
    )

    utility_parser = subparsers.add_parser(
        "train-utility-router",
        help="Train P(Expert x model succeeds | candidate) from measured outcomes",
    )
    utility_parser.add_argument("--train", required=True)
    utility_parser.add_argument("--dev", required=True)
    utility_parser.add_argument(
        "--gate-train",
        default=None,
        help=(
            "Optional project-disjoint outcome JSONL for Escalation Gate fitting. "
            "When omitted, --dev is split by project into gate/calibration halves."
        ),
    )
    utility_parser.add_argument("--env-file", default=".env")
    utility_parser.add_argument("--escalation-threshold", type=float, default=0.85)
    utility_parser.add_argument("--target-truth-recall", type=float, default=0.95)
    utility_parser.add_argument("--gate-fraction", type=float, default=0.50)
    utility_parser.add_argument("--cost-weight", type=float, default=1.0)
    utility_parser.add_argument("--false-positive-weight", type=float, default=0.50)
    utility_parser.add_argument("--unsupported-weight", type=float, default=0.25)
    utility_parser.add_argument(
        "--output", default="artifacts/phase2e/router_top2_full5_v4.pkl"
    )

    evaluate_utility_parser = subparsers.add_parser(
        "evaluate-utility-router",
        help="Evaluate a frozen Top-2/Full-5 Router once on an outcome test split",
    )
    evaluate_utility_parser.add_argument("--artifact", required=True)
    evaluate_utility_parser.add_argument("--test", required=True)
    evaluate_utility_parser.add_argument(
        "--anchor-artifact",
        default="",
        help="Optional AnchorRare baseline artifact evaluated on the same candidates",
    )
    evaluate_utility_parser.add_argument("--output", default="")

    end_to_end_parser = subparsers.add_parser(
        "evaluate-utility-end-to-end",
        help=(
            "Replay analyzer, Candidate Gate, Router, and measured Expert outcomes "
            "on a frozen case split without making LLM calls"
        ),
    )
    end_to_end_parser.add_argument("--cases", required=True)
    end_to_end_parser.add_argument("--outcomes", required=True)
    end_to_end_parser.add_argument("--artifact", required=True)
    end_to_end_parser.add_argument("--env-file", default=".env")
    end_to_end_parser.add_argument("--max-cases", type=int, default=0)
    end_to_end_parser.add_argument("--max-candidates-per-case", type=int, default=0)
    end_to_end_parser.add_argument(
        "--candidate-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CANDIDATE_GATE_ENABLED from .env.",
    )
    end_to_end_parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Override CANDIDATE_GATE_THRESHOLD from .env.",
    )
    end_to_end_parser.add_argument("--output", default="")
    _add_semantic_safety_options(end_to_end_parser)

    plot_parser = subparsers.add_parser(
        "plot-utility-results",
        help="Create the policy comparison CSV and recall/cost SVG figures",
    )
    plot_parser.add_argument("--input", required=True)
    plot_parser.add_argument("--output-dir", default="results/utility_figures")

    collect_parser = subparsers.add_parser(
        "collect-utility-outcomes",
        help="Run an Expert x OpenRouter model matrix and checkpoint Utility Router GT",
    )
    collect_parser.add_argument("--cases", required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--env-file", default=".env")
    collect_parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model IDs; defaults to sweep/per-Expert model settings",
    )
    collect_parser.add_argument(
        "--families",
        default=",".join(family.value for family in ACTIVE_UTILITY_EXPERTS),
        help="Comma-separated active Expert families (E1/E3/E4/E5/E6)",
    )
    collect_parser.add_argument("--max-cases", type=int, default=0)
    collect_parser.add_argument("--max-candidates-per-case", type=int, default=4)
    collect_parser.add_argument("--hard-negatives-per-case", type=int, default=1)
    collect_parser.add_argument("--no-resume", action="store_true")
    _add_semantic_safety_options(collect_parser)

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

    juliet_parser = subparsers.add_parser(
        "prepare-juliet",
        help="Convert an unpacked NIST Juliet C/C++ tree into E3/E4/E6 frozen splits",
    )
    juliet_parser.add_argument("--source", required=True)
    juliet_parser.add_argument("--output-dir", default="data/juliet")
    juliet_parser.add_argument(
        "--families",
        default=",".join(
            family.value
            for family in (
                ExpertFamily.INTEGER_SIZE_TYPE,
                ExpertFamily.TAINT_API_CONTRACT,
                ExpertFamily.CONCURRENCY_TOCTOU,
            )
        ),
    )
    juliet_parser.add_argument("--max-cases", type=int, default=0)
    juliet_parser.add_argument("--max-cases-per-family", type=int, default=100)
    juliet_parser.add_argument("--max-cases-per-cwe", type=int, default=50)
    juliet_parser.add_argument("--max-cases-per-template", type=int, default=10)
    juliet_parser.add_argument("--seed", type=int, default=2026)

    merge_parser = subparsers.add_parser(
        "merge-case-splits",
        help="Merge ARVO/Juliet frozen split directories and reject project leakage",
    )
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output-dir", default="data/benchmark")

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
    _add_analysis_checkpoint_options(phase2e_parser)

    phase2e_prepare_parser = subparsers.add_parser(
        "phase2e-prepare",
        help="Stream ARVO into compact Phase 2E Router JSONL files without training",
    )
    phase2e_prepare_parser.add_argument(
        "--cases", default="data/arvo/cases_all.jsonl"
    )
    phase2e_prepare_parser.add_argument(
        "--frozen-splits-dir",
        default="",
        help="Use existing cases_train/dev/test JSONL without reshuffling them.",
    )
    phase2e_prepare_parser.add_argument("--data-dir", default="data/phase2e")
    phase2e_prepare_parser.add_argument("--seed", type=int, default=2026)
    phase2e_prepare_parser.add_argument(
        "--backend",
        choices=("all", "legacy", "semantic"),
        default="semantic",
        help="Prepare semantic-cwe-v2 by default, or explicitly request legacy/all.",
    )
    _add_semantic_safety_options(phase2e_prepare_parser)
    _add_analysis_checkpoint_options(phase2e_prepare_parser)
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
    if args.command == "train-anchor-router":
        config = AppConfig.from_env(args.env_file)
        router = AnchorRareRouter.fit(
            load_router_samples_jsonl(args.train), seed=config.runtime.seed
        )
        calibration = router.calibrate_threshold(
            load_router_samples_jsonl(args.dev),
            target_rare_recall=args.target_rare_recall,
        )
        metrics = router.evaluate(load_router_samples_jsonl(args.dev))
        router.save(args.output)
        print(
            json.dumps(
                {"calibration": to_dict(calibration), "metrics": to_dict(metrics)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "train-utility-router":
        config = AppConfig.from_env(args.env_file)
        policy = UtilityPolicyConfig(
            escalation_threshold=args.escalation_threshold,
            cost_weight=args.cost_weight,
            false_positive_weight=args.false_positive_weight,
            unsupported_weight=args.unsupported_weight,
        )
        train_rows = load_utility_samples_jsonl(args.train)
        dev_rows = load_utility_samples_jsonl(args.dev)
        assert_project_disjoint(
            train_rows,
            dev_rows,
            first_name="train",
            second_name="dev",
        )
        router = BudgetedUtilityRouter.fit(
            train_rows,
            policy=policy,
            seed=config.runtime.seed,
        )
        if args.gate_train:
            gate_rows = load_utility_samples_jsonl(args.gate_train)
            calibration_rows = dev_rows
            assert_project_disjoint(
                train_rows,
                gate_rows,
                first_name="train",
                second_name="gate-train",
            )
            assert_project_disjoint(
                gate_rows,
                calibration_rows,
                first_name="gate-train",
                second_name="dev",
            )
        else:
            gate_rows, calibration_rows = split_gate_calibration_samples(
                dev_rows,
                seed=config.runtime.seed,
                gate_fraction=args.gate_fraction,
            )
        gate_sample_count = router.fit_escalation_gate(
            gate_rows, seed=config.runtime.seed
        )
        calibration = router.calibrate_threshold(
            calibration_rows,
            target_truth_recall=args.target_truth_recall,
        )
        baseline_calibration = router.calibrate_baselines(calibration_rows)
        metrics = router.evaluate(calibration_rows)
        router.save(args.output)
        print(
            json.dumps(
                {
                    "artifact": args.output,
                    "gate_training_candidates": gate_sample_count,
                    "calibration": to_dict(calibration),
                    "baseline_calibration": to_dict(baseline_calibration),
                    "calibration_metrics": to_dict(metrics),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "evaluate-utility-router":
        router = BudgetedUtilityRouter.load(args.artifact)
        test_rows = load_utility_samples_jsonl(args.test)
        router.assert_test_projects_unseen(test_rows)
        anchor_router = (
            AnchorRareRouter.load(args.anchor_artifact)
            if args.anchor_artifact
            else None
        )
        report = {
            "artifact": args.artifact,
            "test": args.test,
            "policies": to_dict(
                router.evaluate_baselines(test_rows, anchor_router=anchor_router)
            ),
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        print(payload)
        return 0
    if args.command == "evaluate-utility-end-to-end":
        config = AppConfig.from_env(args.env_file)
        router = BudgetedUtilityRouter.load(args.artifact)
        outcome_rows = load_utility_samples_jsonl(args.outcomes)
        router.assert_test_projects_unseen(outcome_rows)
        cases = iter_cases_jsonl(args.cases)
        if args.max_cases:
            cases = islice(cases, args.max_cases)
        threshold = (
            args.gate_threshold
            if args.gate_threshold is not None
            else config.candidate_gate.threshold
        )
        gate_enabled = (
            args.candidate_gate_enabled
            if args.candidate_gate_enabled is not None
            else config.candidate_gate.enabled
        )
        metrics = evaluate_utility_end_to_end(
            cases,
            outcome_rows,
            analyzer=SemanticStaticAnalyzer(
                max_source_bytes=_source_limit_bytes(args.max_source_mb),
                parse_timeout_ms=_parse_timeout_ms(args.parse_timeout_seconds),
            ),
            candidate_gate=CandidateGate(
                enabled=gate_enabled,
                threshold=threshold,
            ),
            router=router,
            max_candidates_per_case=(
                args.max_candidates_per_case or None
            ),
            progress=print,
        )
        report = {
            "artifact": args.artifact,
            "cases": args.cases,
            "outcomes": args.outcomes,
            "candidate_gate_enabled": gate_enabled,
            "candidate_gate_threshold": threshold,
            "metrics": to_dict(metrics),
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        print(payload)
        return 0
    if args.command == "plot-utility-results":
        report = json.loads(Path(args.input).read_text(encoding="utf-8"))
        policies = report.get("policies", report)
        if not isinstance(policies, dict):
            raise ValueError("Utility result JSON must contain a policies object")
        outputs = write_utility_tradeoff_report(policies, args.output_dir)
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return 0
    if args.command == "collect-utility-outcomes":
        config = AppConfig.from_env(args.env_file)
        if not config.runtime.allow_paid_experiments:
            raise RuntimeError(
                "Outcome collection calls OpenRouter. Set RUN_PAID_EXPERIMENTS=1 "
                "after checking the selected model matrix and budget."
            )
        families = tuple(
            ExpertFamily(item.strip()) for item in args.families.split(",") if item.strip()
        )
        explicit_models = tuple(
            item.strip() for item in args.models.split(",") if item.strip()
        )
        model_pool = explicit_models or config.model.sweep_models or tuple(
            dict.fromkeys(
                config.model.expert_models[family]
                for family in ACTIVE_UTILITY_EXPERTS
            )
        )
        assignments = [
            ExpertAssignment(family, model_id)
            for family in families
            for model_id in model_pool
        ]
        cases = iter_cases_jsonl(args.cases)
        if args.max_cases:
            cases = islice(cases, args.max_cases)
        client = build_openrouter_client(config)
        summary = collect_expert_outcomes(
            cases,
            analyzer=SemanticStaticAnalyzer(
                max_source_bytes=_source_limit_bytes(args.max_source_mb),
                parse_timeout_ms=_parse_timeout_ms(args.parse_timeout_seconds),
            ),
            expert_runner=ExpertRunner(
                client=client,
                model=config.model.expert_model,
                context_builder=build_context_builder(config),
                models_by_family=config.model.expert_models,
            ),
            assignments=assignments,
            output_path=args.output,
            max_candidates_per_case=args.max_candidates_per_case,
            hard_negatives_per_case=args.hard_negatives_per_case,
            resume=not args.no_resume,
            progress=print,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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
    if args.command == "prepare-juliet":
        families = tuple(
            ExpertFamily(value.strip())
            for value in args.families.split(",")
            if value.strip()
        )
        manifest = prepare_juliet_dataset(
            args.source,
            args.output_dir,
            families=families,
            seed=args.seed,
            max_cases=args.max_cases,
            max_cases_per_family=args.max_cases_per_family,
            max_cases_per_cwe=args.max_cases_per_cwe,
            max_cases_per_template=args.max_cases_per_template,
            progress=print,
        )
        print(f"Juliet splits saved to {args.output_dir}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "merge-case-splits":
        manifest = merge_case_split_directories(args.inputs, args.output_dir)
        print(f"Merged frozen splits saved to {args.output_dir}")
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
                analysis_checkpoint_every_cases=args.checkpoint_every,
                resume_analysis=not args.no_resume,
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
        preparation = (
            prepare_phase2e_frozen_jsonl
            if args.frozen_splits_dir
            else prepare_phase2e_jsonl
        )
        source = args.frozen_splits_dir or args.cases
        summary = preparation(
            source,
            config=Phase2EConfig(
                seed=args.seed,
                semantic_max_source_bytes=_source_limit_bytes(args.max_source_mb),
                semantic_parse_timeout_ms=_parse_timeout_ms(args.parse_timeout_seconds),
                analysis_checkpoint_every_cases=args.checkpoint_every,
                resume_analysis=not args.no_resume,
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
    errors: list[str] = []
    for router_class in (
        BudgetedUtilityRouter,
        AnchorRareRouter,
        AdaptiveExpertRouter,
    ):
        try:
            return router_class.load(artifact)
        except TypeError as error:
            errors.append(str(error))
    raise TypeError("Unsupported Router artifact: " + " | ".join(errors))


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


def _add_analysis_checkpoint_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Persist semantic-analysis progress after this many completed cases.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard matching Phase 2E analysis checkpoints and start the analysis again.",
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
