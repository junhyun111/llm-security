from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..analysis.semantic_static import SemanticStaticAnalyzer
from ..datasets import RouterSample
from ..models import ProjectCase, to_dict
from ..routing import AdaptiveExpertRouter, RoutingPolicyConfig
from ..static_analysis import LightweightStaticAnalyzer
from .analyzer_eval import AnalyzerEvaluation, evaluate_analyzer
from .dataset import (
    router_samples_from_cached_candidates,
    single_label_samples,
    write_backend_router_datasets,
)
from .end_to_end import evaluate_end_to_end
from .gate_eval import calibrate_gate, evaluate_gate
from .io import write_confusion_csv, write_json
from .router_eval import (
    all_six_baseline,
    confusion_matrix,
    evaluate_full_hybrid_router,
    evaluate_supported_router,
    feature_importance,
)
from .split import (
    FrozenProjectFiles,
    FrozenProjectSplit,
    freeze_project_split,
    freeze_project_split_jsonl,
    load_frozen_project_files,
)
from .streaming import analyze_split_jsonl


@dataclass(slots=True, frozen=True)
class Phase2EConfig:
    seed: int = 2026
    target_gate_retention: float = 0.98
    target_routing_coverage: float = 0.95
    semantic_max_source_bytes: int | None = 2 * 1024 * 1024
    semantic_parse_timeout_ms: int | None = 30_000
    analysis_checkpoint_every_cases: int = 100
    resume_analysis: bool = True
    data_directory: str | Path = "data/phase2e"
    artifact_directory: str | Path = "artifacts/phase2e"
    output_directory: str | Path = "results/phase2e"


@dataclass(slots=True)
class _PreparedPhase2E:
    frozen: FrozenProjectSplit
    analyzer_results: dict[str, dict[str, AnalyzerEvaluation]]
    router_samples: dict[str, dict[str, list[RouterSample]]]


def run_phase2e_jsonl(
    cases_path: str | Path,
    *,
    config: Phase2EConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Memory-bounded Phase 2E entry point used by the CLI."""
    selected = config or Phase2EConfig()
    prepared = _prepare_phase2e_jsonl(
        cases_path,
        config=selected,
        progress=progress,
        retain_for_evaluation=True,
        stage_total=7,
        backends=("legacy", "semantic"),
    )
    return run_phase2e(
        [],
        config=selected,
        progress=progress,
        _prepared=prepared,
    )


def prepare_phase2e_jsonl(
    cases_path: str | Path,
    *,
    config: Phase2EConfig | None = None,
    progress: Callable[[str], None] | None = None,
    backends: tuple[str, ...] = ("legacy", "semantic"),
) -> dict[str, object]:
    """Create compact Router datasets and stop before model training."""
    selected = config or Phase2EConfig()
    prepared = _prepare_phase2e_jsonl(
        cases_path,
        config=selected,
        progress=progress,
        retain_for_evaluation=False,
        stage_total=2,
        backends=backends,
    )
    return _finish_preparation(prepared, selected=selected, progress=progress)


def prepare_phase2e_frozen_jsonl(
    split_directory: str | Path,
    *,
    config: Phase2EConfig | None = None,
    progress: Callable[[str], None] | None = None,
    backends: tuple[str, ...] = ("semantic",),
) -> dict[str, object]:
    """Prepare Router features while preserving an existing frozen split."""
    selected = config or Phase2EConfig()
    _progress(progress, "[1/2] Loading source-frozen train/dev/test splits")
    frozen_files = load_frozen_project_files(split_directory, seed=selected.seed)
    prepared = _prepare_frozen_phase2e_jsonl(
        frozen_files,
        config=selected,
        progress=progress,
        retain_for_evaluation=False,
        stage_total=2,
        backends=backends,
    )
    return _finish_preparation(prepared, selected=selected, progress=progress)


def _finish_preparation(
    prepared: _PreparedPhase2E,
    *,
    selected: Phase2EConfig,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    summary = {
        "seed": selected.seed,
        "streaming": True,
        "checkpointing": {
            "every_cases": selected.analysis_checkpoint_every_cases,
            "resume": selected.resume_analysis,
            "directory": str(Path(selected.data_directory) / "analysis_checkpoints"),
        },
        "training_performed": False,
        "split_manifest": prepared.frozen.manifest,
        "analyzer_metrics": {
            backend: {
                split: to_dict(evaluation.metrics)
                for split, evaluation in results.items()
            }
            for backend, results in prepared.analyzer_results.items()
        },
        "router_sample_counts": {
            backend: {
                split: len(samples)
                for split, samples in split_samples.items()
            }
            for backend, split_samples in prepared.router_samples.items()
        },
    }
    write_json(summary, Path(selected.data_directory) / "preparation_summary.json")
    _progress(
        progress,
        f"Prepared Router JSONL files in {Path(selected.data_directory)}",
    )
    return summary


def _prepare_phase2e_jsonl(
    cases_path: str | Path,
    *,
    config: Phase2EConfig,
    progress: Callable[[str], None] | None,
    retain_for_evaluation: bool,
    stage_total: int,
    backends: tuple[str, ...],
) -> _PreparedPhase2E:
    selected = config
    data_directory = Path(selected.data_directory)
    _progress(
        progress,
        f"[1/{stage_total}] Freezing the project-disjoint split (streaming)",
    )
    frozen_files = freeze_project_split_jsonl(
        cases_path,
        data_directory,
        seed=selected.seed,
        progress=progress,
    )
    return _prepare_frozen_phase2e_jsonl(
        frozen_files,
        config=selected,
        progress=progress,
        retain_for_evaluation=retain_for_evaluation,
        stage_total=stage_total,
        backends=backends,
    )


def _prepare_frozen_phase2e_jsonl(
    frozen_files: FrozenProjectFiles,
    *,
    config: Phase2EConfig,
    progress: Callable[[str], None] | None,
    retain_for_evaluation: bool,
    stage_total: int,
    backends: tuple[str, ...],
) -> _PreparedPhase2E:
    selected = config
    data_directory = Path(selected.data_directory)
    split_items = frozen_files.manifest["splits"]  # type: ignore[index]
    available_analyzers = {
        "legacy": LightweightStaticAnalyzer(max_candidates=None),
        "semantic": SemanticStaticAnalyzer(
            max_source_bytes=selected.semantic_max_source_bytes,
            parse_timeout_ms=selected.semantic_parse_timeout_ms,
        ),
    }
    invalid = set(backends) - set(available_analyzers)
    if not backends or invalid:
        raise ValueError(
            "backends must contain legacy and/or semantic; invalid: "
            + ", ".join(sorted(invalid))
        )
    analyzers = {backend: available_analyzers[backend] for backend in backends}
    analyzer_results: dict[str, dict[str, AnalyzerEvaluation]] = {
        backend: {} for backend in analyzers
    }
    router_samples: dict[str, dict[str, list[RouterSample]]] = {
        backend: {} for backend in analyzers
    }
    retained_cases: dict[str, list[ProjectCase]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    for backend, analyzer in analyzers.items():
        _progress(
            progress,
            f"[2/{stage_total}] Running {backend} analyzer one case at a time",
        )
        for split_name in ("train", "dev", "test"):
            retain = retain_for_evaluation and (
                split_name == "test"
                or (backend == "semantic" and split_name == "dev")
            )
            output = analyze_split_jsonl(
                analyzer,
                frozen_files.files[split_name],
                retain_compact_analysis=retain,
                expected_case_count=int(split_items[split_name]["case_count"]),  # type: ignore[index]
                progress=(
                    lambda done, total, backend=backend, split_name=split_name: _progress(
                        progress,
                        f"      {backend}/{split_name}: {done}/{total} cases",
                    )
                ),
                case_start_progress=(
                    lambda done, total, case, backend=backend, split_name=split_name: _report_large_case(
                        progress,
                        backend=backend,
                        split_name=split_name,
                        done=done,
                        total=total,
                        case=case,
                    )
                ),
                failure_log_path=(
                    data_directory
                    / "analysis_failures"
                    / f"{backend}_{split_name}.jsonl"
                ),
                checkpoint_path=(
                    data_directory
                    / "analysis_checkpoints"
                    / (
                        f"{backend}_{split_name}_"
                        f"{'evaluation' if retain else 'prepare'}.json"
                    )
                ),
                checkpoint_every=selected.analysis_checkpoint_every_cases,
                resume=selected.resume_analysis,
                resume_progress=(
                    lambda done, total, backend=backend, split_name=split_name: _progress(
                        progress,
                        f"      {backend}/{split_name}: resuming from {done}/{total} cases",
                    )
                ),
                checkpoint_progress=(
                    lambda done, total, backend=backend, split_name=split_name: _progress(
                        progress,
                        f"      {backend}/{split_name}: checkpoint saved at {done}/{total} cases",
                    )
                ),
            )
            analyzer_results[backend][split_name] = AnalyzerEvaluation(
                metrics=output.metrics,
                candidates_by_case=output.retained_candidates_by_case,
            )
            router_samples[backend][split_name] = output.router_samples
            if (
                retain_for_evaluation
                and backend == "semantic"
                and split_name in {"dev", "test"}
            ):
                retained_cases[split_name] = output.retained_cases
        write_backend_router_datasets(
            router_samples[backend], data_directory / backend
        )
    frozen = FrozenProjectSplit(
        seed=selected.seed,
        cases=retained_cases,
        manifest=frozen_files.manifest,
    )
    return _PreparedPhase2E(
        frozen=frozen,
        analyzer_results=analyzer_results,
        router_samples=router_samples,
    )


def run_phase2e(
    cases: list[ProjectCase],
    *,
    config: Phase2EConfig | None = None,
    progress: Callable[[str], None] | None = None,
    _prepared: _PreparedPhase2E | None = None,
) -> dict[str, object]:
    """Run the complete Phase 2E experiment without any LLM API calls."""
    selected = config or Phase2EConfig()
    if not cases and _prepared is None:
        raise ValueError("Phase 2E requires at least one ProjectCase")
    data_directory = Path(selected.data_directory)
    artifact_directory = Path(selected.artifact_directory)
    output_directory = Path(selected.output_directory)

    if _prepared is not None:
        frozen = _prepared.frozen
        analyzer_results = _prepared.analyzer_results
        router_samples = _prepared.router_samples
    else:
        _progress(progress, "[1/7] Freezing the project-disjoint split")
        frozen = freeze_project_split(cases, data_directory, seed=selected.seed)

        analyzers = {
            "legacy": LightweightStaticAnalyzer(max_candidates=None),
            "semantic": SemanticStaticAnalyzer(
                max_source_bytes=selected.semantic_max_source_bytes,
                parse_timeout_ms=selected.semantic_parse_timeout_ms,
            ),
        }
        analyzer_results = {backend: {} for backend in analyzers}
        router_samples = {backend: {} for backend in analyzers}
        for backend, analyzer in analyzers.items():
            _progress(progress, f"[2/7] Running {backend} analyzer on the frozen splits")
            for split_name in ("train", "dev", "test"):
                evaluation = evaluate_analyzer(
                    analyzer,
                    frozen.cases[split_name],
                    progress=(
                        lambda done, total, backend=backend, split_name=split_name: _progress(
                            progress,
                            f"      {backend}/{split_name}: {done}/{total} cases",
                        )
                    ),
                )
                analyzer_results[backend][split_name] = evaluation
                router_samples[backend][split_name] = single_label_samples(
                    router_samples_from_cached_candidates(
                        frozen.cases[split_name], evaluation.candidates_by_case
                    )
                )
                if split_name == "train" or (
                    backend == "legacy" and split_name == "dev"
                ):
                    evaluation.candidates_by_case.clear()
            write_backend_router_datasets(
                router_samples[backend], data_directory / backend
            )

    _progress(progress, "[3/7] Calibrating the semantic Candidate Gate on dev only")
    semantic_dev = analyzer_results["semantic"]["dev"]
    semantic_test = analyzer_results["semantic"]["test"]
    gate_dev = calibrate_gate(
        frozen.cases["dev"],
        semantic_dev.candidates_by_case,
        target_retention=selected.target_gate_retention,
    )
    gate_test = evaluate_gate(
        frozen.cases["test"],
        semantic_test.candidates_by_case,
        threshold=gate_dev.threshold,
        target_retention=selected.target_gate_retention,
    )
    semantic_dev.candidates_by_case.clear()

    _progress(progress, "[4/7] Training matching Softmax Routers")
    base_policy = RoutingPolicyConfig()
    legacy_router = _fit_router(
        router_samples["legacy"]["train"], selected.seed, base_policy, "legacy"
    )
    semantic_router = _fit_router(
        router_samples["semantic"]["train"], selected.seed, base_policy, "semantic"
    )
    semantic_calibration_samples = _supported_samples(
        semantic_router, router_samples["semantic"]["dev"]
    )
    if not semantic_calibration_samples:
        raise ValueError(
            "Semantic dev split has no learned-supported Router samples; "
            "cannot calibrate Adaptive Top-K"
        )
    semantic_router.triggers.enabled = False
    policy_calibration = semantic_router.calibrate_policy(
        semantic_calibration_samples,
        target_coverage=selected.target_routing_coverage,
    )
    legacy_router.save(artifact_directory / "router_legacy_v1.pkl")
    semantic_router.save(artifact_directory / "router_semantic_cwe_v2.pkl")

    _progress(progress, "[5/7] Evaluating learned and hybrid routing")
    legacy_test_samples = router_samples["legacy"]["test"]
    semantic_test_samples = router_samples["semantic"]["test"]
    legacy_supported = evaluate_supported_router(legacy_router, legacy_test_samples)
    semantic_supported = evaluate_supported_router(semantic_router, semantic_test_samples)

    sample_ablations = {
        "E0_legacy_adaptive": evaluate_full_hybrid_router(
            legacy_router,
            legacy_test_samples,
            selection_mode="adaptive",
            fallback_enabled=False,
        ),
        "E1_semantic_top1": evaluate_full_hybrid_router(
            semantic_router,
            semantic_test_samples,
            selection_mode="top1",
            fallback_enabled=False,
        ),
        "E2_semantic_top2": evaluate_full_hybrid_router(
            semantic_router,
            semantic_test_samples,
            selection_mode="top2",
            fallback_enabled=False,
        ),
        "E3_semantic_adaptive": evaluate_full_hybrid_router(
            semantic_router,
            semantic_test_samples,
            selection_mode="adaptive",
            fallback_enabled=False,
        ),
        "semantic_score_fusion": evaluate_full_hybrid_router(
            semantic_router,
            semantic_test_samples,
            selection_mode="adaptive",
            fallback_enabled=True,
            fallback_mode="score_fusion",
        ),
        "E5_semantic_forced_fallback": evaluate_full_hybrid_router(
            semantic_router,
            semantic_test_samples,
            selection_mode="adaptive",
            fallback_enabled=True,
            fallback_mode="forced_trigger",
        ),
        "all_6_experts": all_six_baseline(len(semantic_test_samples)),
    }

    _progress(progress, "[6/7] Computing project-level end-to-end recall")
    test_cases = frozen.cases["test"]
    e2e_ablations = {
        "E0_legacy_adaptive": evaluate_end_to_end(
            test_cases,
            analyzer_results["legacy"]["test"].candidates_by_case,
            legacy_router,
            selection_mode="adaptive",
        ),
        "E1_semantic_top1": evaluate_end_to_end(
            test_cases,
            semantic_test.candidates_by_case,
            semantic_router,
            selection_mode="top1",
        ),
        "E2_semantic_top2": evaluate_end_to_end(
            test_cases,
            semantic_test.candidates_by_case,
            semantic_router,
            selection_mode="top2",
        ),
        "E3_semantic_adaptive": evaluate_end_to_end(
            test_cases,
            semantic_test.candidates_by_case,
            semantic_router,
            selection_mode="adaptive",
        ),
        "E4_semantic_calibrated_gate": evaluate_end_to_end(
            test_cases,
            semantic_test.candidates_by_case,
            semantic_router,
            gate_threshold=gate_dev.threshold,
            selection_mode="adaptive",
        ),
        "semantic_gate_score_fusion": evaluate_end_to_end(
            test_cases,
            semantic_test.candidates_by_case,
            semantic_router,
            gate_threshold=gate_dev.threshold,
            selection_mode="adaptive",
            fallback_enabled=True,
            fallback_mode="score_fusion",
        ),
        "E5_semantic_gate_forced_fallback": evaluate_end_to_end(
            test_cases,
            semantic_test.candidates_by_case,
            semantic_router,
            gate_threshold=gate_dev.threshold,
            selection_mode="adaptive",
            fallback_enabled=True,
            fallback_mode="forced_trigger",
        ),
    }

    _progress(progress, "[7/7] Writing deterministic Phase 2E artifacts")
    confusion = confusion_matrix(semantic_router, semantic_test_samples)
    manifest = _experiment_manifest(
        selected,
        frozen.manifest,
        gate_dev.threshold,
        policy_calibration.high_confidence,
        policy_calibration.min_margin,
        router_samples,
    )
    common = {
        "git_commit": manifest["git_commit"],
        "seed": selected.seed,
        "feature_schemas": manifest["feature_schemas"],
        "project_counts": manifest["project_counts"],
        "case_counts": manifest["case_counts"],
        "sample_counts": manifest["sample_counts"],
    }
    analyzer_payload = {
        "metadata": common,
        "legacy": {
            split: to_dict(evaluation.metrics)
            for split, evaluation in analyzer_results["legacy"].items()
        },
        "semantic": {
            split: to_dict(evaluation.metrics)
            for split, evaluation in analyzer_results["semantic"].items()
        },
    }
    gate_payload = {
        "metadata": common,
        "calibration_split": "dev",
        "dev_calibration": gate_dev,
        "test_evaluation_with_frozen_threshold": gate_test,
    }
    legacy_payload = {
        "metadata": common,
        "feature_schema": legacy_router.feature_schema_version,
        "supported_only_test": legacy_supported,
        "available_families": [item.value for item in legacy_router.available_families],
    }
    semantic_payload = {
        "metadata": common,
        "feature_schema": semantic_router.feature_schema_version,
        "supported_only_test": semantic_supported,
        "available_families": [item.value for item in semantic_router.available_families],
        "adaptive_policy_calibration": policy_calibration,
    }
    ablation_payload = {
        "metadata": common,
        "sample_level_router": sample_ablations,
        "project_level_end_to_end": {
            key: value.metrics for key, value in e2e_ablations.items()
        },
    }
    per_family_payload = {
        "metadata": common,
        "variant": "E5_semantic_gate_forced_fallback",
        "families": e2e_ablations[
            "E5_semantic_gate_forced_fallback"
        ].per_family,
    }
    importance_payload = {
        "metadata": common,
        "legacy": feature_importance(legacy_router),
        "semantic": feature_importance(semantic_router),
    }

    write_json(manifest, output_directory / "experiment_manifest.json")
    write_json(analyzer_payload, output_directory / "analyzer_comparison.json")
    write_json(gate_payload, output_directory / "gate_calibration.json")
    write_json(legacy_payload, output_directory / "router_legacy.json")
    write_json(semantic_payload, output_directory / "router_semantic.json")
    write_json(ablation_payload, output_directory / "routing_ablation.json")
    write_json(per_family_payload, output_directory / "per_family.json")
    write_json(
        {"metadata": common, **confusion},
        output_directory / "confusion_matrix.json",
    )
    write_confusion_csv(
        confusion["labels"],  # type: ignore[arg-type]
        confusion["matrix"],  # type: ignore[arg-type]
        output_directory / "confusion_matrix.csv",
    )
    write_json(importance_payload, output_directory / "feature_importance.json")
    return {
        "manifest": manifest,
        "analyzer_comparison": analyzer_payload,
        "gate_calibration": gate_payload,
        "router_legacy": legacy_payload,
        "router_semantic": semantic_payload,
        "routing_ablation": ablation_payload,
        "per_family": per_family_payload,
    }


def _fit_router(
    samples: list[RouterSample],
    seed: int,
    policy: RoutingPolicyConfig,
    backend: str,
) -> AdaptiveExpertRouter:
    if not samples:
        raise ValueError(f"{backend} training split has no routable samples")
    families = {sample.labels[0] for sample in samples}
    if len(families) < 2:
        raise ValueError(
            f"{backend} training split contains only {len(families)} Expert family; "
            "Softmax Router requires at least two"
        )
    return AdaptiveExpertRouter.fit(
        samples,
        policy_config=policy,
        seed=seed,
        use_rule_fallback=True,
    )


def _supported_samples(
    router: AdaptiveExpertRouter, samples: list[RouterSample]
) -> list[RouterSample]:
    supported = set(router.available_families)
    return [sample for sample in samples if sample.labels[0] in supported]


def _experiment_manifest(
    config: Phase2EConfig,
    split_manifest: dict[str, object],
    gate_threshold: float,
    high_confidence: float,
    min_margin: float,
    samples: dict[str, dict[str, list[RouterSample]]],
) -> dict[str, object]:
    split_items = split_manifest["splits"]  # type: ignore[index]
    return {
        "phase": "2E",
        "offline_only": True,
        "llm_api_calls": 0,
        "git_commit": _git_commit(),
        "seed": config.seed,
        "feature_schemas": ["legacy-v1", "semantic-cwe-v3"],
        "router": "multiclass_logistic_regression",
        "classifier_hyperparameters_shared": True,
        "gate_calibration_split": "dev",
        "gate_target_retention": config.target_gate_retention,
        "gate_threshold": gate_threshold,
        "adaptive_calibration_split": "dev",
        "adaptive_target_coverage": config.target_routing_coverage,
        "high_confidence": high_confidence,
        "min_margin": min_margin,
        "fallback_policies_compared": ["score_fusion", "forced_trigger"],
        "final_fallback_policy": "forced_trigger",
        "project_counts": {
            split: split_items[split]["project_count"]  # type: ignore[index]
            for split in ("train", "dev", "test")
        },
        "case_counts": {
            split: split_items[split]["case_count"]  # type: ignore[index]
            for split in ("train", "dev", "test")
        },
        "sample_counts": {
            backend: {
                split: len(samples[backend][split])
                for split in ("train", "dev", "test")
            }
            for backend in ("legacy", "semantic")
        },
        "family_distributions": {
            split: split_items[split]["family_distribution"]  # type: ignore[index]
            for split in ("train", "dev", "test")
        },
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _report_large_case(
    progress: Callable[[str], None] | None,
    *,
    backend: str,
    split_name: str,
    done: int,
    total: int,
    case: ProjectCase,
) -> None:
    source_bytes = sum(
        len(source.encode("utf-8")) for source in case.source_files.values()
    )
    if source_bytes < 1 * 1024 * 1024:
        return
    _progress(
        progress,
        f"      {backend}/{split_name}: starting {done}/{total} "
        f"{case.case_id} ({case.project_id}, {source_bytes / (1024 * 1024):.2f} MiB)",
    )
