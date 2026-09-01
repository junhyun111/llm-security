from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Sequence

from llm_security.config import AppConfig
from llm_security.datasets import _candidate_from_raw
from llm_security.factory import build_batched_web_pipeline, build_openrouter_client
from llm_security.models import ProjectCase, ValidationVerdict, to_dict
from llm_security.patching import LLMBatchPatchAgent
from llm_security.verification import TemporaryPatchVerifier
from llm_security.web.service import (
    PatchBatchRecord,
    WebJobService,
    WebSettings,
    _finding_bundle,
    _finding_from_raw,
    _now,
    _validation_from_raw,
    _write_json_atomic,
    load_project_sources,
    load_router_artifact,
    validate_patch_scope_for_files,
)


@dataclass(frozen=True, slots=True)
class RuntimeJobOptions:
    sensitivity: float = 0.5
    model: str | None = None
    api_key: str | None = None

    def safe_metadata(self) -> dict[str, object]:
        return {
            "sensitivity": self.sensitivity,
            "model_override": self.model,
            "api_key_source": "request" if self.api_key else "runtime_env",
        }


class RequestAwareWebJobService(WebJobService):
    """WebJobService with per-job OpenRouter and sensitivity configuration.

    Secrets are intentionally kept only in process memory. They are never
    serialized into job.json, analysis.json, or the SQLite web database.
    """

    def __init__(self, settings: WebSettings) -> None:
        self._request_options_lock = threading.RLock()
        self._request_options: dict[str, RuntimeJobOptions] = {}
        super().__init__(settings)

    def create_job(
        self,
        project_name: str,
        uploads: Sequence[tuple[str, BinaryIO]],
        *,
        options: RuntimeJobOptions | None = None,
    ):
        selected = options or RuntimeJobOptions()

        # WebJobService submits the worker before returning the JobRecord.
        # Holding this lock creates a barrier so the worker cannot start until
        # the options map contains the newly-created job id.
        with self._request_options_lock:
            record = super().create_job(project_name, uploads)
            self._request_options[record.job_id] = selected
            return record

    def _run_analysis(self, job_id: str) -> None:
        with self._request_options_lock:
            self._request_options.setdefault(job_id, RuntimeJobOptions())
        super()._run_analysis(job_id)

    def request_options(self, job_id: str) -> RuntimeJobOptions:
        with self._request_options_lock:
            return self._request_options.get(job_id, RuntimeJobOptions())

    def _config_for_job(self, job_id: str) -> AppConfig:
        config = AppConfig.from_env(self.settings.env_file)
        options = self.request_options(job_id)

        if options.api_key:
            config.model.api_key = options.api_key
            # Pressing "analyze" with an explicit user key is the per-request
            # authorization to perform the paid OpenRouter request.
            config.runtime.allow_paid_experiments = True

        if options.model:
            selected_model = options.model
            config.model.expert_model = selected_model
            config.model.validator_model = selected_model
            config.model.patch_model = selected_model
            config.model.strong_model = selected_model
            config.model.expert_models = {
                family: selected_model
                for family in config.model.expert_models
            }

        sensitivity = min(1.0, max(0.0, options.sensitivity))

        # 0.5 exactly reproduces the current .env defaults:
        # Candidate Gate 0.40 / minimum validation confidence 0.60.
        # Higher sensitivity lowers both thresholds.
        candidate_threshold = round(0.70 - (0.60 * sensitivity), 4)
        validation_threshold = round(0.85 - (0.50 * sensitivity), 4)

        config.candidate_gate.threshold = candidate_threshold
        config.validation.minimum_confidence = validation_threshold
        config.validation.minimum_confidence_by_expert = {
            expert: validation_threshold
            for expert in config.validation.minimum_confidence_by_expert
        }

        config.validate()
        return config

    def _analyze_project(
        self,
        input_directory: Path,
        job,
        progress: Callable[[int, str], None],
    ) -> dict:
        config = self._config_for_job(job.job_id)
        if not config.runtime.allow_paid_experiments:
            raise RuntimeError(
                "OpenRouter API Key를 입력하거나 model_runtime/.env에서 "
                "RUN_PAID_EXPERIMENTS=1을 설정해주세요."
            )
        if not config.model.api_key:
            raise RuntimeError(
                "OpenRouter API Key가 없습니다. 분석 화면에서 사용자 API Key를 "
                "입력하거나 model_runtime/.env에 OPENROUTER_API_KEY를 설정해주세요."
            )

        config.analysis.backend = "semantic"
        config.candidate_gate.enabled = self.settings.candidate_gate_enabled
        config.model.max_output_tokens = self.settings.detection_max_output_tokens
        config.runtime.max_retries = 0

        progress(20, "Loading C/C++ source files")
        source_files = load_project_sources(
            input_directory,
            max_file_bytes=self.settings.max_source_file_bytes,
            max_total_bytes=self.settings.max_source_total_bytes,
        )

        progress(30, "Loading Router and static analyzer")
        router = load_router_artifact(self.settings.router_artifact)

        case = ProjectCase(
            case_id=f"web-{job.job_id}",
            project_id=job.project_name,
            source_files=source_files,
            split="unlabeled",
            metadata={"source": "web-upload"},
        )

        progress(40, "Running semantic analysis and one batched multi-Expert call")
        result = build_batched_web_pipeline(
            config,
            router,
            max_batch_characters=self.settings.detection_max_prompt_characters,
            max_batch_tasks=self.settings.detection_max_expert_tasks,
        ).run(case)

        progress(95, "Preparing evidence-grounded report")

        candidates = {item.candidate_id: item for item in result.candidates}
        validations = {item.finding_id: item for item in result.validations}
        bundles = [
            {
                "finding": to_dict(finding),
                "validation": to_dict(validations[finding.finding_id]),
                "candidate": to_dict(candidates[finding.candidate_id]),
                "patch": None,
            }
            for finding in result.findings
        ]
        validated = sum(
            item.verdict == ValidationVerdict.VALIDATED
            for item in result.validations
        )

        options = self.request_options(job.job_id)
        return {
            "summary": {
                "router_artifact": str(self.settings.router_artifact.resolve()),
                "router_artifact_sha256": _file_sha256(
                    self.settings.router_artifact
                ),
                "candidate_ranker_artifact": config.analysis.candidate_ranker_path,
                "candidate_ranker_artifact_sha256": (
                    _file_sha256(config.analysis.candidate_ranker_path)
                    if config.analysis.candidate_ranker_path
                    else None
                ),
                "max_candidates": config.analysis.max_candidates_per_project,
                "source_file_count": len(source_files),
                "candidate_count": len(result.candidates),
                "cwe_hypothesis_count": sum(
                    len(item.cwe_hypotheses) for item in result.candidates
                ),
                "finding_count": len(result.findings),
                "validated_finding_count": validated,
                "total_cost": sum(item.cost for item in result.usage),
                "request_count": len(result.usage),
                "expert_task_count": result.expert_task_count,
                "submitted_expert_task_count": result.submitted_expert_task_count,
                "skipped_expert_task_count": result.skipped_expert_task_count,
                "detection_call_limit": 1,
                "request_settings": {
                    **options.safe_metadata(),
                    "effective_model": config.model.expert_model,
                    "candidate_gate_threshold": config.candidate_gate.threshold,
                    "minimum_confidence": config.validation.minimum_confidence,
                },
            },
            "findings": bundles,
            "routes": [to_dict(item) for item in result.routes],
            "errors": result.errors,
            "usage": [to_dict(item) for item in result.usage],
        }

    def propose_patch_batch(
        self,
        job_id: str,
        finding_ids: Sequence[str],
    ) -> PatchBatchRecord:
        """Use the same per-job API key/model for the one patch-generation call."""
        selected_ids = sorted(
            dict.fromkeys(str(item) for item in finding_ids if item)
        )
        if not selected_ids:
            raise ValueError("Select at least one validated finding")

        analysis = self.get_analysis(job_id)
        bundles = [
            _finding_bundle(analysis, finding_id)
            for finding_id in selected_ids
        ]
        for bundle in bundles:
            if bundle["validation"]["verdict"] != ValidationVerdict.VALIDATED.value:
                raise ValueError("Only validated findings can be patched")

        config = self._config_for_job(job_id)
        if not config.runtime.allow_paid_experiments or not config.model.api_key:
            raise RuntimeError(
                "이 작업의 OpenRouter API Key를 사용할 수 없습니다. "
                "Runtime을 재시작했다면 프로젝트를 다시 분석해주세요."
            )

        config.runtime.max_retries = 0

        with self._lock:
            existing = self.get_patch_batch(job_id)
            if existing is not None:
                if existing.finding_ids == selected_ids:
                    return existing
                raise RuntimeError(
                    "This job has already used its one patch-generation call"
                )

            budget_path = self._patch_budget_path(job_id)
            if budget_path.exists():
                raise RuntimeError(
                    "This job has already attempted its one patch-generation call"
                )

            _write_json_atomic(
                budget_path,
                {"finding_ids": selected_ids, "started_at": _now()},
            )

        items = [
            (
                _finding_from_raw(bundle["finding"]),
                _validation_from_raw(bundle["validation"]),
                _candidate_from_raw(bundle["candidate"]),
            )
            for bundle in bundles
        ]

        proposal = LLMBatchPatchAgent(
            build_openrouter_client(config),
            config.model.patch_model,
            max_prompt_characters=self.settings.patch_max_prompt_characters,
        ).propose(items)

        approved_files = {finding.file for finding, _, _ in items}
        validate_patch_scope_for_files(proposal.unified_diff, approved_files)

        verification = TemporaryPatchVerifier().verify(
            self._job_dir(job_id) / "input",
            proposal,
            [],
        )
        if not verification.patch_applied:
            raise RuntimeError(
                verification.error or "Generated patch could not be applied"
            )

        now = _now()
        record = PatchBatchRecord(
            patch_id="PB-"
            + hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest()[:16],
            finding_ids=selected_ids,
            status="proposed",
            summary=proposal.summary,
            unified_diff=proposal.unified_diff,
            model=proposal.model,
            verification=to_dict(verification),
            usage=to_dict(proposal.usage),
            created_at=now,
            updated_at=now,
        )
        self._write_patch_batch(job_id, record)
        return record


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
