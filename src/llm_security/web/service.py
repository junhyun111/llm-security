from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Sequence
from uuid import uuid4

from ..config import AppConfig
from ..datasets import _candidate_from_raw
from ..factory import build_batched_web_pipeline, build_openrouter_client
from ..models import (
    ExpertFamily,
    Finding,
    ProjectCase,
    ValidationResult,
    ValidationVerdict,
    to_dict,
)
from ..patching import LLMBatchPatchAgent, LLMPatchAgent
from ..routing import AdaptiveExpertRouter, AnchorRareRouter, BudgetedUtilityRouter
from ..verification import TemporaryPatchVerifier


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
IGNORED_SOURCE_DIRECTORIES = {
    ".git",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "vendor",
}
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class JobStatus(str, Enum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class WebSettings:
    workspace_root: Path = Path(".web-data")
    router_artifact: Path = Path(
        "Model_Evaluation/artifacts/juliet_utility_router.pkl"
    )
    max_upload_files: int = 10_000
    max_upload_bytes: int = 1024 * 1024 * 1024
    max_source_file_bytes: int = 5 * 1024 * 1024
    max_source_total_bytes: int = 100 * 1024 * 1024
    worker_count: int = 1
    candidate_gate_enabled: bool = True
    detection_max_prompt_characters: int = 120_000
    detection_max_expert_tasks: int = 24
    detection_max_output_tokens: int = 8_192
    patch_max_prompt_characters: int = 120_000
    env_file: Path = Path(".env")

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "WebSettings":
        values = _read_env(env_file)
        values.update(os.environ)
        artifact = values.get("WEB_ROUTER_ARTIFACT", "").strip()
        if not artifact:
            artifact = "Model_Evaluation/artifacts/juliet_utility_router.pkl"
        return cls(
            workspace_root=Path(values.get("WEB_WORKSPACE_ROOT", ".web-data")),
            router_artifact=Path(artifact),
            max_upload_files=int(values.get("WEB_MAX_UPLOAD_FILES", "10000")),
            max_upload_bytes=int(
                float(values.get("WEB_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
            ),
            max_source_file_bytes=int(
                float(values.get("WEB_MAX_SOURCE_FILE_MB", "5")) * 1024 * 1024
            ),
            max_source_total_bytes=int(
                float(values.get("WEB_MAX_SOURCE_TOTAL_MB", "100")) * 1024 * 1024
            ),
            worker_count=max(1, int(values.get("WEB_WORKERS", "1"))),
            candidate_gate_enabled=_as_bool(
                values.get("WEB_CANDIDATE_GATE_ENABLED", "true")
            ),
            detection_max_prompt_characters=int(
                values.get("WEB_DETECTION_MAX_PROMPT_CHARACTERS", "120000")
            ),
            detection_max_expert_tasks=int(
                values.get("WEB_DETECTION_MAX_EXPERT_TASKS", "24")
            ),
            detection_max_output_tokens=int(
                values.get("WEB_DETECTION_MAX_OUTPUT_TOKENS", "8192")
            ),
            patch_max_prompt_characters=int(
                values.get("WEB_PATCH_MAX_PROMPT_CHARACTERS", "120000")
            ),
            env_file=Path(env_file),
        )


@dataclass(slots=True)
class JobRecord:
    job_id: str
    project_name: str
    status: JobStatus
    created_at: str
    updated_at: str
    progress: int = 0
    message: str = ""
    file_count: int = 0
    upload_bytes: int = 0
    source_file_count: int = 0
    finding_count: int = 0
    validated_finding_count: int = 0
    total_cost: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class PatchRecord:
    finding_id: str
    status: str
    summary: str
    unified_diff: str
    model: str
    verification: dict
    created_at: str
    updated_at: str


@dataclass(slots=True)
class PatchBatchRecord:
    patch_id: str
    finding_ids: list[str]
    status: str
    summary: str
    unified_diff: str
    model: str
    verification: dict
    usage: dict
    created_at: str
    updated_at: str


AnalysisCallback = Callable[[Path, JobRecord, Callable[[int, str], None]], dict]


class WebJobService:
    """Persistent single-node job service used by the FastAPI layer.

    Uploaded projects are immutable. Approved patches are applied only under a
    separate ``approved`` directory and downloads are built from that copy.
    """

    def __init__(
        self,
        settings: WebSettings | None = None,
        *,
        analysis_callback: AnalysisCallback | None = None,
    ) -> None:
        self.settings = settings or WebSettings.from_env()
        self.settings.workspace_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.worker_count,
            thread_name_prefix="llm-security-web",
        )
        self._analysis_callback = analysis_callback or self._analyze_project
        self._recover_interrupted_jobs()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def create_job(
        self,
        project_name: str,
        uploads: Sequence[tuple[str, BinaryIO]],
    ) -> JobRecord:
        if not uploads:
            raise ValueError("At least one uploaded file is required")
        if len(uploads) > self.settings.max_upload_files:
            raise ValueError(
                f"Upload contains {len(uploads)} files; limit is "
                f"{self.settings.max_upload_files}"
            )
        normalized = [_safe_relative_path(path) for path, _ in uploads]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Upload contains duplicate relative paths")

        now = _now()
        record = JobRecord(
            job_id=uuid4().hex,
            project_name=_safe_project_name(project_name),
            status=JobStatus.UPLOADING,
            created_at=now,
            updated_at=now,
            message="Uploading project files",
        )
        job_directory = self._job_dir(record.job_id)
        input_directory = job_directory / "input"
        input_directory.mkdir(parents=True)
        total = 0
        try:
            for relative, (_, stream) in zip(normalized, uploads, strict=True):
                destination = input_directory.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    while chunk := stream.read(1024 * 1024):
                        total += len(chunk)
                        if total > self.settings.max_upload_bytes:
                            raise ValueError(
                                "Upload exceeds WEB_MAX_UPLOAD_MB limit"
                            )
                        handle.write(chunk)
            record.file_count = len(uploads)
            record.upload_bytes = total
            record.status = JobStatus.QUEUED
            record.progress = 5
            record.message = "Upload complete; analysis queued"
            record.updated_at = _now()
            self._write_job(record)
        except Exception:
            shutil.rmtree(job_directory, ignore_errors=True)
            raise
        self._executor.submit(self._run_analysis, record.job_id)
        return record

    def list_jobs(self) -> list[JobRecord]:
        records = []
        for path in self.settings.workspace_root.glob("*/job.json"):
            try:
                records.append(_job_from_raw(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def get_job(self, job_id: str) -> JobRecord:
        path = self._job_dir(job_id) / "job.json"
        with self._lock:
            if not path.exists():
                raise KeyError(job_id)
            return _job_from_raw(json.loads(path.read_text(encoding="utf-8")))

    def get_analysis(self, job_id: str) -> dict:
        record = self.get_job(job_id)
        if record.status != JobStatus.COMPLETED:
            raise RuntimeError("Analysis is not complete")
        path = self._job_dir(job_id) / "analysis.json"
        if not path.exists():
            raise RuntimeError("Analysis result is missing")
        return json.loads(path.read_text(encoding="utf-8"))

    def propose_patch_batch(
        self,
        job_id: str,
        finding_ids: Sequence[str],
    ) -> PatchBatchRecord:
        """Spend the job's single patch call on all selected findings."""

        selected_ids = sorted(dict.fromkeys(str(item) for item in finding_ids if item))
        if not selected_ids:
            raise ValueError("Select at least one validated finding")
        analysis = self.get_analysis(job_id)
        bundles = [_finding_bundle(analysis, finding_id) for finding_id in selected_ids]
        for bundle in bundles:
            if bundle["validation"]["verdict"] != ValidationVerdict.VALIDATED.value:
                raise ValueError("Only validated findings can be patched")

        config = AppConfig.from_env(self.settings.env_file)
        if not config.runtime.allow_paid_experiments:
            raise RuntimeError(
                "Patch generation calls OpenRouter; set RUN_PAID_EXPERIMENTS=1"
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
            self._job_dir(job_id) / "input", proposal, []
        )
        if not verification.patch_applied:
            raise RuntimeError(
                verification.error or "Generated patch could not be applied"
            )
        now = _now()
        record = PatchBatchRecord(
            patch_id="PB-" + hashlib.sha256(
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

    def approve_patch_batch(self, job_id: str, patch_id: str) -> PatchBatchRecord:
        record = self.get_patch_batch(job_id)
        if record is None or record.patch_id != patch_id:
            raise KeyError(f"No patch batch for {patch_id}")
        if record.status == "approved":
            return record
        if record.status != "proposed":
            raise ValueError(f"Patch batch is {record.status}, not proposed")
        analysis = self.get_analysis(job_id)
        allowed_files = {
            str(_finding_bundle(analysis, finding_id)["finding"]["file"])
            for finding_id in record.finding_ids
        }
        validate_patch_scope_for_files(record.unified_diff, allowed_files)

        job_directory = self._job_dir(job_id)
        approved = job_directory / "approved"
        if not approved.exists():
            shutil.copytree(job_directory / "input", approved, symlinks=False)
        patch_file = self._patch_batch_path(job_id).with_suffix(".diff")
        _write_text_lf(patch_file, record.unified_diff)
        _git_apply(approved, patch_file, check=True)
        _git_apply(approved, patch_file, check=False)
        record.status = "approved"
        record.updated_at = _now()
        self._write_patch_batch(job_id, record)
        return record

    def reject_patch_batch(self, job_id: str, patch_id: str) -> PatchBatchRecord:
        record = self.get_patch_batch(job_id)
        if record is None or record.patch_id != patch_id:
            raise KeyError(f"No patch batch for {patch_id}")
        if record.status == "approved":
            raise ValueError("An applied patch batch cannot be rejected")
        record.status = "rejected"
        record.updated_at = _now()
        self._write_patch_batch(job_id, record)
        return record

    def get_patch_batch(self, job_id: str) -> PatchBatchRecord | None:
        self.get_job(job_id)
        path = self._patch_batch_path(job_id)
        with self._lock:
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            return PatchBatchRecord(**raw)

    def propose_patch(self, job_id: str, finding_id: str) -> PatchRecord:
        analysis = self.get_analysis(job_id)
        bundle = _finding_bundle(analysis, finding_id)
        if bundle["validation"]["verdict"] != ValidationVerdict.VALIDATED.value:
            raise ValueError("Only validated findings can be patched")
        existing = self._read_patch(job_id, finding_id)
        if existing and existing.status in {"proposed", "approved"}:
            return existing

        config = AppConfig.from_env(self.settings.env_file)
        if not config.runtime.allow_paid_experiments:
            raise RuntimeError(
                "Patch generation calls OpenRouter; set RUN_PAID_EXPERIMENTS=1"
            )
        finding = _finding_from_raw(bundle["finding"])
        validation = _validation_from_raw(bundle["validation"])
        candidate = _candidate_from_raw(bundle["candidate"])
        proposal = LLMPatchAgent(
            build_openrouter_client(config), config.model.patch_model
        ).propose(finding, validation, candidate)
        validate_patch_scope(proposal.unified_diff, finding.file)
        verification = TemporaryPatchVerifier().verify(
            self._job_dir(job_id) / "input", proposal, []
        )
        if not verification.patch_applied:
            raise RuntimeError(
                verification.error or "Generated patch could not be applied"
            )
        now = _now()
        record = PatchRecord(
            finding_id=finding_id,
            status="proposed",
            summary=proposal.summary,
            unified_diff=proposal.unified_diff,
            model=proposal.model,
            verification=to_dict(verification),
            created_at=now,
            updated_at=now,
        )
        self._write_patch(job_id, record)
        return record

    def approve_patch(self, job_id: str, finding_id: str) -> PatchRecord:
        record = self._read_patch(job_id, finding_id)
        if record is None:
            raise KeyError(f"No patch proposal for {finding_id}")
        if record.status == "approved":
            return record
        if record.status != "proposed":
            raise ValueError(f"Patch is {record.status}, not proposed")
        analysis = self.get_analysis(job_id)
        finding = _finding_bundle(analysis, finding_id)["finding"]
        validate_patch_scope(record.unified_diff, str(finding["file"]))

        job_directory = self._job_dir(job_id)
        approved = job_directory / "approved"
        if not approved.exists():
            shutil.copytree(job_directory / "input", approved, symlinks=False)
        patch_file = self._patch_path(job_id, finding_id).with_suffix(".diff")
        _write_text_lf(patch_file, record.unified_diff)
        _git_apply(approved, patch_file, check=True)
        _git_apply(approved, patch_file, check=False)
        record.status = "approved"
        record.updated_at = _now()
        self._write_patch(job_id, record)
        return record

    def reject_patch(self, job_id: str, finding_id: str) -> PatchRecord:
        record = self._read_patch(job_id, finding_id)
        if record is None:
            raise KeyError(f"No patch proposal for {finding_id}")
        if record.status == "approved":
            raise ValueError("An applied patch cannot be rejected")
        record.status = "rejected"
        record.updated_at = _now()
        self._write_patch(job_id, record)
        return record

    def get_patch(self, job_id: str, finding_id: str) -> PatchRecord | None:
        self.get_job(job_id)
        return self._read_patch(job_id, finding_id)

    def create_download(self, job_id: str) -> Path:
        self.get_job(job_id)
        job_directory = self._job_dir(job_id)
        source = (
            job_directory / "approved"
            if (job_directory / "approved").exists()
            else job_directory / "input"
        )
        destination = job_directory / "downloads" / "project.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source).as_posix())
        temporary.replace(destination)
        return destination

    def _run_analysis(self, job_id: str) -> None:
        try:
            record = self.get_job(job_id)
            self._update_job(record, status=JobStatus.ANALYZING, progress=10, message="Starting analysis")

            def progress(value: int, message: str) -> None:
                current = self.get_job(job_id)
                self._update_job(current, progress=value, message=message)

            analysis = self._analysis_callback(
                self._job_dir(job_id) / "input", record, progress
            )
            _write_json_atomic(self._job_dir(job_id) / "analysis.json", analysis)
            completed = self.get_job(job_id)
            summary = analysis.get("summary", {})
            completed.source_file_count = int(summary.get("source_file_count", 0))
            completed.finding_count = int(summary.get("finding_count", 0))
            completed.validated_finding_count = int(
                summary.get("validated_finding_count", 0)
            )
            completed.total_cost = float(summary.get("total_cost", 0.0))
            self._update_job(
                completed,
                status=JobStatus.COMPLETED,
                progress=100,
                message="Analysis complete",
                error=None,
            )
        except Exception as error:  # background boundary must persist the failure
            try:
                failed = self.get_job(job_id)
                self._update_job(
                    failed,
                    status=JobStatus.FAILED,
                    progress=100,
                    message="Analysis failed",
                    error=str(error),
                )
            except Exception:
                pass

    def _analyze_project(
        self,
        input_directory: Path,
        job: JobRecord,
        progress: Callable[[int, str], None],
    ) -> dict:
        config = AppConfig.from_env(self.settings.env_file)
        if not config.runtime.allow_paid_experiments:
            raise RuntimeError(
                "Web analysis calls OpenRouter; set RUN_PAID_EXPERIMENTS=1 in .env"
            )
        config.analysis.backend = "semantic"
        config.candidate_gate.enabled = self.settings.candidate_gate_enabled
        config.model.max_output_tokens = self.settings.detection_max_output_tokens
        # One logical detection call must also mean one physical HTTP attempt.
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
            },
            "findings": bundles,
            "routes": [to_dict(item) for item in result.routes],
            "errors": result.errors,
            "usage": [to_dict(item) for item in result.usage],
        }

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        root = self.settings.workspace_root.resolve()
        target = (root / job_id).resolve()
        if target.parent != root:
            raise ValueError("Invalid job path")
        return target

    def _recover_interrupted_jobs(self) -> None:
        for record in self.list_jobs():
            if record.status in {
                JobStatus.UPLOADING,
                JobStatus.QUEUED,
                JobStatus.ANALYZING,
            }:
                self._update_job(
                    record,
                    status=JobStatus.FAILED,
                    progress=100,
                    message="Analysis interrupted by server restart",
                    error="Restart the upload to run analysis again.",
                )

    def _write_job(self, record: JobRecord) -> None:
        with self._lock:
            _write_json_atomic(self._job_dir(record.job_id) / "job.json", to_dict(record))

    def _update_job(self, record: JobRecord, **changes) -> None:
        for key, value in changes.items():
            setattr(record, key, value)
        record.updated_at = _now()
        self._write_job(record)

    def _patch_path(self, job_id: str, finding_id: str) -> Path:
        tag = hashlib.sha256(finding_id.encode("utf-8")).hexdigest()
        return self._job_dir(job_id) / "patches" / f"{tag}.json"

    def _patch_batch_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "patch-batch.json"

    def _patch_budget_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "patch-request.json"

    def _write_patch_batch(self, job_id: str, record: PatchBatchRecord) -> None:
        _write_json_atomic(self._patch_batch_path(job_id), to_dict(record))

    def _read_patch(self, job_id: str, finding_id: str) -> PatchRecord | None:
        path = self._patch_path(job_id, finding_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PatchRecord(**raw)

    def _write_patch(self, job_id: str, record: PatchRecord) -> None:
        path = self._patch_path(job_id, record.finding_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, to_dict(record))


def load_project_sources(
    project_directory: str | Path,
    *,
    max_file_bytes: int,
    max_total_bytes: int,
) -> dict[str, str]:
    root = Path(project_directory).resolve()
    if not root.is_dir():
        raise ValueError("Uploaded project directory is missing")
    sources: dict[str, str] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_SOURCE_DIRECTORIES for part in relative.parts[:-1]):
            continue
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(
                f"Source file exceeds WEB_MAX_SOURCE_FILE_MB: {relative.as_posix()}"
            )
        total += size
        if total > max_total_bytes:
            raise ValueError("C/C++ source exceeds WEB_MAX_SOURCE_TOTAL_MB")
        sources[relative.as_posix()] = path.read_text(
            encoding="utf-8", errors="replace"
        )
    if not sources:
        raise ValueError("No supported C/C++ source files were found")
    return sources


def load_router_artifact(path: str | Path):
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(
            f"Router artifact not found: {artifact}. Run 01_train_router.ipynb first."
        )
    errors = []
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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_patch_scope(unified_diff: str, expected_file: str) -> None:
    validate_patch_scope_for_files(unified_diff, {expected_file})


def validate_patch_scope_for_files(
    unified_diff: str,
    expected_files: set[str],
) -> None:
    if not unified_diff.strip():
        raise ValueError("Patch is empty")
    forbidden = ("GIT binary patch", "rename from ", "rename to ", "new file mode", "deleted file mode")
    if any(marker in unified_diff for marker in forbidden):
        raise ValueError("Patch may only modify an existing text source file")
    paths = []
    for line in unified_diff.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            token = line[4:].split("\t", 1)[0].strip()
            if token == "/dev/null":
                raise ValueError("Patch may not create or delete files")
            if token.startswith(("a/", "b/")):
                token = token[2:]
            paths.append(_safe_relative_path(token).as_posix())
    if not paths:
        raise ValueError("Patch is not a unified diff")
    expected = {
        _safe_relative_path(expected_file).as_posix()
        for expected_file in expected_files
    }
    if not expected:
        raise ValueError("At least one approved source file is required")
    if any(path not in expected for path in paths):
        raise ValueError("Patch modifies a file outside the approved findings")


def _git_apply(workspace: Path, patch_file: Path, *, check: bool) -> None:
    command = ["git", "apply"]
    if check:
        command.append("--check")
    command.append(str(patch_file.resolve()))
    completed = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        action = "check" if check else "apply"
        raise RuntimeError(
            f"Patch {action} failed: {completed.stderr[-2000:]}"
        )


def _safe_relative_path(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/").strip()
    if (
        not normalized
        or "\x00" in normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise ValueError(f"Unsafe upload path: {value!r}")
    path = PurePosixPath(normalized)
    unsafe_component = any(
        part in {"", ".", ".."}
        or ":" in part
        or part.rstrip(" .") != part
        or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED
        for part in path.parts
    )
    if path.is_absolute() or unsafe_component:
        raise ValueError(f"Unsafe upload path: {value!r}")
    return path


def _safe_project_name(value: str) -> str:
    cleaned = " ".join(value.strip().split())[:120]
    if not cleaned:
        raise ValueError("Project name is required")
    return cleaned


def _finding_bundle(analysis: dict, finding_id: str) -> dict:
    for bundle in analysis.get("findings", []):
        if bundle.get("finding", {}).get("finding_id") == finding_id:
            return bundle
    raise KeyError(finding_id)


def _finding_from_raw(raw: dict) -> Finding:
    return Finding(
        finding_id=str(raw["finding_id"]),
        candidate_id=str(raw["candidate_id"]),
        expert=ExpertFamily(raw["expert"]),
        title=str(raw["title"]),
        root_cause=str(raw["root_cause"]),
        consequence=str(raw["consequence"]),
        file=str(raw["file"]),
        function=str(raw["function"]),
        line_start=int(raw["line_start"]),
        line_end=int(raw["line_end"]),
        cwes=[str(item) for item in raw.get("cwes", [])],
        source=raw.get("source"),
        sink=raw.get("sink"),
        missing_guard=raw.get("missing_guard"),
        trigger_path=[str(item) for item in raw.get("trigger_path", [])],
        evidence_ids=[str(item) for item in raw.get("evidence_ids", [])],
        confidence=float(raw["confidence"]),
        preconditions=[str(item) for item in raw.get("preconditions", [])],
        evidence_for=[str(item) for item in raw.get("evidence_for", [])],
        evidence_against=[str(item) for item in raw.get("evidence_against", [])],
        falsification_test=raw.get("falsification_test"),
        model_id=raw.get("model_id"),
        prompt_version=raw.get("prompt_version"),
        supporting_experts=[
            ExpertFamily(item) for item in raw.get("supporting_experts", [])
        ],
        supporting_models=[str(item) for item in raw.get("supporting_models", [])],
    )


def _validation_from_raw(raw: dict) -> ValidationResult:
    return ValidationResult(
        finding_id=str(raw["finding_id"]),
        verdict=ValidationVerdict(raw["verdict"]),
        confidence=float(raw["confidence"]),
        checks=dict(raw.get("checks", {})),
        reasons=[str(item) for item in raw.get("reasons", [])],
        model_used=raw.get("model_used"),
    )


def _job_from_raw(raw: dict) -> JobRecord:
    return JobRecord(
        job_id=str(raw["job_id"]),
        project_name=str(raw["project_name"]),
        status=JobStatus(raw["status"]),
        created_at=str(raw["created_at"]),
        updated_at=str(raw["updated_at"]),
        progress=int(raw.get("progress", 0)),
        message=str(raw.get("message", "")),
        file_count=int(raw.get("file_count", 0)),
        upload_bytes=int(raw.get("upload_bytes", 0)),
        source_file_count=int(raw.get("source_file_count", 0)),
        finding_count=int(raw.get("finding_count", 0)),
        validated_finding_count=int(raw.get("validated_finding_count", 0)),
        total_cost=float(raw.get("total_cost", 0.0)),
        error=raw.get("error"),
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_text_lf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_env(path: str | Path) -> dict[str, str]:
    destination = Path(path)
    if not destination.exists():
        return {}
    values = {}
    for raw in destination.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
