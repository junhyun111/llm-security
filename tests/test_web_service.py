from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_security.llm import LLMResponse
from llm_security.models import UsageRecord
from llm_security.web.app import create_app
from llm_security.web.service import (
    JobStatus,
    PatchRecord,
    WebJobService,
    WebSettings,
    load_project_sources,
    validate_patch_scope,
)


def _settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        workspace_root=tmp_path / "jobs",
        router_artifact=tmp_path / "router.pkl",
        max_upload_files=10,
        max_upload_bytes=1024 * 1024,
        max_source_file_bytes=1024 * 1024,
        max_source_total_bytes=1024 * 1024,
    )


def _analysis(_project, _job, progress):
    progress(50, "fake analysis")
    return {
        "summary": {
            "source_file_count": 1,
            "candidate_count": 1,
            "finding_count": 1,
            "validated_finding_count": 1,
            "total_cost": 0.0,
        },
        "findings": [
            {
                "finding": {
                    "finding_id": "F-1",
                    "file": "project/main.c",
                },
                "validation": {"verdict": "validated"},
                "candidate": {},
                "patch": None,
            }
        ],
        "routes": [],
        "errors": [],
        "usage": [],
    }


def _patchable_analysis(_project, _job, progress):
    progress(50, "fake analysis")
    candidate = {
        "candidate_id": "C-1",
        "project_id": "project",
        "file": "project/main.c",
        "function": "main",
        "line_start": 1,
        "line_end": 1,
        "code": "int main(void) { return 0; }\n",
        "evidence": [
            {
                "evidence_id": "E-1",
                "kind": "error_path",
                "file": "project/main.c",
                "line": 1,
                "expression": "return 0",
                "function": "main",
                "subject": None,
                "object": None,
                "facts": {},
            }
        ],
        "features": {"error_path_count": 1.0},
        "suspicion_score": 0.9,
        "callers": [],
        "callees": [],
        "feature_schema_version": "semantic-v1",
    }

    def bundle(finding_id):
        return {
            "finding": {
                "finding_id": finding_id,
                "candidate_id": "C-1",
                "expert": "control_state_error",
                "title": "bad return",
                "root_cause": "incorrect return code",
                "consequence": "failure is hidden",
                "file": "project/main.c",
                "function": "main",
                "line_start": 1,
                "line_end": 1,
                "cwes": ["CWE-703"],
                "source": None,
                "sink": "return",
                "missing_guard": "return failure",
                "trigger_path": ["main", "return"],
                "evidence_ids": ["E-1"],
                "confidence": 0.9,
                "preconditions": [],
                "evidence_for": ["E-1"],
                "evidence_against": [],
                "falsification_test": "observe exit code",
                "model_id": "fake",
                "prompt_version": "batch",
                "supporting_experts": ["control_state_error"],
                "supporting_models": ["fake"],
            },
            "validation": {
                "finding_id": finding_id,
                "verdict": "validated",
                "confidence": 0.9,
                "checks": {"evidence_ids_valid": True},
                "reasons": ["static evidence matches"],
                "model_used": None,
            },
            "candidate": candidate,
            "patch": None,
        }

    return {
        "summary": {
            "source_file_count": 1,
            "candidate_count": 1,
            "finding_count": 2,
            "validated_finding_count": 2,
            "total_cost": 0.0,
            "request_count": 1,
        },
        "findings": [bundle("F-1"), bundle("F-2")],
        "routes": [],
        "errors": [],
        "usage": [],
    }


class _PatchClient:
    def __init__(self):
        self.calls = 0

    def complete(self, *, model, messages, response_schema, metadata=None):
        self.calls += 1
        return LLMResponse(
            data={
                "diff": (
                    "--- a/project/main.c\n"
                    "+++ b/project/main.c\n"
                    "@@ -1 +1 @@\n"
                    "-int main(void) { return 0; }\n"
                    "+int main(void) { return 1; }\n"
                ),
                "summary": "Return the failure code.",
            },
            usage=UsageRecord(model=model, prompt_tokens=50, completion_tokens=20),
            raw={},
        )


def _completed_job(service: WebJobService):
    job = service.create_job(
        "project",
        [
            (
                "project/main.c",
                io.BytesIO(b"int main(void) { return 0; }\n"),
            ),
            ("project/app.db", io.BytesIO(b"not source code")),
        ],
    )
    for _ in range(200):
        current = service.get_job(job.job_id)
        if current.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            return current
        time.sleep(0.005)
    raise AssertionError("background analysis did not finish")


def test_upload_is_path_safe_and_only_cpp_is_loaded(tmp_path: Path) -> None:
    service = WebJobService(_settings(tmp_path), analysis_callback=_analysis)
    try:
        with pytest.raises(ValueError, match="Unsafe upload path"):
            service.create_job("project", [("../escape.c", io.BytesIO(b"x"))])
        job = _completed_job(service)
        assert job.status == JobStatus.COMPLETED
        project = tmp_path / "jobs" / job.job_id / "input"
        sources = load_project_sources(
            project,
            max_file_bytes=1024 * 1024,
            max_total_bytes=1024 * 1024,
        )
        assert list(sources) == ["project/main.c"]
        assert (project / "project" / "app.db").exists()
    finally:
        service.close()


def test_approval_changes_only_the_approved_copy(tmp_path: Path) -> None:
    service = WebJobService(_settings(tmp_path), analysis_callback=_analysis)
    try:
        job = _completed_job(service)
        diff = """\
--- a/project/main.c
+++ b/project/main.c
@@ -1 +1 @@
-int main(void) { return 0; }
+int main(void) { return 1; }
"""
        now = "2026-01-01T00:00:00+00:00"
        service._write_patch(
            job.job_id,
            PatchRecord("F-1", "proposed", "test", diff, "fake", {}, now, now),
        )

        patch = service.approve_patch(job.job_id, "F-1")
        job_root = tmp_path / "jobs" / job.job_id
        original = (job_root / "input" / "project" / "main.c").read_text()
        approved = (job_root / "approved" / "project" / "main.c").read_text()

        assert patch.status == "approved"
        assert "return 0" in original
        assert "return 1" in approved
        archive = service.create_download(job.job_id)
        with zipfile.ZipFile(archive) as handle:
            assert "return 1" in handle.read("project/main.c").decode()
    finally:
        service.close()


def test_patch_scope_rejects_other_files_and_file_creation() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_patch_scope(
            "--- a/other.c\n+++ b/other.c\n@@ -1 +1 @@\n-a\n+b\n",
            "main.c",
        )
    with pytest.raises(ValueError, match="create or delete"):
        validate_patch_scope(
            "--- /dev/null\n+++ b/main.c\n@@ -0,0 +1 @@\n+x\n",
            "main.c",
        )


def test_batch_patch_uses_one_call_and_applies_all_selected_findings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    settings.env_file = tmp_path / ".env"
    settings.env_file.write_text(
        "OPENROUTER_API_KEY=test\n"
        "OPENROUTER_PATCH_MODEL=model/patch\n"
        "RUN_PAID_EXPERIMENTS=1\n",
        encoding="utf-8",
    )
    client = _PatchClient()
    monkeypatch.setattr(
        "llm_security.web.service.build_openrouter_client",
        lambda _config: client,
    )
    service = WebJobService(settings, analysis_callback=_patchable_analysis)
    try:
        job = _completed_job(service)
        proposal = service.propose_patch_batch(job.job_id, ["F-1", "F-2"])
        repeated = service.propose_patch_batch(job.job_id, ["F-1", "F-2"])

        assert client.calls == 1
        assert repeated.patch_id == proposal.patch_id
        assert proposal.finding_ids == ["F-1", "F-2"]
        approved = service.approve_patch_batch(job.job_id, proposal.patch_id)
        approved_source = (
            tmp_path / "jobs" / job.job_id / "approved" / "project" / "main.c"
        ).read_text()
        assert approved.status == "approved"
        assert "return 1" in approved_source
    finally:
        service.close()


def test_web_api_uploads_folder_and_returns_analysis(tmp_path: Path) -> None:
    service = WebJobService(_settings(tmp_path), analysis_callback=_analysis)
    try:
        with TestClient(create_app(service)) as client:
            assert client.get("/api/health").json() == {"status": "ok"}
            assert "프로젝트 폴더 업로드" in client.get("/").text
            response = client.post(
                "/api/jobs",
                data={
                    "project_name": "project",
                    "relative_paths": "project/main.c",
                },
                files={
                    "files": (
                        "main.c",
                        b"int main(void) { return 0; }\n",
                        "text/x-c",
                    )
                },
            )
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            for _ in range(200):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "completed":
                    break
                time.sleep(0.005)
            analysis = client.get(f"/api/jobs/{job_id}/analysis")
            assert analysis.status_code == 200
            assert analysis.json()["summary"]["validated_finding_count"] == 1
    finally:
        service.close()
