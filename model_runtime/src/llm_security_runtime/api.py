from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_security.models import to_dict
from llm_security.web.service import WebSettings

from .artifacts import assert_model_compatible, inspect_artifacts
from .paths import RuntimePaths, configure_process_environment
from .request_service import RequestAwareWebJobService, RuntimeJobOptions


class PatchBatchRequest(BaseModel):
    finding_ids: list[str]


def create_runtime_app(
    paths: RuntimePaths | None = None,
) -> tuple[FastAPI, RequestAwareWebJobService]:
    selected = paths or RuntimePaths.discover()
    configure_process_environment(selected)
    startup_metadata = inspect_artifacts(selected)

    settings = WebSettings.from_env(selected.env_file)
    settings.env_file = selected.env_file
    settings.router_artifact = selected.router_artifact
    settings.workspace_root = selected.workspace_root

    service = RequestAwareWebJobService(settings)

    app = FastAPI(
        title="LLM Security Utility Router Runtime",
        version="0.2.0",
    )
    app.state.runtime_service = service
    app.state.runtime_paths = selected

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime_metadata() -> dict[str, object]:
        metadata = inspect_artifacts(selected)
        metadata["request_configuration"] = {
            "supports_sensitivity": True,
            "sensitivity_min": 0.0,
            "sensitivity_max": 1.0,
            "sensitivity_default": 0.5,
            "supports_request_api_key": True,
            "supports_request_model": True,
            "api_key_persistence": "process-memory-only",
        }
        return metadata

    @app.get("/api/runtime/startup")
    def startup() -> dict[str, object]:
        return startup_metadata

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [to_dict(item) for item in service.list_jobs()]

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    def upload_project(
        project_name: Annotated[str, Form()],
        relative_paths: Annotated[list[str], Form()],
        files: Annotated[list[UploadFile], File()],
        sensitivity: Annotated[float, Form()] = 0.5,
        model: Annotated[str | None, Form()] = None,
        api_key: Annotated[str | None, Form()] = None,
    ) -> dict:
        if len(files) != len(relative_paths):
            raise HTTPException(
                status_code=400,
                detail="Each uploaded file must have one relative path",
            )

        if not 0.0 <= sensitivity <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="sensitivity must be between 0.0 and 1.0",
            )

        selected_model = _normalize_optional(model)
        selected_api_key = _normalize_optional(api_key)

        if selected_model is not None:
            try:
                assert_model_compatible(startup_metadata, selected_model)
            except ValueError as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error),
                ) from error

        if selected_api_key is not None:
            if len(selected_api_key) > 512 or any(
                character.isspace()
                for character in selected_api_key
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid OpenRouter API Key format",
                )

        try:
            record = service.create_job(
                project_name,
                [
                    (relative, upload.file)
                    for relative, upload in zip(
                        relative_paths,
                        files,
                        strict=True,
                    )
                ],
                options=RuntimeJobOptions(
                    sensitivity=sensitivity,
                    model=selected_model,
                    api_key=selected_api_key,
                ),
            )
            return to_dict(record)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return to_dict(service.get_job(job_id))
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            ) from error

    @app.get("/api/jobs/{job_id}/analysis")
    def get_analysis(job_id: str) -> dict:
        try:
            analysis = service.get_analysis(job_id)

            for bundle in analysis.get("findings", []):
                finding_id = bundle["finding"]["finding_id"]
                patch = service.get_patch(job_id, finding_id)
                bundle["patch"] = to_dict(patch) if patch else None

            patch_batch = service.get_patch_batch(job_id)
            analysis["patch_batch"] = (
                to_dict(patch_batch)
                if patch_batch
                else None
            )
            return analysis
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=409,
                detail=str(error),
            ) from error

    @app.post("/api/jobs/{job_id}/patches/proposal")
    def propose_patch_batch(
        job_id: str,
        request: PatchBatchRequest,
    ) -> dict:
        try:
            return to_dict(
                service.propose_patch_batch(
                    job_id,
                    request.finding_ids,
                )
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=409,
                detail=str(error),
            ) from error

    @app.post("/api/jobs/{job_id}/patches/{patch_id}/approve")
    def approve_patch_batch(job_id: str, patch_id: str) -> dict:
        try:
            return to_dict(
                service.approve_patch_batch(job_id, patch_id)
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=409,
                detail=str(error),
            ) from error

    @app.post("/api/jobs/{job_id}/patches/{patch_id}/reject")
    def reject_patch_batch(job_id: str, patch_id: str) -> dict:
        try:
            return to_dict(
                service.reject_patch_batch(job_id, patch_id)
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    @app.get("/api/jobs/{job_id}/download")
    def download_project(job_id: str) -> FileResponse:
        try:
            archive = service.create_download(job_id)
            project_name = service.get_job(job_id).project_name
            safe_name = "".join(
                character
                if character.isalnum() or character in "-_"
                else "-"
                for character in project_name
            ).strip("-") or "project"

            return FileResponse(
                archive,
                media_type="application/zip",
                filename=f"{safe_name}-reviewed.zip",
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            ) from error

    static_directory = (
        selected.root
        / "src"
        / "llm_security"
        / "web"
        / "static"
    )
    if static_directory.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=static_directory),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(static_directory / "index.html")

    return app, service


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
