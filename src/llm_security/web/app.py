from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..models import to_dict
from .service import WebJobService, WebSettings


STATIC_DIRECTORY = Path(__file__).with_name("static")


class PatchBatchRequest(BaseModel):
    finding_ids: list[str]


def create_app(service: WebJobService | None = None) -> FastAPI:
    selected_service = service or WebJobService(WebSettings.from_env())
    owns_service = service is None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owns_service:
            selected_service.close()

    app = FastAPI(
        title="LLM Security Review",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.job_service = selected_service

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [to_dict(item) for item in selected_service.list_jobs()]

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    def upload_project(
        project_name: Annotated[str, Form()],
        relative_paths: Annotated[list[str], Form()],
        files: Annotated[list[UploadFile], File()],
    ) -> dict:
        if len(files) != len(relative_paths):
            raise HTTPException(
                status_code=400,
                detail="Each uploaded file must have one relative path",
            )
        try:
            record = selected_service.create_job(
                project_name,
                [
                    (relative, upload.file)
                    for relative, upload in zip(relative_paths, files, strict=True)
                ],
            )
            return to_dict(record)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return to_dict(selected_service.get_job(job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error

    @app.get("/api/jobs/{job_id}/analysis")
    def get_analysis(job_id: str) -> dict:
        try:
            analysis = selected_service.get_analysis(job_id)
            for bundle in analysis.get("findings", []):
                finding_id = bundle["finding"]["finding_id"]
                patch = selected_service.get_patch(job_id, finding_id)
                bundle["patch"] = to_dict(patch) if patch else None
            patch_batch = selected_service.get_patch_batch(job_id)
            analysis["patch_batch"] = to_dict(patch_batch) if patch_batch else None
            return analysis
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/jobs/{job_id}/patches/proposal")
    def propose_patch_batch(job_id: str, request: PatchBatchRequest) -> dict:
        try:
            return to_dict(
                selected_service.propose_patch_batch(job_id, request.finding_ids)
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/jobs/{job_id}/patches/{patch_id}/approve")
    def approve_patch_batch(job_id: str, patch_id: str) -> dict:
        try:
            return to_dict(selected_service.approve_patch_batch(job_id, patch_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/jobs/{job_id}/patches/{patch_id}/reject")
    def reject_patch_batch(job_id: str, patch_id: str) -> dict:
        try:
            return to_dict(selected_service.reject_patch_batch(job_id, patch_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/download")
    def download_project(job_id: str) -> FileResponse:
        try:
            archive = selected_service.create_download(job_id)
            project_name = selected_service.get_job(job_id).project_name
            safe_name = "".join(
                character if character.isalnum() or character in "-_" else "-"
                for character in project_name
            ).strip("-") or "project"
            return FileResponse(
                archive,
                media_type="application/zip",
                filename=f"{safe_name}-reviewed.zip",
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error

    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    return app


app = create_app()
