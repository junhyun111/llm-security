from __future__ import annotations

from fastapi import FastAPI

from llm_security.web.service import WebJobService

from .artifacts import inspect_artifacts
from .paths import RuntimePaths, configure_process_environment


def create_runtime_app(
    paths: RuntimePaths | None = None,
) -> tuple[FastAPI, WebJobService]:
    """Create the existing job API with bundle-owned model paths."""

    selected = paths or RuntimePaths.discover()
    configure_process_environment(selected)
    metadata = inspect_artifacts(selected)

    # Import only after the runtime-owned environment has been installed.
    # llm_security.web.app builds its default application at module import.
    from llm_security.web.app import app

    service = app.state.job_service
    if not isinstance(service, WebJobService):
        raise TypeError("FastAPI application has an invalid job service")
    service.settings.env_file = selected.env_file
    service.settings.router_artifact = selected.router_artifact
    service.settings.workspace_root = selected.workspace_root
    app.state.runtime_service = service
    app.state.runtime_paths = selected

    if not getattr(app.state, "runtime_routes_installed", False):
        @app.get("/api/runtime")
        def runtime_metadata() -> dict[str, object]:
            return inspect_artifacts(selected)

        @app.get("/api/runtime/startup")
        def startup_metadata() -> dict[str, object]:
            return metadata

        app.state.runtime_routes_installed = True

    return app, service
