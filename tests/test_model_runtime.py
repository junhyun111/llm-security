from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).parents[1]
RUNTIME_SOURCE = ROOT / "model_runtime" / "src"
if str(RUNTIME_SOURCE) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SOURCE))

from llm_security_runtime.api import create_runtime_app
from llm_security_runtime.artifacts import assert_model_compatible, inspect_artifacts
from llm_security_runtime.paths import RuntimePaths, configure_process_environment


def test_runtime_artifacts_load_and_match() -> None:
    paths = RuntimePaths.discover()
    metadata = inspect_artifacts(paths)

    assert metadata["status"] == "ready"
    assert metadata["router"]["artifact_version"] == 5
    assert metadata["router"]["backend"] == "multitask_mlp"
    assert metadata["router"]["feature_schema"] == "semantic-cwe-v3"
    assert metadata["router"]["assignment_count"] == 5
    assert metadata["candidate_ranker"]["backend"] == "small_mlp"
    assert_model_compatible(
        metadata, metadata["router"]["expert_model_ids"][0]
    )


def test_runtime_forces_bundle_owned_paths(monkeypatch) -> None:
    paths = RuntimePaths.discover()
    for name in (
        "ANALYSIS_BACKEND",
        "CANDIDATE_RANKER_PATH",
        "CANDIDATE_RANKER_REQUIRED",
        "WEB_ROUTER_ARTIFACT",
        "WEB_WORKSPACE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    configure_process_environment(paths)

    assert paths.router_artifact.name == "router.pkl"
    assert paths.candidate_ranker_artifact.name == "candidate_ranker.pkl"
    assert os.environ["ANALYSIS_BACKEND"] == "semantic"
    assert os.environ["CANDIDATE_RANKER_PATH"] == str(
        paths.candidate_ranker_artifact
    )
    assert os.environ["WEB_ROUTER_ARTIFACT"] == str(paths.router_artifact)


def test_runtime_api_exposes_health_and_metadata(tmp_path: Path) -> None:
    base = RuntimePaths.discover()
    paths = RuntimePaths(
        root=base.root,
        env_file=base.env_file,
        router_artifact=base.router_artifact,
        candidate_ranker_artifact=base.candidate_ranker_artifact,
        workspace_root=tmp_path / "work",
    )
    app, service = create_runtime_app(paths)
    try:
        route_paths = {route.path for route in app.routes}
        assert "/api/health" in route_paths
        assert "/api/jobs" in route_paths
        assert "/api/runtime" in route_paths
        with TestClient(app) as client:
            assert client.get("/api/health").json() == {"status": "ok"}
            runtime = client.get("/api/runtime")
            assert runtime.status_code == 200
            assert runtime.json()["router"]["artifact_version"] == 5
    finally:
        service.close()
