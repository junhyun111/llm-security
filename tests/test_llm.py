from __future__ import annotations

from typing import Any

import pytest

from llm_security.llm import OpenRouterClient, _decode_json_content


class _Response:
    status_code = 200
    is_error = False

    def json(self) -> dict[str, Any]:
        return {
            "model": "model/test",
            "provider": "test-provider",
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {},
        }


def test_decode_json_content_accepts_plain_and_fenced_json() -> None:
    assert _decode_json_content('{"findings": []}') == {"findings": []}
    assert _decode_json_content('```json\n{"findings": []}\n```') == {
        "findings": []
    }


def _capture_request(monkeypatch, *, temperature: float | None) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(
            self,
            endpoint: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> _Response:
            captured.update(json)
            return _Response()

    monkeypatch.setattr("llm_security.llm.httpx.Client", _Client)
    client = OpenRouterClient(
        api_key="test-key",
        temperature=temperature,
        require_parameters=True,
        allow_fallbacks=False,
    )
    client.complete(
        model="model/test",
        messages=[{"role": "user", "content": "test"}],
        response_schema={
            "name": "test",
            "strict": True,
            "schema": {"type": "object"},
        },
    )
    return captured


def test_unset_temperature_is_omitted_from_openrouter_request(monkeypatch) -> None:
    body = _capture_request(monkeypatch, temperature=None)

    assert "temperature" not in body
    assert body["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": False,
    }


def test_explicit_temperature_is_sent_to_openrouter(monkeypatch) -> None:
    body = _capture_request(monkeypatch, temperature=0.0)

    assert body["temperature"] == 0.0


def test_null_content_error_preserves_openrouter_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        _Response,
        "json",
        lambda _self: {
            "id": "gen-null",
            "model": "model/test",
            "provider": "provider/test",
            "choices": [
                {
                    "finish_reason": "length",
                    "native_finish_reason": "max_tokens",
                    "message": {"content": None},
                }
            ],
            "usage": {
                "completion_tokens": 8192,
                "completion_tokens_details": {"reasoning_tokens": 4096},
            },
        },
    )
    class _Client:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *_args: object, **_kwargs: object) -> _Response:
            return _Response()

    monkeypatch.setattr("llm_security.llm.httpx.Client", _Client)
    client = OpenRouterClient(api_key="test-key", max_retries=0)

    with pytest.raises(RuntimeError) as raised:
        client.complete(
            model="model/test",
            messages=[{"role": "user", "content": "test"}],
            response_schema={"name": "test", "schema": {"type": "object"}},
        )

    message = str(raised.value)
    assert "generation_id=gen-null" in message
    assert "provider=provider/test" in message
    assert "finish_reason=length" in message
    assert "completion_tokens=8192" in message
    assert "reasoning_tokens=4096" in message
