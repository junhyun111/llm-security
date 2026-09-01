from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .models import UsageRecord


@dataclass(slots=True)
class LLMResponse:
    data: dict[str, Any]
    usage: UsageRecord
    raw: dict[str, Any]


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        temperature: float | None = None,
        max_output_tokens: int = 2500,
        reasoning_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        provider: str | None = None,
        require_parameters: bool = True,
        allow_fallbacks: bool = False,
        structured_output: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter mode")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.provider = provider
        self.require_parameters = require_parameters
        self.allow_fallbacks = allow_fallbacks
        self.structured_output = structured_output
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if model.startswith("~"):
            raise ValueError("Canonical model IDs are required for reproducible experiments")
        request_messages = messages
        if not self.structured_output:
            schema = response_schema.get("schema", response_schema)
            request_messages = [dict(message) for message in messages]
            request_messages[-1]["content"] += (
                "\n\nReturn only a JSON object matching this JSON Schema. "
                "Do not use Markdown fences:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "max_tokens": self.max_output_tokens,
            "provider": {
                "require_parameters": self.require_parameters,
                "allow_fallbacks": self.allow_fallbacks,
            },
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.structured_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        if self.provider:
            body["provider"]["order"] = [self.provider]
        if self.reasoning_enabled is not None:
            body["reasoning"] = {"enabled": self.reasoning_enabled, "exclude": True}
            if self.reasoning_enabled and self.reasoning_effort:
                body["reasoning"]["effort"] = self.reasoning_effort
        elif self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "LLM Security Conditional Expert Experiment",
        }
        start = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(self.endpoint, headers=headers, json=body)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                if response.is_error:
                    detail = response.text.strip().replace("\n", " ")[:1000]
                    raise RuntimeError(
                        f"OpenRouter HTTP {response.status_code}: "
                        f"{detail or response.reason_phrase}"
                    )
                raw = response.json()
                actual_model = str(raw.get("model", model))
                if actual_model != model:
                    raise RuntimeError(
                        f"Model mismatch: requested {model}, OpenRouter returned {actual_model}"
                    )
                choice = raw["choices"][0]
                message = choice["message"]
                content = message.get("content")
                if content is None:
                    usage_raw = raw.get("usage", {}) or {}
                    details = usage_raw.get("completion_tokens_details", {}) or {}
                    response_error = raw.get("error") or choice.get("error")
                    raise RuntimeError(
                        "Model returned no final content "
                        f"(generation_id={raw.get('id', 'unknown')}, "
                        f"provider={raw.get('provider', 'unknown')}, "
                        f"finish_reason={choice.get('finish_reason', 'unknown')}, "
                        "native_finish_reason="
                        f"{choice.get('native_finish_reason', 'unknown')}, "
                        "completion_tokens="
                        f"{int(usage_raw.get('completion_tokens', 0) or 0)}, "
                        "reasoning_tokens="
                        f"{int(details.get('reasoning_tokens', 0) or 0)}, "
                        f"provider_error={response_error or 'none'}). "
                        "The provider completed the request without a usable answer."
                    )
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if isinstance(content, str):
                    try:
                        data = _decode_json_content(content)
                    except json.JSONDecodeError as error:
                        finish_reason = choice.get("finish_reason", "unknown")
                        raise RuntimeError(
                            "Model returned incomplete or invalid JSON "
                            f"(finish_reason={finish_reason}, content_characters="
                            f"{len(content)}). Increase the output-token budget or reduce "
                            "WEB_DETECTION_MAX_EXPERT_TASKS."
                        ) from error
                elif isinstance(content, dict):
                    data = content
                else:
                    raise TypeError(
                        f"The model returned unsupported content: {type(content).__name__}"
                    )
                usage_raw = raw.get("usage", {})
                details = usage_raw.get("completion_tokens_details", {}) or {}
                usage = UsageRecord(
                    model=actual_model,
                    provider=raw.get("provider"),
                    prompt_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
                    reasoning_tokens=int(details.get("reasoning_tokens", 0) or 0),
                    cost=float(usage_raw.get("cost", 0.0) or 0.0),
                    latency_seconds=time.perf_counter() - start,
                )
                return LLMResponse(data=data, usage=usage, raw=raw)
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                RuntimeError,
            ) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"OpenRouter request failed: {last_error}")


def _decode_json_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise TypeError("The model response must be a JSON object")
    return payload
