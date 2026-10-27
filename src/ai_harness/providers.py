"""Real model providers with explicit configuration and bounded network calls."""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict

from ai_harness.runtime import ProviderResponse, ProviderToolCall


class ProviderError(RuntimeError):
    """Base error for configured model provider failures."""


class ProviderUnavailableError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderOutputError(ProviderError):
    pass


class ProviderHealth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str
    model: str
    available: bool
    model_available: bool
    error: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    duration_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class StructuredProvider(Protocol):
    provider_id: str
    model: str

    async def generate_json(
        self, *, system: str, prompt: str, schema: dict[str, Any]
    ) -> GenerationResult: ...

    async def health(self) -> ProviderHealth: ...


class OllamaProvider:
    """Ollama chat adapter; refuses non-local endpoints and never falls back."""

    provider_id = "ollama"
    mode = "real"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama must use a local HTTP(S) endpoint.")
        if not model.strip():
            raise ValueError("Ollama model must not be blank.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Provider timeout must be positive.")
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with build_opener(ProxyHandler({})).open(
                request, timeout=self.timeout_seconds
            ) as response:
                body = response.read(4 * 1024 * 1024 + 1)
        except (TimeoutError, socket.timeout):
            raise ProviderTimeoutError("Ollama request timed out.") from None
        except HTTPError as exc:
            if exc.code == 404:
                raise ProviderUnavailableError(
                    "Ollama endpoint or configured model was not found."
                ) from None
            raise ProviderError(f"Ollama returned HTTP {exc.code}.") from None
        except (URLError, OSError):
            raise ProviderUnavailableError(
                "Ollama is unavailable at the configured local endpoint."
            ) from None
        if len(body) > 4 * 1024 * 1024:
            raise ProviderOutputError("Ollama response exceeded the size limit.")
        try:
            result = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderOutputError("Ollama returned invalid JSON.") from None
        if not isinstance(result, dict):
            raise ProviderOutputError("Ollama returned an invalid response.")
        if "error" in result:
            detail = str(result["error"]).casefold()
            if "model" in detail and ("not found" in detail or "pull" in detail):
                raise ProviderUnavailableError(
                    "The configured Ollama model is not installed."
                )
            raise ProviderError("Ollama generation failed.")
        return result

    async def health(self) -> ProviderHealth:
        try:
            response = await asyncio.to_thread(self._request, "/api/tags")
            models = response.get("models")
            if not isinstance(models, list):
                raise ProviderOutputError("Ollama returned an invalid model catalog.")
            installed = any(
                isinstance(item, dict)
                and item.get("name") == self.model
                for item in models
            )
            return ProviderHealth(
                provider_id=self.provider_id,
                model=self.model,
                available=True,
                model_available=installed,
                error=None if installed else "Configured model is not installed.",
            )
        except ProviderError as exc:
            return ProviderHealth(
                provider_id=self.provider_id,
                model=self.model,
                available=False,
                model_available=False,
                error=str(exc),
            )

    async def generate_json(
        self, *, system: str, prompt: str, schema: dict[str, Any]
    ) -> GenerationResult:
        started = time.monotonic()
        response = await asyncio.to_thread(
            self._request,
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "format": schema,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 2048},
            },
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderOutputError("Ollama returned no structured text.")
        return GenerationResult(
            text=message["content"],
            duration_ms=(time.monotonic() - started) * 1000,
            input_tokens=_optional_int(response.get("prompt_eval_count")),
            output_tokens=_optional_int(response.get("eval_count")),
        )

    async def complete(
        self,
        *,
        task: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
    ) -> ProviderResponse:
        request_messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": task},
        ]
        for message in messages:
            if message["role"] == "assistant" and "tool_call" in message:
                tool_call = message["tool_call"]
                request_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                            },
                        }],
                    }
                )
                continue
            entry: dict[str, Any] = {
                "role": message["role"],
                "content": _message_content(
                    message.get("content", message.get("result", ""))
                ),
            }
            if message.get("name"):
                entry["tool_name"] = message["name"]
            request_messages.append(entry)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 2048},
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item["description"],
                        "parameters": item["parameters"],
                    },
                }
                for item in tools
            ]
        response = await asyncio.to_thread(self._request, "/api/chat", request)
        message = response.get("message")
        if not isinstance(message, dict):
            raise ProviderOutputError("Ollama returned an invalid chat response.")
        calls = message.get("tool_calls", [])
        if calls:
            if not isinstance(calls, list) or len(calls) != 1:
                raise ProviderOutputError("Ollama returned an unsupported tool call set.")
            function = calls[0].get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    raise ProviderOutputError(
                        "Ollama returned invalid tool arguments."
                    ) from None
            if not isinstance(arguments, dict):
                raise ProviderOutputError("Ollama returned invalid tool arguments.")
            return ProviderResponse(
                tool_call=ProviderToolCall(
                    name=function.get("name", ""),
                    arguments=arguments,
                )
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderOutputError("Ollama returned no response text.")
        return ProviderResponse(text=content)


class AnthropicProvider:
    """Optional adapter using the official SDK and ANTHROPIC_API_KEY."""

    provider_id = "anthropic"
    mode = "real"

    def __init__(self, *, model: str, timeout_seconds: float = 120.0) -> None:
        if not model.strip():
            raise ValueError("Anthropic model must be configured.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderUnavailableError("ANTHROPIC_API_KEY is not configured.")
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ProviderUnavailableError(
                "Install ai-harness[anthropic] to use the Anthropic provider."
            ) from None
        self.model = model.strip()
        self._client = AsyncAnthropic(timeout=timeout_seconds)

    async def health(self) -> ProviderHealth:
        try:
            response = await self._client.models.list(limit=100)
        except Exception as exc:
            return ProviderHealth(
                provider_id=self.provider_id,
                model=self.model,
                available=False,
                model_available=False,
                error=f"Anthropic health check failed ({type(exc).__name__}).",
            )
        available_models = {item.id for item in response.data}
        model_available = self.model in available_models
        return ProviderHealth(
            provider_id=self.provider_id,
            model=self.model,
            available=True,
            model_available=model_available,
            error=None if model_available else "Configured model is not available.",
        )

    async def generate_json(
        self, *, system: str, prompt: str, schema: dict[str, Any]
    ) -> GenerationResult:
        started = time.monotonic()
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Return only JSON conforming to this schema:\n"
                        f"{json.dumps(schema, separators=(',', ':'))}\n\n{prompt}"
                    ),
                }],
            )
        except Exception as exc:
            if type(exc).__name__ in {"APITimeoutError", "TimeoutError"}:
                raise ProviderTimeoutError("Anthropic request timed out.") from None
            raise ProviderError(
                f"Anthropic generation failed ({type(exc).__name__})."
            ) from None
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        if not text:
            raise ProviderOutputError("Anthropic returned no structured text.")
        usage = response.usage
        return GenerationResult(
            text=text,
            duration_ms=(time.monotonic() - started) * 1000,
            input_tokens=_optional_int(usage.input_tokens),
            output_tokens=_optional_int(usage.output_tokens),
        )

    async def complete(
        self,
        *,
        task: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
    ) -> ProviderResponse:
        if tools or any(message["role"] == "tool" for message in messages):
            raise ProviderError(
                "Anthropic adapter does not support runtime tool execution."
            )
        request_messages: list[dict[str, Any]] = []
        for message in messages:
            request_messages.append(
                {
                    "role": message["role"],
                    "content": _message_content(message.get("content")),
                }
            )
        if not request_messages:
            request_messages.append({"role": "user", "content": task})
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": instructions,
            "messages": request_messages,
        }
        try:
            response = await self._client.messages.create(**request)
        except Exception as exc:
            if type(exc).__name__ in {"APITimeoutError", "TimeoutError"}:
                raise ProviderTimeoutError("Anthropic request timed out.") from None
            raise ProviderError(
                f"Anthropic generation failed ({type(exc).__name__})."
            ) from None
        tool_calls = [
            item for item in response.content
            if getattr(item, "type", None) == "tool_use"
        ]
        if tool_calls:
            if len(tool_calls) != 1:
                raise ProviderOutputError(
                    "Anthropic returned an unsupported tool call set."
                )
            call = tool_calls[0]
            return ProviderResponse(
                tool_call=ProviderToolCall(
                    name=call.name,
                    arguments=dict(call.input),
                )
            )
        text = "".join(
            item.text for item in response.content
            if getattr(item, "type", None) == "text"
        )
        if not text:
            raise ProviderOutputError("Anthropic returned no response text.")
        return ProviderResponse(text=text)


def provider_from_environment(
    provider_name: str | None = None,
) -> StructuredProvider:
    selected = (provider_name or os.getenv("AI_HARNESS_PROVIDER", "ollama")).strip()
    model = os.getenv("AI_HARNESS_MODEL", "").strip()
    if selected == "ollama":
        return OllamaProvider(
            model=model or "qwen3.8:27b",
            base_url=os.getenv("AI_HARNESS_OLLAMA_URL", "http://127.0.0.1:11434"),
            timeout_seconds=float(os.getenv("AI_HARNESS_TIMEOUT_SECONDS", "120")),
        )
    if selected == "anthropic":
        if not model:
            raise ValueError("Set AI_HARNESS_MODEL for the Anthropic provider.")
        return AnthropicProvider(
            model=model,
            timeout_seconds=float(os.getenv("AI_HARNESS_TIMEOUT_SECONDS", "120")),
        )
    raise ValueError(
        "AI_HARNESS_PROVIDER must be 'ollama' or 'anthropic'; "
        "simulated mode is selected explicitly in the development CLI."
    )


def _message_content(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _optional_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None
