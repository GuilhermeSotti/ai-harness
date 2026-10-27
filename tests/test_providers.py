import asyncio

import pytest

from ai_harness.providers import (
    OllamaProvider,
    ProviderError,
    ProviderHealth,
    ProviderTimeoutError,
    ProviderUnavailableError,
    provider_from_environment,
)


def test_ollama_health_reports_model_presence(monkeypatch):
    provider = OllamaProvider(model="qwen3.8:27b")
    monkeypatch.setattr(
        provider,
        "_request",
        lambda _path: {"models": [{"name": "qwen3.8:27b"}]},
    )

    health = asyncio.run(provider.health())

    assert health == ProviderHealth(
        provider_id="ollama",
        model="qwen3.8:27b",
        available=True,
        model_available=True,
    )


def test_ollama_health_distinguishes_missing_model(monkeypatch):
    provider = OllamaProvider(model="missing-model")
    monkeypatch.setattr(provider, "_request", lambda _path: {"models": []})

    health = asyncio.run(provider.health())

    assert health.available
    assert not health.model_available
    assert health.error == "Configured model is not installed."


def test_ollama_health_reports_unavailable_provider(monkeypatch):
    provider = OllamaProvider(model="qwen3.8:27b")

    def unavailable(_path):
        raise ProviderUnavailableError("local endpoint unavailable")

    monkeypatch.setattr(provider, "_request", unavailable)
    health = asyncio.run(provider.health())

    assert not health.available
    assert not health.model_available
    assert health.error == "local endpoint unavailable"


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (ProviderTimeoutError("timeout"), ProviderTimeoutError),
        (ProviderError("generation failed"), ProviderError),
    ],
)
def test_generation_errors_are_propagated_without_fallback(
    monkeypatch, failure, error
):
    provider = OllamaProvider(model="qwen3.8:27b")

    def fail(*_args):
        raise failure

    monkeypatch.setattr(provider, "_request", fail)
    with pytest.raises(error):
        asyncio.run(
            provider.generate_json(
                system="system",
                prompt="prompt",
                schema={"type": "object"},
            )
        )


def test_non_local_ollama_endpoint_is_rejected():
    with pytest.raises(ValueError, match="local HTTP"):
        OllamaProvider(
            model="qwen3.8:27b",
            base_url="https://example.invalid",
        )


def test_provider_selection_has_no_simulated_fallback(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_PROVIDER", "unknown-provider")

    with pytest.raises(ValueError, match="AI_HARNESS_PROVIDER"):
        provider_from_environment()


def test_ollama_complete_preserves_tool_exchange_context(monkeypatch):
    provider = OllamaProvider(model="qwen3.8:27b")
    captured = {}

    def respond(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": "Done."}}

    monkeypatch.setattr(provider, "_request", respond)
    response = asyncio.run(
        provider.complete(
            task="Inspect the project.",
            instructions="Use the authorized read tool.",
            messages=[
                {
                    "role": "assistant",
                    "tool_call": {
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    },
                },
                {
                    "role": "tool",
                    "name": "read_file",
                    "result": {"text": "source evidence"},
                },
            ],
            tools=({
                "name": "read_file",
                "description": "Read an allowed file.",
                "parameters": {"type": "object"},
            },),
        )
    )

    messages = captured["payload"]["messages"]
    assert captured["path"] == "/api/chat"
    assert messages[1] == {"role": "user", "content": "Inspect the project."}
    assert messages[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages[3]["tool_name"] == "read_file"
    assert messages[3]["content"] == '{"text": "source evidence"}'
    assert response.text == "Done."
