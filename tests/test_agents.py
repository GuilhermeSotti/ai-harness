import asyncio
import json

import pytest

from ai_harness.agents import (
    AnalystAgent,
    EvidenceCollector,
    ReviewerAgent,
)
from ai_harness.maestro import AgentTask, MaestroError
from ai_harness.providers import GenerationResult
from ai_harness.redaction import redact_sensitive_text


class StubProvider:
    provider_id = "test-provider"
    model = "deterministic-test"

    def __init__(self, *, invalid_citation=False):
        self.calls = []
        self.invalid_citation = invalid_citation

    async def generate_json(self, *, system, prompt, schema):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        citation = "E99" if self.invalid_citation else "E1"
        if "independent reviewer" in system:
            response = {
                "summary": "The analyst omitted an edge case.",
                "facts": [],
                "hypotheses": [],
                "recommendations": [],
                "findings": [{
                    "title": "Missing edge case",
                    "severity": "medium",
                    "detail": "The implementation does not validate the empty input.",
                    "evidence": [citation],
                }],
                "limitations": [],
            }
        else:
            response = {
                "summary": "The source has one validated function.",
                "facts": [{
                    "text": "The module defines a function named calculate.",
                    "evidence": [citation],
                }],
                "hypotheses": [],
                "recommendations": [],
                "findings": [],
                "limitations": [],
            }
        return GenerationResult(
            text=json.dumps(response),
            duration_ms=12.5,
            input_tokens=100,
            output_tokens=40,
        )

    async def health(self):
        raise AssertionError("Agent execution must not silently check/fallback.")


def _task(root, *, context=None, description="Inspect project code"):
    return AgentTask(
        conversation_id="a2b3c4d5-1111-4111-8111-111111111111",
        run_id="a2b3c4d5-2222-4222-8222-222222222222",
        task_id="a2b3c4d5-3333-4333-8333-333333333333",
        description=description,
        user_request="Review the implementation.",
        context={"workspace_path": str(root), **(context or {})},
    )


def test_evidence_collection_is_bounded_and_redacts_secrets(tmp_path):
    (tmp_path / "module.py").write_text(
        'API_KEY = "sk-ant-abcdefghijklmnop"\n'
        '{"access_token": "private value with spaces"}\n'
        "def calculate():\n    return 42\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("API_KEY=not-for-model", encoding="utf-8")
    (tmp_path / "credentials.txt").write_text("private", encoding="utf-8")

    evidence = EvidenceCollector((tmp_path,)).collect(tmp_path, "calculate")

    assert len(evidence.items) == 1
    assert evidence.items[0].path == "module.py"
    assert "sk-ant-" not in evidence.items[0].text
    assert "[REDACTED]" in evidence.items[0].text
    assert "private value" not in evidence.items[0].text


def test_workspace_outside_allowlist_is_rejected(tmp_path):
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()

    with pytest.raises(ValueError, match="outside"):
        EvidenceCollector((allowed,)).collect(denied, "inspect")


def test_redaction_handles_credentials_in_environment_and_json_syntax():
    redacted = redact_sensitive_text(
        'API_KEY="secret value"; {"access_token": "private value"} '
        "Bearer abc.def.ghi"
    )

    assert "secret value" not in redacted
    assert "private value" not in redacted
    assert "abc.def.ghi" not in redacted


def test_analyst_generates_evidence_citing_structured_report(tmp_path):
    (tmp_path / "module.py").write_text(
        "def calculate(value):\n    return value * 2\n", encoding="utf-8"
    )
    provider = StubProvider()
    agent = AnalystAgent(
        provider=provider,
        collector=EvidenceCollector((tmp_path,)),
    )

    result = asyncio.run(agent.execute(_task(tmp_path), cancellation=asyncio.Event()))
    report = json.loads(result.summary)

    assert report["facts"][0]["evidence"] == ["E1"]
    assert result.claims[0].startswith("fact:")
    assert result.artifact_refs[0].startswith("E1:module.py:1-")
    assert result.duration_ms == 12.5
    assert result.tokens_used == 140
    assert result.provider == "test-provider"
    assert result.model == "deterministic-test"


def test_reviewer_receives_analyst_context_and_reports_independent_finding(tmp_path):
    (tmp_path / "module.py").write_text(
        "def calculate(value):\n    if value is None:\n        return 0\n",
        encoding="utf-8",
    )
    provider = StubProvider()
    agent = ReviewerAgent(
        provider=provider,
        collector=EvidenceCollector((tmp_path,)),
    )
    task = _task(
        tmp_path,
        context={
            "dependency_results": {
                "analyst": {"summary": "Prior analyst assertion"}
            },
            "review_context": "Review recent change summary",
        },
    )

    result = asyncio.run(agent.execute(task, cancellation=asyncio.Event()))
    sent = json.loads(provider.calls[0]["prompt"])

    assert sent["dependency_results"]["analyst"]["summary"] == "Prior analyst assertion"
    assert sent["additional_context"]["review_context"] == "Review recent change summary"
    assert json.loads(result.summary)["findings"][0]["severity"] == "medium"


def test_agent_rejects_unprovided_citations(tmp_path):
    (tmp_path / "module.py").write_text("pass\n", encoding="utf-8")
    agent = AnalystAgent(
        provider=StubProvider(invalid_citation=True),
        collector=EvidenceCollector((tmp_path,)),
    )

    with pytest.raises(MaestroError, match="evidence"):
        asyncio.run(agent.execute(_task(tmp_path), cancellation=asyncio.Event()))


def test_agent_requires_evidence_for_factual_claims(tmp_path):
    class NoCitationProvider(StubProvider):
        async def generate_json(self, *, system, prompt, schema):
            return GenerationResult(
                text=json.dumps({
                    "summary": "Unsupported",
                    "facts": [{"text": "An uncited claim.", "evidence": []}],
                    "hypotheses": [],
                    "recommendations": [],
                    "findings": [],
                    "limitations": [],
                }),
                duration_ms=1,
            )

    (tmp_path / "module.py").write_text("pass\n", encoding="utf-8")
    agent = AnalystAgent(
        provider=NoCitationProvider(),
        collector=EvidenceCollector((tmp_path,)),
    )

    with pytest.raises(MaestroError, match="must cite"):
        asyncio.run(agent.execute(_task(tmp_path), cancellation=asyncio.Event()))
