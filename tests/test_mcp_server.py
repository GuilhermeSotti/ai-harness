import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ai_harness.mcp_server import HarnessMCPService, create_server
from ai_harness.providers import GenerationResult, ProviderTimeoutError


class MCPStubProvider:
    provider_id = "test-provider"
    model = "mcp-test-model"

    async def health(self):
        from ai_harness.providers import ProviderHealth

        return ProviderHealth(
            provider_id=self.provider_id,
            model=self.model,
            available=True,
            model_available=True,
        )

    async def generate_json(self, *, system, prompt, schema):
        if "Create a concise task plan" in system:
            output = {
                "tasks": [
                    {
                        "description": "Analyze implementation.",
                        "required_capability": "analysis",
                        "depends_on": [],
                    },
                    {
                        "description": "Review analyst conclusions independently.",
                        "required_capability": "review",
                        "depends_on": [0],
                    },
                ]
            }
        elif "independent reviewer" in system:
            output = {
                "summary": "Independent review complete.",
                "facts": [],
                "hypotheses": [],
                "recommendations": [],
                "findings": [{
                    "title": "Potential edge case",
                    "severity": "low",
                    "detail": "Review the empty input case.",
                    "evidence": ["E1"],
                }],
                "limitations": [],
            }
        else:
            output = {
                "summary": "The project contains a Python implementation.",
                "facts": [{
                    "text": "The file defines a public function.",
                    "evidence": ["E1"],
                }],
                "hypotheses": [],
                "recommendations": [],
                "findings": [],
                "limitations": [],
            }
        return GenerationResult(
            text=json.dumps(output),
            duration_ms=4,
            input_tokens=20,
            output_tokens=10,
        )


def _service(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def run():\n    return True\n", encoding="utf-8")
    return HarnessMCPService(
        provider=MCPStubProvider(),
        allowed_roots=(source,),
        database=tmp_path / "history.sqlite3",
    ), source


def test_mcp_service_plans_and_runs_real_adapters_with_persisted_history(tmp_path):
    service, source = _service(tmp_path)

    async def exercise():
        planned = await service.create_plan("Review source changes.", str(source))
        assert planned["state"] == "awaiting_approval"
        assert len(planned["tasks"]) == 2
        assert planned["tasks"][1]["dependencies"] == [
            planned["tasks"][0]["task_id"]
        ]

        completed = await service.run_agent(
            "analyst", "Inspect app behavior.", str(source)
        )
        assert completed["state"] == "completed"
        assert completed["result"]["mode"] == "unverified"
        assert completed["tasks"][0]["status"] == "completed"
        persisted = service.get_execution(completed["run_id"])
        assert persisted["tasks"][0]["result"]["summary"]
        assert persisted["tasks"][0]["result"]["provider"] == "test-provider"
        assert persisted["tasks"][0]["result"]["model"] == "mcp-test-model"

    asyncio.run(exercise())


def test_mcp_service_rejects_unallowlisted_workspace(tmp_path):
    service, _source = _service(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    try:
        asyncio.run(service.run_agent("analyst", "inspect", str(outside)))
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("MCP accepted a workspace outside its allow-list.")


def test_real_provider_failure_is_persisted_without_simulated_success(tmp_path):
    class FailingProvider(MCPStubProvider):
        async def generate_json(self, *, system, prompt, schema):
            raise ProviderTimeoutError("Ollama request timed out.")

    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("pass\n", encoding="utf-8")
    service = HarnessMCPService(
        provider=FailingProvider(),
        allowed_roots=(source,),
        database=tmp_path / "failed-history.sqlite3",
    )

    result = asyncio.run(service.run_agent("analyst", "Analyze.", str(source)))

    assert result["state"] == "failed"
    assert result["tasks"][0]["status"] == "failed"
    assert result["tasks"][0]["error"] == "ProviderTimeoutError"
    assert result["result"]["mode"] == "unverified"


def test_stdio_mcp_handshake_and_tool_discovery(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "AI_HARNESS_PROVIDER": "ollama",
            "AI_HARNESS_MODEL": "qwen3.8:27b",
            "AI_HARNESS_ALLOWED_ROOTS": str(source),
            "AI_HARNESS_DATABASE": str(tmp_path / "stdio-history.sqlite3"),
        }
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ai_harness.mcp_server"],
        env=environment,
    )

    async def exercise():
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                assert names == {
                    "harness_status",
                    "harness_list_agents",
                    "harness_create_plan",
                    "harness_run_agent",
                    "harness_get_execution",
                    "harness_review_changes",
                }
                result = await session.call_tool("harness_list_agents", {})
                assert not result.isError
                rendered = "\n".join(
                    item.text for item in result.content if hasattr(item, "text")
                )
                assert "analyst" in rendered
                assert "reviewer" in rendered

    asyncio.run(exercise())
