import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from ai_harness import (
    ActionPendingApprovalError,
    AgentCancelledError,
    AgentResult,
    AgentState,
    AgentTask,
    Execution,
    ExecutionState,
    InMemoryAuditLog,
    IntegrationDefinition,
    IntegrationCallError,
    IntegrationPermission,
    IntegrationPermissionError,
    IntegrationRegistry,
    IntegrationResponseLimitError,
    IntegrationTimeoutError,
    IntegrationRuntime,
    IntegrationState,
    IntegrationUnavailableError,
    InvalidToolArgumentsError,
    MCPClient,
    MCPTool,
    SkillConflictError,
    SkillRegistry,
    SkillRegistryError,
    ToolAction,
    ToolApprovalAuthority,
)


class MockTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.delay = 0
        self.result: Any = {"ok": True}
        self.failure: str | None = None

    async def list_tools(self) -> Sequence[MCPTool]:
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if self.delay:
            await asyncio.sleep(self.delay)
        return (
            MCPTool(
                name="list_notes",
                description="Untrusted mock description",
                input_schema={
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "maximum": 100}},
                    "required": ["limit"],
                    "additionalProperties": False,
                },
            ),
            MCPTool(
                name="create_note",
                description="Create a note",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "body": {"type": "string"},
                    },
                    "required": ["title", "body"],
                    "additionalProperties": False,
                },
            ),
        )

    async def call_tool(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> Any:
        self.calls.append((tool_name, dict(arguments)))
        return self.result


class MockAgent:
    agent_id = "mock-agent"
    capabilities = frozenset({"summarize"})
    state = AgentState.READY

    def __init__(self, delay: float = 0) -> None:
        self.calls = 0
        self.delay = delay

    async def execute(
        self, task: AgentTask, *, cancellation: asyncio.Event
    ) -> AgentResult:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return AgentResult(output=f"processed: {task.input}")


class MockPlatform:
    def __init__(self) -> None:
        self.reads = 0
        self.writes = 0
        self.authentications = 0

    async def authenticate(self) -> None:
        self.authentications += 1

    async def read(self, resource: str, query: Mapping[str, Any]) -> Any:
        self.reads += 1
        return {"items": []}

    async def write(self, resource: str, payload: Mapping[str, Any]) -> Any:
        self.writes += 1
        return {"written": True}

    async def execute_action(
        self, action: str, arguments: Mapping[str, Any]
    ) -> Any:
        return {"executed": action}


def _runtime() -> tuple[
    IntegrationRuntime, IntegrationRegistry, ToolApprovalAuthority, InMemoryAuditLog
]:
    registry = IntegrationRegistry()
    definition = IntegrationDefinition.create(
        integration_id="mock",
        resources=(
            "mcp.tools.list",
            "list_notes",
            "create_note",
            "agent.execute.mock-agent",
            "platform.authenticate",
            "records",
            "records.write",
        ),
        permissions=(
            IntegrationPermission(resource="mcp.tools.list", action=ToolAction.READ),
            IntegrationPermission(resource="list_notes", action=ToolAction.READ),
            IntegrationPermission(resource="create_note", action=ToolAction.WRITE),
            IntegrationPermission(
                resource="agent.execute.mock-agent", action=ToolAction.EXTERNAL
            ),
            IntegrationPermission(
                resource="platform.authenticate", action=ToolAction.EXTERNAL
            ),
            IntegrationPermission(resource="records", action=ToolAction.READ),
            IntegrationPermission(
                resource="records.write", action=ToolAction.WRITE
            ),
        ),
        configuration={"endpoint": "https://example.test"},
        credential_ref="mock/credentials",
    )
    registry.register(definition)
    registry.activate("mock")
    registry.mark_connected("mock")
    authority = ToolApprovalAuthority()
    audit = InMemoryAuditLog()
    runtime = IntegrationRuntime(
        registry=registry, approval_authority=authority, audit_sink=audit
    )
    return runtime, registry, authority, audit


def _running_execution() -> Execution:
    return Execution.create().transition(ExecutionState.RUNNING)


def test_integrations_are_disabled_until_explicitly_activated() -> None:
    runtime, registry, _, _ = _runtime()
    registry.deactivate("mock")
    assert registry.get("mock").state is IntegrationState.DISABLED

    with pytest.raises(IntegrationUnavailableError):
        asyncio.run(
            runtime.list_mcp_tools(
                _running_execution(),
                integration_id="mock",
                client=MCPClient(MockTransport()),
            )
        )


def test_new_integrations_start_disabled() -> None:
    registry = IntegrationRegistry()
    registry.register(
        IntegrationDefinition.create(
            integration_id="new",
            resources=("mcp.tools.list",),
            permissions=(
                IntegrationPermission(
                    resource="mcp.tools.list", action=ToolAction.READ
                ),
            ),
        )
    )
    runtime = IntegrationRuntime(registry=registry)

    assert registry.get("new").state is IntegrationState.DISABLED
    with pytest.raises(IntegrationUnavailableError):
        asyncio.run(
            runtime.list_mcp_tools(
                _running_execution(),
                integration_id="new",
                client=MCPClient(MockTransport()),
            )
        )


def test_successful_mcp_discovery_connects_an_activated_integration() -> None:
    runtime, registry, _, _ = _runtime()
    registry.deactivate("mock")
    registry.activate("mock")
    assert registry.get("mock").state is IntegrationState.CONFIGURED

    tools = asyncio.run(
        runtime.list_mcp_tools(
            _running_execution(),
            integration_id="mock",
            client=MCPClient(MockTransport()),
        )
    )

    assert [tool.name for tool in tools] == ["list_notes", "create_note"]
    assert registry.get("mock").state is IntegrationState.CONNECTED


def test_platform_authentication_is_approved_before_it_connects() -> None:
    runtime, registry, authority, _ = _runtime()
    registry.deactivate("mock")
    registry.activate("mock")
    execution = _running_execution()
    platform = MockPlatform()
    request = runtime.create_request(
        execution,
        integration_id="mock",
        resource="platform.authenticate",
        arguments={},
        allow_configured=True,
    )

    with pytest.raises(ActionPendingApprovalError):
        asyncio.run(
            runtime.authenticate_platform(
                execution,
                integration_id="mock",
                adapter=platform,
                request=request,
            )
        )
    assert platform.authentications == 0
    assert registry.get("mock").state is IntegrationState.CONFIGURED

    asyncio.run(
        runtime.authenticate_platform(
            execution,
            integration_id="mock",
            adapter=platform,
            request=request,
            approval=authority.approve(request, approved_by="operator"),
        )
    )
    assert platform.authentications == 1
    assert registry.get("mock").state is IntegrationState.CONNECTED


def test_credentials_are_referenced_not_stored_in_configuration() -> None:
    with pytest.raises(ValueError, match="credential provider"):
        IntegrationDefinition.create(
            integration_id="unsafe",
            resources=(),
            permissions=(),
            configuration={"access_token": "secret-value"},
        )


def test_configuration_and_agent_context_are_defensive_snapshots() -> None:
    settings = {"endpoint": "https://safe.example.test"}
    definition = IntegrationDefinition.create(
        integration_id="snapshot",
        resources=(),
        permissions=(),
        configuration=settings,
    )
    settings["endpoint"] = "https://changed.example.test"
    exposed_settings = definition.configuration
    exposed_settings["endpoint"] = "https://mutated.example.test"

    original_context = {"scope": ["read"]}
    task = AgentTask(input="inspect", context=original_context)
    original_context["scope"].append("write")
    exposed_context = task.context
    exposed_context["scope"].append("external")

    assert definition.configuration == {"endpoint": "https://safe.example.test"}
    assert task.context == {"scope": ["read"]}


def test_skill_discovery_reads_metadata_without_running_scripts(tmp_path) -> None:
    skill_dir = tmp_path / "summarize"
    skill_dir.mkdir()
    marker = tmp_path / "executed.txt"
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: summarize\n"
        "description: Summarize a document\n"
        "version: 1.2.0\n"
        "---\n"
        "Instructions are untrusted content.\n",
        encoding="utf-8",
    )
    (skill_dir / "script.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    registry = SkillRegistry({"local": tmp_path})

    skills = registry.discover()

    assert len(skills) == 1
    assert registry.lookup("summarize").version == "1.2.0"
    assert registry.lookup("summarize").origin == "local"
    assert not marker.exists()


def test_skill_discovery_rejects_conflicting_names_and_invalid_metadata(tmp_path) -> None:
    for directory in ("one", "two"):
        skill_dir = tmp_path / directory
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: same\ndescription: A skill\n---\n",
            encoding="utf-8",
        )
    with pytest.raises(SkillConflictError):
        SkillRegistry({"local": tmp_path}).discover()

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    (invalid_root / "SKILL.md").write_text(
        "---\nname: missing-description\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillRegistryError):
        SkillRegistry({"invalid": invalid_root}).discover()


def test_mcp_write_is_blocked_until_exact_request_is_approved() -> None:
    runtime, _, authority, audit = _runtime()
    transport = MockTransport()
    client = MCPClient(transport)
    execution = _running_execution()
    request = asyncio.run(
        runtime.prepare_mcp_tool_call(
            execution,
            integration_id="mock",
            client=client,
            tool_name="create_note",
            arguments={"title": "hello", "body": "safe"},
        )
    )

    with pytest.raises(ActionPendingApprovalError):
        asyncio.run(
            runtime.execute_mcp_tool(
                execution,
                integration_id="mock",
                client=client,
                request=request,
            )
        )
    assert transport.calls == []

    approval = authority.approve(request, approved_by="operator")
    result = asyncio.run(
        runtime.execute_mcp_tool(
            execution,
            integration_id="mock",
            client=client,
            request=request,
            approval=approval,
        )
    )

    assert result == {"ok": True}
    assert transport.calls == [
        ("create_note", {"title": "hello", "body": "safe"})
    ]
    assert all(not hasattr(event, "arguments") for event in audit.events)
    assert audit.events[-1].outcome == "succeeded"


def test_mcp_approval_cannot_be_reused_after_arguments_change() -> None:
    runtime, _, authority, _ = _runtime()
    transport = MockTransport()
    client = MCPClient(transport)
    execution = _running_execution()
    request = asyncio.run(
        runtime.prepare_mcp_tool_call(
            execution,
            integration_id="mock",
            client=client,
            tool_name="create_note",
            arguments={"title": "hello", "body": "approved"},
        )
    )
    approval = authority.approve(request, approved_by="operator")
    modified_request = request.model_copy(
        update={
            "arguments_json": '{"body":"changed","title":"hello"}',
        }
    )

    with pytest.raises(ActionPendingApprovalError):
        asyncio.run(
            runtime.execute_mcp_tool(
                execution,
                integration_id="mock",
                client=client,
                request=modified_request,
                approval=approval,
            )
        )
    assert transport.calls == []


def test_approval_cannot_cross_execution_boundaries() -> None:
    runtime, _, authority, _ = _runtime()
    transport = MockTransport()
    client = MCPClient(transport)
    first_execution = _running_execution()
    request = asyncio.run(
        runtime.prepare_mcp_tool_call(
            first_execution,
            integration_id="mock",
            client=client,
            tool_name="create_note",
            arguments={"title": "hello", "body": "approved"},
        )
    )
    approval = authority.approve(request, approved_by="operator")

    with pytest.raises(IntegrationPermissionError):
        asyncio.run(
            runtime.execute_mcp_tool(
                _running_execution(),
                integration_id="mock",
                client=client,
                request=request,
                approval=approval,
            )
        )
    assert transport.calls == []


def test_mcp_schema_is_validated_before_tool_execution() -> None:
    runtime, _, _, _ = _runtime()
    transport = MockTransport()
    client = MCPClient(transport)
    execution = _running_execution()

    with pytest.raises(InvalidToolArgumentsError):
        asyncio.run(
            runtime.prepare_mcp_tool_call(
                execution,
                integration_id="mock",
                client=client,
                tool_name="list_notes",
                arguments={"limit": "many"},
            )
        )
    assert transport.calls == []


def test_mcp_timeouts_are_typed_and_do_not_expose_transport_errors() -> None:
    transport = MockTransport()
    transport.delay = 0.05
    runtime, _, _, _ = _runtime()
    client = MCPClient(transport, timeout_seconds=0.001)

    with pytest.raises(IntegrationTimeoutError):
        asyncio.run(
            runtime.list_mcp_tools(
                _running_execution(), integration_id="mock", client=client
            )
        )


def test_mcp_transport_errors_do_not_expose_secret_details() -> None:
    runtime, _, _, _ = _runtime()
    transport = MockTransport()
    transport.failure = "provider token: do-not-leak"

    with pytest.raises(IntegrationCallError) as error:
        asyncio.run(
            runtime.list_mcp_tools(
                _running_execution(),
                integration_id="mock",
                client=MCPClient(transport),
            )
        )
    assert "do-not-leak" not in str(error.value)


def test_mcp_response_size_is_bounded_and_audited() -> None:
    runtime, _, _, audit = _runtime()
    transport = MockTransport()
    transport.result = {"data": "x" * 2000}
    client = MCPClient(transport, max_response_bytes=1024)
    execution = _running_execution()
    request = asyncio.run(
        runtime.prepare_mcp_tool_call(
            execution,
            integration_id="mock",
            client=client,
            tool_name="list_notes",
            arguments={"limit": 1},
        )
    )

    with pytest.raises(IntegrationResponseLimitError):
        asyncio.run(
            runtime.execute_mcp_tool(
                execution,
                integration_id="mock",
                client=client,
                request=request,
            )
        )
    assert transport.calls == [("list_notes", {"limit": 1})]
    assert audit.events[-1].outcome == "failed"


def test_agent_execution_also_requires_execution_bound_approval() -> None:
    runtime, _, authority, _ = _runtime()
    execution = _running_execution()
    agent = MockAgent()
    task = AgentTask(input="summarize", context={"source": "test"})
    request = runtime.create_request(
        execution,
        integration_id="mock",
        resource="agent.execute.mock-agent",
        arguments={
            "task_id": str(task.task_id),
            "input": task.input,
            "context": task.context,
        },
    )

    with pytest.raises(ActionPendingApprovalError):
        asyncio.run(
            runtime.execute_agent_task(
                execution,
                integration_id="mock",
                adapter=agent,
                task=task,
            )
        )
    assert agent.calls == 0

    result = asyncio.run(
        runtime.execute_agent_task(
            execution,
            integration_id="mock",
            adapter=agent,
            task=task,
            request=request,
            approval=authority.approve(request, approved_by="operator"),
        )
    )
    assert result.output == "processed: summarize"
    assert agent.calls == 1


def test_agent_timeouts_and_cancellation_are_enforced() -> None:
    runtime, registry, authority, audit = _runtime()
    execution = _running_execution()
    task = AgentTask(input="slow task")
    agent = MockAgent(delay=0.05)
    limited_runtime = IntegrationRuntime(
        registry=registry,
        approval_authority=authority,
        audit_sink=audit,
        timeout_seconds=0.001,
    )
    request = limited_runtime.create_request(
        execution,
        integration_id="mock",
        resource="agent.execute.mock-agent",
        arguments={
            "task_id": str(task.task_id),
            "input": task.input,
            "context": task.context,
        },
    )
    approval = authority.approve(request, approved_by="operator")

    with pytest.raises(IntegrationTimeoutError):
        asyncio.run(
            limited_runtime.execute_agent_task(
                execution,
                integration_id="mock",
                adapter=agent,
                task=task,
                request=request,
                approval=approval,
            )
        )
    assert agent.calls == 1

    async def cancel_agent() -> None:
        cancellation = asyncio.Event()
        task = AgentTask(input="cancel me")
        request = runtime.create_request(
            execution,
            integration_id="mock",
            resource="agent.execute.mock-agent",
            arguments={
                "task_id": str(task.task_id),
                "input": task.input,
                "context": task.context,
            },
        )
        call = asyncio.create_task(
            runtime.execute_agent_task(
                execution,
                integration_id="mock",
                adapter=agent,
                task=task,
                request=request,
                approval=authority.approve(request, approved_by="operator"),
                cancellation=cancellation,
            )
        )
        await asyncio.sleep(0)
        cancellation.set()
        with pytest.raises(AgentCancelledError):
            await call

    asyncio.run(cancel_agent())


def test_platform_read_uses_runtime_and_write_waits_for_approval() -> None:
    runtime, _, authority, _ = _runtime()
    execution = _running_execution()
    platform = MockPlatform()
    read_request = runtime.create_platform_request(
        execution,
        integration_id="mock",
        resource="records",
        arguments={"limit": 3},
    )
    read_result = asyncio.run(
        runtime.execute_platform_request(
            execution,
            integration_id="mock",
            adapter=platform,
            request=read_request,
        )
    )
    assert read_result == {"items": []}
    assert platform.reads == 1

    write_request = runtime.create_request(
        execution,
        integration_id="mock",
        resource="records.write",
        arguments={"record": "new"},
    )
    with pytest.raises(ActionPendingApprovalError):
        asyncio.run(
            runtime.execute_platform_request(
                execution,
                integration_id="mock",
                adapter=platform,
                request=write_request,
            )
        )
    assert platform.writes == 0

    result = asyncio.run(
        runtime.execute_platform_request(
            execution,
            integration_id="mock",
            adapter=platform,
            request=write_request,
            approval=authority.approve(write_request, approved_by="operator"),
        )
    )
    assert result == {"written": True}
    assert platform.writes == 1
