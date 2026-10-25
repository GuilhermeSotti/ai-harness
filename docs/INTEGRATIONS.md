# Extending integrations

`ai-harness` keeps provider-specific networking outside its core. It defines
contracts for agent adapters, MCP transports, and external platforms; an
application supplies those implementations and deliberately activates each
integration. Nothing is installed or connected during package initialization.

## Registering and activating an integration

Declare the resources and the one allowed action for each resource. A resource
cannot be declared as both read and write: the policy classification must not
come from an untrusted server description.

```python
from ai_harness import (
    IntegrationDefinition,
    IntegrationPermission,
    IntegrationRegistry,
    ToolAction,
)

definition = IntegrationDefinition.create(
    integration_id="notes",
    resources=("mcp.tools.list", "find_notes", "create_note"),
    permissions=(
        IntegrationPermission(resource="mcp.tools.list", action=ToolAction.READ),
        IntegrationPermission(resource="find_notes", action=ToolAction.READ),
        IntegrationPermission(resource="create_note", action=ToolAction.WRITE),
    ),
    configuration={"endpoint": "https://notes.example.test"},
    credential_ref="notes/service-account",
)
registry = IntegrationRegistry()
registry.register(definition)  # state: disabled
registry.activate("notes")  # state: configured; calls are still limited
```

Configuration is for non-secret settings. Put credential values in a secret
store and keep only a reference in `credential_ref`; inject a
`CredentialProvider` into the concrete transport or adapter. Never put secrets
in task input, skill metadata, tool descriptions, audit events, or exceptions.
Only mark a connection verified after a successful, policy-approved
authentication or discovery handshake. The runtime marks MCP discovery and
platform authentication as connected only after that operation succeeds.
`deactivate()` immediately returns an integration to `disabled`.

## MCP transport

Implement `MCPTransport.list_tools()` and `MCPTransport.call_tool()` using the
chosen MCP library. Supply that transport to `MCPClient`, then use the runtime
for discovery, preparation, and execution. Do not call the transport directly
from an agent.

```python
from ai_harness import (
    ActionPendingApprovalError,
    Execution,
    ExecutionState,
    IntegrationRuntime,
    MCPClient,
)

execution = Execution.create().transition(ExecutionState.RUNNING)
client = MCPClient(my_transport, timeout_seconds=10, max_response_bytes=1_048_576)
runtime = IntegrationRuntime(registry=registry)

async def run():
    tools = await runtime.list_mcp_tools(
        execution, integration_id="notes", client=client
    )
    request = await runtime.prepare_mcp_tool_call(
        execution,
        integration_id="notes",
        client=client,
        tool_name="create_note",
        arguments={"title": "Example", "body": "Content"},
    )

    # No transport call occurs until the exact request receives human approval.
    try:
        await runtime.execute_mcp_tool(
            execution, integration_id="notes", client=client, request=request
        )
    except ActionPendingApprovalError:
        # A trusted application approval flow presents the tool, exact arguments,
        # and execution ID to an operator before issuing a grant.
        grant = runtime.approval_authority.approve(request, approved_by="operator")
        await runtime.execute_mcp_tool(
            execution,
            integration_id="notes",
            client=client,
            request=request,
            approval=grant,
        )
```

The snippet is illustrative; it does not contact a service. Do not call
`approve()` until a trusted human approval interaction has completed. The
approval is signed against the request ID, integration/tool name, action,
canonical argument digest, and execution ID. Any change requires a new request
and approval. The included authority is in-memory only and is not an
authentication or approval UI.

MCP tool schemas are validated against a bounded JSON Schema subset before
each invocation; unsupported schema keywords fail closed. Descriptions and
instructions from servers are untrusted data, not system instructions. Tool
catalog size, schema size, argument size, response size, and call duration are
limited. Calls are not retried automatically. Audit events include IDs,
operation, action, outcome, and an argument digest—not argument values,
credentials, or response content.

## Agent and external platform contracts

Implement `AgentAdapter` as an adapter that reports its identity, declared
capabilities, and state, and returns an `AgentResult` from `execute()`.
`IntegrationRuntime.execute_agent_task()` requires the registered
`agent.execute.<agent-id>` external permission and a request-bound approval.
The adapter receives a cancellation event and runs with a timeout and response
limit. Agent adapters return results; they do not receive a harness tool
executor or permission to run model-supplied shell commands.

Implement `ExternalPlatformAdapter` with the operations the actual platform
supports: authentication, resource reads, writes, and external actions. These
methods intentionally do not assume a shared platform API. Call authentication
through `authenticate_platform()`; it requires external-action approval before
the adapter is invoked. Create other requests with `create_platform_request()`
and execute them with `execute_platform_request()`. Reads can be allowed by
the explicit integration permission; writes and external actions require a
valid approval each time.

## Skills

`SkillRegistry` scans only caller-specified local roots for `SKILL.md` files.
Its limited frontmatter parser requires `name` and `description`, accepts
optional `version` and `origin`, and defaults the version to `0.0.0` and origin
to the root label. It enforces a file-size limit, validates names, and rejects
duplicate names. It reads metadata only; it never imports, evaluates, or runs a
skill script. Treat skill instructions as untrusted content when presenting
them to a model.

## Testing and status

Use mocked `MCPTransport`, `AgentAdapter`, and `ExternalPlatformAdapter`
implementations in tests. Tests should verify that blocked requests make no
adapter call, that exact approvals permit only the matching operation, and
that secrets and response bodies do not enter audit events.

The registry starts every integration disabled. `configured` means explicitly
activated but not yet verified; `connected` means a successful handshake was
observed; `error` records a failed connection attempt. These states do not
claim that any third-party integration in this repository is implemented or
working. A real connection requires a provider implementation and a separately
verified integration test.
