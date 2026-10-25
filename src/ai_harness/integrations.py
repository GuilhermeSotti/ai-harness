"""Provider-neutral integration contracts and policy-enforced runtime."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_harness.core import Execution, ExecutionState, HarnessError
from ai_harness.policies import (
    ActionPendingApprovalError,
    ActionRejectedError,
    PolicyResult,
    ToolAction,
    ToolApproval,
    ToolApprovalAuthority,
    ToolPolicy,
    ToolRequest,
)

_MAX_ARGUMENT_BYTES = 64 * 1024
_MAX_SCHEMA_BYTES = 64 * 1024
_MAX_TOOL_DESCRIPTION = 4096
_MAX_SCHEMA_DEPTH = 16


class IntegrationError(HarnessError):
    """Base exception for integration and adapter failures."""


class DuplicateIntegrationError(IntegrationError):
    pass


class IntegrationStateError(IntegrationError):
    pass


class IntegrationUnavailableError(IntegrationError):
    pass


class IntegrationPermissionError(IntegrationError):
    pass


class ToolNotFoundError(IntegrationError):
    pass


class InvalidToolArgumentsError(IntegrationError):
    pass


class IntegrationTimeoutError(IntegrationError):
    pass


class IntegrationCallError(IntegrationError):
    pass


class IntegrationResponseLimitError(IntegrationError):
    pass


class AgentCancelledError(IntegrationError):
    pass


class ExecutionNotRunningError(IntegrationError):
    pass


class SkillRegistryError(IntegrationError):
    pass


class SkillConflictError(SkillRegistryError):
    pass


class SkillNotFoundError(SkillRegistryError):
    pass


class IntegrationState(StrEnum):
    DISABLED = "disabled"
    CONFIGURED = "configured"
    CONNECTED = "connected"
    ERROR = "error"


class AgentState(StrEnum):
    READY = "ready"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class IntegrationPermission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resource: str = Field(min_length=1)
    action: ToolAction

    @field_validator("resource")
    @classmethod
    def validate_resource(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Integration resource cannot be blank.")
        return value


class IntegrationDefinition(BaseModel):
    """Non-secret configuration and the exact resources/actions an integration may use."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    integration_id: str = Field(min_length=1)
    resources: tuple[str, ...] = ()
    permissions: tuple[IntegrationPermission, ...] = ()
    configuration_json: str = Field(default="{}", alias="configuration")
    credential_ref: str | None = None

    @field_validator("integration_id")
    @classmethod
    def validate_integration_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Integration ID cannot be blank.")
        return value

    @field_validator("resources")
    @classmethod
    def validate_resources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not resource.strip() for resource in value):
            raise ValueError("Integration resources cannot be blank.")
        if len(value) != len(set(value)):
            raise ValueError("Integration resources must be unique.")
        return value

    @field_validator("permissions")
    @classmethod
    def validate_permissions(
        cls, value: tuple[IntegrationPermission, ...]
    ) -> tuple[IntegrationPermission, ...]:
        resources = [permission.resource for permission in value]
        if len(resources) != len(set(resources)):
            raise ValueError("Each integration resource must declare exactly one action.")
        return value

    @field_validator("configuration_json", mode="before")
    @classmethod
    def validate_configuration(cls, value: Any) -> str:
        if not isinstance(value, dict):
            raise ValueError("Integration configuration must be a JSON object.")
        secret_keys = {
            "access_token",
            "apikey",
            "api_key",
            "authorization",
            "client_secret",
            "credential",
            "password",
            "refresh_token",
            "secret",
            "token",
        }

        def reject_secret_fields(item: Any) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    normalized = key.casefold().replace("-", "_")
                    if normalized in secret_keys:
                        raise ValueError(
                            "Integration secrets must be stored by a credential provider."
                        )
                    reject_secret_fields(child)
            elif isinstance(item, list):
                for child in item:
                    reject_secret_fields(child)

        _canonical_json(value)
        reject_secret_fields(value)
        return _canonical_json(value)

    @property
    def configuration(self) -> dict[str, Any]:
        return json.loads(self.configuration_json)

    @field_validator("credential_ref")
    @classmethod
    def validate_credential_ref(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Credential reference cannot be blank.")
        return value

    @model_validator(mode="after")
    def validate_permission_resources(self) -> IntegrationDefinition:
        undeclared = {
            permission.resource
            for permission in self.permissions
            if permission.resource not in self.resources
        }
        if undeclared:
            raise ValueError("Every integration permission must name a declared resource.")
        return self

    @classmethod
    def create(
        cls,
        *,
        integration_id: str,
        resources: Sequence[str],
        permissions: Sequence[IntegrationPermission],
        configuration: Mapping[str, Any] | None = None,
        credential_ref: str | None = None,
    ) -> IntegrationDefinition:
        definition = cls.model_validate(
            {
                "integration_id": integration_id,
                "resources": tuple(resources),
                "permissions": tuple(permissions),
                "configuration": dict(configuration or {}),
                "credential_ref": credential_ref,
            }
        )
        return definition


class IntegrationStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    definition: IntegrationDefinition
    state: IntegrationState = IntegrationState.DISABLED


class IntegrationRegistry:
    """Stores independent integration configurations; registrations start disabled."""

    def __init__(self) -> None:
        self._entries: dict[str, IntegrationStatus] = {}

    def register(self, definition: IntegrationDefinition) -> IntegrationStatus:
        if definition.integration_id in self._entries:
            raise DuplicateIntegrationError(
                f"Integration {definition.integration_id!r} is already registered."
            )
        self._entries[definition.integration_id] = IntegrationStatus(
            definition=definition,
            state=IntegrationState.DISABLED,
        )
        return self._entries[definition.integration_id]

    def get(self, integration_id: str) -> IntegrationStatus:
        try:
            return self._entries[integration_id]
        except KeyError:
            raise IntegrationUnavailableError(
                f"Integration {integration_id!r} is not registered."
            ) from None

    def all(self) -> tuple[IntegrationStatus, ...]:
        return tuple(self._entries.values())

    def activate(self, integration_id: str) -> IntegrationStatus:
        status = self.get(integration_id)
        if status.state not in (IntegrationState.DISABLED, IntegrationState.ERROR):
            raise IntegrationStateError("Only a disabled or errored integration can activate.")
        updated = status.model_copy(update={"state": IntegrationState.CONFIGURED})
        self._entries[integration_id] = updated
        return updated

    def mark_connected(self, integration_id: str) -> IntegrationStatus:
        status = self.get(integration_id)
        if status.state is not IntegrationState.CONFIGURED:
            raise IntegrationStateError("Integration must be configured before connecting.")
        updated = status.model_copy(update={"state": IntegrationState.CONNECTED})
        self._entries[integration_id] = updated
        return updated

    def mark_error(self, integration_id: str) -> IntegrationStatus:
        status = self.get(integration_id)
        updated = status.model_copy(update={"state": IntegrationState.ERROR})
        self._entries[integration_id] = updated
        return updated

    def deactivate(self, integration_id: str) -> IntegrationStatus:
        status = self.get(integration_id)
        updated = status.model_copy(update={"state": IntegrationState.DISABLED})
        self._entries[integration_id] = updated
        return updated

    def require_permission(
        self,
        integration_id: str,
        *,
        resource: str,
        action: ToolAction,
        allow_configured: bool = False,
    ) -> IntegrationStatus:
        status = self.get(integration_id)
        if status.state is not IntegrationState.CONNECTED and not (
            allow_configured and status.state is IntegrationState.CONFIGURED
        ):
            raise IntegrationUnavailableError(
                f"Integration {integration_id!r} is not connected."
            )
        definition = status.definition
        if resource not in definition.resources or IntegrationPermission(
            resource=resource, action=action
        ) not in definition.permissions:
            raise IntegrationPermissionError(
                f"Integration {integration_id!r} is not permitted to {action} {resource!r}."
            )
        return status

    def action_for(
        self,
        integration_id: str,
        *,
        resource: str,
        allow_configured: bool = False,
    ) -> ToolAction:
        status = self.get(integration_id)
        actions = {
            permission.action
            for permission in status.definition.permissions
            if permission.resource == resource
        }
        if len(actions) != 1:
            raise IntegrationPermissionError(
                f"Resource {resource!r} must have exactly one declared action."
            )
        action = next(iter(actions))
        self.require_permission(
            integration_id,
            resource=resource,
            action=action,
            allow_configured=allow_configured,
        )
        return action


class SkillDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=2048)
    origin: str = Field(min_length=1)
    version: str = Field(min_length=1, max_length=128)
    path: Path

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None:
            raise ValueError("Skill names must use lowercase letters, digits, '.', '_' or '-'.")
        return value

    @field_validator("description", "origin", "version")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Skill metadata values cannot be blank.")
        return value.strip()


class SkillRegistry:
    """Read-only metadata discovery for local SKILL.md files; never runs scripts."""

    def __init__(
        self,
        roots: Mapping[str, str | Path] | None = None,
        *,
        max_skill_file_bytes: int = 1_048_576,
    ) -> None:
        if type(max_skill_file_bytes) is not int or max_skill_file_bytes <= 0:
            raise ValueError("Skill file limit must be positive.")
        self._roots = {
            origin: Path(path) for origin, path in (roots or {}).items()
        }
        self._max_skill_file_bytes = max_skill_file_bytes
        self._skills: dict[str, SkillDefinition] = {}

    def discover(self) -> tuple[SkillDefinition, ...]:
        discovered: list[SkillDefinition] = []
        for origin, root in self._roots.items():
            if not root.exists() or not root.is_dir():
                raise SkillRegistryError(f"Skill root {origin!r} is not a directory.")
            resolved_root = root.resolve()
            for skill_file in root.rglob("SKILL.md"):
                path = skill_file.resolve()
                if not path.is_relative_to(resolved_root):
                    raise SkillRegistryError("Skill files must remain inside their declared root.")
                if not path.is_file():
                    raise SkillRegistryError("Discovered skill metadata is not a file.")
                try:
                    if path.stat().st_size > self._max_skill_file_bytes:
                        raise SkillRegistryError("Skill metadata file exceeds its size limit.")
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    raise SkillRegistryError("Skill metadata file could not be read.") from None
                metadata = self._parse_frontmatter(content)
                discovered.append(
                    SkillDefinition(
                        name=metadata["name"],
                        description=metadata["description"],
                        version=metadata.get("version", "0.0.0"),
                        origin=metadata.get("origin", origin),
                        path=path,
                    )
                )

        pending: dict[str, SkillDefinition] = {}
        for skill in discovered:
            existing = self._skills.get(skill.name) or pending.get(skill.name)
            if existing is not None and existing.path != skill.path:
                raise SkillConflictError(
                    f"Skill name {skill.name!r} is declared by multiple paths."
                )
            pending[skill.name] = skill
        self._skills.update(pending)
        return self.all()

    def register(self, skill: SkillDefinition) -> None:
        if skill.path.name != "SKILL.md" or not skill.path.is_file():
            raise SkillRegistryError("Registered skill path must be an existing SKILL.md file.")
        existing = self._skills.get(skill.name)
        if existing is not None and existing.path != skill.path:
            raise SkillConflictError(
                f"Skill name {skill.name!r} is already registered from another path."
            )
        self._skills[skill.name] = skill

    def lookup(self, name: str) -> SkillDefinition:
        try:
            return self._skills[name]
        except KeyError:
            raise SkillNotFoundError(f"Skill {name!r} is not registered.") from None

    def all(self) -> tuple[SkillDefinition, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, str]:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillRegistryError("SKILL.md must start with a metadata frontmatter block.")
        try:
            end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
        except StopIteration:
            raise SkillRegistryError("SKILL.md metadata frontmatter is not closed.") from None

        metadata: dict[str, str] = {}
        allowed = {"name", "description", "origin", "version"}
        for line in lines[1:end]:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            key, separator, value = entry.partition(":")
            if not separator or key.strip() not in allowed:
                raise SkillRegistryError("SKILL.md contains unsupported metadata syntax.")
            key = key.strip()
            if key in metadata:
                raise SkillRegistryError(f"SKILL.md repeats metadata field {key!r}.")
            metadata[key] = value.strip().strip("\"'")
        if not metadata.get("name") or not metadata.get("description"):
            raise SkillRegistryError("SKILL.md requires name and description metadata.")
        return metadata


class AgentTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    task_id: UUID = Field(default_factory=uuid4)
    input: str
    context_json: str = Field(default="{}", alias="context")

    @field_validator("context_json", mode="before")
    @classmethod
    def validate_context(cls, value: Any) -> str:
        if not isinstance(value, dict):
            raise ValueError("Agent task context must be a JSON object.")
        return _canonical_json(value)

    @property
    def context(self) -> dict[str, Any]:
        return json.loads(self.context_json)


class AgentResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAdapter(Protocol):
    """External agents return results only; tool execution remains in the harness."""

    @property
    def agent_id(self) -> str: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    @property
    def state(self) -> AgentState: ...

    async def execute(
        self, task: AgentTask, *, cancellation: asyncio.Event
    ) -> AgentResult: ...


class MCPTool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("MCP tool name cannot be blank.")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if len(value) > _MAX_TOOL_DESCRIPTION:
            raise ValueError("MCP tool description exceeds the configured limit.")
        return value

    @field_validator("input_schema")
    @classmethod
    def validate_schema_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("MCP tool input schema must describe an object.")
        if len(_canonical_json(value).encode("utf-8")) > _MAX_SCHEMA_BYTES:
            raise ValueError("MCP tool schema exceeds the configured limit.")
        return json.loads(json.dumps(value, sort_keys=True))


class MCPTransport(Protocol):
    """Low-level transport. Authentication and credential storage belong to its implementation."""

    async def list_tools(self) -> Sequence[MCPTool]: ...

    async def call_tool(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> Any: ...


class ExternalPlatformAdapter(Protocol):
    """Platform-specific auth/read/write/action operations; no common API is assumed."""

    async def authenticate(self) -> None: ...

    async def read(self, resource: str, query: Mapping[str, Any]) -> Any: ...

    async def write(self, resource: str, payload: Mapping[str, Any]) -> Any: ...

    async def execute_action(
        self, action: str, arguments: Mapping[str, Any]
    ) -> Any: ...


class CredentialProvider(Protocol):
    """Resolves secret references without placing secret values in integration config."""

    async def get_credential(self, reference: str) -> str: ...


class MCPClient:
    """Bounded client facade used by IntegrationRuntime, not a direct tool executor."""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 1_048_576,
        max_tools: int = 256,
    ) -> None:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or type(max_response_bytes) is not int
            or max_response_bytes <= 0
            or type(max_tools) is not int
            or max_tools <= 0
        ):
            raise ValueError("MCP limits must be positive.")
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._max_tools = max_tools

    async def _list_tools(self) -> tuple[MCPTool, ...]:
        try:
            raw_tools = await asyncio.wait_for(
                self._transport.list_tools(), timeout=self._timeout_seconds
            )
        except TimeoutError:
            raise IntegrationTimeoutError("MCP tool discovery timed out.") from None
        except Exception:
            raise IntegrationCallError("MCP tool discovery failed.") from None
        if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
            raise IntegrationCallError("MCP returned an invalid tool catalog.")
        if len(raw_tools) > self._max_tools:
            raise IntegrationResponseLimitError("MCP tool catalog exceeds its limit.")
        try:
            tools = tuple(
                MCPTool.model_validate(
                    tool.model_dump(by_alias=True)
                    if isinstance(tool, MCPTool)
                    else tool
                )
                for tool in raw_tools
            )
        except (TypeError, ValueError):
            raise IntegrationCallError("MCP returned an invalid tool catalog.") from None
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise IntegrationCallError("MCP returned conflicting tool names.")
        try:
            catalog = _canonical_json(
                [tool.model_dump(mode="json") for tool in tools]
            ).encode("utf-8")
        except ValueError:
            raise IntegrationCallError("MCP returned an invalid tool catalog.") from None
        if len(catalog) > self._max_response_bytes:
            raise IntegrationResponseLimitError(
                "MCP tool catalog exceeds its configured response limit."
            )
        return tools

    async def _call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        try:
            result = await asyncio.wait_for(
                self._transport.call_tool(tool_name, dict(arguments)),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            raise IntegrationTimeoutError("MCP tool call timed out.") from None
        except Exception:
            raise IntegrationCallError("MCP tool call failed.") from None
        try:
            encoded = _canonical_json({"result": result}).encode("utf-8")
        except (TypeError, ValueError):
            raise IntegrationCallError("MCP returned a non-JSON response.") from None
        if len(encoded) > self._max_response_bytes:
            raise IntegrationResponseLimitError("MCP response exceeds its configured limit.")
        return result


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    execution_id: UUID
    request_id: UUID
    integration_id: str
    operation: str
    action: ToolAction
    outcome: str
    arguments_digest: str


class AuditSink(Protocol):
    async def record(self, event: AuditEvent) -> None: ...


class InMemoryAuditLog:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    async def record(self, event: AuditEvent) -> None:
        self._events.append(event)


class IntegrationRuntime:
    """Policy gate for all harness-mediated calls to external adapters and tools."""

    def __init__(
        self,
        *,
        registry: IntegrationRegistry,
        policy: ToolPolicy | None = None,
        approval_authority: ToolApprovalAuthority | None = None,
        audit_sink: AuditSink | None = None,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1_048_576,
    ) -> None:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or type(max_response_bytes) is not int
            or max_response_bytes <= 0
        ):
            raise ValueError("Runtime limits must be positive.")
        self.registry = registry
        self.policy = policy or ToolPolicy()
        self.approval_authority = approval_authority or ToolApprovalAuthority()
        self.audit_sink = audit_sink or InMemoryAuditLog()
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes

    def create_request(
        self,
        execution: Execution,
        *,
        integration_id: str,
        resource: str,
        arguments: Mapping[str, Any],
        allow_configured: bool = False,
    ) -> ToolRequest:
        self._require_running(execution)
        action = self.registry.action_for(
            integration_id, resource=resource, allow_configured=allow_configured
        )
        return ToolRequest.create(
            tool_name=f"{integration_id}:{resource}",
            action=action,
            arguments=arguments,
            execution_id=execution.context.execution_id,
        )

    async def list_mcp_tools(
        self,
        execution: Execution,
        *,
        integration_id: str,
        client: MCPClient,
    ) -> tuple[MCPTool, ...]:
        request = self._request_for_action(
            execution,
            integration_id=integration_id,
            resource="mcp.tools.list",
            action=ToolAction.READ,
            arguments={},
            allow_configured=True,
        )
        await self._authorize(
            execution,
            integration_id,
            "mcp.tools.list",
            request,
            None,
            allow_configured=True,
        )
        await self._audit(request, integration_id, "started")
        try:
            tools = await asyncio.wait_for(
                client._list_tools(), timeout=self.timeout_seconds
            )
        except TimeoutError:
            await self._audit(request, integration_id, "failed")
            self._mark_connection_error(integration_id)
            raise IntegrationTimeoutError("MCP tool discovery timed out.") from None
        except IntegrationError:
            await self._audit(request, integration_id, "failed")
            self._mark_connection_error(integration_id)
            raise
        try:
            self._check_response_size(
                [tool.model_dump(mode="json") for tool in tools]
            )
        except IntegrationError:
            await self._audit(request, integration_id, "failed")
            self._mark_connection_error(integration_id)
            raise
        if self.registry.get(integration_id).state is IntegrationState.CONFIGURED:
            self.registry.mark_connected(integration_id)
        await self._audit(request, integration_id, "succeeded")
        return tools

    async def prepare_mcp_tool_call(
        self,
        execution: Execution,
        *,
        integration_id: str,
        client: MCPClient,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolRequest:
        tools = await self.list_mcp_tools(
            execution, integration_id=integration_id, client=client
        )
        tool = next((candidate for candidate in tools if candidate.name == tool_name), None)
        if tool is None:
            raise ToolNotFoundError(f"MCP tool {tool_name!r} was not found.")
        self._validate_schema(arguments, tool.input_schema)
        request = self.create_request(
            execution,
            integration_id=integration_id,
            resource=tool_name,
            arguments=arguments,
        )
        if len(request.arguments_json.encode("utf-8")) > _MAX_ARGUMENT_BYTES:
            raise InvalidToolArgumentsError("MCP tool arguments exceed their size limit.")
        return request

    async def execute_mcp_tool(
        self,
        execution: Execution,
        *,
        integration_id: str,
        client: MCPClient,
        request: ToolRequest,
        approval: ToolApproval | None = None,
    ) -> Any:
        resource = self._request_resource(request, integration_id)
        await self._authorize(execution, integration_id, resource, request, approval)
        tools = await self.list_mcp_tools(
            execution, integration_id=integration_id, client=client
        )
        tool = next((candidate for candidate in tools if candidate.name == resource), None)
        if tool is None:
            raise ToolNotFoundError(f"MCP tool {resource!r} was not found.")
        arguments = request.arguments
        self._validate_schema(arguments, tool.input_schema)
        await self._audit(request, integration_id, "started")
        try:
            result = await asyncio.wait_for(
                client._call_tool(resource, arguments), timeout=self.timeout_seconds
            )
        except TimeoutError:
            await self._audit(request, integration_id, "failed")
            raise IntegrationTimeoutError("MCP tool call timed out.") from None
        except IntegrationError:
            await self._audit(request, integration_id, "failed")
            raise
        except Exception:
            await self._audit(request, integration_id, "failed")
            raise IntegrationCallError("MCP tool call failed.") from None
        try:
            self._check_response_size(result)
        except IntegrationError:
            await self._audit(request, integration_id, "failed")
            raise
        await self._audit(request, integration_id, "succeeded")
        return result

    async def authenticate_platform(
        self,
        execution: Execution,
        *,
        integration_id: str,
        adapter: ExternalPlatformAdapter,
        request: ToolRequest | None = None,
        approval: ToolApproval | None = None,
    ) -> None:
        if request is None:
            request = self.create_request(
                execution,
                integration_id=integration_id,
                resource="platform.authenticate",
                arguments={},
                allow_configured=True,
            )
        elif (
            request.tool_name != f"{integration_id}:platform.authenticate"
            or request.action is not ToolAction.EXTERNAL
            or request.arguments != {}
        ):
            raise IntegrationPermissionError(
                "Authentication request does not match this platform."
            )
        try:
            await self._invoke_platform(
                execution,
                integration_id,
                adapter,
                request,
                approval,
                allow_configured=True,
            )
        except (IntegrationCallError, IntegrationTimeoutError):
            self._mark_connection_error(integration_id)
            raise
        if self.registry.get(integration_id).state is IntegrationState.CONFIGURED:
            self.registry.mark_connected(integration_id)

    def create_platform_request(
        self,
        execution: Execution,
        *,
        integration_id: str,
        resource: str,
        arguments: Mapping[str, Any],
    ) -> ToolRequest:
        return self.create_request(
            execution,
            integration_id=integration_id,
            resource=resource,
            arguments=arguments,
        )

    async def execute_platform_request(
        self,
        execution: Execution,
        *,
        integration_id: str,
        adapter: ExternalPlatformAdapter,
        request: ToolRequest,
        approval: ToolApproval | None = None,
    ) -> Any:
        return await self._invoke_platform(
            execution, integration_id, adapter, request, approval
        )

    async def _invoke_platform(
        self,
        execution: Execution,
        integration_id: str,
        adapter: ExternalPlatformAdapter,
        request: ToolRequest,
        approval: ToolApproval | None,
        *,
        allow_configured: bool = False,
    ) -> Any:
        resource = self._request_resource(request, integration_id)
        await self._authorize(
            execution,
            integration_id,
            resource,
            request,
            approval,
            allow_configured=allow_configured,
        )
        arguments = request.arguments
        await self._audit(request, integration_id, "started")
        try:
            if resource == "platform.authenticate":
                result = await asyncio.wait_for(
                    adapter.authenticate(), timeout=self.timeout_seconds
                )
            elif request.action is ToolAction.READ:
                result = await asyncio.wait_for(
                    adapter.read(resource, arguments), timeout=self.timeout_seconds
                )
            elif request.action is ToolAction.WRITE:
                result = await asyncio.wait_for(
                    adapter.write(resource, arguments), timeout=self.timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    adapter.execute_action(resource, arguments),
                    timeout=self.timeout_seconds,
                )
        except TimeoutError:
            await self._audit(request, integration_id, "failed")
            raise IntegrationTimeoutError("Platform operation timed out.") from None
        except Exception:
            await self._audit(request, integration_id, "failed")
            raise IntegrationCallError("Platform operation failed.") from None
        try:
            self._check_response_size(result)
        except IntegrationError:
            await self._audit(request, integration_id, "failed")
            raise
        await self._audit(request, integration_id, "succeeded")
        return result

    async def execute_agent_task(
        self,
        execution: Execution,
        *,
        integration_id: str,
        adapter: AgentAdapter,
        task: AgentTask,
        request: ToolRequest | None = None,
        approval: ToolApproval | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> AgentResult:
        resource = f"agent.execute.{adapter.agent_id}"
        expected_arguments = {
            "task_id": str(task.task_id),
            "input": task.input,
            "context": task.context,
        }
        if request is None:
            request = self.create_request(
                execution,
                integration_id=integration_id,
                resource=resource,
                arguments=expected_arguments,
            )
        elif (
            request.tool_name != f"{integration_id}:{resource}"
            or request.action is not ToolAction.EXTERNAL
            or request.arguments != expected_arguments
        ):
            raise IntegrationPermissionError(
                "Agent request does not match the requested task."
            )
        await self._authorize(execution, integration_id, resource, request, approval)
        if adapter.state is not AgentState.READY:
            raise IntegrationUnavailableError("Agent adapter is not ready.")
        cancel_event = cancellation or asyncio.Event()
        if cancel_event.is_set():
            raise AgentCancelledError("Agent execution was cancelled.")

        await self._audit(request, integration_id, "started")
        operation = asyncio.create_task(
            adapter.execute(task, cancellation=cancel_event)
        )
        cancelled = asyncio.create_task(cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                (operation, cancelled),
                timeout=self.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                await self._audit(request, integration_id, "failed")
                raise IntegrationTimeoutError("Agent execution timed out.")
            if cancelled in done and cancel_event.is_set():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                await self._audit(request, integration_id, "cancelled")
                raise AgentCancelledError("Agent execution was cancelled.")
            try:
                result = operation.result()
            except asyncio.CancelledError:
                await self._audit(request, integration_id, "cancelled")
                raise AgentCancelledError("Agent execution was cancelled.") from None
            except Exception:
                await self._audit(request, integration_id, "failed")
                raise IntegrationCallError("Agent execution failed.") from None
            if not isinstance(result, AgentResult):
                await self._audit(request, integration_id, "failed")
                raise IntegrationCallError("Agent returned an invalid result.")
            try:
                self._check_response_size(result.model_dump(mode="json"))
            except IntegrationError:
                await self._audit(request, integration_id, "failed")
                raise
            await self._audit(request, integration_id, "succeeded")
            return result
        except asyncio.CancelledError:
            if not operation.done():
                operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            await self._audit(request, integration_id, "cancelled")
            raise
        finally:
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)

    def _request_for_action(
        self,
        execution: Execution,
        *,
        integration_id: str,
        resource: str,
        action: ToolAction,
        arguments: Mapping[str, Any],
        allow_configured: bool = False,
    ) -> ToolRequest:
        self._require_running(execution)
        self.registry.require_permission(
            integration_id,
            resource=resource,
            action=action,
            allow_configured=allow_configured,
        )
        return ToolRequest.create(
            tool_name=f"{integration_id}:{resource}",
            action=action,
            arguments=arguments,
            execution_id=execution.context.execution_id,
        )

    async def _authorize(
        self,
        execution: Execution,
        integration_id: str,
        resource: str,
        request: ToolRequest,
        approval: ToolApproval | None,
        *,
        allow_configured: bool = False,
    ) -> PolicyResult:
        self._require_running(execution)
        if request.execution_id != execution.context.execution_id:
            raise IntegrationPermissionError(
                "Tool request is not bound to this execution."
            )
        self.registry.require_permission(
            integration_id,
            resource=resource,
            action=request.action,
            allow_configured=allow_configured,
        )
        result = self.policy.evaluate(
            request,
            approval=approval,
            authority=self.approval_authority,
        )
        try:
            result.require_approved(authority=self.approval_authority)
        except (ActionPendingApprovalError, ActionRejectedError):
            await self._audit(request, integration_id, "blocked")
            raise
        return result

    def _mark_connection_error(self, integration_id: str) -> None:
        if self.registry.get(integration_id).state is IntegrationState.CONFIGURED:
            self.registry.mark_error(integration_id)

    async def _audit(
        self, request: ToolRequest, integration_id: str, outcome: str
    ) -> None:
        if request.execution_id is None:
            raise IntegrationPermissionError(
                "Audited tool requests must be bound to an execution."
            )
        await self.audit_sink.record(
            AuditEvent(
                execution_id=request.execution_id,
                request_id=request.request_id,
                integration_id=integration_id,
                operation=request.tool_name,
                action=request.action,
                outcome=outcome,
                arguments_digest=request.arguments_digest,
            )
        )

    def _require_running(self, execution: Execution) -> None:
        if execution.state is not ExecutionState.RUNNING:
            raise ExecutionNotRunningError(
                "Integration calls require a running execution."
            )

    @staticmethod
    def _request_resource(request: ToolRequest, integration_id: str) -> str:
        prefix = f"{integration_id}:"
        if not request.tool_name.startswith(prefix):
            raise IntegrationPermissionError(
                "Tool request does not name the requested integration."
            )
        resource = request.tool_name[len(prefix) :]
        if not resource:
            raise IntegrationPermissionError("Tool request resource cannot be blank.")
        return resource

    def _check_response_size(self, response: Any) -> None:
        try:
            encoded = _canonical_json({"response": response}).encode("utf-8")
        except (TypeError, ValueError):
            raise IntegrationCallError("Adapter returned a non-JSON response.") from None
        if len(encoded) > self.max_response_bytes:
            raise IntegrationResponseLimitError(
                "Adapter response exceeds its configured limit."
            )

    @classmethod
    def _validate_schema(
        cls,
        arguments: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> None:
        if not isinstance(arguments, Mapping):
            raise InvalidToolArgumentsError("MCP arguments must be a JSON object.")
        if len(_canonical_json(arguments).encode("utf-8")) > _MAX_ARGUMENT_BYTES:
            raise InvalidToolArgumentsError("MCP tool arguments exceed their size limit.")
        try:
            cls._validate_schema_value(dict(arguments), schema, "$", 0)
        except (TypeError, ValueError, KeyError) as exc:
            raise InvalidToolArgumentsError(str(exc)) from None

    @classmethod
    def _validate_schema_value(
        cls, value: Any, schema: Mapping[str, Any], path: str, depth: int
    ) -> None:
        if depth > _MAX_SCHEMA_DEPTH or not isinstance(schema, Mapping):
            raise ValueError(f"{path}: unsupported or deeply nested JSON schema.")
        supported = {
            "type",
            "description",
            "title",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
        }
        if set(schema) - supported:
            raise ValueError(f"{path}: JSON schema contains unsupported keywords.")
        expected = schema.get("type")
        checks: dict[str, Callable[[Any], bool]] = {
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: type(item) is int,
            "number": lambda item: (
                type(item) in (int, float) and math.isfinite(item)
            ),
            "boolean": lambda item: isinstance(item, bool),
            "null": lambda item: item is None,
        }
        if expected is not None:
            if expected not in checks:
                raise ValueError(f"{path}: unsupported JSON schema type.")
            if not checks[expected](value):
                raise ValueError(f"{path}: expected {expected}.")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path}: value is not an allowed enum member.")

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if not isinstance(properties, Mapping) or not isinstance(required, list):
                raise ValueError(f"{path}: malformed object schema.")
            if any(not isinstance(name, str) for name in required):
                raise ValueError(f"{path}: malformed required property list.")
            missing = set(required) - value.keys()
            if missing:
                raise ValueError(f"{path}: missing required properties.")
            additional = schema.get("additionalProperties", True)
            for name, item in value.items():
                if name in properties:
                    cls._validate_schema_value(
                        item, properties[name], f"{path}.{name}", depth + 1
                    )
                elif additional is False:
                    raise ValueError(f"{path}: unexpected property {name!r}.")
                elif isinstance(additional, Mapping):
                    cls._validate_schema_value(
                        item, additional, f"{path}.{name}", depth + 1
                    )
                elif not isinstance(additional, bool):
                    raise ValueError(f"{path}: malformed additionalProperties schema.")
        if isinstance(value, list) and "items" in schema:
            for index, item in enumerate(value):
                cls._validate_schema_value(
                    item, schema["items"], f"{path}[{index}]", depth + 1
                )
        bounds = {
            "minimum": lambda item, bound: item >= bound,
            "maximum": lambda item, bound: item <= bound,
            "minLength": lambda item, bound: len(item) >= bound,
            "maxLength": lambda item, bound: len(item) <= bound,
            "minItems": lambda item, bound: len(item) >= bound,
            "maxItems": lambda item, bound: len(item) <= bound,
        }
        for keyword, check in bounds.items():
            if keyword not in schema:
                continue
            if keyword in ("minimum", "maximum") and type(value) not in (int, float):
                raise ValueError(f"{path}: {keyword} requires a number.")
            if keyword.endswith("Length") and not isinstance(value, str):
                raise ValueError(f"{path}: {keyword} requires a string.")
            if keyword.endswith("Items") and not isinstance(value, list):
                raise ValueError(f"{path}: {keyword} requires an array.")
            if not check(value, schema[keyword]):
                raise ValueError(f"{path}: {keyword} constraint was not met.")


def _canonical_json(value: Any) -> str:
    def validate(item: Any, depth: int = 0) -> None:
        if depth > 64:
            raise ValueError("JSON value exceeds its nesting limit.")
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite.")
            return
        if isinstance(item, list):
            for child in item:
                validate(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings.")
                validate(child, depth + 1)
            return
        raise ValueError("Value must contain only finite JSON-compatible data.")

    try:
        validate(value)
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError):
        raise ValueError("Value must contain only finite JSON-compatible data.") from None
