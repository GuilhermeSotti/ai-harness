"""Task orchestration, safe tool dispatch, and SQLite execution history."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
import math
import os
import sqlite3
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_harness.policies import (
    ActionPendingApprovalError,
    ActionRejectedError,
    PolicyDecision,
    ToolAction,
    ToolApprovalAuthority,
    ToolPolicy,
    ToolRequest,
)
from ai_harness.redaction import redact_sensitive_text


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunError(Exception):
    """Base class for task runtime errors."""


class ToolValidationError(RunError):
    pass


class ProviderTimeoutError(RunError):
    pass


class OperationTimeoutError(RunError):
    pass


class RunCancelledError(RunError):
    pass


class ProviderToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ProviderResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str | None = None
    tool_call: ProviderToolCall | None = None

    @model_validator(mode="after")
    def validate_response(self) -> ProviderResponse:
        if (self.text is None) == (self.tool_call is None):
            raise ValueError("Provider response must contain text or one tool call.")
        return self


class ModelProvider(Protocol):
    """Provider contract: adapters return text or a proposed registered tool call."""

    @property
    def provider_id(self) -> str: ...

    @property
    def mode(self) -> str: ...

    async def complete(
        self,
        *,
        task: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
    ) -> ProviderResponse: ...


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=2048)
    action: ToolAction
    parameters: dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip() or any(char.isspace() for char in value):
            raise ValueError("Tool names must be non-blank and contain no spaces.")
        return value


ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class ToolRegistry:
    """Only explicitly registered tools can be exposed or called by the runtime."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool {definition.name!r} is already registered.")
        if not inspect.iscoroutinefunction(handler):
            raise TypeError("Tool handlers must be async so runtime timeouts can apply.")
        self._validate_schema(definition.parameters)
        self._tools[definition.name] = (
            ToolDefinition.model_validate(
                definition.model_dump(mode="python")
            ).model_copy(update={"parameters": deepcopy(definition.parameters)}),
            handler,
        )

    def get(self, name: str) -> tuple[ToolDefinition, ToolHandler]:
        try:
            definition, handler = self._tools[name]
            return definition.model_copy(
                update={"parameters": deepcopy(definition.parameters)}
            ), handler
        except KeyError:
            raise ToolValidationError("The requested tool is not authorized.") from None

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            definition.model_copy(update={"parameters": deepcopy(definition.parameters)})
            for definition, _ in self._tools.values()
        )

    def provider_tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": definition.name,
                "description": definition.description,
                "parameters": deepcopy(definition.parameters),
            }
            for definition, _ in self._tools.values()
        )

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        definition, handler = self.get(name)
        self.validate_arguments(definition, arguments)
        return await handler(dict(arguments))

    @classmethod
    def validate_arguments(
        cls, definition: ToolDefinition, arguments: dict[str, Any]
    ) -> None:
        if not isinstance(arguments, dict):
            raise ToolValidationError("Tool arguments must be a JSON object.")
        cls._validate_value(definition.parameters, arguments, "arguments")

    @classmethod
    def _validate_schema(cls, schema: dict[str, Any]) -> None:
        allowed_by_type = {
            "object": {
                "type", "properties", "required", "additionalProperties", "enum",
            },
            "string": {"type", "enum", "minLength", "maxLength"},
            "integer": {"type", "enum", "minimum", "maximum"},
            "number": {"type", "enum", "minimum", "maximum"},
            "boolean": {"type", "enum"},
            "array": {"type", "items", "enum"},
        }
        if not isinstance(schema, dict):
            raise ValueError("Tool schema contains an unsupported definition.")
        kind = schema.get("type")
        if not isinstance(kind, str) or kind not in allowed_by_type:
            raise ValueError("Tool schema contains an unsupported definition.")
        if set(schema) - allowed_by_type[kind]:
            raise ValueError("Tool schema contains an unsupported definition.")
        enum = schema.get("enum")
        if enum is not None and (not isinstance(enum, list) or not enum):
            raise ValueError("Tool enum schemas must contain a non-empty list.")
        if kind == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if (
                not isinstance(properties, dict)
                or not isinstance(required, list)
                or schema.get("additionalProperties", False) is not False
                or any(key not in properties for key in required)
            ):
                raise ValueError("Tool object schemas must be strict JSON objects.")
            for property_schema in properties.values():
                cls._validate_schema(property_schema)
        elif kind == "array":
            if "items" not in schema:
                raise ValueError("Tool array schemas must declare item types.")
            cls._validate_schema(schema["items"])
        elif kind == "string":
            minimum = schema.get("minLength", 0)
            maximum = schema.get("maxLength", float("inf"))
            if (
                type(minimum) is not int
                or minimum < 0
                or type(maximum) not in (int, float)
                or maximum < minimum
            ):
                raise ValueError("Tool string length limits are invalid.")
        elif kind in {"integer", "number"}:
            minimum = schema.get("minimum", float("-inf"))
            maximum = schema.get("maximum", float("inf"))
            if (
                type(minimum) not in (int, float)
                or type(maximum) not in (int, float)
                or not math.isfinite(minimum) and minimum != float("-inf")
                or not math.isfinite(maximum) and maximum != float("inf")
                or maximum < minimum
            ):
                raise ValueError("Tool numeric limits are invalid.")

    @classmethod
    def _validate_value(cls, schema: dict[str, Any], value: Any, path: str) -> None:
        kind = schema["type"]
        valid = {
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: type(item) is int,
            "number": lambda item: type(item) in (int, float),
            "boolean": lambda item: type(item) is bool,
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
        }[kind]
        if not valid(value):
            raise ToolValidationError(f"{path} does not match its declared type.")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolValidationError(f"{path} is not an allowed value.")
        if kind == "object":
            properties = schema.get("properties", {})
            if any(key not in properties for key in value):
                raise ToolValidationError(f"{path} contains an unauthorized field.")
            if any(key not in value for key in schema.get("required", [])):
                raise ToolValidationError(f"{path} is missing a required field.")
            for key, item in value.items():
                cls._validate_value(properties[key], item, f"{path}.{key}")
        elif kind == "array":
            for index, item in enumerate(value):
                cls._validate_value(schema["items"], item, f"{path}[{index}]")
        elif kind == "string":
            if len(value) < schema.get("minLength", 0) or len(value) > schema.get(
                "maxLength", float("inf")
            ):
                raise ToolValidationError(f"{path} is outside its length limits.")
        elif kind in {"integer", "number"}:
            if value < schema.get("minimum", float("-inf")) or value > schema.get(
                "maximum", float("inf")
            ):
                raise ToolValidationError(f"{path} is outside its numeric limits.")


def make_simulated_registry() -> ToolRegistry:
    """Return a deterministic tool that has no external or persistent side effects."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Return a simulated acknowledgement without external effects.",
            action=ToolAction.READ,
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        _simulated_echo,
    )
    return registry


class SimulatedProvider:
    """Deterministic fake provider; its output is never represented as model output."""

    provider_id = "simulated"
    mode = "simulated"

    async def complete(
        self,
        *,
        task: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
    ) -> ProviderResponse:
        _ = instructions
        if not any(message["role"] == "tool" for message in messages):
            if not any(tool["name"] == "echo" for tool in tools):
                return ProviderResponse(
                    text="Simulated run completed without an authorized tool."
                )
            return ProviderResponse(
                tool_call=ProviderToolCall(
                    name="echo", arguments={"text": task}
                )
            )
        return ProviderResponse(text="Simulated run completed; no real model was used.")


class SQLiteExecutionStore:
    """Small synchronous SQLite store; task content is persisted, secrets are not logged."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    state TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    agent_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    result TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    timestamp TEXT NOT NULL,
                    agent_id TEXT,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL,
                    tool_name TEXT,
                    summary TEXT
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    tool_name TEXT NOT NULL,
                    arguments_digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_summary TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    call_id TEXT NOT NULL REFERENCES tool_calls(call_id),
                    tool_name TEXT NOT NULL,
                    arguments_digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decided_by TEXT,
                    decided_at TEXT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_execution(
        self, execution_id: UUID, task: str, agent_id: str, provider_id: str, mode: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO executions
                (execution_id, task, state, agent_id, provider_id, mode)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (str(execution_id), task, RunState.QUEUED, agent_id, provider_id, mode),
            )

    def set_state(
        self,
        execution_id: UUID,
        state: RunState,
        *,
        result: str | None = None,
        error: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        updates = ["state = ?"]
        values: list[Any] = [state.value]
        if result is not None:
            updates.append("result = ?")
            values.append(result)
        if error is not None:
            updates.append("error = ?")
            values.append(error)
        if started:
            updates.append("started_at = ?")
            values.append(_timestamp())
        if finished:
            updates.append("finished_at = ?")
            values.append(_timestamp())
        values.append(str(execution_id))
        with self._connect() as connection:
            connection.execute(
                f"UPDATE executions SET {', '.join(updates)} WHERE execution_id = ?",
                values,
            )

    def add_event(
        self,
        *,
        execution_id: UUID,
        agent_id: str,
        event_type: str,
        status: str,
        duration_ms: float | None = None,
        tool_name: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        timestamp = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO events
                (execution_id, timestamp, agent_id, event_type, status, duration_ms,
                 tool_name, summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(execution_id), timestamp, agent_id, event_type, status,
                    duration_ms, tool_name, summary,
                ),
            )
            return {
                "sequence": cursor.lastrowid,
                "timestamp": timestamp,
                "execution_id": str(execution_id),
                "agent_id": agent_id,
                "event_type": event_type,
                "status": status,
                "duration_ms": duration_ms,
                "tool_name": tool_name,
                "summary": summary,
            }

    def add_tool_call(
        self,
        *,
        call_id: UUID,
        execution_id: UUID,
        tool_name: str,
        arguments_digest: str,
        status: str,
        result_summary: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO tool_calls
                (call_id, execution_id, tool_name, arguments_digest, status,
                 result_summary, error) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(call_id), str(execution_id), tool_name, arguments_digest,
                    status, result_summary, error,
                ),
            )

    def update_tool_call(
        self, call_id: UUID, *, status: str, result_summary: str | None = None,
        error: str | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE tool_calls SET status = ?, result_summary = ?, error = ?
                WHERE call_id = ?""",
                (status, result_summary, error, str(call_id)),
            )

    def add_approval(
        self,
        *,
        approval_id: UUID,
        execution_id: UUID,
        call_id: UUID,
        tool_name: str,
        arguments_digest: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO approvals
                (approval_id, execution_id, call_id, tool_name, arguments_digest, status)
                VALUES (?, ?, ?, ?, ?, 'pending')""",
                (
                    str(approval_id), str(execution_id), str(call_id),
                    tool_name, arguments_digest,
                ),
            )

    def decide_approval(
        self, approval_id: UUID, *, approved: bool | None, decided_by: str
    ) -> None:
        status = (
            "cancelled" if approved is None else "approved" if approved else "rejected"
        )
        with self._connect() as connection:
            connection.execute(
                """UPDATE approvals SET status = ?, decided_by = ?, decided_at = ?
                WHERE approval_id = ? AND status = 'pending'""",
                (
                    status, decided_by,
                    _timestamp(), str(approval_id),
                ),
            )

    def get(self, execution_id: UUID | str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM executions WHERE execution_id = ?", (str(execution_id),)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["events"] = [
                dict(event)
                for event in connection.execute(
                    "SELECT * FROM events WHERE execution_id = ? ORDER BY sequence",
                    (str(execution_id),),
                )
            ]
            result["tool_calls"] = [
                dict(call)
                for call in connection.execute(
                    "SELECT * FROM tool_calls WHERE execution_id = ? ORDER BY rowid",
                    (str(execution_id),),
                )
            ]
            result["approvals"] = [
                dict(approval)
                for approval in connection.execute(
                    "SELECT * FROM approvals WHERE execution_id = ? ORDER BY rowid",
                    (str(execution_id),),
                )
            ]
            return result

    def history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """SELECT execution_id, task, state, started_at, finished_at,
                    agent_id, provider_id, mode FROM executions
                    ORDER BY rowid DESC LIMIT ?""",
                    (limit,),
                )
            ]


class Harness:
    """Runs a task with a provider, authorized tools, policy, and live event observer."""

    instructions = (
        "Use only tools explicitly listed by the harness. Never propose shell commands "
        "or call tools not present in the authorized registry."
    )

    def __init__(
        self,
        *,
        store: SQLiteExecutionStore,
        provider: ModelProvider,
        tools: ToolRegistry,
        policy: ToolPolicy | None = None,
        agent_id: str = "starter-agent",
        max_iterations: int = 8,
        timeout_seconds: float = 30.0,
        observer: Callable[[dict[str, Any]], None] | None = None,
        approval_handler: Callable[[ToolRequest], Awaitable[tuple[bool, str] | None]]
        | None = None,
    ) -> None:
        if max_iterations <= 0 or timeout_seconds <= 0:
            raise ValueError("Iteration and timeout limits must be positive.")
        self.store = store
        self.provider = provider
        self.tools = tools
        self.policy = policy or ToolPolicy()
        self.agent_id = agent_id
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.observer = observer
        self.approval_handler = approval_handler
        self.approval_authority = ToolApprovalAuthority()

    def _emit(
        self,
        execution_id: UUID,
        event_type: str,
        status: str,
        *,
        duration_ms: float | None = None,
        tool_name: str | None = None,
        summary: str | None = None,
    ) -> None:
        event = self.store.add_event(
            execution_id=execution_id,
            agent_id=self.agent_id,
            event_type=event_type,
            status=status,
            duration_ms=duration_ms,
            tool_name=tool_name,
            summary=summary,
        )
        if self.observer is not None:
            self.observer(event)

    async def run(
        self, task: str, *, cancellation: asyncio.Event | None = None
    ) -> dict[str, Any]:
        task = redact_sensitive_text(task)
        if not task.strip():
            raise ValueError("Task cannot be blank.")
        execution_id = uuid4()
        cancel_event = cancellation or asyncio.Event()
        self.store.create_execution(
            execution_id, task, self.agent_id, self.provider.provider_id, self.provider.mode
        )
        self._emit(execution_id, "execution.queued", RunState.QUEUED)
        self.store.set_state(execution_id, RunState.RUNNING, started=True)
        execution_started = time.monotonic()
        self._emit(execution_id, "execution.started", RunState.RUNNING)
        messages: list[dict[str, Any]] = []
        try:
            for iteration in range(1, self.max_iterations + 1):
                await self._check_cancel(cancel_event)
                self._emit(
                    execution_id, "agent.iteration.started", RunState.RUNNING,
                    summary=f"Iteration {iteration}/{self.max_iterations}",
                )
                started = time.monotonic()
                try:
                    response = await self._controlled(
                        self.provider.complete(
                            task=task,
                            instructions=self.instructions,
                            messages=messages,
                            tools=self.tools.provider_tools(),
                        ),
                        cancel_event,
                    )
                except OperationTimeoutError as exc:
                    raise ProviderTimeoutError("Provider request timed out.") from exc
                elapsed = (time.monotonic() - started) * 1000
                self._emit(
                    execution_id, "provider.response", RunState.RUNNING,
                    duration_ms=elapsed,
                    summary=f"Provider mode: {self.provider.mode}",
                )
                if not isinstance(response, ProviderResponse):
                    raise RunError("Provider returned an invalid response.")
                if response.text is not None:
                    self.store.set_state(
                        execution_id, RunState.COMPLETED,
                        result=response.text, finished=True,
                    )
                    self._emit(
                        execution_id, "execution.completed", RunState.COMPLETED,
                        duration_ms=(time.monotonic() - execution_started) * 1000,
                        summary=f"Mode: {self.provider.mode}",
                    )
                    return self._snapshot(execution_id)

                proposed = response.tool_call
                if proposed is None:
                    raise RunError("Provider returned an invalid response.")
                definition, _ = self.tools.get(proposed.name)
                self.tools.validate_arguments(definition, proposed.arguments)
                call_id = uuid4()
                request = ToolRequest.create(
                    tool_name=proposed.name,
                    action=definition.action,
                    arguments=proposed.arguments,
                    execution_id=execution_id,
                )
                approval = None
                decision = self.policy.evaluate(request)
                if decision.decision is PolicyDecision.REJECTED:
                    self.store.add_tool_call(
                        call_id=call_id, execution_id=execution_id,
                        tool_name=proposed.name,
                        arguments_digest=request.arguments_digest,
                        status="rejected", error="ActionRejectedError",
                    )
                    self._emit(
                        execution_id, "tool.blocked", RunState.RUNNING,
                        tool_name=proposed.name,
                        summary="Blocked by runtime policy; tool was not run.",
                    )
                    raise ActionRejectedError(request.action)
                pending_call = decision.decision is PolicyDecision.PENDING
                if pending_call:
                    self.store.add_tool_call(
                        call_id=call_id, execution_id=execution_id,
                        tool_name=proposed.name,
                        arguments_digest=request.arguments_digest, status="pending",
                    )
                    approval_id = uuid4()
                    self.store.add_approval(
                        approval_id=approval_id, execution_id=execution_id,
                        call_id=call_id, tool_name=proposed.name,
                        arguments_digest=request.arguments_digest,
                    )
                    self.store.set_state(execution_id, RunState.WAITING_APPROVAL)
                    self._emit(
                        execution_id, "approval.pending", RunState.WAITING_APPROVAL,
                        duration_ms=(time.monotonic() - execution_started) * 1000,
                        tool_name=proposed.name,
                        summary=f"Approval {approval_id} required; tool was not run.",
                    )
                    if self.approval_handler is None:
                        return self._snapshot(execution_id)
                    try:
                        approval_result = await self._controlled(
                            self.approval_handler(request), cancel_event
                        )
                    except RunCancelledError:
                        self.store.update_tool_call(call_id, status="cancelled")
                        self.store.decide_approval(
                            approval_id, approved=None, decided_by="runtime"
                        )
                        raise
                    if approval_result is None:
                        return self._snapshot(execution_id)
                    approved, decided_by = approval_result
                    if approved:
                        approval = self.approval_authority.approve(
                            request, approved_by=decided_by
                        )
                        decision = self.policy.evaluate(
                            request, approval=approval,
                            authority=self.approval_authority,
                        )
                        self.store.decide_approval(
                            approval_id, approved=True, decided_by=decided_by
                        )
                        self.store.set_state(execution_id, RunState.RUNNING)
                        self._emit(
                            execution_id, "approval.approved", RunState.RUNNING,
                            tool_name=proposed.name,
                            summary="Exact tool request approved.",
                        )
                    else:
                        self.store.update_tool_call(call_id, status="rejected")
                        self.store.decide_approval(
                            approval_id, approved=False, decided_by=decided_by
                        )
                        self.store.set_state(
                            execution_id, RunState.FAILED,
                            error="Approval rejected.", finished=True,
                        )
                        self._emit(
                            execution_id, "approval.rejected", RunState.FAILED,
                            duration_ms=(time.monotonic() - execution_started) * 1000,
                            tool_name=proposed.name,
                            summary="Tool was not run.",
                        )
                        return self._snapshot(execution_id)
                if decision.decision is PolicyDecision.REJECTED:
                    raise ActionRejectedError(request.action)
                try:
                    decision.require_approved(authority=self.approval_authority)
                except ActionPendingApprovalError:
                    raise

                if pending_call:
                    self.store.update_tool_call(call_id, status="running")
                else:
                    self.store.add_tool_call(
                        call_id=call_id, execution_id=execution_id,
                        tool_name=proposed.name,
                        arguments_digest=request.arguments_digest, status="running",
                    )
                self._emit(
                    execution_id, "tool.started", RunState.RUNNING, tool_name=proposed.name
                )
                tool_started = time.monotonic()
                try:
                    result = await self._controlled(
                        self.tools.invoke(proposed.name, proposed.arguments), cancel_event
                    )
                except (RunCancelledError, asyncio.CancelledError):
                    self.store.update_tool_call(call_id, status="cancelled")
                    self._emit(
                        execution_id, "tool.cancelled", RunState.RUNNING,
                        tool_name=proposed.name,
                    )
                    raise
                except Exception as exc:
                    safe_error = type(exc).__name__
                    self.store.update_tool_call(call_id, status="failed", error=safe_error)
                    self._emit(
                        execution_id, "tool.failed", RunState.RUNNING,
                        tool_name=proposed.name, summary=safe_error,
                    )
                    raise
                duration = (time.monotonic() - tool_started) * 1000
                summary = _safe_result_summary(result)
                self.store.update_tool_call(
                    call_id, status="completed", result_summary=summary
                )
                self._emit(
                    execution_id, "tool.completed", RunState.RUNNING,
                    duration_ms=duration, tool_name=proposed.name, summary=summary,
                )
                messages.append(
                    {
                        "role": "assistant",
                        "tool_call": {
                            "name": proposed.name,
                            "arguments": proposed.arguments,
                        },
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": proposed.name,
                        "result": _safe_provider_result(),
                    }
                )
            raise RunError("Maximum agent iteration count reached.")
        except (RunCancelledError, asyncio.CancelledError):
            self.store.set_state(
                execution_id, RunState.CANCELLED, error="Execution cancelled.",
                finished=True,
            )
            self._emit(
                execution_id, "execution.cancelled", RunState.CANCELLED,
                duration_ms=(time.monotonic() - execution_started) * 1000,
            )
            return self._snapshot(execution_id)
        except Exception as exc:
            safe_error = type(exc).__name__
            self.store.set_state(
                execution_id, RunState.FAILED, error=safe_error, finished=True
            )
            self._emit(
                execution_id, "execution.failed", RunState.FAILED,
                duration_ms=(time.monotonic() - execution_started) * 1000,
                summary=safe_error,
            )
            return self._snapshot(execution_id)

    async def _controlled(self, operation: Awaitable[Any], cancellation: asyncio.Event) -> Any:
        task = asyncio.ensure_future(operation)
        cancelled = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                (task, cancelled), timeout=self.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise OperationTimeoutError("Operation timed out.")
            if cancelled in done and cancellation.is_set():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise RunCancelledError("Execution cancelled.")
            return task.result()
        finally:
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    async def _check_cancel(cancellation: asyncio.Event) -> None:
        if cancellation.is_set():
            raise RunCancelledError("Execution cancelled.")

    def _snapshot(self, execution_id: UUID) -> dict[str, Any]:
        result = self.store.get(execution_id)
        if result is None:
            raise RuntimeError("Execution disappeared from the persistent store.")
        return result


def _safe_provider_result() -> Any:
    return {"result": "tool result omitted"}


def _safe_result_summary(result: Any) -> str:
    if isinstance(result, dict):
        return f"Object result with {len(result)} fields; values omitted."
    if isinstance(result, list):
        return f"Array result with {len(result)} items; values omitted."
    if isinstance(result, str):
        return f"Text result with {len(result)} characters; content omitted."
    if isinstance(result, bool):
        return f"Boolean result: {str(result).lower()}."
    if isinstance(result, (int, float)):
        return "Numeric result; value omitted."
    if result is None:
        return "Empty result."
    return f"Non-JSON result of type {type(result).__name__}; content omitted."


async def _simulated_echo(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "character_count": len(arguments["text"])}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def default_database_path() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if root is None:
        root = str(Path.home() / ".local" / "share")
    return Path(root) / "ai-harness" / "history.sqlite3"
