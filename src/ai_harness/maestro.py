"""Policy-bounded coordination for provider-backed and explicit simulated agents."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_harness.policies import ToolAction, ToolPolicy
from ai_harness.runtime import (
    Harness,
    ModelProvider,
    RunCancelledError,
    RunState,
    SimulatedProvider,
    SQLiteExecutionStore,
    ToolRegistry,
    make_simulated_registry,
)
from ai_harness.redaction import redact_sensitive_text


class MaestroError(Exception):
    """Base error for planning and coordination failures."""


class PlanValidationError(MaestroError):
    """Raised when a plan has missing tasks, invalid dependencies, or cycles."""


class MessageRejectedError(MaestroError):
    """Raised when a message violates the centralized communication policy."""


class MaestroRunState(StrEnum):
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class MessageType(StrEnum):
    CONTROL = "control"
    TASK = "task"
    RESULT = "result"
    REVIEW_REQUEST = "review_request"
    APPROVAL_NOTIFICATION = "approval_notification"
    ERROR = "error"


class MessageStatus(StrEnum):
    CREATED = "created"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class AgentState(StrEnum):
    READY = "ready"
    BUSY = "busy"
    DISABLED = "disabled"
    ERROR = "error"


class CommunicationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allow_direct_agent_messages: bool = False


class AgentRegistration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=2048)
    specialty: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    capabilities: tuple[str, ...] = ()
    authorized_tools: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    enabled: bool = True
    max_concurrent_tasks: int = Field(default=1, ge=1)


class AgentResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str = Field(min_length=1)
    claims: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    duration_ms: float | None = Field(default=None, ge=0)
    tokens_used: int | None = Field(default=None, ge=0)
    provider: str | None = None
    model: str | None = None


class AgentTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_id: UUID
    run_id: UUID
    task_id: UUID
    description: str
    user_request: str
    context: dict[str, Any] = Field(default_factory=dict)


class PlanTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: UUID = Field(default_factory=uuid4)
    description: str = Field(min_length=1, max_length=4096)
    required_capability: str = Field(min_length=1)
    dependencies: tuple[UUID, ...] = ()
    agent_id: str | None = None


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: tuple[PlanTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dependencies(self) -> ExecutionPlan:
        task_ids = {task.task_id for task in self.tasks}
        if len(task_ids) != len(self.tasks):
            raise PlanValidationError("Task IDs must be unique.")
        for task in self.tasks:
            if task.task_id in task.dependencies or not set(task.dependencies) <= task_ids:
                raise PlanValidationError("Task dependency is missing or self-referential.")
        visiting: set[UUID] = set()
        visited: set[UUID] = set()
        by_id = {task.task_id: task for task in self.tasks}

        def visit(task_id: UUID) -> None:
            if task_id in visiting:
                raise PlanValidationError("Task dependency cycle detected.")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in by_id[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)
        return self


class AgentAdapter(Protocol):
    async def execute(
        self, task: AgentTask, *, cancellation: asyncio.Event
    ) -> AgentResult: ...


class AgentRegistry:
    """Explicit allow-list of agent registrations and their adapters."""

    def __init__(self) -> None:
        self._agents: dict[str, tuple[AgentRegistration, AgentAdapter, AgentState]] = {}

    def register(self, registration: AgentRegistration, adapter: AgentAdapter) -> None:
        if registration.agent_id in self._agents:
            raise ValueError(f"Agent {registration.agent_id!r} is already registered.")
        self._agents[registration.agent_id] = (
            registration,
            adapter,
            AgentState.READY if registration.enabled else AgentState.DISABLED,
        )

    def get(self, agent_id: str) -> tuple[AgentRegistration, AgentAdapter, AgentState]:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise MaestroError(f"Agent {agent_id!r} is not registered.") from None

    def all(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **registration.model_dump(mode="json"),
                "state": state.value,
            }
            for registration, _, state in self._agents.values()
        )

    def select(self, capability: str) -> tuple[AgentRegistration, AgentAdapter]:
        candidates = [
            (registration, adapter)
            for registration, adapter, state in self._agents.values()
            if state is AgentState.READY
            and registration.enabled
            and capability in registration.capabilities
        ]
        if not candidates:
            raise MaestroError(f"No available agent has capability {capability!r}.")
        return candidates[0]

    def set_state(self, agent_id: str, state: AgentState) -> None:
        registration, adapter, _ = self.get(agent_id)
        self._agents[agent_id] = (registration, adapter, state)

    def deactivate(self, agent_id: str) -> None:
        registration, adapter, _ = self.get(agent_id)
        self._agents[agent_id] = (
            registration.model_copy(update={"enabled": False}),
            adapter,
            AgentState.DISABLED,
        )


class MaestroLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrent_agents: int = Field(default=2, ge=1)
    max_concurrent_tasks: int = Field(default=8, ge=1)
    max_tasks_per_run: int = Field(default=32, ge=1)
    max_delegations: int = Field(default=32, ge=1)
    max_iterations: int = Field(default=1, ge=1)
    task_timeout_seconds: float = Field(default=30, gt=0)


class SQLiteMaestroStore:
    """Maestro records share the runtime's SQLite file and survive process restarts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS maestro_runs (
                    run_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    user_request TEXT NOT NULL, state TEXT NOT NULL,
                    plan_json TEXT NOT NULL, result_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS maestro_tasks (
                    task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    description TEXT NOT NULL, capability TEXT NOT NULL,
                    agent_id TEXT NOT NULL, dependencies_json TEXT NOT NULL,
                    status TEXT NOT NULL, result_json TEXT, error TEXT,
                    started_at TEXT, finished_at TEXT,
                    UNIQUE(run_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS maestro_messages (
                    message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, task_id TEXT,
                    sender_id TEXT NOT NULL, recipient_id TEXT NOT NULL,
                    parent_message_id TEXT, message_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL, status TEXT NOT NULL,
                    payload_json TEXT NOT NULL, artifact_refs_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS maestro_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL, run_id TEXT NOT NULL,
                    task_id TEXT, timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL, status TEXT NOT NULL,
                    agent_id TEXT, duration_ms REAL, tokens_used INTEGER,
                    summary TEXT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def create_run(
        self,
        run_id: UUID,
        conversation_id: UUID,
        user_request: str,
        plan: ExecutionPlan,
        assignments: dict[UUID, str],
    ) -> None:
        now = _timestamp()
        plan_json = plan.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO maestro_runs
                (run_id, conversation_id, user_request, state, plan_json,
                 created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (str(run_id), str(conversation_id), user_request,
                 MaestroRunState.AWAITING_APPROVAL, plan_json, now, now),
            )
            for task in plan.tasks:
                connection.execute(
                    """INSERT INTO maestro_tasks
                    (task_id, run_id, description, capability, agent_id,
                     dependencies_json, status) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (str(task.task_id), str(run_id), task.description,
                     task.required_capability, assignments[task.task_id],
                     json.dumps([str(value) for value in task.dependencies]),
                     TaskState.PENDING),
                )

    def run(self, run_id: UUID | str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM maestro_runs WHERE run_id = ?", (str(run_id),)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["plan"] = json.loads(result.pop("plan_json"))
            raw_result = result.pop("result_json")
            result["result"] = json.loads(raw_result) if raw_result else None
            result["tasks"] = [
                {**dict(task), "dependencies": json.loads(task["dependencies_json"]),
                 "result": json.loads(task["result_json"]) if task["result_json"] else None}
                for task in connection.execute(
                    "SELECT * FROM maestro_tasks WHERE run_id = ? ORDER BY rowid",
                    (str(run_id),),
                )
            ]
            result["messages"] = self.messages(run_id)
            result["events"] = [
                dict(event) for event in connection.execute(
                    "SELECT * FROM maestro_events WHERE run_id = ? ORDER BY sequence",
                    (str(run_id),),
                )
            ]
            return result

    def set_run_state(
        self, run_id: UUID, state: MaestroRunState, *, result: dict[str, Any] | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE maestro_runs SET state = ?, result_json = COALESCE(?, result_json),
                updated_at = ? WHERE run_id = ?""",
                (state.value, json.dumps(result, ensure_ascii=False) if result else None,
                 _timestamp(), str(run_id)),
            )

    def claim_run(self, run_id: UUID) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                 """UPDATE maestro_runs SET state = ?, updated_at = ?
                 WHERE run_id = ? AND state = ? AND cancel_requested = 0""",
                 (_enum_value(MaestroRunState.RUNNING), _timestamp(), str(run_id),
                  _enum_value(MaestroRunState.AWAITING_APPROVAL)),
            )
            return cursor.rowcount == 1

    def set_task_state(
        self,
        task_id: UUID,
        state: TaskState,
        *,
        result: AgentResult | None = None,
        error: str | None = None,
    ) -> None:
        started = _timestamp() if state is TaskState.RUNNING else None
        finished = _timestamp() if state in {
            TaskState.COMPLETED, TaskState.FAILED, TaskState.BLOCKED, TaskState.CANCELLED
        } else None
        with self._connect() as connection:
            connection.execute(
                """UPDATE maestro_tasks SET status = ?, result_json = COALESCE(?, result_json),
                error = ?, started_at = COALESCE(?, started_at),
                finished_at = COALESCE(?, finished_at) WHERE task_id = ?""",
                (state.value, result.model_dump_json() if result else None, error,
                 started, finished, str(task_id)),
            )

    def add_message(
        self, *, conversation_id: UUID, run_id: UUID, task_id: UUID | None,
        sender_id: str, recipient_id: str, message_type: MessageType,
        status: MessageStatus, payload: dict[str, Any], parent_message_id: UUID | None = None,
        artifact_refs: Sequence[str] = (),
    ) -> UUID:
        message_id = uuid4()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO maestro_messages
                (message_id, conversation_id, run_id, task_id, sender_id, recipient_id,
                 parent_message_id, message_type, timestamp, status, payload_json,
                 artifact_refs_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(message_id), str(conversation_id), str(run_id),
                 str(task_id) if task_id else None, sender_id, recipient_id,
                 str(parent_message_id) if parent_message_id else None,
                 message_type.value, _timestamp(), status.value,
                 json.dumps(payload, ensure_ascii=False),
                 json.dumps(list(artifact_refs), ensure_ascii=False)),
            )
        return message_id

    def messages(self, run_id: UUID | str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            result = []
            for row in connection.execute(
                "SELECT * FROM maestro_messages WHERE run_id = ? ORDER BY timestamp, rowid",
                (str(run_id),),
            ):
                message = dict(row)
                message["payload"] = json.loads(message.pop("payload_json"))
                message["artifact_refs"] = json.loads(message.pop("artifact_refs_json"))
                result.append(message)
            return result

    def add_event(
        self, conversation_id: UUID, run_id: UUID, event_type: str, status: str,
        *, task_id: UUID | None = None, agent_id: str | None = None,
        duration_ms: float | None = None, tokens_used: int | None = None,
        summary: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO maestro_events
                (conversation_id, run_id, task_id, timestamp, event_type, status,
                 agent_id, duration_ms, tokens_used, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(conversation_id), str(run_id), str(task_id) if task_id else None,
                 _timestamp(), event_type, status, agent_id, duration_ms, tokens_used,
                 summary),
            )

    def request_cancel(self, run_id: UUID | str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE maestro_runs SET cancel_requested = 1, updated_at = ?
                WHERE run_id = ? AND state IN (?, ?)""",
                (_timestamp(), str(run_id), MaestroRunState.AWAITING_APPROVAL,
                 MaestroRunState.RUNNING),
            )
            return cursor.rowcount > 0

    def cancel_requested(self, run_id: UUID) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM maestro_runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                """SELECT run_id, conversation_id, state, created_at, updated_at
                FROM maestro_runs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            )]

    def pending_approvals(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                """SELECT run_id, conversation_id, user_request, created_at
                FROM maestro_runs WHERE state = ? ORDER BY created_at""",
                (MaestroRunState.AWAITING_APPROVAL,),
            )]


class SimulatedSpecialist:
    """Runs through the existing harness; this adapter does not contact a model."""

    def __init__(
        self, *, store: SQLiteExecutionStore, agent_id: str,
        provider: ModelProvider | None = None, tools: ToolRegistry | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self.store = store
        self.agent_id = agent_id
        self.provider = provider or SimulatedProvider()
        self.tools = tools or make_simulated_registry()
        self.policy = policy or ToolPolicy()

    async def execute(
        self, task: AgentTask, *, cancellation: asyncio.Event
    ) -> AgentResult:
        execution = await Harness(
            store=self.store, provider=self.provider, tools=self.tools,
            policy=self.policy, agent_id=self.agent_id, max_iterations=2,
        ).run(f"{task.description}: {task.user_request}", cancellation=cancellation)
        if execution["state"] != RunState.COMPLETED:
            raise MaestroError(execution["error"] or execution["state"])
        return AgentResult(
            summary=(
                f"Simulated {self.agent_id} completed its assigned work; "
                "no real model or external service was used."
            ),
            claims=("Execution was handled by the local simulated runtime.",),
            limitations=("Simulated output is not an expert assessment.",),
            artifact_refs=(f"execution:{execution['execution_id']}",),
            duration_ms=_duration(execution),
            provider=self.provider.provider_id,
            model="deterministic",
        )


class Maestro:
    """Plans and coordinates work; all agent messages pass through this component."""

    def __init__(
        self,
        *,
        store: SQLiteExecutionStore,
        registry: AgentRegistry | None = None,
        limits: MaestroLimits | None = None,
        planner: Callable[[str], ExecutionPlan] | None = None,
        tools: ToolRegistry | None = None,
        policy: ToolPolicy | None = None,
        agent_config: str | Path | None = None,
        task_context: dict[str, Any] | None = None,
        simulation: bool = False,
    ) -> None:
        self.runtime_store = store
        self.store = SQLiteMaestroStore(store.path)
        self.limits = limits or MaestroLimits()
        if registry is None and agent_config is None and not simulation:
            raise ValueError(
                "Provide an agent registry or explicitly set simulation=True."
            )
        if planner is None and agent_config is None and not simulation:
            raise ValueError(
                "A provider-backed planner is required outside explicit simulation."
            )
        self.planner = planner or _default_plan
        self.task_context = dict(task_context or {})
        self.tools = (
            tools
            if tools is not None
            else make_simulated_registry()
            if simulation or agent_config is not None
            else ToolRegistry()
        )
        self.policy = policy or ToolPolicy()
        self.registry = registry or (
            load_simulated_agent_registry(
                store, agent_config, tools=self.tools, policy=self.policy
            )
            if agent_config is not None
            else make_simulated_agent_registry(store, tools=self.tools, policy=self.policy)
        )
        self.communication_policy = CommunicationPolicy()
        self._cancellations: dict[UUID, asyncio.Event] = {}
        self._active_agents: dict[str, asyncio.Semaphore] = {}
        self._agent_activity: dict[str, int] = {}
        self._agent_activity_lock = asyncio.Lock()

    def create_plan(self, user_request: str) -> dict[str, Any]:
        user_request = redact_sensitive_text(user_request)
        if not user_request.strip():
            raise ValueError("Task cannot be blank.")
        plan = self.planner(user_request)
        if len(plan.tasks) > self.limits.max_tasks_per_run:
            raise PlanValidationError("Plan exceeds the configured task limit.")
        if len(plan.tasks) > self.limits.max_delegations:
            raise PlanValidationError("Plan exceeds the configured delegation limit.")
        assignments: dict[UUID, str] = {}
        assigned_tasks: list[PlanTask] = []
        for task in plan.tasks:
            if task.agent_id:
                registration, _, state = self.registry.get(task.agent_id)
                if state is not AgentState.READY or not registration.enabled:
                    raise MaestroError(f"Agent {task.agent_id!r} is unavailable.")
                if task.required_capability not in registration.capabilities:
                    raise MaestroError("Assigned agent lacks the required capability.")
            else:
                registration, _ = self.registry.select(task.required_capability)
            assignments[task.task_id] = registration.agent_id
            assigned_tasks.append(task.model_copy(update={"agent_id": registration.agent_id}))
        final_plan = ExecutionPlan(tasks=tuple(assigned_tasks))
        run_id = uuid4()
        conversation_id = uuid4()
        self.store.create_run(run_id, conversation_id, user_request, final_plan, assignments)
        self.store.add_event(
            conversation_id, run_id, "plan.created", MaestroRunState.AWAITING_APPROVAL,
            summary=f"{len(final_plan.tasks)} task(s) planned.",
        )
        self.store.add_message(
            conversation_id=conversation_id, run_id=run_id, task_id=None,
            sender_id="user", recipient_id="maestro", message_type=MessageType.CONTROL,
            status=MessageStatus.DELIVERED, payload={"request": user_request},
        )
        return self.store.run(run_id) or {}

    async def approve_plan(
        self, run_id: UUID | str, *, approved: bool, approved_by: str = "operator",
        cancellation: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        snapshot = self.store.run(run_id)
        if snapshot is None:
            raise MaestroError("Run was not found.")
        if snapshot["state"] != MaestroRunState.AWAITING_APPROVAL:
            raise MaestroError("Only a plan awaiting approval can be decided.")
        run_uuid = UUID(str(run_id))
        conversation_id = UUID(snapshot["conversation_id"])
        if not approved:
            self.store.set_run_state(run_uuid, MaestroRunState.CANCELLED)
            self.store.add_event(
                conversation_id, run_uuid, "plan.rejected", MaestroRunState.CANCELLED,
                summary="Plan rejected by operator.",
            )
            self.store.add_message(
                conversation_id=conversation_id, run_id=run_uuid, task_id=None,
                sender_id="operator", recipient_id="maestro",
                message_type=MessageType.APPROVAL_NOTIFICATION,
                status=MessageStatus.REJECTED, payload={"approved": False, "by": approved_by},
            )
            return self.store.run(run_uuid) or {}
        if self.store.cancel_requested(run_uuid):
            self.store.set_run_state(run_uuid, MaestroRunState.CANCELLED)
            return self.store.run(run_uuid) or {}
        if not self.store.claim_run(run_uuid):
            raise MaestroError("Plan approval was already decided or cancelled.")
        event = cancellation or asyncio.Event()
        self._cancellations[run_uuid] = event
        self.store.add_event(
            conversation_id, run_uuid, "plan.approved", MaestroRunState.RUNNING,
            summary="Plan approved by operator.",
        )
        self.store.add_event(
            conversation_id, run_uuid, "run.started", MaestroRunState.RUNNING
        )
        self.store.add_message(
            conversation_id=conversation_id, run_id=run_uuid, task_id=None,
            sender_id="operator", recipient_id="maestro",
            message_type=MessageType.APPROVAL_NOTIFICATION,
            status=MessageStatus.COMPLETED, payload={"approved": True, "by": approved_by},
        )
        monitor = asyncio.create_task(self._watch_cancellation(run_uuid, event))
        started = time.monotonic()
        try:
            await self._execute_plan(snapshot, event)
        except asyncio.CancelledError:
            event.set()
            current = self.store.run(run_uuid)
            if current is not None:
                for task in current["tasks"]:
                    if task["status"] in {TaskState.PENDING, TaskState.RUNNING}:
                        self.store.set_task_state(
                            UUID(task["task_id"]), TaskState.CANCELLED
                        )
            self.store.set_run_state(run_uuid, MaestroRunState.CANCELLED)
            self.store.add_event(
                conversation_id, run_uuid, "run.cancelled",
                MaestroRunState.CANCELLED,
                summary="Coordinator task was cancelled.",
            )
            return self.store.run(run_uuid) or {}
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            self._cancellations.pop(run_uuid, None)
        result = self._consolidate(self.store.run(run_uuid) or {})
        current = self.store.run(run_uuid) or {}
        if current["state"] == MaestroRunState.CANCELLED:
            return current
        finished_tasks = [task for task in current["tasks"] if task["status"] == TaskState.COMPLETED]
        state = (
            MaestroRunState.COMPLETED
            if len(finished_tasks) == len(current["tasks"])
            else MaestroRunState.PARTIAL if finished_tasks
            else MaestroRunState.FAILED
        )
        self.store.set_run_state(run_uuid, state, result=result)
        self.store.add_event(
            conversation_id, run_uuid, "run.finished", state,
            duration_ms=(time.monotonic() - started) * 1000,
            summary=f"{len(finished_tasks)}/{len(current['tasks'])} task(s) completed.",
        )
        return self.store.run(run_uuid) or {}

    async def cancel(self, run_id: UUID | str) -> bool:
        accepted = self.store.request_cancel(run_id)
        if accepted:
            event = self._cancellations.get(UUID(str(run_id)))
            if event is not None:
                event.set()
            snapshot = self.store.run(run_id)
            if snapshot is not None and snapshot["state"] == MaestroRunState.AWAITING_APPROVAL:
                self.store.set_run_state(UUID(str(run_id)), MaestroRunState.CANCELLED)
                self.store.add_event(
                    UUID(snapshot["conversation_id"]), UUID(str(run_id)),
                    "run.cancelled", MaestroRunState.CANCELLED,
                )
        return accepted

    async def request_review(
        self, run_id: UUID | str, task_id: UUID | str, *,
        reviewer_id: str, request: str,
    ) -> UUID:
        snapshot = self.store.run(run_id)
        if snapshot is None:
            raise MaestroError("Run was not found.")
        task = next((item for item in snapshot["tasks"] if item["task_id"] == str(task_id)), None)
        if task is None:
            raise MaestroError("Task was not found.")
        review_count = sum(
            message["message_type"] == MessageType.REVIEW_REQUEST
            for message in snapshot["messages"]
        )
        delegation_count = sum(
            message["message_type"] in {MessageType.TASK, MessageType.REVIEW_REQUEST}
            for message in snapshot["messages"]
        )
        if review_count >= self.limits.max_iterations:
            raise MaestroError("The configured review iteration limit was reached.")
        if delegation_count >= self.limits.max_delegations:
            raise MaestroError("The configured delegation limit was reached.")
        registration, _, state = self.registry.get(reviewer_id)
        if state is not AgentState.READY or not registration.enabled:
            raise MaestroError("Reviewer is unavailable.")
        if "review" not in registration.capabilities:
            raise MaestroError("Reviewer does not declare the review capability.")
        conversation_id = UUID(snapshot["conversation_id"])
        run_uuid = UUID(str(run_id))
        task_uuid = UUID(str(task_id))
        message_id = self.store.add_message(
            conversation_id=conversation_id, run_id=run_uuid,
            task_id=task_uuid, sender_id="maestro", recipient_id=reviewer_id,
            message_type=MessageType.REVIEW_REQUEST, status=MessageStatus.DELIVERED,
            payload={"request": request},
        )
        self.store.add_event(
            conversation_id, run_uuid, "review.requested", "delivered",
            task_id=task_uuid, agent_id=reviewer_id,
        )
        _, adapter, _ = self.registry.get(reviewer_id)
        cancellation = asyncio.Event()
        self.registry.set_state(reviewer_id, AgentState.BUSY)
        started = time.monotonic()
        try:
            result = await self._execute_adapter(
                adapter,
                AgentTask(
                    conversation_id=conversation_id, run_id=run_uuid, task_id=task_uuid,
                    description=request, user_request=snapshot["user_request"],
                    context={"review_of_task": str(task_uuid)},
                ),
                cancellation,
            )
            self.store.add_message(
                conversation_id=conversation_id, run_id=run_uuid, task_id=task_uuid,
                sender_id=reviewer_id, recipient_id="maestro",
                parent_message_id=message_id, message_type=MessageType.RESULT,
                status=MessageStatus.COMPLETED, payload=result.model_dump(mode="json"),
                artifact_refs=result.artifact_refs,
            )
            self.store.add_event(
                conversation_id, run_uuid, "review.received", "completed",
                task_id=task_uuid, agent_id=reviewer_id,
                duration_ms=result.duration_ms or (time.monotonic() - started) * 1000,
                tokens_used=result.tokens_used, summary="Review result recorded.",
            )
            self._refresh_consolidation(run_uuid)
        except Exception as exc:
            error = type(exc).__name__
            self.store.add_message(
                conversation_id=conversation_id, run_id=run_uuid, task_id=task_uuid,
                sender_id=reviewer_id, recipient_id="maestro",
                parent_message_id=message_id, message_type=MessageType.ERROR,
                status=MessageStatus.FAILED, payload={"error": error},
            )
            self.store.add_event(
                conversation_id, run_uuid, "review.failed", "failed",
                task_id=task_uuid, agent_id=reviewer_id, summary=error,
            )
            self._refresh_consolidation(run_uuid)
            raise
        finally:
            self.registry.set_state(reviewer_id, AgentState.READY)
        return message_id

    def send_direct_message(
        self, *, sender_id: str, recipient_id: str, run_id: UUID | str,
        task_id: UUID | str,
    ) -> None:
        if not self.communication_policy.allow_direct_agent_messages:
            snapshot = self.store.run(run_id)
            if snapshot is not None:
                self.store.add_message(
                    conversation_id=UUID(snapshot["conversation_id"]),
                    run_id=UUID(str(run_id)), task_id=UUID(str(task_id)),
                    sender_id=sender_id, recipient_id=recipient_id,
                    message_type=MessageType.CONTROL, status=MessageStatus.REJECTED,
                    payload={"error": "Direct agent communication is disabled."},
                )
            raise MessageRejectedError("Direct agent communication is disabled.")

    async def _watch_cancellation(
        self, run_id: UUID, event: asyncio.Event
    ) -> None:
        while not event.is_set():
            if self.store.cancel_requested(run_id):
                event.set()
                return
            await asyncio.sleep(0.1)

    async def _execute_plan(
        self, snapshot: dict[str, Any], cancellation: asyncio.Event
    ) -> None:
        run_id = UUID(snapshot["run_id"])
        conversation_id = UUID(snapshot["conversation_id"])
        tasks = snapshot["tasks"]
        by_id = {task["task_id"]: task for task in tasks}
        semaphore = asyncio.Semaphore(min(
            self.limits.max_concurrent_agents, self.limits.max_concurrent_tasks
        ))
        completed: set[str] = set()
        task_locks = {task_id: asyncio.Lock() for task_id in by_id}

        async def execute_task(task_id: str) -> None:
            item = by_id[task_id]
            async with task_locks[task_id]:
                latest = self.store.run(run_id)
                if latest is None:
                    raise MaestroError("Run was not found during task execution.")
                current_item = next(
                    task for task in latest["tasks"] if task["task_id"] == task_id
                )
                if current_item["status"] != TaskState.PENDING:
                    by_id[task_id] = current_item
                    return
                dependencies = item["dependencies"]
                if dependencies:
                    await asyncio.gather(*(execute_task(dependency) for dependency in dependencies))
                latest = self.store.run(run_id)
                if latest is None:
                    raise MaestroError("Run was not found during task execution.")
                current_item = next(
                    task for task in latest["tasks"] if task["task_id"] == task_id
                )
                if cancellation.is_set():
                    self.store.set_task_state(UUID(task_id), TaskState.CANCELLED)
                    by_id[task_id]["status"] = TaskState.CANCELLED
                    return
                if any(
                    next(task for task in latest["tasks"] if task["task_id"] == dependency)["status"]
                    != TaskState.COMPLETED
                    for dependency in dependencies
                ):
                    self.store.set_task_state(
                        UUID(task_id), TaskState.BLOCKED, error="Dependency did not complete."
                    )
                    by_id[task_id]["status"] = TaskState.BLOCKED
                    return
                async with semaphore:
                    if cancellation.is_set():
                        self.store.set_task_state(UUID(task_id), TaskState.CANCELLED)
                        by_id[task_id]["status"] = TaskState.CANCELLED
                        return
                    await self._run_task(item, snapshot, cancellation)
                    latest = self.store.run(run_id)
                    if latest is None:
                        raise MaestroError("Run was not found during task execution.")
                    by_id[task_id] = next(
                        task for task in latest["tasks"] if task["task_id"] == task_id
                    )
                    if by_id[task_id]["status"] == TaskState.COMPLETED:
                        completed.add(task_id)

        await asyncio.gather(*(execute_task(task_id) for task_id in by_id))
        current = self.store.run(run_id) or {}
        if cancellation.is_set():
            for task in current["tasks"]:
                if task["status"] == TaskState.PENDING:
                    self.store.set_task_state(UUID(task["task_id"]), TaskState.CANCELLED)
            self.store.set_run_state(run_id, MaestroRunState.CANCELLED)
            self.store.add_event(
                conversation_id, run_id, "run.cancelled", MaestroRunState.CANCELLED
            )

    async def _run_task(
        self, item: dict[str, Any], snapshot: dict[str, Any],
        cancellation: asyncio.Event,
    ) -> None:
        run_id = UUID(snapshot["run_id"])
        conversation_id = UUID(snapshot["conversation_id"])
        task_id = UUID(item["task_id"])
        agent_id = item["agent_id"]
        registration, adapter, state = self.registry.get(agent_id)
        if state not in (AgentState.READY, AgentState.BUSY) or not registration.enabled:
            self.store.set_task_state(task_id, TaskState.FAILED, error="Agent unavailable.")
            return
        if registration.agent_id not in self._active_agents:
            self._active_agents[registration.agent_id] = asyncio.Semaphore(
                registration.max_concurrent_tasks
            )
        task_semaphore = self._active_agents[registration.agent_id]
        latest = self.store.run(run_id)
        if latest is None:
            raise MaestroError("Run was not found before delegating a task.")
        dependency_context = {
            task["task_id"]: task["result"]
            for task in latest["tasks"]
            if task["task_id"] in item["dependencies"]
        }
        context = {
            **self.task_context,
            "user_request": snapshot["user_request"],
            "dependency_results": dependency_context,
        }
        message_id = self.store.add_message(
            conversation_id=conversation_id, run_id=run_id, task_id=task_id,
            sender_id="maestro", recipient_id=agent_id, message_type=MessageType.TASK,
            status=MessageStatus.DELIVERED,
            payload={
                "description": item["description"],
                "context": context,
            },
        )
        self.store.add_event(
            conversation_id, run_id, "task.delegated", TaskState.RUNNING,
            task_id=task_id, agent_id=agent_id,
        )
        self.store.set_task_state(task_id, TaskState.RUNNING)
        async with self._agent_activity_lock:
            self._agent_activity[agent_id] = self._agent_activity.get(agent_id, 0) + 1
            self.registry.set_state(agent_id, AgentState.BUSY)
        task_input = AgentTask(
            conversation_id=conversation_id, run_id=run_id, task_id=task_id,
            description=item["description"], user_request=snapshot["user_request"],
            context=context,
        )
        started = time.monotonic()
        try:
            async with task_semaphore:
                result = await self._execute_adapter(
                    adapter, task_input, cancellation
                )
                if cancellation.is_set():
                    self.store.set_task_state(task_id, TaskState.CANCELLED)
                    return
            if not isinstance(result, AgentResult):
                raise MaestroError("Agent returned an invalid result.")
            self.store.set_task_state(task_id, TaskState.COMPLETED, result=result)
            self.store.add_message(
                conversation_id=conversation_id, run_id=run_id, task_id=task_id,
                sender_id=agent_id, recipient_id="maestro", parent_message_id=message_id,
                message_type=MessageType.RESULT, status=MessageStatus.COMPLETED,
                payload=result.model_dump(mode="json"), artifact_refs=result.artifact_refs,
            )
            self.store.add_event(
                conversation_id, run_id, "result.received", TaskState.COMPLETED,
                task_id=task_id, agent_id=agent_id,
                duration_ms=result.duration_ms if result.duration_ms is not None
                else (time.monotonic() - started) * 1000,
                tokens_used=result.tokens_used, summary="Agent result recorded.",
            )
        except RunCancelledError:
            self.store.set_task_state(task_id, TaskState.CANCELLED)
            return
        except asyncio.CancelledError:
            self.store.set_task_state(task_id, TaskState.CANCELLED)
            raise
        except Exception as exc:
            error = "TimeoutError" if isinstance(exc, TimeoutError) else type(exc).__name__
            self.store.set_task_state(task_id, TaskState.FAILED, error=error)
            self.store.add_message(
                conversation_id=conversation_id, run_id=run_id, task_id=task_id,
                sender_id=agent_id, recipient_id="maestro", parent_message_id=message_id,
                message_type=MessageType.ERROR, status=MessageStatus.FAILED,
                payload={"error": error},
            )
            self.store.add_event(
                conversation_id, run_id, "task.failed", TaskState.FAILED,
                task_id=task_id, agent_id=agent_id,
                duration_ms=(time.monotonic() - started) * 1000, summary=error,
            )
        finally:
            async with self._agent_activity_lock:
                remaining = self._agent_activity.get(agent_id, 1) - 1
                current_state = self.registry.get(agent_id)[2]
                if remaining:
                    self._agent_activity[agent_id] = remaining
                    if current_state is not AgentState.DISABLED:
                        self.registry.set_state(agent_id, AgentState.BUSY)
                else:
                    self._agent_activity.pop(agent_id, None)
                    if current_state is not AgentState.DISABLED:
                        self.registry.set_state(agent_id, AgentState.READY)

    async def _execute_adapter(
        self, adapter: AgentAdapter, task_input: AgentTask,
        cancellation: asyncio.Event,
    ) -> AgentResult:
        operation = asyncio.create_task(
            adapter.execute(task_input, cancellation=cancellation)
        )
        cancellation_waiter = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                (operation, cancellation_waiter),
                timeout=self.limits.task_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_waiter in done and cancellation.is_set():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise RunCancelledError("Run cancelled.")
            if operation not in done:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise TimeoutError("Agent task timed out.")
            return operation.result()
        finally:
            cancellation_waiter.cancel()
            await asyncio.gather(cancellation_waiter, return_exceptions=True)
            if not operation.done():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)

    def _consolidate(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        completed = [task for task in snapshot["tasks"] if task["status"] == TaskState.COMPLETED]
        incomplete = [
            {"task_id": task["task_id"], "status": task["status"], "error": task["error"]}
            for task in snapshot["tasks"] if task["status"] != TaskState.COMPLETED
        ]
        claims: dict[str, list[str]] = {}
        claim_values: dict[str, dict[str, list[str]]] = {}
        summaries = []
        limitations: set[str] = set()
        for task in completed:
            value = task["result"]
            if value is None:
                continue
            summaries.append({"task_id": task["task_id"], "agent_id": task["agent_id"],
                              "summary": value["summary"]})
            for claim in value["claims"]:
                claims.setdefault(claim, []).append(task["agent_id"])
                key, separator, claim_value = claim.partition(":")
                if separator:
                    claim_values.setdefault(key.strip().casefold(), {}).setdefault(
                        claim_value.strip().casefold(), []
                    ).append(task["agent_id"])
            limitations.update(value["limitations"])
        review_ids = {
            message["message_id"]
            for message in snapshot["messages"]
            if message["message_type"] == MessageType.REVIEW_REQUEST
        }
        review_results = {
            message["parent_message_id"]: message
            for message in snapshot["messages"]
            if message["message_type"] == MessageType.RESULT
            and message["parent_message_id"] in review_ids
        }
        for message in review_results.values():
            result = message["payload"]
            summaries.append({
                    "task_id": message["task_id"], "agent_id": message["sender_id"],
                    "summary": result["summary"], "review": True,
            })
            for claim in result["claims"]:
                    claims.setdefault(claim, []).append(message["sender_id"])
                    key, separator, claim_value = claim.partition(":")
                    if separator:
                        claim_values.setdefault(key.strip().casefold(), {}).setdefault(
                            claim_value.strip().casefold(), []
                        ).append(message["sender_id"])
            limitations.update(result["limitations"])
        for review in snapshot["messages"]:
            if (
                    review["message_type"] == MessageType.REVIEW_REQUEST
                    and review["message_id"] not in review_results
            ):
                    incomplete.append({
                        "task_id": review["task_id"], "status": "review_incomplete",
                        "error": "Review result was not received.",
                    })
        divergences = [
            {"topic": topic, "positions": positions}
            for topic, positions in claim_values.items() if len(positions) > 1
        ]
        providers = {
            self.registry.get(task["agent_id"])[0].provider
            for task in snapshot["tasks"]
        }
        known_real_providers = {"ollama", "anthropic"}
        if providers == {"simulated"}:
            mode = "simulated"
        elif providers and providers <= known_real_providers:
            mode = "real" if len(providers) == 1 else "mixed"
        elif "simulated" in providers:
            mode = "mixed"
        else:
            mode = "unverified"
        limitations.add(
            "Results depend on configured adapters and should be validated independently."
        )
        if mode == "simulated":
            limitations.add(
                "Results depend on the configured adapters; simulated adapters are not expert evidence."
            )
        elif mode == "unverified":
            limitations.add(
                "Provider identity is not verified by the built-in adapters."
            )
        return {
            "summary": (
                "Consolidated simulated agent results."
                if mode == "simulated"
                else "Consolidated results from unverified adapters."
                if mode == "unverified"
                else "Consolidated agent results."
            ),
            "task_results": summaries,
            "incomplete_tasks": incomplete,
            "divergences": divergences,
            "shared_claims": [
                {"claim": claim, "agents": agents}
                for claim, agents in claims.items() if len(set(agents)) > 1
            ],
            "limitations": sorted(limitations),
            "mode": mode,
        }

    def _refresh_consolidation(self, run_id: UUID) -> None:
        snapshot = self.store.run(run_id)
        if snapshot is None:
            raise MaestroError("Run disappeared while consolidating a review.")
        self.store.set_run_state(
            run_id, MaestroRunState(snapshot["state"]),
            result=self._consolidate(snapshot),
        )


def make_simulated_agent_registry(
    store: SQLiteExecutionStore,
    *,
    tools: ToolRegistry | None = None,
    policy: ToolPolicy | None = None,
) -> AgentRegistry:
    registry = AgentRegistry()
    available_tools = tools or make_simulated_registry()
    registrations = (
        AgentRegistration(
            agent_id="analyst", name="Simulated Analyst",
            description="Analyzes the request using the local simulated runtime.",
            specialty="analysis", provider="simulated", model="deterministic",
            capabilities=("analysis",), authorized_tools=("echo",),
            permissions=("runtime.read",),
        ),
        AgentRegistration(
            agent_id="reviewer", name="Simulated Reviewer",
            description="Independently reviews the request using the local simulated runtime.",
            specialty="review", provider="simulated", model="deterministic",
            capabilities=("review",), authorized_tools=("echo",),
            permissions=("runtime.read",),
        ),
    )
    for registration in registrations:
        authorized_tools = _authorized_tool_registry(registration, available_tools)
        registry.register(
            registration,
            SimulatedSpecialist(
                store=store, agent_id=registration.agent_id,
                tools=authorized_tools, policy=policy,
            ),
        )
    return registry


def load_simulated_agent_registry(
    store: SQLiteExecutionStore,
    config_path: str | Path,
    *,
    tools: ToolRegistry | None = None,
    policy: ToolPolicy | None = None,
) -> AgentRegistry:
    """Load only explicit simulated-agent configuration; no external is marked connected."""
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or set(configuration) != {"agents"}:
        raise ValueError("Agent configuration must contain only an 'agents' list.")
    configured_agents = configuration["agents"]
    if not isinstance(configured_agents, list) or not configured_agents:
        raise ValueError("Agent configuration must contain at least one agent.")
    registry = AgentRegistry()
    available_tools = tools or make_simulated_registry()
    for value in configured_agents:
        registration = AgentRegistration.model_validate(value)
        if registration.provider != "simulated":
            raise MaestroError(
                "Only provider='simulated' is supported; no external provider is connected."
            )
        adapter_tools = _authorized_tool_registry(registration, available_tools)
        registry.register(
            registration,
            SimulatedSpecialist(
                store=store, agent_id=registration.agent_id,
                tools=adapter_tools, policy=policy,
            ),
        )
    return registry


def _authorized_tool_registry(
    registration: AgentRegistration, available_tools: ToolRegistry
) -> ToolRegistry:
    authorized = ToolRegistry()
    for definition in available_tools.definitions():
        explicit_permission = (
            f"tool:{definition.name}:{definition.action.value}" in registration.permissions
        )
        read_permission = (
            definition.action is ToolAction.READ
            and "runtime.read" in registration.permissions
        )
        if (
            definition.name in registration.authorized_tools
            and (explicit_permission or read_permission)
        ):
            _, handler = available_tools.get(definition.name)
            authorized.register(definition, handler)
    return authorized


def _default_plan(user_request: str) -> ExecutionPlan:
    _ = user_request
    analysis = PlanTask(
        description="Analyze the user request and identify the core work and limitations.",
        required_capability="analysis",
    )
    review = PlanTask(
        description="Independently review the request and identify risks or omissions.",
        required_capability="review",
    )
    return ExecutionPlan(tasks=(analysis, review))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _enum_value(value: StrEnum) -> str:
    return value.value


def _duration(execution: dict[str, Any]) -> float | None:
    events = execution.get("events", [])
    if not events:
        return None
    start = datetime.fromisoformat(events[0]["timestamp"])
    end = datetime.fromisoformat(events[-1]["timestamp"])
    return max(0.0, (end - start).total_seconds() * 1000)
