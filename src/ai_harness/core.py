"""Execution lifecycle types and storage contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class HarnessError(Exception):
    """Base exception for errors raised by the harness."""


class ExecutionState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionEventType(StrEnum):
    CREATED = "execution.created"
    STARTED = "execution.started"
    WAITING_APPROVAL = "execution.waiting_approval"
    RESUMED = "execution.resumed"
    SUCCEEDED = "execution.succeeded"
    FAILED = "execution.failed"
    CANCELLED = "execution.cancelled"


class ExecutionContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    execution_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: ExecutionEventType
    data: dict[str, Any] = Field(default_factory=dict)


class InvalidStateTransitionError(HarnessError):
    def __init__(self, current: ExecutionState, target: ExecutionState) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Cannot transition execution from {current} to {target}.")


_TRANSITIONS: dict[ExecutionState, dict[ExecutionState, ExecutionEventType]] = {
    ExecutionState.CREATED: {
        ExecutionState.RUNNING: ExecutionEventType.STARTED,
        ExecutionState.CANCELLED: ExecutionEventType.CANCELLED,
    },
    ExecutionState.RUNNING: {
        ExecutionState.WAITING_APPROVAL: ExecutionEventType.WAITING_APPROVAL,
        ExecutionState.SUCCEEDED: ExecutionEventType.SUCCEEDED,
        ExecutionState.FAILED: ExecutionEventType.FAILED,
        ExecutionState.CANCELLED: ExecutionEventType.CANCELLED,
    },
    ExecutionState.WAITING_APPROVAL: {
        ExecutionState.FAILED: ExecutionEventType.FAILED,
        ExecutionState.CANCELLED: ExecutionEventType.CANCELLED,
    },
    ExecutionState.SUCCEEDED: {},
    ExecutionState.FAILED: {},
    ExecutionState.CANCELLED: {},
}


class Execution(BaseModel):
    """Immutable execution snapshot with validated state transitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: ExecutionContext
    state: ExecutionState = ExecutionState.CREATED
    events: tuple[ExecutionEvent, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        metadata: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
    ) -> Execution:
        if context is not None and metadata is not None:
            raise ValueError("Pass either context or metadata, not both.")
        execution_context = context or ExecutionContext(metadata=metadata or {})
        created_event = ExecutionEvent(
            execution_id=execution_context.execution_id,
            event_type=ExecutionEventType.CREATED,
        )
        return cls(context=execution_context, events=(created_event,))

    def transition(
        self,
        target: ExecutionState,
        *,
        data: dict[str, Any] | None = None,
    ) -> Execution:
        event_type = _TRANSITIONS[self.state].get(target)
        if event_type is None:
            raise InvalidStateTransitionError(self.state, target)

        event = ExecutionEvent(
            execution_id=self.context.execution_id,
            event_type=event_type,
            data=data or {},
        )
        return self.model_copy(
            update={"state": target, "events": (*self.events, event)}
        )


class ExecutionStore(Protocol):
    """Persistence contract for execution snapshots."""

    async def save(self, execution: Execution) -> None: ...

    async def get(self, execution_id: UUID) -> Execution | None: ...
