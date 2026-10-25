from datetime import UTC

import pytest

from ai_harness import (
    Execution,
    ExecutionEventType,
    ExecutionState,
    InvalidStateTransitionError,
)


def test_execution_creation_and_transitions_emit_utc_events() -> None:
    execution = Execution.create(metadata={"source": "test"})

    assert execution.state is ExecutionState.CREATED
    assert execution.events[0].event_type is ExecutionEventType.CREATED
    assert execution.events[0].execution_id == execution.context.execution_id
    assert execution.events[0].timestamp.tzinfo == UTC

    running = execution.transition(ExecutionState.RUNNING)
    waiting = running.transition(ExecutionState.WAITING_APPROVAL)
    failed = waiting.transition(
        ExecutionState.FAILED, data={"reason": "approval unavailable"}
    )

    assert execution.state is ExecutionState.CREATED
    assert failed.state is ExecutionState.FAILED
    assert [event.event_type for event in failed.events] == [
        ExecutionEventType.CREATED,
        ExecutionEventType.STARTED,
        ExecutionEventType.WAITING_APPROVAL,
        ExecutionEventType.FAILED,
    ]
    assert failed.events[-1].data == {"reason": "approval unavailable"}


def test_terminal_execution_cannot_transition() -> None:
    execution = Execution.create().transition(ExecutionState.CANCELLED)

    with pytest.raises(InvalidStateTransitionError):
        execution.transition(ExecutionState.RUNNING)


def test_waiting_approval_cannot_resume_without_approval_evidence() -> None:
    execution = (
        Execution.create()
        .transition(ExecutionState.RUNNING)
        .transition(ExecutionState.WAITING_APPROVAL)
    )

    with pytest.raises(InvalidStateTransitionError):
        execution.transition(ExecutionState.RUNNING)
