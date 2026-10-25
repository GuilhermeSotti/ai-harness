import asyncio
from pathlib import Path
from typing import Any

import pytest

from ai_harness import (
    Harness,
    ProviderResponse,
    ProviderToolCall,
    RunState,
    SimulatedProvider,
    SQLiteExecutionStore,
    ToolAction,
    ToolDefinition,
    ToolRegistry,
    make_simulated_registry,
)


def _store(tmp_path: Path) -> SQLiteExecutionStore:
    return SQLiteExecutionStore(tmp_path / "history.sqlite3")


def _harness(
    tmp_path: Path,
    *,
    provider: Any | None = None,
    tools: ToolRegistry | None = None,
    **kwargs: Any,
) -> Harness:
    return Harness(
        store=_store(tmp_path),
        provider=provider or SimulatedProvider(),
        tools=tools or make_simulated_registry(),
        **kwargs,
    )


def test_simulated_run_completes_and_tool_result_is_recorded(tmp_path: Path) -> None:
    execution = asyncio.run(_harness(tmp_path).run("count these words"))

    assert execution["state"] == RunState.COMPLETED
    assert execution["mode"] == "simulated"
    assert execution["provider_id"] == "simulated"
    assert execution["result"].startswith("Simulated run completed")
    assert execution["tool_calls"][0]["tool_name"] == "echo"
    assert execution["tool_calls"][0]["status"] == "completed"
    assert execution["tool_calls"][0]["result_summary"] == (
        "Object result with 2 fields; values omitted."
    )


def test_history_and_status_survive_store_reopen(tmp_path: Path) -> None:
    execution = asyncio.run(_harness(tmp_path).run("persist this"))
    reopened = SQLiteExecutionStore(tmp_path / "history.sqlite3")

    persisted = reopened.get(execution["execution_id"])
    assert persisted is not None
    assert persisted["task"] == "persist this"
    assert reopened.history()[0]["execution_id"] == execution["execution_id"]


def test_events_are_persisted_and_observed_in_sequence(tmp_path: Path) -> None:
    observed: list[dict[str, Any]] = []
    execution = asyncio.run(
        _harness(tmp_path, observer=observed.append).run("ordered events")
    )
    stored = execution["events"]

    assert [event["sequence"] for event in observed] == sorted(
        event["sequence"] for event in observed
    )
    assert [event["event_type"] for event in stored] == [
        event["event_type"] for event in observed
    ]
    assert stored[0]["event_type"] == "execution.queued"
    assert stored[-1]["event_type"] == "execution.completed"


def test_provider_failure_is_persisted_without_raw_exception_text(tmp_path: Path) -> None:
    class FailingProvider:
        provider_id = "failure-test"
        mode = "simulated"

        async def complete(self, **_: Any) -> ProviderResponse:
            raise RuntimeError("credential=very-secret")

    execution = asyncio.run(_harness(tmp_path, provider=FailingProvider()).run("task"))

    assert execution["state"] == RunState.FAILED
    assert execution["error"] == "RuntimeError"
    assert "very-secret" not in repr(execution["events"])


def test_provider_timeout_is_recorded(tmp_path: Path) -> None:
    class SlowProvider:
        provider_id = "slow-test"
        mode = "simulated"

        async def complete(self, **_: Any) -> ProviderResponse:
            await asyncio.sleep(0.2)
            return ProviderResponse(text="too late")

    execution = asyncio.run(
        _harness(tmp_path, provider=SlowProvider(), timeout_seconds=0.01).run("task")
    )

    assert execution["state"] == RunState.FAILED
    assert execution["error"] == "ProviderTimeoutError"


def test_tool_timeout_is_recorded_and_handler_is_cancelled(tmp_path: Path) -> None:
    cancelled: list[bool] = []
    registry = ToolRegistry()

    async def slow_tool(_: dict[str, Any]) -> dict[str, bool]:
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return {"finished": True}

    registry.register(
        ToolDefinition(
            name="slow",
            description="Wait for a long time.",
            action=ToolAction.READ,
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        ),
        slow_tool,
    )

    class SlowToolProvider:
        provider_id = "slow-tool-provider"
        mode = "simulated"

        async def complete(self, *, messages: list[dict[str, Any]], **_: Any) -> ProviderResponse:
            if messages:
                return ProviderResponse(text="unexpected")
            return ProviderResponse(
                tool_call=ProviderToolCall(name="slow", arguments={})
            )

    execution = asyncio.run(
        _harness(
            tmp_path,
            provider=SlowToolProvider(),
            tools=registry,
            timeout_seconds=0.01,
        ).run("run slow tool")
    )

    assert execution["state"] == RunState.FAILED
    assert execution["error"] == "OperationTimeoutError"
    assert execution["tool_calls"][0]["status"] == "failed"
    assert cancelled == [True]


def test_cancellation_is_persisted(tmp_path: Path) -> None:
    class SlowProvider:
        provider_id = "slow-test"
        mode = "simulated"

        async def complete(self, **_: Any) -> ProviderResponse:
            await asyncio.sleep(2)
            return ProviderResponse(text="too late")

    async def run_cancelled() -> dict[str, Any]:
        cancellation = asyncio.Event()
        harness = _harness(tmp_path, provider=SlowProvider(), timeout_seconds=3)
        task = asyncio.create_task(harness.run("cancel this", cancellation=cancellation))
        await asyncio.sleep(0.02)
        cancellation.set()
        return await task

    execution = asyncio.run(run_cancelled())
    assert execution["state"] == RunState.CANCELLED


def _write_tool(calls: list[dict[str, Any]]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="write_record",
            description="Write a record.",
            action=ToolAction.WRITE,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        _write_and_record(calls),
    )
    return registry


def _write_and_record(calls: list[dict[str, Any]]) -> Any:
    async def handler(arguments: dict[str, Any]) -> dict[str, bool]:
        calls.append(arguments)
        return {"written": True}

    return handler


class _WriteProvider:
    provider_id = "test-provider"
    mode = "simulated"

    async def complete(self, *, messages: list[dict[str, Any]], **_: Any) -> ProviderResponse:
        if any(message["role"] == "tool" for message in messages):
            return ProviderResponse(text="done")
        return ProviderResponse(
            tool_call=ProviderToolCall(
                name="write_record", arguments={"value": "exact value"}
            )
        )


def test_write_tool_is_blocked_until_exact_request_is_approved(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    execution = asyncio.run(
        _harness(
            tmp_path, provider=_WriteProvider(), tools=_write_tool(calls)
        ).run("write something")
    )

    assert execution["state"] == RunState.WAITING_APPROVAL
    assert calls == []
    assert execution["tool_calls"][0]["status"] == "pending"
    assert execution["approvals"][0]["status"] == "pending"


@pytest.mark.parametrize(("approved", "expected_state", "expected_calls"), [
    (True, RunState.COMPLETED, 1),
    (False, RunState.FAILED, 0),
])
def test_approval_and_rejection_are_recorded(
    tmp_path: Path,
    approved: bool,
    expected_state: RunState,
    expected_calls: int,
) -> None:
    calls: list[dict[str, Any]] = []

    async def approve(_: Any) -> tuple[bool, str]:
        return approved, "test-operator"

    execution = asyncio.run(
        _harness(
            tmp_path,
            provider=_WriteProvider(),
            tools=_write_tool(calls),
            approval_handler=approve,
        ).run("write something")
    )

    assert execution["state"] == expected_state
    assert len(calls) == expected_calls
    assert execution["approvals"][0]["status"] == (
        "approved" if approved else "rejected"
    )
    assert execution["approvals"][0]["decided_by"] == "test-operator"
    assert execution["tool_calls"][0]["status"] == (
        "completed" if approved else "rejected"
    )


def test_tool_arguments_are_validated_before_handler_is_called(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class InvalidProvider:
        provider_id = "invalid-test"
        mode = "simulated"

        async def complete(self, **_: Any) -> ProviderResponse:
            return ProviderResponse(
                tool_call=ProviderToolCall(
                    name="echo",
                    arguments={"text": "valid", "unauthorized": "field"},
                )
            )

    execution = asyncio.run(
        _harness(tmp_path, provider=InvalidProvider()).run("task")
    )
    assert execution["state"] == RunState.FAILED
    assert execution["error"] == "ToolValidationError"
    assert calls == []


def test_iteration_limit_fails_closed(tmp_path: Path) -> None:
    class RepeatingProvider:
        provider_id = "repeating-test"
        mode = "simulated"

        async def complete(self, **_: Any) -> ProviderResponse:
            return ProviderResponse(
                tool_call=ProviderToolCall(name="echo", arguments={"text": "repeat"})
            )

    execution = asyncio.run(
        _harness(
            tmp_path, provider=RepeatingProvider(), max_iterations=2
        ).run("repeat")
    )

    assert execution["state"] == RunState.FAILED
    assert execution["error"] == "RunError"
    assert len(execution["tool_calls"]) == 2


def test_secrets_in_task_or_tool_arguments_do_not_enter_event_logs(tmp_path: Path) -> None:
    secret = "not-for-logs-123"
    execution = asyncio.run(
        _harness(tmp_path).run(f"password={secret}")
    )

    assert secret not in repr(execution["events"])
    assert secret not in repr(execution["tool_calls"])
