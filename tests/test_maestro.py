import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from ai_harness.maestro import (
    AgentRegistration,
    AgentRegistry,
    AgentResult,
    AgentTask,
    ExecutionPlan,
    Maestro,
    MaestroError,
    MaestroLimits,
    MaestroRunState,
    MessageRejectedError,
    PlanTask,
    PlanValidationError,
    SimulatedSpecialist,
    SQLiteMaestroStore,
    TaskState,
    make_simulated_agent_registry,
)
from ai_harness.cli import main
from ai_harness.policies import ToolAction
from ai_harness.runtime import (
    SQLiteExecutionStore,
    ToolDefinition,
    ToolRegistry,
    ToolValidationError,
)


def _store(tmp_path: Path) -> SQLiteExecutionStore:
    return SQLiteExecutionStore(tmp_path / "history.sqlite3")


def _maestro(tmp_path: Path, **kwargs: Any) -> Maestro:
    return Maestro(store=_store(tmp_path), **kwargs)


def test_maestro_creates_identifiable_plan_before_execution(tmp_path: Path) -> None:
    run = _maestro(tmp_path).create_plan("Review this proposal.")

    assert run["state"] == MaestroRunState.AWAITING_APPROVAL
    assert len(run["plan"]["tasks"]) == 2
    assert all(task["task_id"] and task["agent_id"] for task in run["tasks"])
    assert all(task["status"] == TaskState.PENDING for task in run["tasks"])


def test_task_delegation_routes_message_and_result_to_maestro(tmp_path: Path) -> None:
    maestro = _maestro(tmp_path)
    plan = maestro.create_plan("Analyze and review.")
    finished = asyncio.run(maestro.approve_plan(plan["run_id"], approved=True))

    delegated = [message for message in finished["messages"] if message["message_type"] == "task"]
    returned = [message for message in finished["messages"] if message["message_type"] == "result"]
    assert {message["recipient_id"] for message in delegated} == {"analyst", "reviewer"}
    assert {message["sender_id"] for message in returned} == {"analyst", "reviewer"}
    assert all(message["recipient_id"] == "maestro" for message in returned)
    assert all(message["parent_message_id"] for message in returned)
    assert all(message["artifact_refs"][0].startswith("execution:") for message in returned)
    assert finished["state"] == MaestroRunState.COMPLETED


def test_run_history_reconstructs_messages_after_store_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    maestro = Maestro(store=store)
    plan = maestro.create_plan("Persist communication.")
    finished = asyncio.run(maestro.approve_plan(plan["run_id"], approved=True))
    reopened = SQLiteMaestroStore(store.path).run(finished["run_id"])

    assert reopened is not None
    assert [m["message_id"] for m in reopened["messages"]] == [
        m["message_id"] for m in finished["messages"]
    ]
    assert [event["event_type"] for event in reopened["events"]][:2] == [
        "plan.created", "plan.approved"
    ]


def test_independent_tasks_respect_configured_concurrency(tmp_path: Path) -> None:
    class SlowAgent:
        def __init__(self, tracker: dict[str, int]) -> None:
            self.tracker = tracker

        async def execute(self, task: AgentTask, *, cancellation: asyncio.Event) -> AgentResult:
            self.tracker["active"] += 1
            self.tracker["maximum"] = max(
                self.tracker["maximum"], self.tracker["active"]
            )
            try:
                await asyncio.sleep(0.03)
                return AgentResult(summary=task.description)
            finally:
                self.tracker["active"] -= 1

    tracker = {"active": 0, "maximum": 0}
    registry = AgentRegistry()
    for agent_id, capability in (("analyst", "analysis"), ("reviewer", "review")):
        registry.register(
            AgentRegistration(
                agent_id=agent_id, name=agent_id, description="simulated",
                specialty=capability, provider="test", model="test",
                capabilities=(capability,),
            ),
            SlowAgent(tracker),
        )
    maestro = _maestro(
        tmp_path, registry=registry,
        limits=MaestroLimits(max_concurrent_agents=2, max_concurrent_tasks=2),
    )
    plan = maestro.create_plan("Run independent work.")
    finished = asyncio.run(maestro.approve_plan(plan["run_id"], approved=True))

    assert tracker["maximum"] == 2
    assert finished["state"] == MaestroRunState.COMPLETED


def test_agent_failure_is_recorded_without_losing_other_results(tmp_path: Path) -> None:
    class Agent:
        def __init__(self, fails: bool) -> None:
            self.fails = fails

        async def execute(self, task: AgentTask, *, cancellation: asyncio.Event) -> AgentResult:
            if self.fails:
                raise RuntimeError("private detail must not be persisted")
            return AgentResult(summary="Completed.", claims=("result: available",))

    registry = AgentRegistry()
    for agent_id, capability, fails in (
        ("analyst", "analysis", True), ("reviewer", "review", False)
    ):
        registry.register(
            AgentRegistration(
                agent_id=agent_id, name=agent_id, description="test",
                specialty=capability, provider="test", model="test",
                capabilities=(capability,),
            ),
            Agent(fails),
        )
    maestro = _maestro(tmp_path, registry=registry)
    plan = maestro.create_plan("Exercise failure handling.")
    finished = asyncio.run(maestro.approve_plan(plan["run_id"], approved=True))

    assert finished["state"] == MaestroRunState.PARTIAL
    assert any(message["message_type"] == "error" for message in finished["messages"])
    assert "private detail" not in repr(finished)
    assert finished["result"]["incomplete_tasks"]


def test_task_dependency_cycle_is_rejected() -> None:
    first_id, second_id = UUID(int=1), UUID(int=2)
    with pytest.raises(PlanValidationError, match="cycle"):
        ExecutionPlan(tasks=(
            PlanTask(
                task_id=first_id, description="one",
                required_capability="analysis", dependencies=(second_id,),
            ),
            PlanTask(
                task_id=second_id, description="two",
                required_capability="review", dependencies=(first_id,),
            ),
        ))


def test_agent_cannot_invoke_tool_not_in_its_authorized_registry(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def write_record(arguments: dict[str, Any]) -> dict[str, bool]:
        calls.append(arguments)
        return {"written": True}

    tools = ToolRegistry()
    tools.register(
        ToolDefinition(
            name="write_record", description="Protected write.", action=ToolAction.WRITE,
            parameters={
                "type": "object", "properties": {"value": {"type": "string"}},
                "required": ["value"], "additionalProperties": False,
            },
        ),
        write_record,
    )
    runtime_store = _store(tmp_path)
    registry = make_simulated_agent_registry(runtime_store, tools=tools)
    _, adapter, _ = registry.get("analyst")

    assert isinstance(adapter, SimulatedSpecialist)
    assert adapter.tools.definitions() == ()
    with pytest.raises(ToolValidationError):
        asyncio.run(adapter.tools.invoke("write_record", {"value": "not approved"}))
    assert calls == []


def test_cancellation_interrupts_active_adapter(tmp_path: Path) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowAgent:
        async def execute(self, task: AgentTask, *, cancellation: asyncio.Event) -> AgentResult:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return AgentResult(summary="too late")

    registry = AgentRegistry()
    for agent_id, capability in (("analyst", "analysis"), ("reviewer", "review")):
        registry.register(
            AgentRegistration(
                agent_id=agent_id, name=agent_id, description="test",
                specialty=capability, provider="test", model="test",
                capabilities=(capability,),
            ),
            SlowAgent(),
        )
    maestro = _maestro(tmp_path, registry=registry)
    plan = maestro.create_plan("Cancel this.")

    async def cancel_run() -> dict[str, Any]:
        running = asyncio.create_task(
            maestro.approve_plan(plan["run_id"], approved=True)
        )
        await started.wait()
        assert await maestro.cancel(plan["run_id"])
        return await running

    finished = asyncio.run(cancel_run())
    assert finished["state"] == MaestroRunState.CANCELLED
    assert cancelled.is_set()


def test_consolidated_answer_reports_divergent_claims(tmp_path: Path) -> None:
    class PositionAgent:
        def __init__(self, position: str) -> None:
            self.position = position

        async def execute(self, task: AgentTask, *, cancellation: asyncio.Event) -> AgentResult:
            return AgentResult(
                summary="Review complete.", claims=(f"recommendation: {self.position}",)
            )

    registry = AgentRegistry()
    for agent_id, capability, position in (
        ("analyst", "analysis", "proceed"),
        ("reviewer", "review", "hold"),
    ):
        registry.register(
            AgentRegistration(
                agent_id=agent_id, name=agent_id, description="test",
                specialty=capability, provider="test", model="test",
                capabilities=(capability,),
            ),
            PositionAgent(position),
        )
    maestro = _maestro(tmp_path, registry=registry)
    plan = maestro.create_plan("Return independent positions.")
    finished = asyncio.run(maestro.approve_plan(plan["run_id"], approved=True))

    assert finished["result"]["divergences"] == [{
        "topic": "recommendation",
        "positions": {"proceed": ["analyst"], "hold": ["reviewer"]},
    }]
    assert finished["result"]["limitations"]


def test_agent_direct_messages_are_rejected_and_recorded(tmp_path: Path) -> None:
    maestro = _maestro(tmp_path)
    run = maestro.create_plan("Central communication only.")
    task_id = run["tasks"][0]["task_id"]

    with pytest.raises(MessageRejectedError):
        maestro.send_direct_message(
            sender_id="analyst", recipient_id="reviewer",
            run_id=run["run_id"], task_id=task_id,
        )
    messages = maestro.store.messages(run["run_id"])
    assert messages[-1]["status"] == "rejected"


def test_plan_can_be_rejected_without_delegating(tmp_path: Path) -> None:
    maestro = _maestro(tmp_path)
    plan = maestro.create_plan("Do not execute yet.")
    rejected = asyncio.run(maestro.approve_plan(plan["run_id"], approved=False))

    assert rejected["state"] == MaestroRunState.CANCELLED
    assert not any(message["message_type"] == "task" for message in rejected["messages"])


def test_review_request_is_relayed_to_reviewer_and_result_is_persisted(
    tmp_path: Path,
) -> None:
    maestro = _maestro(tmp_path)
    plan = maestro.create_plan("Request a separate review.")
    completed = asyncio.run(maestro.approve_plan(plan["run_id"], approved=True))

    review_message = asyncio.run(
        maestro.request_review(
            completed["run_id"], completed["tasks"][0]["task_id"],
            reviewer_id="reviewer", request="Check for omissions.",
        )
    )
    messages = maestro.store.messages(completed["run_id"])
    request = next(message for message in messages if message["message_id"] == str(review_message))
    result = next(message for message in messages if message["parent_message_id"] == str(review_message))

    assert request["message_type"] == "review_request"
    assert request["recipient_id"] == "reviewer"
    assert result["message_type"] == "result"
    assert result["recipient_id"] == "maestro"
    refreshed = maestro.store.run(completed["run_id"])
    assert refreshed is not None
    assert any(item.get("review") for item in refreshed["result"]["task_results"])


def test_agent_registry_can_be_configured_and_external_provider_is_rejected(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "agents.json"
    config_path.write_text(json.dumps({
        "agents": [{
            "agent_id": "custom-reviewer", "name": "Configured Reviewer",
            "description": "Test specialist.", "specialty": "review",
            "provider": "simulated", "model": "deterministic",
            "capabilities": ["review"], "authorized_tools": [],
            "permissions": [], "enabled": False,
        }]
    }), encoding="utf-8")
    configured = Maestro(store=_store(tmp_path), agent_config=config_path)

    assert configured.registry.all()[0]["state"] == "disabled"
    config_path.write_text(json.dumps({
        "agents": [{
            "agent_id": "real-model", "name": "External",
            "description": "Not actually connected.", "specialty": "analysis",
            "provider": "openai", "model": "example",
        }]
    }), encoding="utf-8")
    with pytest.raises(MaestroError, match="no external provider is connected"):
        Maestro(store=_store(tmp_path), agent_config=config_path)


def test_maestro_cli_plan_approve_and_status_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "cli.sqlite3"
    assert main([
        "--db", str(database), "maestro", "run", "CLI smoke test"
    ]) == 0
    plan_output = capsys.readouterr().out
    run_id = next(
        line.split('"run_id": "')[1].split('"')[0]
        for line in plan_output.splitlines() if '"run_id": "' in line
    )

    assert main(["--db", str(database), "maestro", "approve", run_id]) == 0
    assert '"state": "completed"' in capsys.readouterr().out
    assert main(["--db", str(database), "maestro", "status", run_id]) == 0
    assert '"messages"' in capsys.readouterr().out


def test_dependent_task_runs_only_after_prerequisite(tmp_path: Path) -> None:
    order: list[str] = []
    contexts: list[dict[str, Any]] = []

    class OrderedAgent:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        async def execute(self, task: AgentTask, *, cancellation: asyncio.Event) -> AgentResult:
            order.append(self.marker)
            contexts.append(task.context)
            return AgentResult(summary=self.marker)

    dependency_id = UUID(int=9)
    plan_spec = ExecutionPlan(tasks=(
        PlanTask(
            task_id=dependency_id, description="First", required_capability="analysis"
        ),
        PlanTask(
            task_id=UUID(int=10), description="Second", required_capability="review",
            dependencies=(dependency_id,),
        ),
    ))
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            agent_id="analyst", name="analyst", description="test",
            specialty="analysis", provider="test", model="test",
            capabilities=("analysis",),
        ),
        OrderedAgent("first"),
    )
    registry.register(
        AgentRegistration(
            agent_id="reviewer", name="reviewer", description="test",
            specialty="review", provider="test", model="test",
            capabilities=("review",),
        ),
        OrderedAgent("second"),
    )
    maestro = _maestro(tmp_path, registry=registry, planner=lambda _: plan_spec)
    planned = maestro.create_plan("Execute dependency order.")
    finished = asyncio.run(maestro.approve_plan(planned["run_id"], approved=True))

    assert order == ["first", "second"]
    assert list(contexts[1]["dependency_results"].values())[0]["summary"] == "first"
    assert finished["state"] == MaestroRunState.COMPLETED
