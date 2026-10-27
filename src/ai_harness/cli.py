"""Command-line entry point for the provider-backed harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_harness.agents import allowed_roots_from_environment, make_real_agent_registry
from ai_harness.planning import create_dynamic_plan
from ai_harness.providers import provider_from_environment
from ai_harness.runtime import (
    Harness,
    SQLiteExecutionStore,
    SimulatedProvider,
    ToolRegistry,
    default_database_path,
    make_simulated_registry,
)
from ai_harness.maestro import Maestro, MaestroError, SQLiteMaestroStore


def _print_event(event: dict[str, Any]) -> None:
    timestamp = datetime.fromisoformat(event["timestamp"]).astimezone().strftime("%H:%M:%S")
    details = [f"execution={event['execution_id']}", f"status={event['status']}"]
    if event["tool_name"]:
        details.append(f"tool={event['tool_name']}")
    if event["duration_ms"] is not None:
        details.append(f"duration_ms={event['duration_ms']:.1f}")
    if event["summary"]:
        details.append(event["summary"])
    print(f"[{timestamp}] {event['event_type']} " + " ".join(details), flush=True)


def _format_run(
    store: SQLiteExecutionStore, task: str, database: Path, provider_name: str
) -> int:
    print(f"History database: {database}", flush=True)
    if provider_name == "simulated":
        print("Mode: simulated (no model or external API was called).", flush=True)
        provider = SimulatedProvider()
        tools = make_simulated_registry()
    else:
        provider = provider_from_environment(provider_name)
        print(
            f"Mode: real provider={provider.provider_id} model={provider.model}.",
            flush=True,
        )
        tools = ToolRegistry()
    harness = Harness(
        store=store,
        provider=provider,
        tools=tools,
        observer=_print_event,
    )
    execution = asyncio.run(harness.run(task))
    print(f"Execution ID: {execution['execution_id']}")
    print(f"State: {execution['state']}")
    if execution["result"]:
        print(f"Result: {execution['result']}")
    if execution["error"]:
        print(f"Error: {execution['error']}")
        return 1
    return 0


def _format_status(store: SQLiteExecutionStore, execution_id: str) -> int:
    execution = store.get(execution_id)
    if execution is None:
        print(f"Execution not found: {execution_id}")
        return 1
    print(json.dumps(execution, ensure_ascii=False, indent=2, default=str))
    return 0


def _format_history(store: SQLiteExecutionStore, limit: int) -> int:
    records = store.history(limit=limit)
    if not records:
        print("No executions found.")
        return 0
    for record in records:
        print(
            f"{record['execution_id']}  {record['state']}  "
            f"{record['mode']}  {record['task']}"
        )
    return 0


def _maestro_command(
    store: SQLiteExecutionStore,
    database: Path,
    operation: str,
    run_id: str | None = None,
    *,
    task: str | None = None,
    approve: bool = False,
    reject: bool = False,
    agent_id: str | None = None,
    limit: int = 20,
    agent_config: Path | None = None,
    provider_name: str = "ollama",
) -> int:
    provider = None
    plan = None
    if provider_name == "simulated":
        maestro = Maestro(
            store=store,
            agent_config=agent_config,
            simulation=True,
        )
    else:
        if agent_config is not None:
            raise ValueError("--agents-config is supported only with --provider simulated.")
        provider = provider_from_environment(provider_name)
        roots = allowed_roots_from_environment()
        if operation == "run" and task is not None:
            plan = asyncio.run(create_dynamic_plan(provider, task))
            planner = lambda _objective: plan
        else:
            def planner(_objective: str):
                raise MaestroError("No plan generation was requested for this command.")

        maestro = Maestro(
            store=store,
            planner=planner,
            registry=make_real_agent_registry(
                provider=provider,
                allowed_roots=roots,
            ),
            tools=ToolRegistry(),
            task_context={"workspace_path": str(roots[0])},
        )
    history = SQLiteMaestroStore(database)
    if operation == "agents":
        print(json.dumps(maestro.registry.all(), ensure_ascii=False, indent=2))
        return 0
    if operation == "approvals":
        print(json.dumps(history.pending_approvals(), ensure_ascii=False, indent=2))
        return 0
    if operation == "history":
        print(json.dumps(history.list_runs(limit), ensure_ascii=False, indent=2))
        return 0
    if operation == "run":
        if task is None:
            return 2
        execution = maestro.create_plan(task)
        print(json.dumps(execution, ensure_ascii=False, indent=2))
        if not approve:
            print("Plan awaits approval. Approve with: harness maestro approve <run_id>")
            return 0
        finished = asyncio.run(
            maestro.approve_plan(execution["run_id"], approved=True)
        )
        print(json.dumps(finished, ensure_ascii=False, indent=2))
        return 0 if finished["state"] == "completed" else 1
    if run_id is None:
        return 2
    if operation == "approve":
        finished = asyncio.run(
            maestro.approve_plan(run_id, approved=not reject)
        )
        print(json.dumps(finished, ensure_ascii=False, indent=2))
        return 0 if finished["state"] in {"completed", "cancelled"} else 1
    if operation == "cancel":
        cancelled = asyncio.run(maestro.cancel(run_id))
        print("Cancellation requested." if cancelled else "Run is not cancellable.")
        return 0 if cancelled else 1
    execution = history.run(run_id)
    if execution is None:
        print(f"Maestro run not found: {run_id}")
        return 1
    if operation in {"status", "plan", "tasks", "messages"}:
        if operation == "plan":
            output: Any = execution["plan"]
        elif operation == "tasks":
            output = [
                task for task in execution["tasks"]
                if agent_id is None or task["agent_id"] == agent_id
            ]
        elif operation == "messages":
            output = execution["messages"]
        else:
            output = execution
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument(
        "--db",
        type=Path,
        default=default_database_path(),
        help="SQLite history path (default: per-user application data directory).",
    )
    parser.add_argument(
        "--agents-config",
        type=Path,
        help="Optional JSON file configuring simulated agents.",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "anthropic", "simulated"),
        default=os.getenv("AI_HARNESS_PROVIDER", "ollama"),
        help="Model provider for new runs (default: AI_HARNESS_PROVIDER or ollama).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a task using the selected provider.")
    run.add_argument("task")
    status = commands.add_parser("status", help="Show a persisted execution.")
    status.add_argument("execution_id")
    history = commands.add_parser("history", help="List persisted executions.")
    history.add_argument("--limit", type=int, default=20)
    tools = commands.add_parser("tools", help="Inspect authorized tools.")
    tools.add_argument("tools_command", choices=("list",))
    maestro = commands.add_parser("maestro", help="Plan and coordinate agents.")
    operations = maestro.add_subparsers(dest="maestro_command", required=True)
    maestro_run = operations.add_parser("run", help="Plan a task; approve to execute it.")
    maestro_run.add_argument("task")
    maestro_run.add_argument("--approve", action="store_true")
    maestro_approve = operations.add_parser("approve", help="Approve or reject a plan.")
    maestro_approve.add_argument("run_id")
    maestro_approve.add_argument("--reject", action="store_true")
    maestro_status = operations.add_parser("status", help="Show a run and its history.")
    maestro_status.add_argument("run_id")
    maestro_plan = operations.add_parser("plan", help="Show a persisted execution plan.")
    maestro_plan.add_argument("run_id")
    maestro_tasks = operations.add_parser("tasks", help="List tasks, optionally by agent.")
    maestro_tasks.add_argument("run_id")
    maestro_tasks.add_argument("--agent")
    maestro_messages = operations.add_parser("messages", help="List run communications.")
    maestro_messages.add_argument("run_id")
    maestro_history = operations.add_parser("history", help="List Maestro runs.")
    maestro_history.add_argument("--limit", type=int, default=20)
    operations.add_parser("agents", help="List configured agents and their states.")
    operations.add_parser("approvals", help="List plans awaiting approval.")
    maestro_cancel = operations.add_parser("cancel", help="Cancel a pending or running plan.")
    maestro_cancel.add_argument("run_id")
    args = parser.parse_args(argv)
    store = SQLiteExecutionStore(args.db)

    if args.command == "run":
        return _format_run(store, args.task, args.db, args.provider)
    if args.command == "status":
        return _format_status(store, args.execution_id)
    if args.command == "history":
        if args.limit < 1:
            parser.error("--limit must be greater than zero.")
        return _format_history(store, args.limit)
    if args.command == "tools":
        if args.provider == "simulated":
            for definition in make_simulated_registry().definitions():
                print(
                    f"{definition.name} [{definition.action}]: {definition.description}"
                )
        else:
            print("No task tools are enabled for real providers; agents are read-only.")
        return 0
    if args.command == "maestro":
        if args.maestro_command == "history" and args.limit < 1:
            parser.error("--limit must be greater than zero.")
        return _maestro_command(
            store,
            args.db,
            args.maestro_command,
            getattr(args, "run_id", None),
            task=getattr(args, "task", None),
            approve=getattr(args, "approve", False),
            reject=getattr(args, "reject", False),
            agent_id=getattr(args, "agent", None),
            limit=getattr(args, "limit", 20),
            agent_config=args.agents_config,
            provider_name=args.provider,
        )
    return 2
