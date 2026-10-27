"""Read-only stdio MCP server for real AI-Harness agents and Maestro executions."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ai_harness.agents import (
    allowed_roots_from_environment,
    make_real_agent_registry,
)
from ai_harness.maestro import (
    ExecutionPlan,
    Maestro,
    PlanTask,
    SQLiteMaestroStore,
)
from ai_harness.planning import create_dynamic_plan
from ai_harness.providers import StructuredProvider, provider_from_environment
from ai_harness.redaction import redact_sensitive_text
from ai_harness.runtime import (
    SQLiteExecutionStore,
    ToolRegistry,
    default_database_path,
)


class HarnessMCPService:
    def __init__(
        self,
        *,
        provider: StructuredProvider,
        allowed_roots: tuple[str | Path, ...],
        database: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.allowed_roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
        self.evidence_root = self.allowed_roots[0]
        self.database = Path(database) if database else default_database_path()
        self.runtime_store = SQLiteExecutionStore(self.database)
        self.maestro_store = SQLiteMaestroStore(self.database)

    def workspace(self, requested: str | None) -> Path:
        if requested is None:
            return self.evidence_root
        candidate = Path(requested).resolve(strict=True)
        if not candidate.is_dir() or not any(
            candidate == root or root in candidate.parents
            for root in self.allowed_roots
        ):
            raise ValueError("Workspace is outside AI_HARNESS_ALLOWED_ROOTS.")
        return candidate

    def maestro(
        self,
        workspace: Path,
        *,
        planner: Callable[[str], ExecutionPlan],
        context: dict[str, Any] | None = None,
    ) -> Maestro:
        return Maestro(
            store=self.runtime_store,
            planner=planner,
            registry=make_real_agent_registry(
                provider=self.provider,
                allowed_roots=self.allowed_roots,
            ),
            tools=ToolRegistry(),
            task_context={
                "workspace_path": str(workspace),
                **(context or {}),
            },
        )

    async def status(self) -> dict[str, Any]:
        health = await self.provider.health()
        agents = make_real_agent_registry(
            provider=self.provider,
            allowed_roots=self.allowed_roots,
        ).all()
        return {
            "state": "available" if health.available and health.model_available else "degraded",
            "provider": health.model_dump(mode="json"),
            "agents": agents,
            "allowed_roots": [str(root) for root in self.allowed_roots],
            "database": str(self.database),
            "limitations": [
                "Agents inspect bounded text files only; no shell, file writes, or network tools.",
                "Provider status checks endpoint/model presence, not successful inference or performance.",
            ],
        }

    async def list_agents(self) -> list[dict[str, Any]]:
        return list(
            make_real_agent_registry(
                provider=self.provider,
                allowed_roots=self.allowed_roots,
            ).all()
        )

    async def create_plan(
        self, objective: str, workspace_path: str | None
    ) -> dict[str, Any]:
        workspace = self.workspace(workspace_path)
        objective = redact_sensitive_text(objective)
        plan = await create_dynamic_plan(self.provider, objective)
        maestro = self.maestro(workspace, planner=lambda _objective: plan)
        return maestro.create_plan(objective)

    async def run_agent(
        self,
        agent_id: str,
        objective: str,
        workspace_path: str | None,
        review_context: str | None = None,
    ) -> dict[str, Any]:
        if agent_id not in {"analyst", "reviewer"}:
            raise ValueError("agent_id must be 'analyst' or 'reviewer'.")
        workspace = self.workspace(workspace_path)
        objective = redact_sensitive_text(objective)
        if review_context:
            review_context = redact_sensitive_text(review_context)
        capability = "analysis" if agent_id == "analyst" else "review"
        plan = ExecutionPlan(
            tasks=(
                PlanTask(
                    description=objective,
                    required_capability=capability,
                    agent_id=agent_id,
                ),
            )
        )
        context = {"review_context": review_context} if review_context else None
        maestro = self.maestro(
            workspace, planner=lambda _objective: plan, context=context
        )
        created = maestro.create_plan(objective)
        result = await maestro.approve_plan(
            created["run_id"], approved=True, approved_by="mcp-caller-request"
        )
        return result

    def get_execution(self, run_id: str) -> dict[str, Any]:
        identifier = UUID(run_id)
        execution = self.maestro_store.run(identifier)
        if execution is None:
            raise ValueError("Maestro execution was not found.")
        return execution


def create_server(service: HarnessMCPService | None = None) -> FastMCP:
    runtime = service or HarnessMCPService(
        provider=provider_from_environment(),
        allowed_roots=allowed_roots_from_environment(),
        database=os.getenv("AI_HARNESS_DATABASE") or None,
    )
    server = FastMCP("ai-harness")

    @server.tool()
    async def harness_status() -> dict[str, Any]:
        """Return the actual provider health, agent capabilities, and filesystem limits."""
        return await runtime.status()

    @server.tool()
    async def harness_list_agents() -> list[dict[str, Any]]:
        """List the real, configured analyst and reviewer agents."""
        return await runtime.list_agents()

    @server.tool()
    async def harness_create_plan(
        objective: Annotated[str, Field(min_length=1, max_length=8192)],
        workspace_path: Annotated[str | None, Field(max_length=4096)] = None,
    ) -> dict[str, Any]:
        """Generate, validate, and persist a provider-backed plan awaiting approval."""
        return await runtime.create_plan(objective, workspace_path)

    @server.tool()
    async def harness_run_agent(
        agent_id: Literal["analyst", "reviewer"],
        objective: Annotated[str, Field(min_length=1, max_length=8192)],
        workspace_path: Annotated[str | None, Field(max_length=4096)] = None,
    ) -> dict[str, Any]:
        """Run the selected read-only real agent and persist its result."""
        return await runtime.run_agent(agent_id, objective, workspace_path)

    @server.tool()
    async def harness_get_execution(run_id: UUID) -> dict[str, Any]:
        """Read the persisted Maestro plan, state, task results, and execution events."""
        return runtime.get_execution(str(run_id))

    @server.tool()
    async def harness_review_changes(
        objective: Annotated[str, Field(min_length=1, max_length=8192)],
        change_summary: Annotated[str, Field(min_length=1, max_length=16384)],
        workspace_path: Annotated[str | None, Field(max_length=4096)] = None,
    ) -> dict[str, Any]:
        """Run the independent reviewer against allowed source files and supplied change context."""
        return await runtime.run_agent(
            "reviewer",
            objective,
            workspace_path,
            review_context=change_summary,
        )

    return server


def main() -> None:
    logging.basicConfig(
        level=os.getenv("AI_HARNESS_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
