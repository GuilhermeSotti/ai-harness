"""Initial tool risk classification and request-bound approval decisions."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_harness.core import HarnessError


class ToolAction(StrEnum):
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


class ActionPendingApprovalError(HarnessError):
    def __init__(self, action: ToolAction) -> None:
        self.action = action
        super().__init__(f"Action {action} is pending human approval.")


class ActionRejectedError(HarnessError):
    def __init__(self, action: ToolAction) -> None:
        self.action = action
        super().__init__(f"Action {action} was rejected by policy.")


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Tool arguments cannot contain non-finite numbers.")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Tool argument object keys must be strings.")
            _validate_json_value(item)
        return
    raise ValueError(f"Unsupported tool argument value: {type(value).__name__}.")


def _canonical_json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("Tool arguments must be a JSON object.")
    arguments = dict(value)
    _validate_json_value(arguments)
    return json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class ToolRequest(BaseModel):
    """Immutable snapshot of the exact tool call requiring policy evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    execution_id: UUID | None = None
    tool_name: str = Field(min_length=1)
    action: ToolAction
    arguments_json: str

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Tool name cannot be blank.")
        return value

    @field_validator("arguments_json")
    @classmethod
    def validate_arguments_json(cls, value: str) -> str:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments must be valid JSON.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return _canonical_json(decoded)

    @classmethod
    def create(
        cls,
        *,
        tool_name: str,
        action: ToolAction,
        arguments: Mapping[str, Any],
        execution_id: UUID | None = None,
    ) -> ToolRequest:
        return cls(
            execution_id=execution_id,
            tool_name=tool_name,
            action=action,
            arguments_json=_canonical_json(arguments),
        )

    @property
    def arguments(self) -> dict[str, Any]:
        """Return a fresh copy so callers cannot mutate the approved snapshot."""
        return json.loads(self.arguments_json)

    @property
    def arguments_digest(self) -> str:
        return hashlib.sha256(self.arguments_json.encode("utf-8")).hexdigest()


class ToolApproval(BaseModel):
    """Signed approval bound to one request and one approver identity label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: UUID
    execution_id: UUID | None = None
    tool_name: str
    action: ToolAction
    arguments_digest: str
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    signature: str


class ToolApprovalAuthority:
    """In-process capability for a trusted human-approval integration."""

    __slots__ = ("__secret",)

    def __init__(self) -> None:
        self.__secret = secrets.token_bytes(32)

    @staticmethod
    def _payload(
        *,
        request_id: UUID,
        execution_id: UUID | None,
        tool_name: str,
        action: ToolAction,
        arguments_digest: str,
        approved_by: str,
        approved_at: datetime,
    ) -> bytes:
        return json.dumps(
            {
                "request_id": str(request_id),
                "execution_id": (
                    str(execution_id) if execution_id is not None else None
                ),
                "tool_name": tool_name,
                "action": action.value,
                "arguments_digest": arguments_digest,
                "approved_by": approved_by,
                "approved_at": approved_at.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def approve(self, request: ToolRequest, *, approved_by: str) -> ToolApproval:
        """Issue a grant only after the trusted caller receives human approval."""
        approver = approved_by.strip()
        if not approver:
            raise ValueError("An approver identity is required.")

        approved_at = datetime.now(UTC)
        payload = self._payload(
            request_id=request.request_id,
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            action=request.action,
            arguments_digest=request.arguments_digest,
            approved_by=approver,
            approved_at=approved_at,
        )
        signature = hmac.new(self.__secret, payload, hashlib.sha256).hexdigest()
        return ToolApproval(
            request_id=request.request_id,
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            action=request.action,
            arguments_digest=request.arguments_digest,
            approved_by=approver,
            approved_at=approved_at,
            signature=signature,
        )

    def verifies(self, request: ToolRequest, approval: ToolApproval) -> bool:
        if (
            approval.request_id != request.request_id
            or approval.execution_id != request.execution_id
            or approval.tool_name != request.tool_name
            or approval.action is not request.action
            or approval.arguments_digest != request.arguments_digest
            or approval.approved_at.tzinfo is None
        ):
            return False

        payload = self._payload(
            request_id=approval.request_id,
            execution_id=approval.execution_id,
            tool_name=approval.tool_name,
            action=approval.action,
            arguments_digest=approval.arguments_digest,
            approved_by=approval.approved_by,
            approved_at=approval.approved_at,
        )
        expected = hmac.new(self.__secret, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, approval.signature)


class PolicyResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: ToolAction
    risk: RiskLevel
    decision: PolicyDecision
    reason: str
    request: ToolRequest | None = None
    approval: ToolApproval | None = None

    def require_approved(
        self, *, authority: ToolApprovalAuthority | None = None
    ) -> None:
        if self.decision is PolicyDecision.PENDING:
            raise ActionPendingApprovalError(self.action)
        if self.decision is PolicyDecision.REJECTED:
            raise ActionRejectedError(self.action)
        if self.action is ToolAction.READ:
            return
        if (
            self.request is None
            or self.approval is None
            or authority is None
            or not authority.verifies(self.request, self.approval)
        ):
            raise ActionPendingApprovalError(self.action)


class ToolPolicy(BaseModel):
    """Conservative defaults; sensitive actions require a request-bound grant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocked_actions: frozenset[ToolAction] = Field(default_factory=frozenset)

    def evaluate(
        self,
        action: ToolAction | ToolRequest,
        *,
        approval: ToolApproval | None = None,
        authority: ToolApprovalAuthority | None = None,
    ) -> PolicyResult:
        if isinstance(action, ToolRequest):
            request = action
            resolved_action = action.action
        else:
            request = None
            resolved_action = action
        risk = {
            ToolAction.READ: RiskLevel.LOW,
            ToolAction.WRITE: RiskLevel.HIGH,
            ToolAction.EXTERNAL: RiskLevel.CRITICAL,
        }[resolved_action]

        if resolved_action in self.blocked_actions:
            return PolicyResult(
                action=resolved_action,
                risk=risk,
                decision=PolicyDecision.REJECTED,
                reason="This action is explicitly blocked by policy.",
                request=request,
            )

        if resolved_action is ToolAction.READ:
            return PolicyResult(
                action=resolved_action,
                risk=risk,
                decision=PolicyDecision.APPROVED,
                reason="Read actions are approved by the initial policy.",
                request=request,
            )

        if (
            request is not None
            and approval is not None
            and authority is not None
            and authority.verifies(request, approval)
        ):
            return PolicyResult(
                action=resolved_action,
                risk=risk,
                decision=PolicyDecision.APPROVED,
                reason="A valid approval matches this exact tool request.",
                request=request,
                approval=approval,
            )

        return PolicyResult(
            action=resolved_action,
            risk=risk,
            decision=PolicyDecision.PENDING,
            reason=(
                "This action requires human approval for this exact tool "
                "request before execution."
            ),
            request=request,
        )
