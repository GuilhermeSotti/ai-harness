import pytest

from ai_harness import (
    ActionPendingApprovalError,
    ActionRejectedError,
    PolicyDecision,
    PolicyResult,
    RiskLevel,
    ToolAction,
    ToolApprovalAuthority,
    ToolPolicy,
    ToolRequest,
)


def test_read_action_is_low_risk_and_approved() -> None:
    result = ToolPolicy().evaluate(ToolAction.READ)

    assert result.risk is RiskLevel.LOW
    assert result.decision is PolicyDecision.APPROVED
    result.require_approved()


@pytest.mark.parametrize(
    ("action", "risk"),
    [
        (ToolAction.WRITE, RiskLevel.HIGH),
        (ToolAction.EXTERNAL, RiskLevel.CRITICAL),
    ],
)
def test_sensitive_actions_remain_pending(action: ToolAction, risk: RiskLevel) -> None:
    result = ToolPolicy().evaluate(action)

    assert result.risk is risk
    assert result.decision is PolicyDecision.PENDING
    with pytest.raises(ActionPendingApprovalError):
        result.require_approved()


def test_explicitly_blocked_action_is_rejected() -> None:
    result = ToolPolicy(blocked_actions=frozenset({ToolAction.WRITE})).evaluate(
        ToolAction.WRITE
    )

    assert result.decision is PolicyDecision.REJECTED
    with pytest.raises(ActionRejectedError):
        result.require_approved()


def test_approval_is_bound_to_exact_request_tool_arguments_and_id() -> None:
    authority = ToolApprovalAuthority()
    policy = ToolPolicy()
    request = ToolRequest.create(
        tool_name="write_file",
        action=ToolAction.WRITE,
        arguments={"path": "report.txt", "content": "approved"},
    )
    approval = authority.approve(request, approved_by="operator")

    approved = policy.evaluate(request, approval=approval, authority=authority)
    assert approved.decision is PolicyDecision.APPROVED
    approved.require_approved(authority=authority)

    mismatched_requests = (
        ToolRequest(
            request_id=request.request_id,
            tool_name="delete_file",
            action=ToolAction.WRITE,
            arguments_json=request.arguments_json,
        ),
        ToolRequest(
            request_id=request.request_id,
            tool_name="write_file",
            action=ToolAction.WRITE,
            arguments_json='{"content":"changed","path":"report.txt"}',
        ),
        ToolRequest.create(
            tool_name="write_file",
            action=ToolAction.WRITE,
            arguments={"path": "report.txt", "content": "approved"},
        ),
        ToolRequest.create(
            tool_name="write_file",
            action=ToolAction.WRITE,
            arguments={"content": "approved", "path": "report.txt"},
        ),
    )

    for mismatched_request in mismatched_requests:
        result = policy.evaluate(
            mismatched_request, approval=approval, authority=authority
        )
        assert result.decision is PolicyDecision.PENDING
        with pytest.raises(ActionPendingApprovalError):
            result.require_approved(authority=authority)


def test_approval_cannot_be_used_without_its_authority() -> None:
    authority = ToolApprovalAuthority()
    request = ToolRequest.create(
        tool_name="send_message",
        action=ToolAction.EXTERNAL,
        arguments={"recipient": "user@example.test", "body": "hello"},
    )
    approval = authority.approve(request, approved_by="operator")

    result = ToolPolicy().evaluate(request, approval=approval, authority=authority)

    with pytest.raises(ActionPendingApprovalError):
        result.require_approved()

    with pytest.raises(ActionPendingApprovalError):
        result.require_approved(authority=ToolApprovalAuthority())


def test_forged_approved_policy_result_fails_closed() -> None:
    authority = ToolApprovalAuthority()
    request = ToolRequest.create(
        tool_name="write_file",
        action=ToolAction.WRITE,
        arguments={"path": "report.txt"},
    )
    approval = authority.approve(request, approved_by="operator")
    forged_result = PolicyResult(
        action=ToolAction.WRITE,
        risk=RiskLevel.HIGH,
        decision=PolicyDecision.APPROVED,
        reason="forged decision",
        request=request,
        approval=approval.model_copy(update={"signature": "0" * 64}),
    )

    with pytest.raises(ActionPendingApprovalError):
        forged_result.require_approved(authority=authority)


def test_request_snapshots_arguments_and_rejects_non_json_values() -> None:
    arguments = {"path": "original.txt"}
    request = ToolRequest.create(
        tool_name="write_file",
        action=ToolAction.WRITE,
        arguments=arguments,
    )
    arguments["path"] = "changed.txt"

    assert request.arguments == {"path": "original.txt"}

    with pytest.raises(ValueError, match="Unsupported tool argument value"):
        ToolRequest.create(
            tool_name="write_file",
            action=ToolAction.WRITE,
            arguments={"path": object()},
        )


def test_blocked_action_remains_rejected_even_with_valid_approval() -> None:
    authority = ToolApprovalAuthority()
    request = ToolRequest.create(
        tool_name="write_file",
        action=ToolAction.WRITE,
        arguments={"path": "report.txt"},
    )
    approval = authority.approve(request, approved_by="operator")
    policy = ToolPolicy(blocked_actions=frozenset({ToolAction.WRITE}))

    result = policy.evaluate(request, approval=approval, authority=authority)

    assert result.decision is PolicyDecision.REJECTED
    with pytest.raises(ActionRejectedError):
        result.require_approved(authority=authority)
