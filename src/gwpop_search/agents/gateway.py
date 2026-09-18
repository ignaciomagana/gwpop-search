"""Deterministic gateway from agent proposals to recorded tasks."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from gwpop_search.grammar import ModelGraph

from .contracts import AgentDecision, AgentTaskProposal, validate_agent_proposal


def task_id_for_proposal(proposal: AgentTaskProposal) -> str:
    payload = {
        "proposal_id": proposal.proposal_id,
        "role": proposal.role.value,
        "kind": proposal.kind.value,
        "payload": dict(proposal.payload),
        "rationale": proposal.rationale,
        "source_artifacts": list(proposal.source_artifacts),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"task-{digest[:24]}"


def review_agent_proposal(
    proposal: AgentTaskProposal,
    *,
    graph: ModelGraph,
    agents_enabled: bool,
) -> AgentDecision:
    """Validate a proposal and optionally turn it into a deterministic task id."""
    if not agents_enabled:
        return AgentDecision(
            proposal_id=proposal.proposal_id,
            accepted=False,
            reason="agents_disabled",
        )

    try:
        validate_agent_proposal(proposal, graph=graph)
    except ValueError as exc:
        return AgentDecision(
            proposal_id=proposal.proposal_id,
            accepted=False,
            reason=f"invalid_proposal:{exc}",
        )

    return AgentDecision(
        proposal_id=proposal.proposal_id,
        accepted=True,
        reason="valid_typed_proposal",
        resulting_task_id=task_id_for_proposal(proposal),
    )


def append_agent_log(
    path: str | Path,
    proposal: AgentTaskProposal,
    decision: AgentDecision,
) -> None:
    """Append an immutable JSONL audit record for an agent proposal/decision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "proposal": {
            "proposal_id": proposal.proposal_id,
            "role": proposal.role.value,
            "kind": proposal.kind.value,
            "payload": dict(proposal.payload),
            "rationale": proposal.rationale,
            "source_artifacts": list(proposal.source_artifacts),
            "status": proposal.status,
        },
        "decision": asdict(decision),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
