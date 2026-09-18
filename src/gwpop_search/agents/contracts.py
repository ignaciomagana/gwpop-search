"""Typed contracts for optional search agents.

Agents may propose legal tasks. They do not execute inference, edit likelihoods,
or create arbitrary model code through this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Mapping

from gwpop_search.grammar import DEFAULT_MUTATIONS, ModelGraph


class AgentRole(str, Enum):
    SEARCH_PROPOSER = "search_proposer"
    NUMERICAL_CRITIC = "numerical_critic"
    RESIDUAL_PROPOSER = "residual_proposer"
    ADVERSARIAL_VALIDATOR = "adversarial_validator"
    SYNTHESIZER = "synthesizer"


class AgentTaskKind(str, Enum):
    EVALUATE_MODEL = "evaluate_model"
    PROPOSE_MUTATION = "propose_mutation"
    REQUEST_VALIDATION = "request_validation"
    REVIEW_SCOUT = "review_scout"
    SYNTHESIZE_RESULTS = "synthesize_results"


_ALLOWED_PAYLOAD_KEYS = {
    AgentTaskKind.EVALUATE_MODEL: {
        "model_hash",
        "requested_fidelity",
        "reason",
    },
    AgentTaskKind.PROPOSE_MUTATION: {
        "parent_hash",
        "mutation_id",
        "reason",
    },
    AgentTaskKind.REQUEST_VALIDATION: {
        "model_hash",
        "validation",
        "reason",
    },
    AgentTaskKind.REVIEW_SCOUT: {
        "proposal_id",
        "mutation_id",
        "reason",
    },
    AgentTaskKind.SYNTHESIZE_RESULTS: {
        "artifact_ids",
        "question",
    },
}

_FORBIDDEN_KEY_FRAGMENTS = (
    "code",
    "script",
    "shell",
    "command",
    "patch",
    "diff",
    "source",
)


@dataclass(frozen=True)
class AgentTaskProposal:
    proposal_id: str
    role: AgentRole
    kind: AgentTaskKind
    payload: Mapping[str, object]
    rationale: str
    source_artifacts: tuple[str, ...] = ()
    status: str = "proposed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", AgentRole(self.role))
        object.__setattr__(self, "kind", AgentTaskKind(self.kind))
        if not self.proposal_id:
            raise ValueError("proposal_id cannot be empty")
        if self.status not in {"proposed", "accepted", "rejected"}:
            raise ValueError("agent proposal status is invalid")
        if not self.rationale.strip():
            raise ValueError("agent proposal rationale cannot be empty")

        payload = {str(key): value for key, value in self.payload.items()}
        allowed = _ALLOWED_PAYLOAD_KEYS[self.kind]
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"agent payload has unrecognized key(s): {sorted(unknown)}"
            )
        for key in payload:
            lowered = key.lower()
            if any(fragment in lowered for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError(
                    f"agent payload key {key!r} is not allowed in typed task contracts"
                )

        # Require the payload itself to be inert JSON data.
        try:
            json.dumps(payload, sort_keys=True)
        except TypeError as exc:
            raise ValueError("agent payload must contain JSON-serializable data") from exc
        object.__setattr__(self, "payload", payload)
        object.__setattr__(
            self,
            "source_artifacts",
            tuple(str(item) for item in self.source_artifacts),
        )


def validate_agent_proposal(
    proposal: AgentTaskProposal,
    *,
    graph: ModelGraph,
) -> None:
    """Reject proposals that cannot be represented by the frozen deterministic core."""
    model_hashes = set(graph.by_hash)
    mutation_ids = {item.mutation_id for item in DEFAULT_MUTATIONS}
    payload = proposal.payload

    if proposal.kind is AgentTaskKind.EVALUATE_MODEL:
        if payload.get("model_hash") not in model_hashes:
            raise ValueError("agent requested evaluation of unknown model hash")
    elif proposal.kind is AgentTaskKind.PROPOSE_MUTATION:
        if payload.get("parent_hash") not in model_hashes:
            raise ValueError("agent proposed mutation from unknown parent")
        if payload.get("mutation_id") not in mutation_ids:
            raise ValueError("agent proposed unregistered mutation")
    elif proposal.kind is AgentTaskKind.REQUEST_VALIDATION:
        if payload.get("model_hash") not in model_hashes:
            raise ValueError("agent requested validation of unknown model")
        allowed_validation = {
            "holdout",
            "seed_repeat",
            "importance_ess",
            "prior_sensitivity",
            "nearby_baseline",
            "null_replay",
        }
        if payload.get("validation") not in allowed_validation:
            raise ValueError("agent requested unknown validation type")
    elif proposal.kind is AgentTaskKind.REVIEW_SCOUT:
        if payload.get("mutation_id") not in mutation_ids:
            raise ValueError("agent scout review references unregistered mutation")
    elif proposal.kind is AgentTaskKind.SYNTHESIZE_RESULTS:
        artifacts = payload.get("artifact_ids", [])
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("synthesis requires a non-empty artifact_ids list")


@dataclass(frozen=True)
class AgentDecision:
    proposal_id: str
    accepted: bool
    reason: str
    resulting_task_id: str | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.reason:
            raise ValueError("agent decisions require proposal_id and reason")
        if self.accepted and not self.resulting_task_id:
            raise ValueError("accepted proposals require resulting_task_id")
        if not self.accepted and self.resulting_task_id is not None:
            raise ValueError("rejected proposals cannot create a task")
