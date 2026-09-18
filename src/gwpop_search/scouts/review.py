"""Explicit human-review boundary for HSGP scout descendants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

from gwpop_search.grammar import ModelSpec

from .proposals import (
    StructureProposal,
    descendant_for_proposal,
)


@dataclass(frozen=True)
class ReviewedScoutProposal:
    proposal_id: str
    mutation_id: str
    decision: str
    parent_model_hash: str
    child_model_hash: str | None
    note: str = ""
    format_version: str = "gwpop-search-scout-review-1.0"

    def __post_init__(self) -> None:
        if self.format_version != "gwpop-search-scout-review-1.0":
            raise ValueError("unsupported scout review format")
        if self.decision not in {"accepted", "rejected"}:
            raise ValueError("decision must be accepted or rejected")
        if self.decision == "accepted" and not self.child_model_hash:
            raise ValueError("accepted proposal requires a child model hash")
        if self.decision == "rejected" and self.child_model_hash is not None:
            raise ValueError("rejected proposal cannot carry a child model hash")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _proposal_from_payload(payload: Mapping[str, object]) -> StructureProposal:
    return StructureProposal(
        proposal_id=str(payload["proposal_id"]),
        mutation_id=str(payload["mutation_id"]),
        target=str(payload["target"]),
        covariate=str(payload["covariate"]),
        score=float(payload["score"]),
        evidence=dict(payload.get("evidence", {})),
        status=str(payload.get("status", "proposed")),
    )


def select_validated_scout_proposal(
    scout_summary: Mapping[str, object],
    proposal_id: str,
) -> StructureProposal:
    if not bool(dict(scout_summary["numerical"])["passed"]):
        raise ValueError(
            "cannot review a scout proposal whose numerical gate failed"
        )
    matches = [
        item
        for item in scout_summary.get("validated_proposals", [])
        if str(item["proposal_id"]) == str(proposal_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one validated proposal {proposal_id!r}; "
            f"found {len(matches)}"
        )
    return _proposal_from_payload(matches[0])


def review_scout_proposal(
    scout_summary: Mapping[str, object],
    parent: ModelSpec,
    *,
    proposal_id: str,
    decision: str,
    note: str = "",
) -> tuple[ReviewedScoutProposal, ModelSpec | None]:
    """Accept/reject one validated proposal without altering the frozen grammar."""
    proposal = select_validated_scout_proposal(
        scout_summary,
        proposal_id,
    )
    if str(scout_summary["base_model_hash"]) != parent.model_hash:
        raise ValueError(
            "scout summary base model hash does not match supplied parent"
        )

    decision = str(decision)
    if decision == "accepted":
        child = descendant_for_proposal(parent, proposal)
        review = ReviewedScoutProposal(
            proposal_id=proposal.proposal_id,
            mutation_id=proposal.mutation_id,
            decision="accepted",
            parent_model_hash=parent.model_hash,
            child_model_hash=child.model_hash,
            note=str(note),
        )
        return review, child
    if decision == "rejected":
        review = ReviewedScoutProposal(
            proposal_id=proposal.proposal_id,
            mutation_id=proposal.mutation_id,
            decision="rejected",
            parent_model_hash=parent.model_hash,
            child_model_hash=None,
            note=str(note),
        )
        return review, None
    raise ValueError("decision must be accepted or rejected")


def write_scout_review(
    path: str | Path,
    review: ReviewedScoutProposal,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review.to_dict(), sort_keys=True, indent=2))
