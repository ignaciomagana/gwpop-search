"""Typed scout summaries and human-reviewable grammar proposals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from gwpop_search.grammar import DEFAULT_MUTATIONS, ModelSpec, MutationSpec, apply_mutation


@dataclass(frozen=True)
class ResidualDependenceSummary:
    target: str
    covariate: str
    slope: float
    slope_error: float
    z_score: float
    n_effective: float
    method: str = "weighted-linear-residual-v1"

    def __post_init__(self) -> None:
        for name in ("slope", "slope_error", "z_score", "n_effective"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.slope_error <= 0.0:
            raise ValueError("slope_error must be positive")
        if self.n_effective <= 0.0:
            raise ValueError("n_effective must be positive")


@dataclass(frozen=True)
class StructureProposal:
    proposal_id: str
    mutation_id: str
    target: str
    covariate: str
    score: float
    evidence: Mapping[str, object]
    status: str = "proposed"

    def __post_init__(self) -> None:
        if self.status not in {"proposed", "accepted", "rejected"}:
            raise ValueError("invalid proposal status")
        if not math.isfinite(self.score):
            raise ValueError("proposal score must be finite")


_MUTATION_BY_DEPENDENCE = {
    ("pairing", "m1"): "pairing.beta.linear_m1",
    ("chieff_mean", "m1"): "chieff.mean.linear_m1",
    ("chieff_mean", "q"): "chieff.mean.linear_q",
    ("chieff_mean", "z"): "chieff.mean.linear_z",
    ("chieff_width", "m1"): "chieff.width.linear_m1",
    ("chieff_width", "q"): "chieff.width.linear_q",
    ("chieff_width", "z"): "chieff.width.linear_z",
}
_DEFAULT_MUTATION_TABLE = {item.mutation_id: item for item in DEFAULT_MUTATIONS}


def weighted_linear_dependence(
    x,
    residual,
    *,
    weights=None,
    target: str,
    covariate: str,
) -> ResidualDependenceSummary:
    """Weighted least-squares slope for a residual diagnostic.

    This is a structure summary, not a population likelihood or Bayes factor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(residual, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and residual must be one-dimensional with equal shape")

    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != x.shape:
            raise ValueError("weights must match x shape")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    x = x[valid]
    y = y[valid]
    w = w[valid]
    if x.size < 3:
        raise ValueError("at least three finite weighted points are required")

    w = w / w.sum()
    x_bar = np.sum(w * x)
    y_bar = np.sum(w * y)
    dx = x - x_bar
    dy = y - y_bar
    xx = np.sum(w * dx**2)
    if xx <= 0.0:
        raise ValueError("covariate has zero weighted variance")
    slope = np.sum(w * dx * dy) / xx
    intercept = y_bar - slope * x_bar
    residual_fit = y - (intercept + slope * x)

    n_eff = 1.0 / np.sum(w**2)
    dof = max(n_eff - 2.0, 1.0)
    sigma2 = np.sum(w * residual_fit**2) * n_eff / dof
    slope_error = np.sqrt(sigma2 / (xx * n_eff))
    if not np.isfinite(slope_error):
        raise ValueError("unable to estimate a finite slope uncertainty")
    if slope_error <= 0.0:
        # A deterministic scout residual can be exactly linear in tests or
        # generated summaries. Keep the diagnostic finite rather than encoding
        # an infinite significance that downstream JSON/results cannot represent.
        scale = max(abs(float(slope)), 1.0)
        slope_error = np.sqrt(np.finfo(float).eps) * scale
    z_score = slope / slope_error

    return ResidualDependenceSummary(
        target=target,
        covariate=covariate,
        slope=float(slope),
        slope_error=float(slope_error),
        z_score=float(z_score),
        n_effective=float(n_eff),
    )


def proposal_from_summary(
    summary: ResidualDependenceSummary,
    *,
    minimum_abs_z: float = 2.0,
) -> StructureProposal | None:
    """Map a scout residual to an existing legal mutation, pending review."""
    if abs(summary.z_score) < minimum_abs_z:
        return None

    key = (summary.target, summary.covariate)
    mutation_id = _MUTATION_BY_DEPENDENCE.get(key)
    if mutation_id is None:
        return None
    if mutation_id not in _DEFAULT_MUTATION_TABLE:
        raise RuntimeError(
            f"scout mapping points to unavailable mutation {mutation_id!r}"
        )

    proposal_id = (
        f"{summary.method}:{summary.target}:{summary.covariate}:"
        f"{summary.z_score:+.6f}"
    )
    return StructureProposal(
        proposal_id=proposal_id,
        mutation_id=mutation_id,
        target=summary.target,
        covariate=summary.covariate,
        score=abs(summary.z_score),
        evidence={
            "slope": summary.slope,
            "slope_error": summary.slope_error,
            "z_score": summary.z_score,
            "n_effective": summary.n_effective,
            "method": summary.method,
        },
    )


def mutation_for_proposal(
    proposal: StructureProposal,
) -> MutationSpec:
    """Resolve a reviewed scout proposal to the frozen legal mutation registry."""
    try:
        return _DEFAULT_MUTATION_TABLE[proposal.mutation_id]
    except KeyError as exc:
        raise ValueError(
            f"proposal references unknown mutation {proposal.mutation_id!r}"
        ) from exc



def descendant_for_proposal(
    parent: ModelSpec,
    proposal: StructureProposal,
) -> ModelSpec:
    """Compile one typed scout proposal to its exact declarative child model."""
    return apply_mutation(parent, mutation_for_proposal(proposal))


def descendant_payload_for_proposal(
    parent: ModelSpec,
    proposal: StructureProposal,
) -> dict[str, object]:
    child = descendant_for_proposal(parent, proposal)
    return {
        "proposal_id": proposal.proposal_id,
        "mutation_id": proposal.mutation_id,
        "parent_model_hash": parent.model_hash,
        "child_model_hash": child.model_hash,
        "child_model_spec": child.to_dict(),
    }
