"""Scientific model scoring from evidence plus explicit model priors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol

import numpy as np

from gwpop_search.grammar import ModelGraph, ModelSpec, structural_diff_axes


@dataclass(frozen=True)
class ModelEvidence:
    model_hash: str
    log_evidence: float
    log_evidence_error: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_evidence):
            raise ValueError("log_evidence must be finite")
        if not math.isfinite(self.log_evidence_error) or self.log_evidence_error < 0:
            raise ValueError("log_evidence_error must be finite and non-negative")


class ModelPrior(Protocol):
    version: str

    def log_prior(self, model: ModelSpec, *, root: ModelSpec) -> float: ...


@dataclass(frozen=True)
class UniformModelPrior:
    version: str = "uniform-v1"

    def log_prior(self, model: ModelSpec, *, root: ModelSpec) -> float:
        return 0.0


@dataclass(frozen=True)
class ComplexityModelPrior:
    """Exponential structural-complexity prior relative to a declared root."""

    penalty_per_axis: float = math.log(2.0)
    version: str = "axis-complexity-v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.penalty_per_axis) or self.penalty_per_axis < 0.0:
            raise ValueError("penalty_per_axis must be finite and non-negative")

    def log_prior(self, model: ModelSpec, *, root: ModelSpec) -> float:
        n_axes = len(structural_diff_axes(root, model))
        return -float(self.penalty_per_axis) * n_axes


@dataclass(frozen=True)
class ModelScore:
    model_hash: str
    log_evidence: float
    log_evidence_error: float
    log_model_prior: float
    log_unnormalized_posterior: float
    posterior_probability: float


@dataclass(frozen=True)
class EdgeComparison:
    parent_hash: str
    child_hash: str
    mutation_id: str
    log_bayes_factor: float
    log_bayes_factor_error: float
    log_posterior_odds: float


@dataclass(frozen=True)
class ScoredModelGraph:
    model_prior_version: str
    scores: tuple[ModelScore, ...]
    edge_comparisons: tuple[EdgeComparison, ...]

    @property
    def by_hash(self) -> dict[str, ModelScore]:
        return {item.model_hash: item for item in self.scores}

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": "gwpop-search-scored-graph-1.0",
            "model_prior_version": self.model_prior_version,
            "scores": [
                {
                    "model_hash": item.model_hash,
                    "log_evidence": item.log_evidence,
                    "log_evidence_error": item.log_evidence_error,
                    "log_model_prior": item.log_model_prior,
                    "log_unnormalized_posterior": item.log_unnormalized_posterior,
                    "posterior_probability": item.posterior_probability,
                }
                for item in self.scores
            ],
            "edge_comparisons": [
                {
                    "parent_hash": item.parent_hash,
                    "child_hash": item.child_hash,
                    "mutation_id": item.mutation_id,
                    "log_bayes_factor": item.log_bayes_factor,
                    "log_bayes_factor_error": item.log_bayes_factor_error,
                    "log_posterior_odds": item.log_posterior_odds,
                }
                for item in self.edge_comparisons
            ],
        }


def score_model_graph(
    graph: ModelGraph,
    evidences: Mapping[str, ModelEvidence],
    *,
    model_prior: ModelPrior | None = None,
) -> ScoredModelGraph:
    """Normalize posterior mass over all graph nodes with available evidence."""
    model_prior = UniformModelPrior() if model_prior is None else model_prior
    by_hash = graph.by_hash
    root = by_hash[graph.root_hash]

    unknown = set(evidences) - set(by_hash)
    if unknown:
        raise ValueError(f"evidence supplied for unknown model hash(es): {sorted(unknown)}")
    if not evidences:
        raise ValueError("at least one evidence estimate is required")

    ordered_models = [
        model for model in graph.nodes
        if model.model_hash in evidences
    ]
    log_weights = []
    prior_values = []
    for model in ordered_models:
        evidence = evidences[model.model_hash]
        log_prior = float(model_prior.log_prior(model, root=root))
        if not math.isfinite(log_prior):
            raise ValueError(
                f"model prior returned non-finite log probability for {model.model_hash}"
            )
        prior_values.append(log_prior)
        log_weights.append(evidence.log_evidence + log_prior)

    weights = np.asarray(log_weights, dtype=float)
    maximum = float(np.max(weights))
    log_norm = maximum + float(np.log(np.exp(weights - maximum).sum()))

    scores: list[ModelScore] = []
    for model, log_prior, log_weight in zip(
        ordered_models, prior_values, log_weights, strict=True
    ):
        evidence = evidences[model.model_hash]
        scores.append(
            ModelScore(
                model_hash=model.model_hash,
                log_evidence=evidence.log_evidence,
                log_evidence_error=evidence.log_evidence_error,
                log_model_prior=log_prior,
                log_unnormalized_posterior=log_weight,
                posterior_probability=float(np.exp(log_weight - log_norm)),
            )
        )

    score_by_hash = {item.model_hash: item for item in scores}
    edge_comparisons: list[EdgeComparison] = []
    for edge in graph.edges:
        if edge.parent_hash not in score_by_hash or edge.child_hash not in score_by_hash:
            continue
        parent = score_by_hash[edge.parent_hash]
        child = score_by_hash[edge.child_hash]
        log_bf = child.log_evidence - parent.log_evidence
        log_bf_error = math.hypot(
            child.log_evidence_error,
            parent.log_evidence_error,
        )
        log_prior_odds = child.log_model_prior - parent.log_model_prior
        edge_comparisons.append(
            EdgeComparison(
                parent_hash=edge.parent_hash,
                child_hash=edge.child_hash,
                mutation_id=edge.mutation_id,
                log_bayes_factor=log_bf,
                log_bayes_factor_error=log_bf_error,
                log_posterior_odds=log_bf + log_prior_odds,
            )
        )

    return ScoredModelGraph(
        model_prior_version=model_prior.version,
        scores=tuple(scores),
        edge_comparisons=tuple(edge_comparisons),
    )


def posterior_mass_for_structure_axis(
    graph: ModelGraph,
    scored: ScoredModelGraph,
    axis: str,
) -> float:
    """Posterior mass in models that differ from the root on one structural axis."""
    by_hash = graph.by_hash
    root = by_hash[graph.root_hash]
    total = 0.0
    for score in scored.scores:
        model = by_hash[score.model_hash]
        if axis in structural_diff_axes(root, model):
            total += score.posterior_probability
    return float(total)
