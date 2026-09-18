"""Flexible residual scouts that propose, but never auto-promote, structure."""

from .conditional import (
    ConditionalHSGPConfig,
    ConditionalHSGPResidualModel,
    coefficient_priors,
)
from .inference import (
    ConditionalScoutRunConfig,
    ScoutNumericalCriteria,
    assess_scout_numerics,
    run_conditional_hsgp_scout,
)
from .summary import (
    ConditionalMomentSummaryConfig,
    summarize_conditional_hsgp,
)
from .hsgp import (
    HSGPAxis,
    HSGPDesign,
    basis_matrix,
    laplacian_frequencies,
    squared_exponential_spectral_weights,
)
from .proposals import (
    ResidualDependenceSummary,
    StructureProposal,
    mutation_for_proposal,
    proposal_from_summary,
    weighted_linear_dependence,
)

__all__ = [
    "ConditionalHSGPConfig",
    "ConditionalHSGPResidualModel",
    "ConditionalMomentSummaryConfig",
    "ConditionalScoutRunConfig",
    "HSGPAxis",
    "HSGPDesign",
    "ResidualDependenceSummary",
    "ScoutNumericalCriteria",
    "StructureProposal",
    "assess_scout_numerics",
    "basis_matrix",
    "coefficient_priors",
    "laplacian_frequencies",
    "mutation_for_proposal",
    "proposal_from_summary",
    "run_conditional_hsgp_scout",
    "squared_exponential_spectral_weights",
    "summarize_conditional_hsgp",
    "weighted_linear_dependence",
]
