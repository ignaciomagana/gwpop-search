"""Flexible residual scouts that propose, but never auto-promote, structure."""

from .config import (
    ScoutCampaignConfig,
    default_scout_campaign_config,
    load_scout_campaign_config,
    save_scout_campaign_config,
)
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
    descendant_for_proposal,
    descendant_payload_for_proposal,
    mutation_for_proposal,
    proposal_from_summary,
    weighted_linear_dependence,
)

__all__ = [
    "ConditionalHSGPConfig",
    "ScoutCampaignConfig",
    "ConditionalHSGPResidualModel",
    "ConditionalMomentSummaryConfig",
    "ConditionalScoutRunConfig",
    "HSGPAxis",
    "HSGPDesign",
    "ResidualDependenceSummary",
    "ScoutNumericalCriteria",
    "StructureProposal",
    "assess_scout_numerics",
    "default_scout_campaign_config",
    "descendant_for_proposal",
    "descendant_payload_for_proposal",
    "basis_matrix",
    "coefficient_priors",
    "laplacian_frequencies",
    "load_scout_campaign_config",
    "mutation_for_proposal",
    "proposal_from_summary",
    "run_conditional_hsgp_scout",
    "save_scout_campaign_config",
    "squared_exponential_spectral_weights",
    "summarize_conditional_hsgp",
    "weighted_linear_dependence",
]
