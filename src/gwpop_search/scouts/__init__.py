"""Flexible residual scouts that propose, but never auto-promote, structure."""

from .comparison import (
    compare_scout_descendant_evidence,
    scout_comparison_seed_root,
)
from .campaign import (
    assess_structured_scout_campaign,
    build_structured_scout_campaign_plan,
    reachable_mutation_ids,
    run_structured_scout_campaign,
    structured_scout_seed,
)
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
from .synthetic import (
    StructuredScoutInjection,
    generate_structured_scout_dataset,
)
from .review import (
    ReviewedScoutProposal,
    review_scout_proposal,
    select_validated_scout_proposal,
    write_scout_review,
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
    "ReviewedScoutProposal",
    "ScoutCampaignConfig",
    "StructuredScoutInjection",
    "ConditionalHSGPResidualModel",
    "ConditionalMomentSummaryConfig",
    "ConditionalScoutRunConfig",
    "HSGPAxis",
    "HSGPDesign",
    "ResidualDependenceSummary",
    "ScoutNumericalCriteria",
    "StructureProposal",
    "assess_scout_numerics",
    "compare_scout_descendant_evidence",
    "assess_structured_scout_campaign",
    "build_structured_scout_campaign_plan",
    "default_scout_campaign_config",
    "descendant_for_proposal",
    "descendant_payload_for_proposal",
    "basis_matrix",
    "coefficient_priors",
    "laplacian_frequencies",
    "load_scout_campaign_config",
    "mutation_for_proposal",
    "proposal_from_summary",
    "generate_structured_scout_dataset",
    "reachable_mutation_ids",
    "review_scout_proposal",
    "run_conditional_hsgp_scout",
    "run_structured_scout_campaign",
    "save_scout_campaign_config",
    "scout_comparison_seed_root",
    "select_validated_scout_proposal",
    "squared_exponential_spectral_weights",
    "structured_scout_seed",
    "summarize_conditional_hsgp",
    "weighted_linear_dependence",
    "write_scout_review",
]
