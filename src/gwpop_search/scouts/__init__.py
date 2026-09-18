"""Flexible residual scouts that propose, but never auto-promote, structure."""

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
    "HSGPAxis",
    "HSGPDesign",
    "ResidualDependenceSummary",
    "StructureProposal",
    "basis_matrix",
    "laplacian_frequencies",
    "mutation_for_proposal",
    "proposal_from_summary",
    "squared_exponential_spectral_weights",
    "weighted_linear_dependence",
]
