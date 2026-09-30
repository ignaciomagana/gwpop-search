"""Robustness suites and faithful model-comparison analyses.

Implements the recommendations of MODEL_COMPARISON_MATH.md on saved dynesty
results (the likelihood, data semantics and priors are never changed):

* :mod:`.edge_mc_error` — Monte-Carlo-likelihood error and bias of ``ln Z`` and
  ``ln BF`` from posterior-averaged weights (common random numbers), plus the
  empirical bootstrap-reweighting estimator;
* :mod:`.prior_sensitivity` — exact narrowed-prior reweighting, the Occam
  relation for widened priors (contained posteriors only) and model-prior
  variants;
* :mod:`.sddr` — Savage–Dickey (x Verdinelli–Wasserman) cross-checks with
  boundary-corrected densities, and the nesting classification of graph edges;
* :mod:`.model_comparison` — per-edge ``ln BF`` with the full error budget,
  posterior model probabilities with propagated errors, structural masses,
  alias merging and the claim criteria (JSON + Markdown);
* :mod:`.psis_loo` — PSIS leave-one-out influence on the exact factorization
  ``L = prod_i ell_i / A`` from weighted nested-sampling points;
* :mod:`.posterior_gates` — cross-run R-hat, Kish and importance gates of
  repeated dynesty runs (used by the holdout refits).

Imports are lazy: the JAX-dependent modules load on first use.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    # common
    "AnalysisInputError": "._common",
    "WeightedPosterior": "._common",
    "discover_dynesty_results": "._common",
    "trajectory_configuration": "._common",
    "trajectory_rung": "._common",
    "pool_dynesty_results": "._common",
    "require_same_data": "._common",
    # terms
    "BatchedCatalogTerms": ".terms",
    "CatalogWeightEvaluator": ".terms",
    "pad_catalog": ".terms",
    # MC error
    "BootstrapEdgeMCError": ".edge_mc_error",
    "EdgeMCError": ".edge_mc_error",
    "InsufficientReweightingESSError": ".edge_mc_error",
    "ModelMCWeights": ".edge_mc_error",
    "bootstrap_edge_mc_error": ".edge_mc_error",
    "compute_model_mc_weights": ".edge_mc_error",
    "edge_mc_error": ".edge_mc_error",
    "load_model_mc_weights": ".edge_mc_error",
    "mc_covariance": ".edge_mc_error",
    "mc_covariance_matrix": ".edge_mc_error",
    "save_model_mc_weights": ".edge_mc_error",
    # prior sensitivity
    "MIN_REWEIGHTING_ESS": ".prior_sensitivity",
    "edge_prior_sensitivity": ".prior_sensitivity",
    "null_anchors": ".prior_sensitivity",
    "model_prior_variants": ".prior_sensitivity",
    "occam_widening": ".prior_sensitivity",
    "prior_reweighting": ".prior_sensitivity",
    "rescaled_prior": ".prior_sensitivity",
    # SDDR
    "EdgeNesting": ".sddr",
    "SDDR_METHOD_SYSTEMATIC": ".sddr",
    "method_systematic_for": ".sddr",
    "classify_edge": ".sddr",
    "classify_graph_edges": ".sddr",
    "density_at_null": ".sddr",
    "edge_classification_report": ".sddr",
    "sddr_edge_check": ".sddr",
    "verify_nesting": ".sddr",
    "vw_factor": ".sddr",
    # structure
    "atoms_of": ".structure",
    "effective_model": ".structure",
    "find_alias_groups": ".structure",
    "structural_masses": ".structure",
    "used_hyperparameters": ".structure",
    # model comparison
    "ClaimCriteria": ".model_comparison",
    "ComparisonConfig": ".model_comparison",
    "EvidenceSummary": ".model_comparison",
    "NullCalibration": ".model_comparison",
    "build_model_comparison": ".model_comparison",
    "render_markdown": ".model_comparison",
    # PSIS-LOO
    "PSISLOOConfig": ".psis_loo",
    "evaluate_model_loo_terms": ".psis_loo",
    "psis_loo_estimate": ".psis_loo",
    "psis_loo_model": ".psis_loo",
    "psis_loo_report": ".psis_loo",
    "psis_smooth": ".psis_loo",
    # posterior gates
    "PosteriorGateCriteria": ".posterior_gates",
    "cross_run_rhat": ".posterior_gates",
    "evaluate_posterior_gates": ".posterior_gates",
    "rank_normalized_split_rhat": ".posterior_gates",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value
