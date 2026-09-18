"""Standardized hierarchical Bayesian inference for GW population models."""
from .common import HBIError, PopulationDensityError, SelectionSupportError
from .numpy_backend import (
    catalog_log_likelihood,
    evaluate_catalog_terms,
    evaluate_events,
    evaluate_selection,
    poisson_log_likelihood,
    shape_log_likelihood,
)
from .types import (
    CampaignSelectionResult,
    CatalogLikelihoodResult,
    CatalogTerms,
    EventLikelihoodResult,
    HBIConfig,
    ImportanceDiagnostics,
    LikelihoodVarianceDiagnostics,
    PopulationLogDensity,
    RateTreatment,
    SelectionResult,
)

def build_jax_shape_log_likelihood(*args, **kwargs):
    from .jax_backend import build_shape_log_likelihood
    return build_shape_log_likelihood(*args, **kwargs)

def build_jax_poisson_log_likelihood(*args, **kwargs):
    from .jax_backend import build_poisson_log_likelihood
    return build_poisson_log_likelihood(*args, **kwargs)

__all__ = [
    'HBIError','PopulationDensityError','SelectionSupportError','HBIConfig','RateTreatment',
    'ImportanceDiagnostics','EventLikelihoodResult','SelectionResult','CampaignSelectionResult',
    'LikelihoodVarianceDiagnostics','CatalogTerms','CatalogLikelihoodResult','PopulationLogDensity',
    'evaluate_events','evaluate_selection','evaluate_catalog_terms','catalog_log_likelihood','shape_log_likelihood',
    'poisson_log_likelihood','build_jax_shape_log_likelihood','build_jax_poisson_log_likelihood'
]
