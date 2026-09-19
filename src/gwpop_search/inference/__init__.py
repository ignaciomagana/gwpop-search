"""Inference backends, recovery campaigns, and checkpoint utilities."""

from .numpyro import (
    NUTSConfig,
    NUTSResult,
    NumPyroUnavailableError,
    build_numpyro_model,
    build_run_manifest,
    load_result,
    run_nuts,
    run_resumable_chains,
    save_result,
)
from .evidence import (
    EvidenceBackendUnavailableError,
    EvidenceRepeatSummary,
    EvidenceResult,
    NestedSamplingConfig,
    load_evidence_result,
    run_hbi_evidence,
    run_numpyro_nested_model,
    save_evidence_result,
    summarize_evidence_repeats,
)
from .dynesty_backend import (
    BatchedShapeLogLikelihood,
    DirtyCodeWarning,
    DynestyConfig,
    DynestyResult,
    DynestyUnavailableError,
    ImportanceDiagnosticsBatch,
    ImportanceDiagnosticsFunction,
    PoolCancelledError,
    PooledPointLogLikelihood,
    PosteriorImportanceSummary,
    PriorTransform,
    SelectionSupportWarning,
    ThreadBatchPool,
    build_batched_log_likelihood,
    build_dynesty_manifest,
    build_importance_diagnostics,
    build_likelihood_identity,
    dynesty_result_exists,
    equal_weight_resample,
    importance_diagnostics_over_posterior,
    load_dynesty_result,
    prior_transform_for,
    run_dynesty,
    run_dynesty_population,
    save_dynesty_result,
)
from .model_spec import prior_specs_from_model_spec
from .priors import BASELINE_SYNTHETIC_PRIORS, PriorSpec, serialize_prior_map


def generate_baseline_synthetic_dataset(*args, **kwargs):
    from .synthetic import generate_baseline_synthetic_dataset as _generate

    return _generate(*args, **kwargs)


def run_synthetic_baseline_recovery(*args, **kwargs):
    from .recovery import run_synthetic_baseline_recovery as _run

    return _run(*args, **kwargs)


def run_recovery_campaign(*args, **kwargs):
    from .campaign import run_recovery_campaign as _run

    return _run(*args, **kwargs)


def assess_recovery_campaign(*args, **kwargs):
    from .campaign import assess_recovery_campaign as _assess

    return _assess(*args, **kwargs)


__all__ = [
    "BASELINE_SYNTHETIC_PRIORS",
    "BatchedShapeLogLikelihood",
    "DirtyCodeWarning",
    "DynestyConfig",
    "DynestyResult",
    "DynestyUnavailableError",
    "EvidenceBackendUnavailableError",
    "EvidenceRepeatSummary",
    "EvidenceResult",
    "ImportanceDiagnosticsBatch",
    "ImportanceDiagnosticsFunction",
    "NUTSConfig",
    "NestedSamplingConfig",
    "NUTSResult",
    "NumPyroUnavailableError",
    "PoolCancelledError",
    "PooledPointLogLikelihood",
    "PosteriorImportanceSummary",
    "PriorSpec",
    "PriorTransform",
    "SelectionSupportWarning",
    "ThreadBatchPool",
    "assess_recovery_campaign",
    "build_batched_log_likelihood",
    "build_dynesty_manifest",
    "build_importance_diagnostics",
    "build_likelihood_identity",
    "build_numpyro_model",
    "build_run_manifest",
    "dynesty_result_exists",
    "equal_weight_resample",
    "generate_baseline_synthetic_dataset",
    "importance_diagnostics_over_posterior",
    "load_dynesty_result",
    "load_evidence_result",
    "load_result",
    "prior_specs_from_model_spec",
    "prior_transform_for",
    "run_dynesty",
    "run_dynesty_population",
    "run_hbi_evidence",
    "run_numpyro_nested_model",
    "run_nuts",
    "run_recovery_campaign",
    "run_resumable_chains",
    "run_synthetic_baseline_recovery",
    "save_dynesty_result",
    "save_evidence_result",
    "save_result",
    "serialize_prior_map",
    "summarize_evidence_repeats",
]


# -- Phase-3 v2 (dynesty recovery campaign) and Phase-3c evidence check ---------
# Lazy wrappers: importing gwpop_search.inference does not import the campaign
# modules (they pull in the synthetic survey and the population models).


def run_ns_recovery_campaign(*args, **kwargs):
    from .phase3_ns import run_ns_recovery_campaign as _run

    return _run(*args, **kwargs)


def run_ns_synthetic_recovery(*args, **kwargs):
    from .phase3_ns import run_ns_synthetic_recovery as _run

    return _run(*args, **kwargs)


def assess_ns_recovery_campaign(*args, **kwargs):
    from .phase3_ns import assess_ns_recovery_campaign as _assess

    return _assess(*args, **kwargs)


def ns_run_fingerprints(*args, **kwargs):
    from .phase3_ns import ns_run_fingerprints as _fingerprints

    return _fingerprints(*args, **kwargs)


def run_evidence_check(*args, **kwargs):
    from .phase3_evidence import run_evidence_check as _run

    return _run(*args, **kwargs)


__all__ += [
    "assess_ns_recovery_campaign",
    "ns_run_fingerprints",
    "run_evidence_check",
    "run_ns_recovery_campaign",
    "run_ns_synthetic_recovery",
]
