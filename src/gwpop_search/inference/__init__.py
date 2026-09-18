"""Inference backends and checkpointable run utilities."""

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
from .priors import BASELINE_SYNTHETIC_PRIORS, PriorSpec, serialize_prior_map


def generate_baseline_synthetic_dataset(*args, **kwargs):
    """Lazily import the JAX-backed Phase-3 synthetic dataset generator."""
    from .synthetic import generate_baseline_synthetic_dataset as _generate

    return _generate(*args, **kwargs)


def run_synthetic_baseline_recovery(*args, **kwargs):
    """Lazily import the Phase-3 synthetic NUTS recovery campaign."""
    from .recovery import run_synthetic_baseline_recovery as _run

    return _run(*args, **kwargs)


__all__ = [
    "BASELINE_SYNTHETIC_PRIORS",
    "NUTSConfig",
    "NUTSResult",
    "NumPyroUnavailableError",
    "PriorSpec",
    "build_numpyro_model",
    "build_run_manifest",
    "generate_baseline_synthetic_dataset",
    "load_result",
    "run_nuts",
    "run_resumable_chains",
    "run_synthetic_baseline_recovery",
    "save_result",
    "serialize_prior_map",
]
