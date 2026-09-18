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
    "NUTSConfig",
    "NUTSResult",
    "NumPyroUnavailableError",
    "PriorSpec",
    "assess_recovery_campaign",
    "build_numpyro_model",
    "build_run_manifest",
    "generate_baseline_synthetic_dataset",
    "load_result",
    "prior_specs_from_model_spec",
    "run_nuts",
    "run_recovery_campaign",
    "run_resumable_chains",
    "run_synthetic_baseline_recovery",
    "save_result",
    "serialize_prior_map",
]
