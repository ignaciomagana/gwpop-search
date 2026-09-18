"""Baseline declarative model specification used to seed Phase-4 enumeration."""

from __future__ import annotations

from .registry import DEFAULT_COMPONENT_REGISTRY
from .schema import ModelSpec, PriorConfig


def _uniform(low: float, high: float) -> PriorConfig:
    return PriorConfig("uniform", {"low": low, "high": high})


def _log_uniform(low: float, high: float) -> PriorConfig:
    return PriorConfig("log_uniform", {"low": low, "high": high})


def baseline_model_spec() -> ModelSpec:
    """Return the declarative equivalent of the Phase-3 BBH baseline."""
    registry = DEFAULT_COMPONENT_REGISTRY
    model = ModelSpec(
        mass=registry.block("mass", "pl_peak"),
        pairing=registry.block("pairing", "powerlaw_q"),
        chieff=registry.block("chieff", "truncated_gaussian"),
        redshift=registry.block("redshift", "powerlaw"),
        mixture=registry.block("mixture", "single"),
        priors={
            "alpha": _uniform(0.0, 8.0),
            "mmin": _uniform(2.0, 10.0),
            "mmax": _uniform(60.0, 120.0),
            "peak_fraction": _uniform(0.0, 0.5),
            "peak_mu": _uniform(20.0, 50.0),
            "peak_sigma": _log_uniform(1.0, 15.0),
            "beta_q": _uniform(-4.0, 12.0),
            "kappa": _uniform(-6.0, 12.0),
            "chi_mu": _uniform(-0.3, 0.3),
            "chi_sigma": _log_uniform(0.03, 0.5),
        },
    )
    registry.validate_model(model)
    return model
