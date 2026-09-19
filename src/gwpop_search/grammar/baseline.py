"""Baseline declarative model specification used to seed Phase-4 enumeration."""

from __future__ import annotations

from .registry import DEFAULT_COMPONENT_REGISTRY
from .schema import ModelSpec, PriorConfig


def _uniform(low: float, high: float) -> PriorConfig:
    return PriorConfig("uniform", {"low": low, "high": high})


def _log_uniform(low: float, high: float) -> PriorConfig:
    return PriorConfig("log_uniform", {"low": low, "high": high})


#: Default profile. It reproduces the original baseline byte-for-byte, so every
#: existing model hash, graph and campaign is unchanged.
DEFAULT_HYPERPRIOR_PROFILE = "phase3"

#: Versioned root hyperprior profiles. A profile only replaces root priors; the
#: atom (mutation) priors are unchanged, and because priors are part of every
#: ModelSpec the profile is carried by every model hash of a graph enumerated
#: from it.
#:
#: ``phase3``   the Phase-3 synthetic-recovery hyperpriors (mmax ~ U(60, 120)).
#: ``gwtc5-v1`` GWTC-5 production (frozen 2026-09-19, before any real-data
#:              population inference): identical except mmax ~ U(60, 200). The
#:              Phase-3 ceiling of 120 Msun would truncate GW231123_135430, which
#:              has only 7.9% of its PE below 120 Msun; the O3+O4ab detected
#:              injections cover m1_source to ~980 Msun.
_PROFILE_OVERRIDES: dict[str, dict[str, PriorConfig]] = {
    "phase3": {},
    "gwtc5-v1": {"mmax": _uniform(60.0, 200.0)},
}
HYPERPRIOR_PROFILES = tuple(_PROFILE_OVERRIDES)


def baseline_model_spec(profile: str = DEFAULT_HYPERPRIOR_PROFILE) -> ModelSpec:
    """Return the declarative BBH baseline under a registered hyperprior profile."""
    if profile not in _PROFILE_OVERRIDES:
        raise ValueError(
            f"unknown hyperprior profile {profile!r}; "
            f"registered={list(HYPERPRIOR_PROFILES)}"
        )
    registry = DEFAULT_COMPONENT_REGISTRY
    priors = {
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
    }
    priors.update(_PROFILE_OVERRIDES[profile])
    model = ModelSpec(
        mass=registry.block("mass", "pl_peak"),
        pairing=registry.block("pairing", "powerlaw_q"),
        chieff=registry.block("chieff", "truncated_gaussian"),
        redshift=registry.block("redshift", "powerlaw"),
        mixture=registry.block("mixture", "single"),
        priors=priors,
    )
    registry.validate_model(model)
    return model


def baseline_hyperprior_profile(spec: ModelSpec) -> str | None:
    """Name of the registered profile whose baseline equals ``spec``, else None."""
    for profile in HYPERPRIOR_PROFILES:
        if baseline_model_spec(profile).model_hash == spec.model_hash:
            return profile
    return None
