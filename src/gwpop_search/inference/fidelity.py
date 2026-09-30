"""Fidelity ladder v2 (F0 -> F3 -> F4) on dynesty nested sampling.

Rungs
-----
``F0`` (sanity; every graph node)
    ``F0SanityConfig.prior_draws`` seeded prior draws on the FULL data through
    the production batched likelihood (the function dynesty calls): it must
    compile and never return NaN/``+inf`` (those raise
    :class:`gwpop_search.hbi.PopulationDensityError`, a bug, not a gate); the
    finite fraction must be ``>= min_finite_fraction``; the selection
    exposure must have population support on every draw (zero-exposure
    fraction ``<= max_zero_exposure_fraction``, i.e. 0: ``log A = -inf`` means
    the injections do not cover a hyperprior population); the jitted
    importance diagnostics must reproduce the batched likelihood; and the JAX
    likelihood must agree with the NumPy reference on a small thinned
    PE/selection pair (``|dlogL| <= parity_rtol * max(1, |logL|)``). Screen
    value ``0.0`` (``sanity_constant``).
``F3`` (evidence; the search beam and, through evidence completion, every node)
``F4`` (production precision; at most ``max_f4_models`` models)
    ``EvidenceCampaignConfig.repeats`` independent static dynesty runs
    (bound ``multi``, sample ``rslice``, ``slices = 2 (3 + ndim)``) on the
    full data; screen value = mean ``ln Z`` (``log_evidence_mean``).
``F1``/``F2``
    Not part of ladder v2 (the enum values stay for store compatibility);
    the evaluator refuses them.

Gates (:class:`NumericalCriteria` v2; thresholds from D3)
----------------------------------------------------------
Nested-sampling validity replaces the MCMC diagnostics: every run
terminated by ``dlogz`` (not by a budget), no evaluation without selection
support, cross-run rank-normalized split R-hat, Kish ESS per run, and
repeat consistency. Evidence precision: repeat std (ddof 1), conservative
error ``max(repeat std, max logzerr)`` and pairwise
``|dlnZ| / sqrt(err_r^2 + err_s^2)``. Importance sampling (event ESS,
selection ESS, maximum event/selection weights, ``Var[log L]``) is gated at
the pooled-posterior median point AND on the median AND the tail
(``q10`` for ESS-type, ``q90`` for weights/variance) over
``FidelityRunConfig.posterior_draws`` pooled posterior draws, all with the
run's own HBI configuration. The insertion-index test is advisory.

The pooled posterior is the equal-weight mixture of separately normalized
runs (never ``merge_runs``/``jitter_run``/``resample_run``). Models with
exactly exchangeable components (``chieff.family.gaussian_mixture``) are
sampled with the evidence-preserving order-statistic transform of
:mod:`~gwpop_search.inference.label_switching`, so every posterior summary
and gate is on canonical labels.

Formats: fidelity configuration ``gwpop-search-fidelity-config-2.0``
(NUTS/JAXNS-era 1.x configurations are refused), evaluation
``gwpop-search-fidelity-evaluation-2.0`` with the pooled equal-weight
posterior saved next to it (``pooled_posterior.npz``).
``EvaluationRecord.compute_cost`` is wall-clock hours of the evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dataclass_fields
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from gwpop_search.data import thin_catalog_pair
from gwpop_search.grammar import ModelSpec
from gwpop_search.hbi import HBIConfig, RateTreatment, shape_log_likelihood
from gwpop_search.models import compile_model_spec
from gwpop_search.search.scheduler import EvaluationRecord, Fidelity

from .dynesty_backend import (
    DynestyConfig,
    build_batched_log_likelihood,
    build_importance_diagnostics,
    equal_weight_resample,
)
from .evidence import (
    NoFiniteSupportError,
    SAMPLER_BACKEND,
    run_diagnostics,
    summarize_evidence_repeats,
)
from .evidence_campaign import EvidenceCampaignConfig, run_model_evidence_repeats
from .label_switching import ModelParameterization, parameterization_for_spec
from .model_spec import prior_specs_from_model_spec
from .ns_diagnostics import (
    cross_run_rhat,
    pooled_weighted_samples,
    prior_edge_mass,
    weighted_quantiles,
)
from .synthetic import require_population_proxy_coverage

FIDELITY_CONFIG_FORMAT_VERSION = "gwpop-search-fidelity-config-2.0"
EVALUATION_FORMAT_VERSION = "gwpop-search-fidelity-evaluation-2.0"
# Machine-readable reason the F4 insertion-index advisory carries no value.
INSERTION_INDEX_UNAVAILABLE_REASON = "backend_does_not_persist_birth_iterations"
POOLED_POSTERIOR_FORMAT_VERSION = "gwpop-search-pooled-posterior-1.0"
POOLED_POSTERIOR_FILENAME = "pooled_posterior.npz"
LADDER_V2 = (Fidelity.F0_SANITY, Fidelity.F3_EVIDENCE, Fidelity.F4_PRODUCTION)
EVIDENCE_RUNGS = (Fidelity.F3_EVIDENCE, Fidelity.F4_PRODUCTION)

# D1: static NestedSampler, bound 'multi', sample 'rslice' for every model/repeat.
REQUIRED_BOUND = "multi"
REQUIRED_SAMPLE = "rslice"

_F0_STREAM = 0x46305341  # "F0SA"
_IMPORTANCE_STREAM = 0x494D5054  # "IMPT"


class LegacyFidelityConfigError(ValueError):
    """A NUTS/JAXNS-era (1.x) fidelity configuration was presented to ladder v2."""


class LegacyEvaluationError(ValueError):
    """An evaluation artifact of a retired (NUTS/JAXNS-era 1.x) format was presented."""


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _as_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _as_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


_LEGACY_CRITERIA_FIELDS = ("max_r_hat", "min_mcmc_n_eff", "max_divergences")

# Check families of a repeated dynesty fit (:func:`summarize_dynesty_fit`).
# A check's family is its name without the ``.point``/``.draw_median``/
# ``.draw_tail`` suffix; ``NumericalCriteria.binding`` sets, per family,
# whether a failing check fails the evaluation (``stage == "gate"``) or is
# recorded as advisory only.
CHECK_FAMILIES = (
    "nested_sampling.all_runs_terminated_by_dlogz",
    "nested_sampling.selection_unsupported_evaluations",
    "nested_sampling.min_kish_ess_per_run",
    "nested_sampling.cross_run_r_hat",
    "evidence.conservative_error",
    "evidence.repeat_std",
    "evidence.max_pairwise_z",
    "importance.min_event_ess",
    "importance.selection_ess",
    "importance.max_event_weight_fraction",
    "importance.selection_max_weight_fraction",
    "importance.shape_log_likelihood_variance",
    "taper.posterior_mass_in_taper_region",
)
_CHECK_SUFFIXES = (".point", ".draw_median", ".draw_tail")


def check_family(name: str) -> str:
    """The binding family of a check name (suffixes ``.point``/``.draw_*`` removed)."""
    for suffix in _CHECK_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@dataclass(frozen=True)
class NumericalCriteria:
    """Nested-sampling, evidence and importance-sampling gates of one rung.

    ``None`` disables a gate. ``importance_tail_quantile`` ``q`` selects the
    tail statistic over posterior draws: the ``q`` quantile for ESS-type
    metrics ("larger is better") and the ``1 - q`` quantile for maximum
    weights and ``Var[log L]``. With ``gate_importance_over_posterior=False``
    the draw median/tail are recorded as advisory only (the posterior-median
    point is always gated).

    ``insertion_index_advisory`` (D3: advisory at F4) requests the
    insertion-index KS test. It is never gated, and with the current dynesty
    backend it is *not computed*: the reconstruction needs each dead point's
    birth iteration (``samples_it``), which :class:`DynestyResult` does not
    persist. The evaluation then records ``available=False`` with
    ``reason_code=INSERTION_INDEX_UNAVAILABLE_REASON`` on both the check and
    the ``nested_sampling.insertion_index`` block, so the absence is explicit
    rather than an empty advisory. The test itself is implemented and tested
    (:func:`gwpop_search.inference.ns_diagnostics.insertion_index_ranks`,
    :func:`~gwpop_search.inference.ns_diagnostics.insertion_index_test`) and
    becomes live as soon as the backend persists the birth iterations.

    ``binding`` (per-check binding flags) maps a check family
    (:data:`CHECK_FAMILIES`) to ``True`` (binding: a failure fails the
    evaluation) or ``False`` (recorded with ``stage="advisory"`` and
    ``binding=False``; never fails the evaluation). Families not listed are
    binding. With the likelihood variance taper inside the likelihood (v2),
    the ``importance.*`` variance/ESS families are typically non-binding: the
    taper, not a post-hoc gate, handles an unreliable estimate.

    ``max_posterior_taper_mass`` (``None``: report only) limits the posterior
    fraction inside the variance-taper region
    (``taper.posterior_mass_in_taper_region``); it needs a tapered HBI
    configuration.

    Serialization omits ``binding`` when empty and
    ``max_posterior_taper_mass`` when ``None``, so the hashes of configurations
    written before these fields existed are unchanged.
    """

    max_cross_run_r_hat: float | None = None
    min_kish_ess_per_run: float | None = None
    require_dlogz_termination: bool = True
    require_selection_support: bool = True
    max_evidence_error: float | None = None
    max_evidence_repeat_std: float | None = None
    max_repeat_consistency_z: float | None = None
    min_event_ess: float = 1.0
    min_selection_ess: float = 1.0
    max_event_weight_fraction: float = 1.0
    max_selection_weight_fraction: float = 1.0
    max_shape_log_likelihood_variance: float = math.inf
    importance_tail_quantile: float = 0.1
    gate_importance_over_posterior: bool = True
    insertion_index_advisory: bool = False
    binding: Mapping[str, bool] = field(default_factory=dict)
    max_posterior_taper_mass: float | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if not isinstance(self.binding, Mapping):
            raise TypeError("binding must be a mapping of check family -> bool")
        binding = {str(key): value for key, value in self.binding.items()}
        unknown = sorted(set(binding) - set(CHECK_FAMILIES))
        if unknown:
            raise ValueError(
                f"unknown check families in binding: {unknown}; known: {list(CHECK_FAMILIES)}"
            )
        for key, value in binding.items():
            if not isinstance(value, bool):
                raise TypeError(f"binding[{key!r}] must be a bool")
        set_(self, "binding", dict(sorted(binding.items())))
        if self.max_posterior_taper_mass is not None:
            mass = _as_float("max_posterior_taper_mass", self.max_posterior_taper_mass)
            if not 0.0 <= mass <= 1.0:
                raise ValueError("max_posterior_taper_mass must lie in [0, 1]")
            set_(self, "max_posterior_taper_mass", mass)
        if self.max_cross_run_r_hat is not None:
            value = _as_float("max_cross_run_r_hat", self.max_cross_run_r_hat)
            if not value > 1.0:
                raise ValueError("max_cross_run_r_hat must exceed one")
            set_(self, "max_cross_run_r_hat", value)
        for name in (
            "min_kish_ess_per_run",
            "max_evidence_error",
            "max_evidence_repeat_std",
            "max_repeat_consistency_z",
        ):
            value = getattr(self, name)
            if value is not None:
                value = _as_float(name, value)
                if not (math.isfinite(value) and value > 0.0):
                    raise ValueError(f"{name} must be finite and positive when supplied")
                set_(self, name, value)
        for name in (
            "require_dlogz_termination",
            "require_selection_support",
            "gate_importance_over_posterior",
            "insertion_index_advisory",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("min_event_ess", "min_selection_ess"):
            value = _as_float(name, getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError("importance ESS thresholds must be finite and positive")
            set_(self, name, value)
        for name in ("max_event_weight_fraction", "max_selection_weight_fraction"):
            value = _as_float(name, getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
            set_(self, name, value)
        variance = _as_float(
            "max_shape_log_likelihood_variance", self.max_shape_log_likelihood_variance
        )
        if not variance > 0.0:
            raise ValueError("max_shape_log_likelihood_variance must be positive")
        set_(self, "max_shape_log_likelihood_variance", variance)
        tail = _as_float("importance_tail_quantile", self.importance_tail_quantile)
        if not 0.0 < tail < 0.5:
            raise ValueError("importance_tail_quantile must lie in (0, 0.5)")
        set_(self, "importance_tail_quantile", tail)

    @property
    def needs_repeats(self) -> bool:
        """True when a gate compares independent runs (needs at least two)."""
        return any(
            value is not None
            for value in (
                self.max_cross_run_r_hat,
                self.max_evidence_repeat_std,
                self.max_repeat_consistency_z,
            )
        )

    def is_binding(self, name: str) -> bool:
        """Whether the check ``name`` (or its family) is binding."""
        return bool(self.binding.get(check_family(name), True))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["binding"] = dict(self.binding)
        if not payload["binding"]:
            payload.pop("binding")
        if payload["max_posterior_taper_mass"] is None:
            payload.pop("max_posterior_taper_mass")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NumericalCriteria":
        payload = dict(payload)
        legacy = sorted(set(payload) & set(_LEGACY_CRITERIA_FIELDS))
        if legacy:
            raise LegacyFidelityConfigError(
                f"MCMC-era numerical criteria {legacy}; NUTS/JAXNS-era config: re-freeze "
                "with write-default-fidelity-config"
            )
        known = {item.name for item in dataclass_fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown NumericalCriteria field(s): {unknown}")
        return cls(**payload)


def default_f3_criteria() -> NumericalCriteria:
    return NumericalCriteria(
        max_cross_run_r_hat=1.05,
        min_kish_ess_per_run=500.0,
        max_evidence_error=0.50,
        max_evidence_repeat_std=0.50,
        max_repeat_consistency_z=3.5,
        min_event_ess=10.0,
        min_selection_ess=100.0,
        max_event_weight_fraction=0.35,
        max_selection_weight_fraction=0.15,
        max_shape_log_likelihood_variance=2.0,
    )


def default_f4_criteria() -> NumericalCriteria:
    return NumericalCriteria(
        max_cross_run_r_hat=1.01,
        min_kish_ess_per_run=2000.0,
        max_evidence_error=0.20,
        max_evidence_repeat_std=0.20,
        max_repeat_consistency_z=3.0,
        min_event_ess=20.0,
        min_selection_ess=200.0,
        max_event_weight_fraction=0.25,
        max_selection_weight_fraction=0.10,
        max_shape_log_likelihood_variance=1.0,
        insertion_index_advisory=True,
    )


def default_f3_evidence() -> EvidenceCampaignConfig:
    return EvidenceCampaignConfig(
        repeats=2,
        dynesty=DynestyConfig(
            nlive=1000, bound=REQUIRED_BOUND, sample=REQUIRED_SAMPLE, dlogz=0.1
        ),
        slices_multiplier=2,
    )


def default_f4_evidence() -> EvidenceCampaignConfig:
    return EvidenceCampaignConfig(
        repeats=3,
        dynesty=DynestyConfig(
            nlive=2000, bound=REQUIRED_BOUND, sample=REQUIRED_SAMPLE, dlogz=0.05
        ),
        slices_multiplier=2,
    )


@dataclass(frozen=True)
class F0SanityConfig:
    """F0 prior-support scan and JAX/NumPy parity settings.

    ``prior_draws`` seeded prior draws are evaluated on the FULL data with the
    production batched likelihood (``batch_size`` rows per device call). The
    NumPy parity uses a thinned pair (``parity_pe_samples_per_event`` PE
    samples per event, ``parity_selected_per_campaign`` injections per
    campaign) at up to ``parity_points`` draws with finite thinned likelihood.
    """

    prior_draws: int = 4096
    batch_size: int = 64
    min_finite_fraction: float = 1.0e-3
    max_zero_exposure_fraction: float = 0.0
    parity_pe_samples_per_event: int = 32
    parity_selected_per_campaign: int = 512
    parity_points: int = 2
    parity_rtol: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in (
            "prior_draws",
            "batch_size",
            "parity_pe_samples_per_event",
            "parity_selected_per_campaign",
            "parity_points",
        ):
            object.__setattr__(self, name, _as_positive_int(name, getattr(self, name)))
        fraction = _as_float("min_finite_fraction", self.min_finite_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("min_finite_fraction must lie in (0, 1]")
        object.__setattr__(self, "min_finite_fraction", fraction)
        exposure = _as_float("max_zero_exposure_fraction", self.max_zero_exposure_fraction)
        if not 0.0 <= exposure < 1.0:
            raise ValueError("max_zero_exposure_fraction must lie in [0, 1)")
        object.__setattr__(self, "max_zero_exposure_fraction", exposure)
        rtol = _as_float("parity_rtol", self.parity_rtol)
        if not (math.isfinite(rtol) and rtol > 0.0):
            raise ValueError("parity_rtol must be finite and positive")
        object.__setattr__(self, "parity_rtol", rtol)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "F0SanityConfig":
        known = {item.name for item in dataclass_fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown F0SanityConfig field(s): {unknown}")
        return cls(**dict(payload))


def _hbi_to_dict(hbi: HBIConfig) -> dict[str, object]:
    # ``variance_taper`` appears only when configured, so pre-v2 configuration
    # hashes are unchanged.
    return hbi.to_dict()


def _hbi_from_dict(payload: Mapping[str, object]) -> HBIConfig:
    return HBIConfig.from_dict(payload)


@dataclass(frozen=True)
class FidelityRunConfig:
    """Frozen numerical configuration of the v2 ladder (format 2.0).

    ``hbi`` is the likelihood configuration of every rung (D5: one selection
    chunk). ``posterior_draws`` pooled posterior draws feed the importance
    gates; ``rhat_draws_per_run`` caps the draws per run in the cross-run
    R-hat. ``canonicalize_exchangeable_components`` samples exactly
    exchangeable mixture components on canonical labels (evidence unchanged).
    """

    f0: F0SanityConfig = field(default_factory=F0SanityConfig)
    f3_evidence: EvidenceCampaignConfig = field(default_factory=default_f3_evidence)
    f4_evidence: EvidenceCampaignConfig = field(default_factory=default_f4_evidence)
    f3_criteria: NumericalCriteria = field(default_factory=default_f3_criteria)
    f4_criteria: NumericalCriteria = field(default_factory=default_f4_criteria)
    hbi: HBIConfig = field(default_factory=lambda: HBIConfig(selection_chunk_size=None))
    posterior_draws: int = 512
    rhat_draws_per_run: int = 2000
    diagnostics_batch_size: int = 16
    canonicalize_exchangeable_components: bool = True
    format_version: str = FIDELITY_CONFIG_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FIDELITY_CONFIG_FORMAT_VERSION:
            raise LegacyFidelityConfigError(
                f"unsupported fidelity config format {self.format_version!r}; expected "
                f"{FIDELITY_CONFIG_FORMAT_VERSION}"
            )
        if not isinstance(self.f0, F0SanityConfig):
            raise TypeError("f0 must be an F0SanityConfig")
        if not isinstance(self.hbi, HBIConfig):
            raise TypeError("hbi must be an HBIConfig")
        if self.hbi.rate_treatment is not RateTreatment.SHAPE:
            raise ValueError("the dynesty ladder evaluates the rate-marginalized shape likelihood")
        for rung in EVIDENCE_RUNGS:
            evidence = self.evidence_config(rung)
            criteria = self.criteria(rung)
            if not isinstance(evidence, EvidenceCampaignConfig):
                raise TypeError(f"{rung.value} evidence must be an EvidenceCampaignConfig")
            if not isinstance(criteria, NumericalCriteria):
                raise TypeError(f"{rung.value} criteria must be NumericalCriteria")
            dynesty = evidence.dynesty
            if dynesty.bound != REQUIRED_BOUND or dynesty.sample != REQUIRED_SAMPLE:
                raise ValueError(
                    f"{rung.value} must use dynesty bound={REQUIRED_BOUND!r}, "
                    f"sample={REQUIRED_SAMPLE!r} (D1); got bound="
                    f"{dynesty.bound!r}, sample={dynesty.sample!r}"
                )
            if criteria.needs_repeats and evidence.repeats < 2:
                raise ValueError(
                    f"{rung.value} gates compare independent runs (R-hat, repeat std, "
                    "pairwise z) and need repeats >= 2"
                )
        for name in ("posterior_draws", "diagnostics_batch_size"):
            object.__setattr__(self, name, _as_positive_int(name, getattr(self, name)))
        draws = _as_positive_int("rhat_draws_per_run", self.rhat_draws_per_run)
        if draws < 4:
            raise ValueError("rhat_draws_per_run must be at least 4")
        object.__setattr__(self, "rhat_draws_per_run", draws)
        if not isinstance(self.canonicalize_exchangeable_components, bool):
            raise TypeError("canonicalize_exchangeable_components must be a bool")

    def evidence_config(self, fidelity) -> EvidenceCampaignConfig:
        fidelity = Fidelity(fidelity)
        if fidelity is Fidelity.F3_EVIDENCE:
            return self.f3_evidence
        if fidelity is Fidelity.F4_PRODUCTION:
            return self.f4_evidence
        raise ValueError(f"{fidelity.value} is not an evidence rung")

    def criteria(self, fidelity) -> NumericalCriteria:
        fidelity = Fidelity(fidelity)
        if fidelity is Fidelity.F3_EVIDENCE:
            return self.f3_criteria
        if fidelity is Fidelity.F4_PRODUCTION:
            return self.f4_criteria
        raise ValueError(f"{fidelity.value} has no nested-sampling criteria")

    # -- legacy compatibility ------------------------------------------------

    @property
    def f4_nuts(self):
        """Deprecated NUTS settings for the NUTS-era holdout refits only.

        Not part of fidelity config 2.0 (never serialized) and never used by
        the ladder. ``validation/holdout_campaign.py`` still refits folds with
        NUTS; this returns exactly the frozen 1.x F4 NUTS settings until that
        module moves to dynesty.
        """
        from .numpyro import NUTSConfig

        return NUTSConfig(**LEGACY_HOLDOUT_NUTS_SETTINGS)


def fidelity_run_config_to_dict(config: FidelityRunConfig) -> dict[str, object]:
    return {
        "format_version": config.format_version,
        "ladder_fidelities": [item.value for item in LADDER_V2],
        "sampler_backend": SAMPLER_BACKEND,
        "f0": config.f0.to_dict(),
        "f3_evidence": config.f3_evidence.to_dict(),
        "f4_evidence": config.f4_evidence.to_dict(),
        "f3_criteria": config.f3_criteria.to_dict(),
        "f4_criteria": config.f4_criteria.to_dict(),
        "hbi": _hbi_to_dict(config.hbi),
        "posterior_draws": int(config.posterior_draws),
        "rhat_draws_per_run": int(config.rhat_draws_per_run),
        "diagnostics_batch_size": int(config.diagnostics_batch_size),
        "canonicalize_exchangeable_components": bool(
            config.canonicalize_exchangeable_components
        ),
    }


_LEGACY_CONFIG_KEYS = (
    "f1_nuts",
    "f2_nuts",
    "f4_nuts",
    "f1_pe_samples_per_event",
    "f1_selected_per_campaign",
    "f0_pe_samples_per_event",
    "f2_criteria",
)


def fidelity_run_config_from_dict(payload: Mapping[str, object]) -> FidelityRunConfig:
    """Load a 2.0 configuration; NUTS/JAXNS-era 1.x configurations are refused."""
    payload = dict(payload)
    version = payload.get("format_version")
    legacy_keys = sorted(set(payload) & set(_LEGACY_CONFIG_KEYS))
    if version != FIDELITY_CONFIG_FORMAT_VERSION or legacy_keys:
        found = "no format_version (1.x)" if version is None else repr(version)
        raise LegacyFidelityConfigError(
            f"NUTS/JAXNS-era config; re-freeze: fidelity config has {found}"
            + (f" and legacy keys {legacy_keys}" if legacy_keys else "")
            + f"; ladder v2 requires {FIDELITY_CONFIG_FORMAT_VERSION} "
            "(gwpop-search write-default-fidelity-config)"
        )
    ladder = payload.pop("ladder_fidelities", None)
    if ladder is not None and list(ladder) != [item.value for item in LADDER_V2]:
        raise ValueError(f"unsupported ladder fidelities {ladder}")
    backend = payload.pop("sampler_backend", SAMPLER_BACKEND)
    if backend != SAMPLER_BACKEND:
        raise ValueError(f"unsupported sampler backend {backend!r}")
    known = {
        "format_version",
        "f0",
        "f3_evidence",
        "f4_evidence",
        "f3_criteria",
        "f4_criteria",
        "hbi",
        "posterior_draws",
        "rhat_draws_per_run",
        "diagnostics_batch_size",
        "canonicalize_exchangeable_components",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown fidelity config field(s): {unknown}")
    return FidelityRunConfig(
        f0=F0SanityConfig.from_dict(dict(payload["f0"])),
        f3_evidence=EvidenceCampaignConfig.from_dict(dict(payload["f3_evidence"])),
        f4_evidence=EvidenceCampaignConfig.from_dict(dict(payload["f4_evidence"])),
        f3_criteria=NumericalCriteria.from_dict(dict(payload["f3_criteria"])),
        f4_criteria=NumericalCriteria.from_dict(dict(payload["f4_criteria"])),
        hbi=_hbi_from_dict(dict(payload["hbi"])),
        posterior_draws=payload["posterior_draws"],
        rhat_draws_per_run=payload["rhat_draws_per_run"],
        diagnostics_batch_size=payload["diagnostics_batch_size"],
        canonicalize_exchangeable_components=payload["canonicalize_exchangeable_components"],
        format_version=str(payload["format_version"]),
    )


def fidelity_config_sha256(config: FidelityRunConfig) -> str:
    """SHA-256 of the canonical JSON of a fidelity configuration."""
    text = json.dumps(fidelity_run_config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_fidelity_run_config(path: str | Path, config: FidelityRunConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fidelity_run_config_to_dict(config), sort_keys=True, indent=2))


def load_fidelity_run_config(path: str | Path) -> FidelityRunConfig:
    return fidelity_run_config_from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _prior_center(spec) -> float:
    parameters = spec.parameters
    if spec.family == "uniform":
        return 0.5 * (parameters["low"] + parameters["high"])
    if spec.family == "log_uniform":
        return float(np.sqrt(parameters["low"] * parameters["high"]))
    if spec.family == "normal":
        return float(parameters["loc"])
    raise ValueError(f"unsupported prior family {spec.family!r}")


def prior_center(model: ModelSpec) -> dict[str, float]:
    """Centre of every declared hyperprior (a reference point for tools and tests)."""
    return {name: _prior_center(prior) for name, prior in model.priors.items()}


def posterior_median(samples: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Coordinate-wise median of equal-weight samples keyed by parameter name."""
    return {
        name: float(np.median(np.asarray(values, dtype=float)))
        for name, values in samples.items()
    }


def _check(name: str, value, limit, *, comparison: str, stage: str = "gate") -> dict[str, object]:
    if value is None or not np.isfinite(float(value)):
        passed = False
    elif comparison == "le":
        passed = float(value) <= float(limit)
    elif comparison == "ge":
        passed = float(value) >= float(limit)
    else:  # pragma: no cover - programming error
        raise ValueError(comparison)
    return {
        "name": name,
        "stage": stage,
        "value": None if value is None else float(value),
        "comparison": comparison,
        "limit": float(limit),
        "passed": bool(passed),
    }


def _bool_check(name: str, passed: bool, *, value=None, stage: str = "gate", note=None):
    payload = {
        "name": name,
        "stage": stage,
        "value": value,
        "comparison": "is_true",
        "limit": None,
        "passed": bool(passed),
    }
    if note is not None:
        payload["note"] = str(note)
    return payload


def apply_binding(
    checks: Sequence[Mapping[str, object]], criteria: "NumericalCriteria"
) -> list[dict[str, object]]:
    """Copies of ``checks`` with the per-family binding flags applied.

    Every check gets ``binding`` (``True``/``False``). A ``stage == "gate"``
    check of a non-binding family becomes ``stage = "advisory"``; advisory
    checks stay advisory (and non-binding).
    """
    out = []
    for check in checks:
        item = dict(check)
        stage = item.get("stage", "gate")
        binding = stage == "gate" and criteria.is_binding(str(item["name"]))
        if stage == "gate" and not binding:
            item["stage"] = "advisory"
        item["binding"] = bool(binding)
        out.append(item)
    return out


def _gates_passed(checks: Sequence[Mapping[str, object]]) -> bool:
    gates = [item for item in checks if item.get("stage", "gate") == "gate"]
    return bool(gates) and all(bool(item["passed"]) for item in gates)


# Non-finite floats are written as these strings: ``json.dumps`` would
# otherwise emit the non-standard tokens Infinity/-Infinity/NaN, which is
# exactly what happens when an importance gate fails (a NaN metric is mapped
# to +inf so that it fails), leaving evaluation.json unreadable by any strict
# JSON consumer. ``float()`` of each sentinel reproduces the value, and
# :func:`read_evaluation` restores them.
NON_FINITE_JSON_SENTINELS = {
    math.inf: "Infinity",
    -math.inf: "-Infinity",
}
NAN_JSON_SENTINEL = "NaN"
_JSON_SENTINEL_VALUES = {
    "Infinity": math.inf,
    "-Infinity": -math.inf,
    "NaN": math.nan,
}


def _json_float(value: float):
    value = float(value)
    if math.isnan(value):
        return NAN_JSON_SENTINEL
    if math.isinf(value):
        return NON_FINITE_JSON_SENTINELS[math.inf if value > 0 else -math.inf]
    return value


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _json_float(value)
    return value


def _json_restore(value):
    """Inverse of :func:`_json_ready` for the non-finite sentinels."""
    if isinstance(value, Mapping):
        return {str(k): _json_restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_restore(v) for v in value]
    if isinstance(value, str) and value in _JSON_SENTINEL_VALUES:
        return _JSON_SENTINEL_VALUES[value]
    return value


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
    )
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_chunk_size(identity: Mapping[str, object]) -> dict[str, object]:
    payload = json.loads(json.dumps(identity))
    hbi = payload.get("hbi_config")
    if isinstance(hbi, dict):
        hbi.pop("selection_chunk_size", None)
    return payload


def _identity_differences(a: Mapping, b: Mapping, prefix: str = "") -> list[str]:
    diffs = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key, "<missing>"), b.get(key, "<missing>")
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            diffs.extend(_identity_differences(x, y, prefix=f"{prefix}{key}."))
        elif x != y:
            diffs.append(f"{prefix}{key}")
    return diffs


def _installed_dynesty_version() -> str:
    import importlib.metadata

    try:
        return str(importlib.metadata.version("dynesty"))
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - optional extra
        return "unavailable"


# ---------------------------------------------------------------------------
# Importance diagnostics over the pooled posterior
# ---------------------------------------------------------------------------

# (gate metric, comparison, NumericalCriteria attribute, summary_statistics key)
_IMPORTANCE_GATES = (
    ("min_event_ess", "ge", "min_event_ess", "min_event_ess"),
    ("selection_ess", "ge", "min_selection_ess", "selection_ess"),
    ("max_event_weight_fraction", "le", "max_event_weight_fraction", "max_event_max_weight"),
    (
        "selection_max_weight_fraction",
        "le",
        "max_selection_weight_fraction",
        "selection_max_weight",
    ),
    (
        "shape_log_likelihood_variance",
        "le",
        "max_shape_log_likelihood_variance",
        "shape_log_likelihood_variance",
    ),
)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="inverted_cdf"))


def pooled_posterior_importance(
    results: Sequence[object],
    posterior,
    selection,
    population_model,
    *,
    hbi_config: HBIConfig,
    criteria: NumericalCriteria,
    median: Mapping[str, float],
    n_draws: int,
    seed: int,
    batch_size: int = 16,
    diagnostics_fn=None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Importance diagnostics at the posterior median and over pooled posterior draws.

    One jitted diagnostics function (the run's HBI configuration; built here
    unless a compiled ``diagnostics_fn`` for the same data, model and names is
    supplied) is evaluated at the pooled-posterior median point and at
    ``n_draws`` systematic-resampled draws of the equal-weight mixture of the
    runs (generator ``default_rng([seed, IMPT])``). Every run must carry the
    likelihood identity of exactly this estimator (``selection_chunk_size``
    aside, which only changes the summation order). Returns the importance
    block and its checks (point, draw median and draw tail per metric).
    """
    results = tuple(results)
    names = tuple(results[0].names)
    if diagnostics_fn is None:
        fn = build_importance_diagnostics(
            posterior,
            selection,
            population_model,
            names,
            hbi_config=hbi_config,
            batch_size=batch_size,
        )
    else:
        fn = diagnostics_fn
        if tuple(fn.names) != names:
            raise ValueError("diagnostics_fn parameter names differ from the runs")
    requested = _without_chunk_size(fn.likelihood_identity())
    for index, result in enumerate(results):
        stored = getattr(result, "likelihood_identity", None)
        if stored is None:
            raise ValueError(
                f"run {index} carries no likelihood identity; its importance diagnostics "
                "cannot be verified to describe the estimator it sampled"
            )
        diffs = _identity_differences(_without_chunk_size(stored), requested)
        if diffs:
            raise ValueError(
                f"run {index} sampled a different likelihood than the one diagnosed; "
                f"differing keys: {diffs}"
            )
    samples, weights = pooled_weighted_samples(results)
    rng = np.random.default_rng([int(seed), _IMPORTANCE_STREAM])
    draws = equal_weight_resample(samples, weights, int(n_draws), rng)
    point = np.asarray([float(median[name]) for name in names], dtype=float)
    batch = fn(np.vstack([point[None, :], draws]))
    stats = batch.summary_statistics()

    tail_q = float(criteria.importance_tail_quantile)
    at_point: dict[str, object] = {
        "hyperparameters": {name: float(point[k]) for k, name in enumerate(names)},
        "log_likelihood": float(stats["log_likelihood"][0]),
        "event_variance_total": float(stats["event_variance_total"][0]),
        "selection_variance_term": float(stats["selection_variance_term"][0]),
        "selection_ess_fraction": float(stats["selection_ess_fraction"][0]),
        "worst_event": str(batch.event_names[int(np.argmin(batch.event_ess[0]))]),
    }
    over: dict[str, dict[str, float]] = {}
    checks: list[dict[str, object]] = []
    draw_stage = "gate" if criteria.gate_importance_over_posterior else "advisory"
    for metric, comparison, attribute, key in _IMPORTANCE_GATES:
        limit = float(getattr(criteria, attribute))
        values = np.asarray(stats[key], dtype=float)
        if comparison == "le":
            values = np.where(np.isnan(values), np.inf, values)
        point_value = float(values[0])
        draw_values = values[1:]
        q = tail_q if comparison == "ge" else 1.0 - tail_q
        failing = draw_values < limit if comparison == "ge" else draw_values > limit
        entry = {
            "median": _quantile(draw_values, 0.5),
            "tail": _quantile(draw_values, q),
            "tail_quantile": q,
            "min": float(np.min(draw_values)),
            "max": float(np.max(draw_values)),
            "fraction_failing": float(np.mean(failing)),
        }
        at_point[metric] = point_value
        over[metric] = entry
        checks.append(
            _check(f"importance.{metric}.point", point_value, limit, comparison=comparison)
        )
        checks.append(
            _check(
                f"importance.{metric}.draw_median",
                entry["median"],
                limit,
                comparison=comparison,
                stage=draw_stage,
            )
        )
        checks.append(
            _check(
                f"importance.{metric}.draw_tail",
                entry["tail"],
                limit,
                comparison=comparison,
                stage=draw_stage,
            )
        )
    log_like = np.asarray(stats["log_likelihood"][1:], dtype=float)
    over["log_likelihood"] = {
        "median": _quantile(log_like, 0.5),
        "min": float(np.min(log_like)),
        "max": float(np.max(log_like)),
    }
    block = {
        "hbi_config": _hbi_to_dict(hbi_config),
        "at_posterior_median": at_point,
        "over_posterior": {
            "n_draws": int(draws.shape[0]),
            "seed": int(seed),
            "construction": "systematic resampling of the equal-weight run mixture",
            "metrics": over,
        },
    }
    return block, checks


# ---------------------------------------------------------------------------
# Summary of a repeated dynesty fit (F3/F4 and any posterior refit)
# ---------------------------------------------------------------------------


def pooled_posterior_summary(
    results: Sequence[object],
    *,
    priors: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Weighted quantiles of the equal-weight run mixture (canonical labels)."""
    results = tuple(results)
    names = tuple(results[0].names)
    samples, weights = pooled_weighted_samples(results)
    qs = (0.05, 0.5, 0.95)
    table = {
        name: weighted_quantiles(samples[:, k], weights, qs) for k, name in enumerate(names)
    }
    summary: dict[str, object] = {
        "parameter_order": list(names),
        "q05": {name: float(values[0]) for name, values in table.items()},
        "median": {name: float(values[1]) for name, values in table.items()},
        "q95": {name: float(values[2]) for name, values in table.items()},
        "construction": (
            "equal-weight mixture of separately normalized runs; weighted quantiles over "
            "dead and final live points"
        ),
    }
    if priors is not None:
        summary["prior_edge_mass_1pct"] = prior_edge_mass(samples, weights, names, priors)
    return summary


def summarize_dynesty_fit(
    results: Sequence[object],
    posterior,
    selection,
    population_model,
    *,
    hbi_config: HBIConfig,
    criteria: NumericalCriteria,
    priors: Mapping[str, object] | None = None,
    n_draws: int = 512,
    draw_seed: int = 0,
    rhat_seed: int = 0,
    rhat_draws_per_run: int = 2000,
    diagnostics_batch_size: int = 16,
    diagnostics_fn=None,
    taper_loglike=None,
    taper_batch_size: int = 64,
) -> dict[str, object]:
    """All numerical gates of a repeated dynesty fit.

    ``results`` are independent runs of one model on one dataset (each a
    :class:`~gwpop_search.inference.dynesty_backend.DynestyResult`). Returns
    ``passed`` (every ``stage == "gate"`` check), ``checks``,
    ``nested_sampling`` (per-run diagnostics, cross-run R-hat, insertion
    index), ``evidence``, ``posterior``, the flat ``posterior_median`` and
    ``importance``. ``diagnostics_fn`` optionally supplies an already
    compiled :func:`~gwpop_search.inference.dynesty_backend.build_importance_diagnostics`
    function (its likelihood identity is still verified against every run).

    Variance taper (``hbi_config.variance_taper``): the ``taper`` block is the
    posterior taper-mass diagnostic of
    :func:`~gwpop_search.inference.dynesty_backend.posterior_taper_mass`
    (per run and pooled; the fraction of posterior mass inside the taper
    region), evaluated with ``taper_loglike`` (a tapered batched likelihood
    of the same estimator; built here with ``taper_batch_size`` rows per
    call when omitted). It is always reported; the check
    ``taper.posterior_mass_in_taper_region`` is added only when
    ``criteria.max_posterior_taper_mass`` is set. Without a taper the block
    is ``None``.

    Binding: ``criteria.binding`` is applied to every check (each carries
    ``binding``; non-binding families are recorded as advisory).
    """
    results = tuple(results)
    if not results:
        raise ValueError("at least one dynesty result is required")
    names = tuple(results[0].names)
    if any(tuple(result.names) != names for result in results):
        raise ValueError("runs have different parameter names")
    checks: list[dict[str, object]] = []
    runs = [run_diagnostics(result, repeat=index) for index, result in enumerate(results)]

    if criteria.require_dlogz_termination:
        stopped = [row["repeat"] for row in runs if row["termination"] != "dlogz"]
        checks.append(
            _bool_check(
                "nested_sampling.all_runs_terminated_by_dlogz",
                not stopped,
                value=len(stopped),
                note=None if not stopped else f"runs {stopped} stopped before dlogz",
            )
        )
    if criteria.require_selection_support:
        counts = [row["n_selection_unsupported"] for row in runs]
        total = None if any(count is None for count in counts) else int(sum(counts))
        checks.append(
            _check(
                "nested_sampling.selection_unsupported_evaluations",
                total,
                0,
                comparison="le",
            )
        )
    min_kish = float(min(row["kish_ess"] for row in runs))
    if criteria.min_kish_ess_per_run is not None:
        checks.append(
            _check(
                "nested_sampling.min_kish_ess_per_run",
                min_kish,
                criteria.min_kish_ess_per_run,
                comparison="ge",
            )
        )
    rhat = (
        cross_run_rhat(results, max_draws=rhat_draws_per_run, seed=rhat_seed)
        if len(results) >= 2
        else None
    )
    if criteria.max_cross_run_r_hat is not None:
        checks.append(
            _check(
                "nested_sampling.cross_run_r_hat",
                None if rhat is None else rhat["max"],
                criteria.max_cross_run_r_hat,
                comparison="le",
            )
        )

    repeat = summarize_evidence_repeats(results)
    predicted = [row["predicted_error"] for row in runs if row["predicted_error"] is not None]
    evidence = {
        **repeat.to_dict(),
        "predicted_error": float(np.mean(predicted)) if predicted else None,
        "semantics": (
            "mean over independent dynesty runs of ln Z of the rate-marginalized shape "
            "likelihood (p(R) ~ 1/R); conservative_error = max(repeat std, max logzerr)"
        ),
    }
    if criteria.max_evidence_error is not None:
        checks.append(
            _check(
                "evidence.conservative_error",
                repeat.conservative_error,
                criteria.max_evidence_error,
                comparison="le",
            )
        )
    if criteria.max_evidence_repeat_std is not None:
        checks.append(
            _check(
                "evidence.repeat_std",
                repeat.repeat_std if repeat.n_repeats > 1 else None,
                criteria.max_evidence_repeat_std,
                comparison="le",
            )
        )
    if criteria.max_repeat_consistency_z is not None:
        checks.append(
            _check(
                "evidence.max_pairwise_z",
                repeat.max_pairwise_z,
                criteria.max_repeat_consistency_z,
                comparison="le",
            )
        )

    posterior_block = pooled_posterior_summary(results, priors=priors)
    importance, importance_checks = pooled_posterior_importance(
        results,
        posterior,
        selection,
        population_model,
        hbi_config=hbi_config,
        criteria=criteria,
        median=posterior_block["median"],
        n_draws=n_draws,
        seed=draw_seed,
        batch_size=diagnostics_batch_size,
        diagnostics_fn=diagnostics_fn,
    )
    checks.extend(importance_checks)

    taper_block = None
    taper = hbi_config.variance_taper
    if taper is None:
        if criteria.max_posterior_taper_mass is not None:
            raise ValueError(
                "max_posterior_taper_mass is set but the HBI configuration has no variance taper"
            )
    else:
        from .dynesty_backend import posterior_taper_mass

        loglike = taper_loglike
        if loglike is None:
            loglike = build_batched_log_likelihood(
                posterior,
                selection,
                population_model,
                names,
                hbi_config=hbi_config,
                batch_size=taper_batch_size,
            )
        taper_block = posterior_taper_mass(results, loglike)
        if criteria.max_posterior_taper_mass is not None:
            checks.append(
                _check(
                    "taper.posterior_mass_in_taper_region",
                    taper_block["pooled"]["posterior_mass_in_taper_region"],
                    criteria.max_posterior_taper_mass,
                    comparison="le",
                )
            )

    insertion = {
        "available": False,
        "reason_code": INSERTION_INDEX_UNAVAILABLE_REASON,
        "reason": (
            "the dynesty backend results do not persist samples_it (the birth "
            "iteration of each dead point), which the insertion-index "
            "reconstruction needs"
        ),
        "requires": "DynestyResult.samples_it",
        "implementation": (
            "gwpop_search.inference.ns_diagnostics.insertion_index_ranks / "
            "insertion_index_test (ready; unused until the backend persists the "
            "birth iterations)"
        ),
    }
    if criteria.insertion_index_advisory:
        checks.append(
            {
                "name": "nested_sampling.insertion_index_ks",
                "stage": "advisory",
                "available": False,
                "value": None,
                "comparison": "p_value_ge",
                "limit": None,
                "passed": None,
                "reason_code": INSERTION_INDEX_UNAVAILABLE_REASON,
                "note": insertion["reason"],
            }
        )

    checks = apply_binding(checks, criteria)
    summary = {
        "passed": _gates_passed(checks),
        "checks": checks,
        "nested_sampling": {
            "n_runs": len(results),
            "parameter_order": list(names),
            "runs": runs,
            "min_kish_ess_per_run": min_kish,
            "cross_run_r_hat": rhat,
            "insertion_index": insertion,
        },
        "evidence": evidence,
        "posterior": posterior_block,
        "posterior_median": dict(posterior_block["median"]),
        "importance": importance,
    }
    if taper_block is not None:
        summary["taper"] = taper_block
    return summary


def save_pooled_posterior(path: str | Path, results: Sequence[object]) -> dict[str, object]:
    """Write the pooled equal-weight posterior (each run's equal-weight draws).

    Every run contributes its own ``posterior_samples`` (systematic
    resampling of that run); equal counts per run make the concatenation an
    equal-weight draw from the equal mixture of separately normalized runs.
    Returns the artifact description recorded in the evaluation.
    """
    results = tuple(results)
    names = tuple(results[0].names)
    counts = {int(np.asarray(result.posterior_samples).shape[0]) for result in results}
    if len(counts) != 1:
        raise ValueError(f"runs carry different numbers of equal-weight draws: {sorted(counts)}")
    samples = np.vstack([np.asarray(result.posterior_samples, dtype=float) for result in results])
    run_index = np.repeat(np.arange(len(results), dtype=np.int64), counts.pop())
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray(POOLED_POSTERIOR_FORMAT_VERSION),
            parameter_names=np.asarray(names),
            samples=samples,
            run_index=run_index,
        )
    os.replace(tmp, path)
    return {
        "artifact": path.name,
        "format_version": POOLED_POSTERIOR_FORMAT_VERSION,
        "sha256": _sha256_file(path),
        "n_draws": int(samples.shape[0]),
        "n_runs": len(results),
        "parameter_order": list(names),
        "construction": (
            "concatenated equal-weight draws of each run (equal counts per run): an "
            "equal-weight sample of the equal mixture of separately normalized runs"
        ),
    }


def load_pooled_posterior(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``pooled_posterior.npz`` (``parameter_names``, ``samples``, ``run_index``)."""
    with np.load(Path(path), allow_pickle=False) as archive:
        version = str(archive["format_version"])
        if version != POOLED_POSTERIOR_FORMAT_VERSION:
            raise ValueError(f"unsupported pooled posterior format {version!r}")
        return {
            "parameter_names": np.asarray(archive["parameter_names"]).astype(str),
            "samples": np.asarray(archive["samples"], dtype=float),
            "run_index": np.asarray(archive["run_index"], dtype=np.int64),
        }


# ---------------------------------------------------------------------------
# F0 sanity
# ---------------------------------------------------------------------------


def run_f0_sanity(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, object],
    *,
    seed: int,
    config: F0SanityConfig,
    hbi_config: HBIConfig,
    parameterization: ModelParameterization,
    diagnostics_batch_size: int = 16,
) -> dict[str, object]:
    """Prior-support scan on the full data plus JAX/NumPy parity on a thinned pair.

    NaN/``+inf`` in any likelihood evaluation raises
    :class:`gwpop_search.hbi.PopulationDensityError` (never a gate value).
    """
    names, transform = parameterization.prior_transform(priors)
    rng = np.random.default_rng([int(seed), _F0_STREAM])
    theta = np.asarray(transform(rng.random((config.prior_draws, len(names)))), dtype=float)

    # 1. The production batched likelihood on the full data.
    start = time.perf_counter()
    loglike = build_batched_log_likelihood(
        posterior,
        selection,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=config.batch_size,
    )
    first = loglike(theta[: config.batch_size])
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rest = (
        loglike(theta[config.batch_size :])
        if theta.shape[0] > config.batch_size
        else np.empty(0, dtype=float)
    )
    evaluation_seconds = time.perf_counter() - start
    values = np.concatenate([first, rest])
    finite = np.isfinite(values)
    finite_fraction = float(np.mean(finite))
    stats = loglike.stats()

    # 2. Jitted importance diagnostics on the same draws: exposure support.
    diagnostics_fn = build_importance_diagnostics(
        posterior,
        selection,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=diagnostics_batch_size,
    )
    batch = diagnostics_fn(theta)
    zero_exposure = ~np.isfinite(np.asarray(batch.log_exposure, dtype=float))
    zero_exposure_fraction = float(np.mean(zero_exposure))
    diag_values = np.asarray(batch.log_likelihood, dtype=float)
    same_support = bool(np.array_equal(np.isfinite(diag_values), finite))
    both = finite & np.isfinite(diag_values)
    consistency = (
        float(
            np.max(
                np.abs(diag_values[both] - values[both])
                / np.maximum(1.0, np.abs(values[both]))
            )
        )
        if np.any(both)
        else 0.0
    )
    event_zero = ~np.isfinite(np.asarray(batch.event_log_likelihoods, dtype=float))
    event_fraction = np.mean(event_zero, axis=0)
    top = np.argsort(-event_fraction, kind="stable")[:10]

    # 3. JAX vs NumPy parity on a small thinned pair.
    pe_thin, sel_thin = thin_catalog_pair(
        posterior,
        selection,
        max_samples_per_event=config.parity_pe_samples_per_event,
        max_selected_per_campaign=config.parity_selected_per_campaign,
        seed=seed,
    )
    thin_loglike = build_batched_log_likelihood(
        pe_thin,
        sel_thin,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=config.batch_size,
    )
    thin_values = thin_loglike(theta)
    points = []
    for index in np.flatnonzero(np.isfinite(thin_values))[: config.parity_points]:
        hyperparameters = {name: float(theta[index, k]) for k, name in enumerate(names)}
        reference = float(
            shape_log_likelihood(
                pe_thin, sel_thin, population_model, hyperparameters, config=hbi_config
            ).log_likelihood
        )
        jax_value = float(thin_values[index])
        difference = abs(reference - jax_value)
        points.append(
            {
                "draw": int(index),
                "hyperparameters": hyperparameters,
                "jax": jax_value,
                "numpy": reference,
                "abs_diff": difference,
                "rel_diff": difference / max(1.0, abs(reference)),
            }
        )
    parity_max = max((item["rel_diff"] for item in points), default=None)

    best = int(np.argmax(np.where(finite, values, -np.inf))) if np.any(finite) else None
    checks = [
        _bool_check("f0.batched_likelihood_compiles_without_nan_or_posinf", True),
        _check(
            "f0.finite_fraction",
            finite_fraction,
            config.min_finite_fraction,
            comparison="ge",
        ),
        _check(
            "f0.zero_exposure_fraction",
            zero_exposure_fraction,
            config.max_zero_exposure_fraction,
            comparison="le",
        ),
        _bool_check(
            "f0.importance_diagnostics_reproduce_likelihood",
            same_support and consistency <= config.parity_rtol,
            value=consistency,
        ),
        _check(
            "f0.jax_numpy_parity_rel_diff",
            parity_max,
            config.parity_rtol,
            comparison="le",
        ),
    ]
    throughput = {
        "batch_size": int(config.batch_size),
        "n_draws": int(theta.shape[0]),
        "compile_and_first_batch_seconds": float(compile_seconds),
        "evaluation_seconds": float(evaluation_seconds),
        "evaluations_per_second": (
            float(rest.size / evaluation_seconds) if evaluation_seconds > 0 and rest.size else None
        ),
        "device_seconds": float(stats.get("device_seconds", 0.0)),
    }
    support = {
        "n_draws": int(theta.shape[0]),
        "seed_stream": [int(seed), _F0_STREAM],
        "n_finite": int(np.count_nonzero(finite)),
        "finite_fraction": finite_fraction,
        "n_zero_exposure": int(np.count_nonzero(zero_exposure)),
        "zero_exposure_fraction": zero_exposure_fraction,
        "n_selection_unsupported": int(stats.get("n_selection_unsupported", 0)),
        "max_log_likelihood": None if best is None else float(values[best]),
        "argmax_hyperparameters": (
            None
            if best is None
            else {name: float(theta[best, k]) for k, name in enumerate(names)}
        ),
        "event_zero_support_fraction_top": [
            {"event": str(batch.event_names[i]), "fraction": float(event_fraction[i])}
            for i in top
        ],
        "diagnostics_likelihood_max_rel_diff": consistency,
        "diagnostics_likelihood_same_support": same_support,
    }
    parity = {
        "pe_samples_per_event": int(config.parity_pe_samples_per_event),
        "selected_per_campaign": int(config.parity_selected_per_campaign),
        "n_thinned_finite": int(np.count_nonzero(np.isfinite(thin_values))),
        "n_points_compared": len(points),
        "points": points,
        "max_rel_diff": parity_max,
        "rtol": float(config.parity_rtol),
    }
    return {
        "passed": _gates_passed(checks),
        "checks": checks,
        "parameter_order": list(names),
        "parameterization": parameterization.to_dict(),
        "support": support,
        "parity": parity,
        "throughput": throughput,
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _screen_semantics(fidelity: Fidelity) -> str:
    if fidelity in EVIDENCE_RUNGS:
        return "log_evidence_mean"
    return "sanity_constant"


@dataclass
class DeterministicHBIEvaluator:
    """F0/F3/F4 evaluator of the standardized HBI likelihood (ladder v2).

    ``evaluate`` writes ``run_dir/evaluation.json`` (format 2.0) and, for F3
    and F4, the evidence runs under ``run_dir/evidence/repeat_###`` and the
    pooled equal-weight posterior ``run_dir/pooled_posterior.npz``. Typed
    numerical failures (no finite prior support for dynesty) complete with
    ``diagnostics_pass=False`` and a ``failure`` block; NaN/``+inf``
    population densities propagate as errors.
    """

    posterior: object
    selection: object
    config: FidelityRunConfig = field(default_factory=FidelityRunConfig)
    dataset_identity: str = "unspecified"

    supported_fidelities = tuple(item.value for item in LADDER_V2)

    @property
    def fidelity_config_sha256(self) -> str:
        """Hash of the frozen numerical configuration of every evaluation.

        The search executor records it with each stored evaluation row and
        refuses to reuse a row produced under a different one (evaluation seeds
        do not depend on the configuration).
        """
        return fidelity_config_sha256(self.config)

    def _write_evaluation(
        self,
        run_dir: Path,
        *,
        model: ModelSpec,
        fidelity: Fidelity,
        diagnostics: Mapping[str, object],
        screen_value: float | None,
        elapsed_seconds: float,
    ) -> None:
        payload = {
            "format_version": EVALUATION_FORMAT_VERSION,
            "model_hash": model.model_hash,
            "fidelity": fidelity.value,
            "dataset_identity": self.dataset_identity,
            "sampler_backend": {"name": SAMPLER_BACKEND, "version": _installed_dynesty_version()},
            "fidelity_config_sha256": fidelity_config_sha256(self.config),
            "screen_value": None if screen_value is None else float(screen_value),
            "screen_value_semantics": _screen_semantics(fidelity),
            "elapsed_seconds": float(elapsed_seconds),
            "diagnostics": dict(diagnostics),
        }
        _atomic_write_json(run_dir / "evaluation.json", payload)

    def _evidence_rung(
        self,
        model: ModelSpec,
        fidelity: Fidelity,
        population_model,
        priors,
        parameterization: ModelParameterization,
        *,
        seed: int,
        run_dir: Path,
    ) -> tuple[dict[str, object], float | None]:
        evidence_config = self.config.evidence_config(fidelity)
        criteria = self.config.criteria(fidelity)
        sampling = {
            "evidence_config": evidence_config.to_dict(),
            "resolved_dynesty_config": evidence_config.dynesty_config_for(len(priors)).to_dict(),
            "root_seed": int(seed),
            "parameterization": parameterization.to_dict(),
            "evidence_dir": "evidence",
        }
        try:
            results, _ = run_model_evidence_repeats(
                run_dir / "evidence",
                model,
                self.posterior,
                self.selection,
                root_seed=seed,
                config=evidence_config,
                hbi_config=self.config.hbi,
                dataset_identity=self.dataset_identity,
                parameterization=parameterization,
            )
        except NoFiniteSupportError as exc:
            failure = {"type": "no_finite_support", "message": str(exc)}
            checks = [
                _bool_check(
                    "nested_sampling.initialization_found_finite_support",
                    False,
                    note=str(exc),
                )
            ]
            return (
                {
                    "passed": False,
                    "checks": checks,
                    "failure": failure,
                    "nested_sampling": sampling,
                },
                None,
            )
        diagnostics = summarize_dynesty_fit(
            results,
            self.posterior,
            self.selection,
            population_model,
            hbi_config=self.config.hbi,
            criteria=criteria,
            priors=priors,
            n_draws=self.config.posterior_draws,
            draw_seed=_derived_seed(seed, "importance-draws"),
            rhat_seed=_derived_seed(seed, "cross-run-rhat"),
            rhat_draws_per_run=self.config.rhat_draws_per_run,
            diagnostics_batch_size=self.config.diagnostics_batch_size,
        )
        diagnostics["nested_sampling"] = {**sampling, **diagnostics["nested_sampling"]}
        artifact = save_pooled_posterior(run_dir / POOLED_POSTERIOR_FILENAME, results)
        diagnostics["posterior"] = {**diagnostics["posterior"], "pooled_equal_weight": artifact}
        diagnostics["failure"] = None
        return diagnostics, float(diagnostics["evidence"]["log_evidence_mean"])

    def evaluate(
        self,
        model: ModelSpec,
        fidelity: Fidelity,
        *,
        seed: int,
        run_dir: Path,
    ) -> EvaluationRecord:
        fidelity = Fidelity(fidelity)
        if fidelity not in LADDER_V2:
            raise ValueError(
                f"{fidelity.value} is not part of fidelity ladder v2 "
                f"({[item.value for item in LADDER_V2]}); F1/F2 were NUTS-era rungs"
            )
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()

        population_model = compile_model_spec(model)
        data_support = None
        if getattr(population_model, "is_v2", False):
            # v2 models declare their support (zmax, q_floor, mmin/mmax, sky,
            # cosmology); the dataset must match it before anything is sampled
            from gwpop_search.models.data_support import require_v2_data_support

            data_support = require_v2_data_support(
                population_model, self.posterior, self.selection,
                context=f"model {model.model_hash}",
            )
        priors = prior_specs_from_model_spec(model)
        require_population_proxy_coverage(
            self.selection,
            priors=priors,
            context=f"model {model.model_hash}",
        )
        parameterization = parameterization_for_spec(
            model, canonicalize=self.config.canonicalize_exchangeable_components
        )

        if fidelity is Fidelity.F0_SANITY:
            diagnostics = run_f0_sanity(
                self.posterior,
                self.selection,
                population_model,
                priors,
                seed=seed,
                config=self.config.f0,
                hbi_config=self.config.hbi,
                parameterization=parameterization,
                diagnostics_batch_size=self.config.diagnostics_batch_size,
            )
            screen_value: float | None = 0.0
        else:
            diagnostics, screen_value = self._evidence_rung(
                model,
                fidelity,
                population_model,
                priors,
                parameterization,
                seed=seed,
                run_dir=run_dir,
            )

        if data_support is not None:
            diagnostics = {**diagnostics, "v2_data_support": data_support}
        elapsed = time.perf_counter() - start
        self._write_evaluation(
            run_dir,
            model=model,
            fidelity=fidelity,
            diagnostics=diagnostics,
            screen_value=screen_value,
            elapsed_seconds=elapsed,
        )
        return EvaluationRecord(
            model_hash=model.model_hash,
            fidelity=fidelity,
            diagnostics_pass=bool(diagnostics["passed"]),
            screen_value=screen_value,
            compute_cost=float(elapsed / 3600.0),
            status="complete",
        )


def read_evaluation(path: str | Path) -> dict[str, object]:
    """Load an ``evaluation.json`` of format 2.0; legacy (1.x) evaluations are refused.

    The artifact is strict JSON: non-finite numbers are stored as the strings
    ``"Infinity"``, ``"-Infinity"`` and ``"NaN"`` and are restored to floats
    here.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    version = payload.get("format_version")
    if version != EVALUATION_FORMAT_VERSION:
        raise LegacyEvaluationError(
            f"{path} is a {version!r} evaluation; only {EVALUATION_FORMAT_VERSION} "
            "(dynesty ladder v2) evaluations are accepted"
        )
    return _json_restore(payload)


# ---------------------------------------------------------------------------
# Legacy NUTS compatibility (validation/holdout_campaign.py only)
# ---------------------------------------------------------------------------
# The NUTS-era holdout refits still import ``summarize_nuts_fit`` and read
# ``FidelityRunConfig.f4_nuts``. Neither is part of ladder v2 or of fidelity
# config 2.0; they keep that module working unchanged until it is moved to
# dynesty, and nothing in the production ladder calls them.

LEGACY_HOLDOUT_NUTS_SETTINGS = {
    "num_warmup": 2_000,
    "num_samples": 2_000,
    "num_chains": 4,
    "target_accept_prob": 0.95,
    "max_tree_depth": 12,
    "progress_bar": False,
}
LEGACY_NUTS_MIN_N_EFF = 400.0
LEGACY_NUTS_MAX_DIVERGENCES = 0


def _numpy_importance_metrics(
    posterior, selection, population_model, hyperparameters, *, hbi_config
):
    evaluated = shape_log_likelihood(
        posterior, selection, population_model, hyperparameters, config=hbi_config
    )
    event = evaluated.terms.events.diagnostics
    sel = evaluated.terms.selection.diagnostics
    return {
        "log_likelihood": float(evaluated.log_likelihood),
        "min_event_ess": float(min(item.ess for item in event)),
        "max_event_weight_fraction": float(max(item.max_weight_fraction for item in event)),
        "selection_ess": float(sel.ess),
        "selection_max_weight_fraction": float(sel.max_weight_fraction),
        "shape_log_likelihood_variance": float(
            evaluated.terms.variance.shape_log_likelihood_variance
        ),
    }


def summarize_nuts_fit(
    result,
    posterior,
    selection,
    population_model,
    *,
    hbi_config: HBIConfig,
    criteria: NumericalCriteria,
    min_n_eff: float = LEGACY_NUTS_MIN_N_EFF,
    max_divergences: int = LEGACY_NUTS_MAX_DIVERGENCES,
) -> dict[str, object]:
    """Deprecated NUTS diagnostics (legacy holdout refits only; not a ladder rung).

    Importance thresholds come from ``criteria`` at the NUTS posterior median;
    split R-hat uses ``criteria.max_cross_run_r_hat`` (the same threshold
    values as the retired ``max_r_hat``), ``min_n_eff`` and
    ``max_divergences`` default to the frozen 1.x F4 values.
    """
    from .recovery import chain_diagnostics

    chain = chain_diagnostics(result.samples)
    divergent = np.asarray(result.extra_fields.get("diverging", []), dtype=bool)
    median = posterior_median(result.samples)
    importance = _numpy_importance_metrics(
        posterior, selection, population_model, median, hbi_config=hbi_config
    )
    checks = [
        _check(
            "min_event_ess",
            importance["min_event_ess"],
            criteria.min_event_ess,
            comparison="ge",
        ),
        _check(
            "selection_ess",
            importance["selection_ess"],
            criteria.min_selection_ess,
            comparison="ge",
        ),
        _check(
            "max_event_weight_fraction",
            importance["max_event_weight_fraction"],
            criteria.max_event_weight_fraction,
            comparison="le",
        ),
        _check(
            "selection_max_weight_fraction",
            importance["selection_max_weight_fraction"],
            criteria.max_selection_weight_fraction,
            comparison="le",
        ),
        _check(
            "shape_log_likelihood_variance",
            importance["shape_log_likelihood_variance"],
            criteria.max_shape_log_likelihood_variance,
            comparison="le",
        ),
        _check("min_mcmc_n_eff", chain["min_n_eff"], min_n_eff, comparison="ge"),
        _check("n_divergent", int(divergent.sum()), max_divergences, comparison="le"),
    ]
    if criteria.max_cross_run_r_hat is not None:
        checks.append(
            _check("max_r_hat", chain["max_r_hat"], criteria.max_cross_run_r_hat, comparison="le")
        )
    return {
        "passed": _gates_passed(checks),
        "checks": checks,
        "chain": chain,
        "n_divergent": int(divergent.sum()),
        "posterior_median": median,
        "importance": importance,
        "legacy": "nuts_holdout_refit",
    }
