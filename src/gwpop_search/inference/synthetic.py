"""Closed synthetic BBH data for Phase-3 end-to-end recovery tests.

This is deliberately a simple validation survey, not an astrophysical detector
simulation. It produces PE samples and a raw-draw selection campaign in the
exact gwcat-v2 chi_eff density basis so the common HBI engine can be tested
without release-specific data.

Two independent choices define a survey.

``SyntheticSurveyConfig.observation_model`` -- how events are observed:

``truth_centered`` (default, unchanged legacy behavior)
    Detection is applied to the true parameters and each event's PE samples are
    a truncated normal centred on the true parameters (zero-noise PE). This is
    not a physical data-generating process: detection is not a function of the
    data, and the truth sits at the same quantile of every event's PE. Width
    hyperparameters are therefore biased low (Essick & Fishbach 2023,
    arXiv:2310.02017). Kept only so existing campaigns reproduce.

``noisy_observation`` (DAG-consistent)
    Every system (event or injection) receives exactly one measurement-noise
    realisation, which produces its observed data
    ``d = (ln m1_detector, q, ln d_L, chi_eff) + N(0, diag(s^2))`` with
    ``s = (pe_m1_fractional_sigma, pe_q_sigma, pe_d_l_fractional_sigma,
    pe_chi_eff_sigma)``; the sky is observed exactly. Detection is the same
    chirp-mass-scaled reach evaluated on the observed data. PE samples are
    exact draws from the posterior ``p(theta | d) proportional to
    L(d | theta) pi(theta)`` under the stored uniform detector-box prior
    ``pi``, so the PE is centred on the observed, not the true, values.
    Selection injections are detected with the same statistic on their own
    noise realisation, so the raw-draw estimator measures
    ``P_det(theta) = P(d in detected | theta)``, the probability that enters
    the likelihood.

``SyntheticSurveyConfig.injection_draw`` -- how selection injections are drawn:

``uniform_detector_box`` (default, unchanged legacy behavior)
    Uniform in (m1_detector, q, luminosity_distance, chi_eff) over the detector
    prior box, isotropic sky. Simple, but the population occupies a tiny corner
    of the box, so the selection effective sample size is only ~1-2% of the
    detected injections.

``population_proxy``
    Injections are drawn from the baseline population at fixed proxy
    hyperparameters (``injection_draw_hyperparameters``) with
    ``_draw_population``, mixed with a defensive pairing component: for
    ``m1_source < POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX`` a fraction
    ``injection_draw_defensive_fraction`` of draws replace q by
    ``q = 1 - u`` with u log-uniform on ``[r u_max, u_max]``,
    ``u_max = 1 - max(q_floor, mmin_proxy/m1_source)``,
    ``r = POPULATION_PROXY_DEFENSIVE_U_RATIO``. The stored raw-draw density is
    exactly that mixture: the baseline model's own normalized detector-basis
    log density at the proxy with its pairing factor ``p_q(q | m1)`` replaced
    by ``(1 - eps) p_q + eps g``. It includes the source-to-detector Jacobian
    and the 1/(4 pi) sky factor. The proxy's support contains the support of
    every population inside the Phase-3 hyperprior, so
    ``A = T/N_draw * sum_detected p_pop/p_draw`` stays unbiased for every trial
    hyperparameter point. The defensive component exists because the target
    pairing density ``q**beta / Z(m1)`` diverges like ``1/(1 - mmin/m1)`` as
    ``m1 -> mmin`` for every target with ``mmin > mmin_proxy``; no single
    population proxy can follow that, so without it the importance weights
    have an infinite (log-divergent) second moment and rare single rows move
    ``N log A`` by several nats. With it the corner weights are bounded by
    ``ln(1/r)/eps`` times the other density ratios. ``eps = 0`` reproduces the
    pure proxy exactly.

In every mode the campaign stores the true total number of draws ``n_draw``,
keeps only detected rows and records the TRUE parameters of those rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Callable, Mapping

import numpy as np
from scipy.stats import truncnorm

from ..data import Campaign, PosteriorCatalog, SelectionCatalog, SelectionMode, validate_pair
from ..data.adapters import gwcat_v2_basis_for_spin
from ..models import DEFAULT_BASELINE_HYPERPARAMETERS, GwcatChiEffBBHModel
from ..models.components import mass_ratio_logpdf
from .priors import BASELINE_SYNTHETIC_PRIORS, PriorSpec

INJECTION_DRAW_UNIFORM_DETECTOR_BOX = "uniform_detector_box"
INJECTION_DRAW_POPULATION_PROXY = "population_proxy"
INJECTION_DRAWS = (
    INJECTION_DRAW_UNIFORM_DETECTOR_BOX,
    INJECTION_DRAW_POPULATION_PROXY,
)

OBSERVATION_MODEL_TRUTH_CENTERED = "truth_centered"
OBSERVATION_MODEL_NOISY = "noisy_observation"
OBSERVATION_MODELS = (
    OBSERVATION_MODEL_TRUTH_CENTERED,
    OBSERVATION_MODEL_NOISY,
)


class FrozenHyperparameters(dict):
    """Read-only, hashable ``name -> float`` mapping.

    Frozen configs store validated hyperparameters in this type so they cannot
    be changed after validation (item assignment raises ``TypeError``) and the
    config stays hashable. It is a ``dict`` subclass, so it compares equal to
    plain dicts with the same items and serializes to JSON unchanged.
    """

    def __init__(self, values: Mapping[str, float] = (), /) -> None:
        super().__init__(
            (str(name), float(value)) for name, value in dict(values).items()
        )

    def _read_only(self, *args, **kwargs):
        raise TypeError(f"{type(self).__name__} is read-only")

    __setitem__ = _read_only
    __delitem__ = _read_only
    __ior__ = _read_only
    clear = _read_only
    pop = _read_only
    popitem = _read_only
    setdefault = _read_only
    update = _read_only

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))

    def __reduce__(self):
        return (type(self), (dict(self),))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict.__repr__(self)})"


# Default proxy population for ``injection_draw="population_proxy"``.
#
# Support (required for an unbiased raw-draw estimator at every trial point):
# mmin=2 and mmax=120 are the Phase-3 hyperprior bounds (mmin U(2,10), mmax
# U(60,120)), so [mmin_proxy, mmax_proxy] contains every population mass range
# and the pairing support q >= max(q_floor, mmin_proxy/m1) contains every
# population pairing support. Redshift (same model zmax), chi_eff ([-1, 1]) and
# the isotropic sky have full support for any proxy value.
#
# Efficiency (a choice, not a correctness requirement): the selection integral
# only needs injections where *detected* systems of plausible populations live.
# Detection favours heavy, nearby, near-equal-mass binaries, so the proxy is a
# detection-tilted version of the Phase-3 truth: a shallow mass power law
# (alpha 1.25 vs 3), a larger 35 Msun peak (0.35 vs 0.10, same location and
# width), pairing tilted toward q=1 (beta_q 2 vs 1) and a redshift density
# weighted to low z (kappa -1.5 vs 2). chi_eff does not affect detection; its
# proxy keeps the truth mean and is wider (0.25 vs 0.20) so chi weights stay
# bounded over the whole chi_sigma prior (efficiency factor >= 0.025 for
# chi_sigma up to 0.5 and 0.93 at the truth). Selected by Monte-Carlo scans of
# the selection effective sample size per draw at the truth, at the hyperprior
# median, over posterior-like clouds and over random hyperprior draws; see
# docs/phase3_recovery.md.
DEFAULT_POPULATION_PROXY_HYPERPARAMETERS = FrozenHyperparameters(
    {
        "alpha": 1.25,
        "mmin": 2.0,
        "mmax": 120.0,
        "peak_fraction": 0.35,
        "peak_mu": 35.0,
        "peak_sigma": 4.0,
        "beta_q": 2.0,
        "kappa": -1.5,
        "chi_mu": 0.05,
        "chi_sigma": 0.25,
    }
)

# Defensive pairing component of the population proxy (see module docstring).
# A log-uniform density in u = 1 - q is the density that bounds
# p_q(target)/p_q(draw) uniformly over every support width
# 1 - mmin_target/m1; the ratio r trades the bound ln(1/r)/eps against the
# smallest width covered (widths below r*u_max, i.e. about 1e-6, keep the
# pure-proxy tail but are never reached by practical injection counts). The
# component is only needed where a target support edge can sit, m1_source <
# sup(prior mmin) = 10; 12 leaves a margin. Values chosen by replicate scans
# (400 independent 1e5-draw sets): eps=0.3 removed every set whose largest
# single row carried more than 1% of the selection sum (pure proxy: up to 5.5%
# of sets, one row moving 48 log A by 3.5 nats) without lowering the
# selection ESS at the truth or the hyperprior median.
DEFAULT_POPULATION_PROXY_DEFENSIVE_FRACTION = 0.3
POPULATION_PROXY_DEFENSIVE_U_RATIO = 1.0e-6
POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX = 12.0

# population_proxy stores an exact draw density, so the redshift sampler (a
# linear-CDF inversion on ``redshift_sampling_grid`` points) must follow the
# model density closely: the importance-weight bias max|E[w_z] - 1| over the
# kappa prior is about 7e-7 at 4096 points (1.6e-7 at the default 8192) but
# 1.9e-4 at 256 and 5e-2 at 16.
POPULATION_PROXY_MIN_REDSHIFT_SAMPLING_GRID = 4096

# Smallest injection count recommended for population_proxy campaigns; the CLI
# uses it as the population_proxy default. See docs/phase3_recovery.md.
RECOMMENDED_POPULATION_PROXY_N_INJECTIONS = 100_000

_SELECTION_FIELDS = (
    "m1_detector",
    "q",
    "luminosity_distance",
    "ra",
    "dec",
    "chi_eff",
)
_OBSERVED_FIELDS = (
    "log_m1_detector",
    "q",
    "log_luminosity_distance",
    "chi_eff",
)
_V2_FIELDS = (
    "injection_draw",
    "injection_draw_hyperparameters",
    "injection_draw_defensive_fraction",
    "observation_model",
)


def _prior_bounds(spec: PriorSpec) -> tuple[float, float] | None:
    if spec.family in {"uniform", "log_uniform"}:
        return float(spec.low), float(spec.high)
    return None


# Upper edge of the Phase-3 mmax hyperprior; the noisy_observation PE prior box
# covers every hyperprior population's detector-frame mass range.
_HYPERPRIOR_MMAX_SUP = _prior_bounds(BASELINE_SYNTHETIC_PRIORS["mmax"])[1]


def population_proxy_support_violations(
    proxy_hyperparameters: Mapping[str, float],
    priors: Mapping[str, PriorSpec] = BASELINE_SYNTHETIC_PRIORS,
) -> tuple[str, ...]:
    """Reasons a baseline proxy draw fails to cover every population in ``priors``.

    The support of every population family in the model grammar is
    m1_source in [mmin, mmax], q in [max(q_floor, mmin/m1_source), 1],
    z in (0, zmax], chi_eff in [-1, 1] and the full sky. Every proxy value
    other than mmin/mmax gives a strictly positive density on that support
    (the defensive pairing component only adds density), so coverage of all
    trial populations reduces to ``mmin_proxy <= inf(prior mmin)`` and
    ``mmax_proxy >= sup(prior mmax)``, which requires bounded priors on both.
    An empty tuple means the proxy covers the hyperprior.
    """
    problems: list[str] = []
    for name, side in (("mmin", "low"), ("mmax", "high")):
        if name not in priors:
            problems.append(f"prior has no {name!r} entry; support cannot be verified")
            continue
        bounds = _prior_bounds(priors[name])
        if bounds is None:
            problems.append(
                f"prior on {name!r} is {priors[name].family!r} (unbounded); no fixed "
                "proxy can cover its support"
            )
            continue
        proxy_value = float(proxy_hyperparameters[name])
        if side == "low" and proxy_value > bounds[0]:
            problems.append(
                f"proxy mmin={proxy_value} exceeds the smallest prior mmin={bounds[0]}"
            )
        if side == "high" and proxy_value < bounds[1]:
            problems.append(
                f"proxy mmax={proxy_value} is below the largest prior mmax={bounds[1]}"
            )
    return tuple(problems)


def population_proxy_point_violations(
    proxy_hyperparameters: Mapping[str, float],
    hyperparameters: Mapping[str, float],
) -> tuple[str, ...]:
    """Reasons a proxy fails to cover the population at one fixed point.

    Same support argument as ``population_proxy_support_violations``: the
    population at ``hyperparameters`` is covered iff
    ``mmin_proxy <= mmin`` and ``mmax_proxy >= mmax``.
    """
    problems: list[str] = []
    for name in ("mmin", "mmax"):
        if name not in hyperparameters:
            problems.append(
                f"hyperparameters have no {name!r} entry; support cannot be verified"
            )
            continue
        value = float(hyperparameters[name])
        if not math.isfinite(value):
            problems.append(f"hyperparameter {name}={value} is not finite")
            continue
        proxy_value = float(proxy_hyperparameters[name])
        if name == "mmin" and value < proxy_value:
            problems.append(f"mmin={value} is below the proxy mmin={proxy_value}")
        if name == "mmax" and value > proxy_value:
            problems.append(f"mmax={value} exceeds the proxy mmax={proxy_value}")
    return tuple(problems)


def require_population_proxy_coverage(
    selection: SelectionCatalog,
    *,
    priors: Mapping[str, PriorSpec] | None = None,
    hyperparameters: Mapping[str, float] | None = None,
    context: str = "",
) -> None:
    """Fail loudly if a population_proxy selection misses an evaluated population.

    Every consumer that evaluates the selection integral must call this with
    exactly what it evaluates: the hyperprior it samples (``priors``) and/or
    the fixed hyperparameter point it holds (``hyperparameters``). Outside the
    proxy support the raw-draw estimator silently drops population mass, so a
    violation is an error, never a warning. Selections whose metadata do not
    declare a population-proxy draw (the uniform box, real injections) are
    accepted unchanged.
    """
    metadata = getattr(selection, "metadata", None) or {}
    if metadata.get("injection_draw") != INJECTION_DRAW_POPULATION_PROXY:
        return
    if metadata.get("injection_draw_hyperparameters") is None:
        raise ValueError(
            "population_proxy selection does not record its proxy hyperparameters"
        )
    proxy = dict(metadata["injection_draw_hyperparameters"])
    if priors is None and hyperparameters is None:
        raise ValueError(
            "population_proxy coverage check needs the priors or the fixed "
            "hyperparameters that will be evaluated"
        )
    problems: list[str] = []
    if priors is not None:
        problems.extend(population_proxy_support_violations(proxy, priors))
    if hyperparameters is not None:
        problems.extend(population_proxy_point_violations(proxy, hyperparameters))
    if problems:
        where = f" ({context})" if context else ""
        raise ValueError(
            "population_proxy selection does not cover the evaluated "
            f"populations{where}: " + "; ".join(problems)
        )


def _validated_proxy_hyperparameters(values: Mapping[str, float]) -> dict[str, float]:
    names = tuple(DEFAULT_BASELINE_HYPERPARAMETERS)
    missing = sorted(set(names) - set(values))
    unknown = sorted(set(values) - set(names))
    if missing or unknown:
        raise ValueError(
            "injection_draw_hyperparameters must name exactly the baseline "
            f"hyperparameters; missing={missing}, unknown={unknown}"
        )
    hp = {name: float(values[name]) for name in names}
    bad = [name for name, value in hp.items() if not math.isfinite(value)]
    if bad:
        raise ValueError(f"injection_draw_hyperparameters must be finite; bad={bad}")
    if not hp["mmax"] > hp["mmin"] > 0.0:
        raise ValueError("proxy mass support requires 0 < mmin < mmax")
    if not 0.0 <= hp["peak_fraction"] <= 1.0:
        raise ValueError("proxy peak_fraction must lie in [0, 1]")
    if not hp["mmin"] <= hp["peak_mu"] <= hp["mmax"]:
        raise ValueError("proxy peak_mu must lie inside [mmin, mmax]")
    if hp["peak_sigma"] <= 0.0 or hp["chi_sigma"] <= 0.0:
        raise ValueError("proxy peak_sigma and chi_sigma must be positive")
    if not -1.0 <= hp["chi_mu"] <= 1.0:
        raise ValueError("proxy chi_mu must lie inside [-1, 1]")
    return hp


@dataclass(frozen=True)
class SyntheticSurveyConfig:
    """Closed Phase-3 survey settings.

    ``observation_model`` and ``injection_draw`` are described in the module
    docstring. ``injection_draw_hyperparameters`` and
    ``injection_draw_defensive_fraction`` are only accepted for
    ``population_proxy``; ``None`` there resolves to
    ``DEFAULT_POPULATION_PROXY_HYPERPARAMETERS`` and
    ``DEFAULT_POPULATION_PROXY_DEFENSIVE_FRACTION`` so the resolved draw is
    always recorded in manifests. The resolved proxy is stored read-only. A
    proxy must cover the Phase-3 hyperprior
    (``population_proxy_support_violations``) and needs
    ``redshift_sampling_grid >= POPULATION_PROXY_MIN_REDSHIFT_SAMPLING_GRID``,
    otherwise it is rejected.

    Under ``noisy_observation`` the four ``pe_*_sigma`` fields are the
    measurement-noise scales (fractional ones act on ln m1_detector and
    ln d_L); under ``truth_centered`` they are the widths of the truth-centred
    PE draws, as before.
    """

    n_events: int = 48
    posterior_samples_per_event: int = 256
    n_injections: int = 20_000
    observing_time_yr: float = 1.0
    reference_chirp_mass: float = 20.0
    reference_horizon_mpc: float = 5_000.0
    m1_detector_min: float = 2.0
    m1_detector_max: float = 400.0
    pe_m1_fractional_sigma: float = 0.08
    pe_q_sigma: float = 0.06
    pe_d_l_fractional_sigma: float = 0.12
    pe_chi_eff_sigma: float = 0.12
    population_batch_size: int = 2048
    redshift_sampling_grid: int = 8192
    injection_draw: str = INJECTION_DRAW_UNIFORM_DETECTOR_BOX
    injection_draw_hyperparameters: Mapping[str, float] | None = None
    injection_draw_defensive_fraction: float | None = None
    observation_model: str = OBSERVATION_MODEL_TRUTH_CENTERED

    def __post_init__(self) -> None:
        for name in (
            "n_events",
            "posterior_samples_per_event",
            "n_injections",
            "population_batch_size",
            "redshift_sampling_grid",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.observing_time_yr <= 0:
            raise ValueError("observing_time_yr must be positive")
        if self.reference_chirp_mass <= 0 or self.reference_horizon_mpc <= 0:
            raise ValueError("detection reference scales must be positive")
        if not self.m1_detector_max > self.m1_detector_min > 0:
            raise ValueError("detector-frame mass prior bounds are invalid")
        for name in (
            "pe_m1_fractional_sigma",
            "pe_q_sigma",
            "pe_d_l_fractional_sigma",
            "pe_chi_eff_sigma",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        observation = str(self.observation_model)
        if observation not in OBSERVATION_MODELS:
            raise ValueError(
                f"observation_model must be one of {OBSERVATION_MODELS}; "
                f"got {observation!r}"
            )
        object.__setattr__(self, "observation_model", observation)

        draw = str(self.injection_draw)
        if draw not in INJECTION_DRAWS:
            raise ValueError(
                f"injection_draw must be one of {INJECTION_DRAWS}; got {draw!r}"
            )
        object.__setattr__(self, "injection_draw", draw)
        if draw == INJECTION_DRAW_UNIFORM_DETECTOR_BOX:
            for name in (
                "injection_draw_hyperparameters",
                "injection_draw_defensive_fraction",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name} is only used by "
                        f"injection_draw={INJECTION_DRAW_POPULATION_PROXY!r}"
                    )
            return

        if int(self.redshift_sampling_grid) < POPULATION_PROXY_MIN_REDSHIFT_SAMPLING_GRID:
            raise ValueError(
                "population_proxy stores an exact draw density, so its redshift "
                "sampler needs redshift_sampling_grid >= "
                f"{POPULATION_PROXY_MIN_REDSHIFT_SAMPLING_GRID}; got "
                f"{int(self.redshift_sampling_grid)}"
            )
        proxy = _validated_proxy_hyperparameters(
            DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
            if self.injection_draw_hyperparameters is None
            else self.injection_draw_hyperparameters
        )
        violations = population_proxy_support_violations(proxy)
        if violations:
            raise ValueError(
                "population_proxy injections must cover every Phase-3 hyperprior "
                "population: " + "; ".join(violations)
            )
        fraction = (
            DEFAULT_POPULATION_PROXY_DEFENSIVE_FRACTION
            if self.injection_draw_defensive_fraction is None
            else float(self.injection_draw_defensive_fraction)
        )
        if not (math.isfinite(fraction) and 0.0 <= fraction < 1.0):
            raise ValueError(
                "injection_draw_defensive_fraction must lie in [0, 1); "
                f"got {fraction}"
            )
        object.__setattr__(
            self, "injection_draw_hyperparameters", FrozenHyperparameters(proxy)
        )
        object.__setattr__(self, "injection_draw_defensive_fraction", fraction)

    @property
    def uses_v2_options(self) -> bool:
        """True when a survey-v2 option (injection draw or observation) is set."""
        return (
            self.injection_draw != INJECTION_DRAW_UNIFORM_DETECTOR_BOX
            or self.observation_model != OBSERVATION_MODEL_TRUTH_CENTERED
        )

    def to_dict(self) -> dict[str, object]:
        """JSON manifest payload.

        Survey-v2 fields at their legacy defaults (``uniform_detector_box``,
        ``truth_centered``) are omitted, so manifests written before the options
        existed stay byte-identical; ``population_proxy`` records the resolved
        proxy and defensive fraction.
        """
        payload: dict[str, object] = {}
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            if (
                name in _V2_FIELDS[:3]
                and self.injection_draw == INJECTION_DRAW_UNIFORM_DETECTOR_BOX
            ):
                continue
            if name == "observation_model" and value == OBSERVATION_MODEL_TRUTH_CENTERED:
                continue
            if name == "injection_draw_hyperparameters":
                value = {key: float(v) for key, v in value.items()}
            payload[name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SyntheticSurveyConfig:
        values = dict(payload)
        if (
            values.get("injection_draw") == INJECTION_DRAW_POPULATION_PROXY
            and "injection_draw_defensive_fraction" not in values
        ):
            raise ValueError(
                "population_proxy survey payload has no "
                "injection_draw_defensive_fraction: it predates the defensive "
                "pairing component, so resolving it now would change the "
                "injection draw it describes; set the fraction explicitly "
                "(0.0 reproduces the pure proxy)"
            )
        return cls(**values)

    def dataset_identity_suffix(self) -> str:
        """Suffix that separates dataset identities of survey-v2 configurations.

        Legacy configurations return ``""`` so existing identities (and the
        manifests keyed by them) are unchanged; survey-v2 configurations return
        ``":survey-<12 hex>"``, a digest of ``to_dict()``.
        """
        if not self.uses_v2_options:
            return ""
        digest = hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f":survey-{digest[:12]}"


@dataclass(frozen=True)
class SyntheticDataset:
    posterior: PosteriorCatalog
    selection: SelectionCatalog
    event_truths: Mapping[str, np.ndarray]
    truth_hyperparameters: Mapping[str, float]
    config: SyntheticSurveyConfig
    seed: int
    event_observations: Mapping[str, np.ndarray] | None = None


def _sample_powerlaw(rng, n: int, *, alpha: float, low: float, high: float):
    exponent = 1.0 - float(alpha)
    u = rng.random(n)
    if abs(exponent) < 1e-10:
        return low * np.power(high / low, u)
    return np.power(
        np.power(low, exponent)
        + u * (np.power(high, exponent) - np.power(low, exponent)),
        1.0 / exponent,
    )


def _sample_primary_mass(rng, n: int, hp: Mapping[str, float]):
    peak = rng.random(n) < float(hp["peak_fraction"])
    out = np.empty(n, dtype=float)

    n_power = int((~peak).sum())
    if n_power:
        out[~peak] = _sample_powerlaw(
            rng,
            n_power,
            alpha=float(hp["alpha"]),
            low=float(hp["mmin"]),
            high=float(hp["mmax"]),
        )

    n_peak = int(peak.sum())
    if n_peak:
        mu = float(hp["peak_mu"])
        sigma = float(hp["peak_sigma"])
        a = (float(hp["mmin"]) - mu) / sigma
        b = (float(hp["mmax"]) - mu) / sigma
        out[peak] = truncnorm.rvs(
            a,
            b,
            loc=mu,
            scale=sigma,
            size=n_peak,
            random_state=rng,
        )
    return out


def _sample_q(rng, m1_source, hp: Mapping[str, float], q_floor: float):
    qmin = np.maximum(q_floor, float(hp["mmin"]) / m1_source)
    exponent = float(hp["beta_q"]) + 1.0
    u = rng.random(m1_source.size)
    if abs(exponent) < 1e-10:
        return qmin * np.power(1.0 / qmin, u)
    return np.power(
        np.power(qmin, exponent)
        + u * (1.0 - np.power(qmin, exponent)),
        1.0 / exponent,
    )


def _sample_redshift(rng, n: int, hp, model, grid_size: int):
    z_grid = np.linspace(1e-7, model.zmax, int(grid_size))
    shape = np.array(model.cosmology.dVc_dz(z_grid), dtype=float, copy=True)
    shape = shape * np.power(1.0 + z_grid, float(hp["kappa"]) - 1.0)

    dz = np.diff(z_grid)
    cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (shape[1:] + shape[:-1]) * dz))
    )
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("invalid synthetic redshift density")
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, z_grid)


def _sample_chi_eff(rng, n: int, hp):
    mu = float(hp["chi_mu"])
    sigma = float(hp["chi_sigma"])
    a = (-1.0 - mu) / sigma
    b = (1.0 - mu) / sigma
    return truncnorm.rvs(
        a,
        b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng,
    )


def _draw_population(rng, n: int, hp, model, config):
    m1_source = _sample_primary_mass(rng, n, hp)
    q = _sample_q(rng, m1_source, hp, model.q_floor)
    z = _sample_redshift(
        rng,
        n,
        hp,
        model,
        config.redshift_sampling_grid,
    )
    chi_eff = _sample_chi_eff(rng, n, hp)
    ra = rng.uniform(0.0, 2.0 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n))

    d_l = np.asarray(model.cosmology.dL_of_z(z), dtype=float)
    m1_detector = m1_source * (1.0 + z)

    return {
        "m1_source": m1_source,
        "q": q,
        "z": z,
        "chi_eff": chi_eff,
        "ra": ra,
        "dec": dec,
        "luminosity_distance": d_l,
        "m1_detector": m1_detector,
    }


def detection_mask(samples: Mapping[str, np.ndarray], config: SyntheticSurveyConfig):
    """Deterministic synthetic detection cut based on a chirp-mass-scaled reach.

    Applied to the true parameters under ``truth_centered`` (legacy). Under
    ``noisy_observation`` the same reach is evaluated on the observed data by
    ``observed_detection_mask``.
    """
    m1 = np.asarray(samples["m1_detector"], dtype=float)
    q = np.asarray(samples["q"], dtype=float)
    d_l = np.asarray(samples["luminosity_distance"], dtype=float)

    chirp_mass = m1 * np.power(q, 3.0 / 5.0) / np.power(1.0 + q, 1.0 / 5.0)
    reach = config.reference_horizon_mpc * np.power(
        chirp_mass / config.reference_chirp_mass,
        5.0 / 6.0,
    )
    return d_l <= reach


def _observation_sigmas(config: SyntheticSurveyConfig) -> tuple[float, float, float, float]:
    return (
        float(config.pe_m1_fractional_sigma),
        float(config.pe_q_sigma),
        float(config.pe_d_l_fractional_sigma),
        float(config.pe_chi_eff_sigma),
    )


def _observe(rng, systems: Mapping[str, np.ndarray], config: SyntheticSurveyConfig):
    """One measurement-noise realisation per system (``noisy_observation``).

    Returns the observed data ``ln m1_detector + s_m e1``, ``q + s_q e2``,
    ``ln d_L + s_d e3`` and ``chi_eff + s_chi e4`` with independent standard
    normal ``e`` (one ``(4, n)`` draw). The observed q and chi_eff may leave
    their physical ranges; they are data, not parameters.
    """
    m1 = np.asarray(systems["m1_detector"], dtype=float)
    noise = rng.standard_normal((4, m1.size))
    s_m, s_q, s_d, s_c = _observation_sigmas(config)
    with np.errstate(divide="ignore"):
        log_m1 = np.log(m1)
        log_d_l = np.log(np.asarray(systems["luminosity_distance"], dtype=float))
    return {
        "log_m1_detector": log_m1 + s_m * noise[0],
        "q": np.asarray(systems["q"], dtype=float) + s_q * noise[1],
        "log_luminosity_distance": log_d_l + s_d * noise[2],
        "chi_eff": np.asarray(systems["chi_eff"], dtype=float) + s_c * noise[3],
    }


def observed_detection_mask(
    observed: Mapping[str, np.ndarray],
    config: SyntheticSurveyConfig,
    bounds: Mapping[str, float],
):
    """``noisy_observation`` detection: the chirp-mass-scaled reach on observed data.

    Detected iff ``ln d_obs <= ln R0 + 5/6 (ln Mc_obs - ln Mc0)``, with the
    observed chirp mass built from ``exp(x_m)`` and the observed q clipped to
    the PE prior range ``[bounds["q_min"], bounds["q_max"]]`` (a fixed function
    of the data, so the rule is defined for every observation). This is the
    ``detection_mask`` rule applied to observed instead of true values; events
    and injections use this same function.
    """
    q = np.clip(
        np.asarray(observed["q"], dtype=float),
        float(bounds["q_min"]),
        float(bounds["q_max"]),
    )
    log_chirp_mass = (
        np.asarray(observed["log_m1_detector"], dtype=float)
        + 0.6 * np.log(q)
        - 0.2 * np.log1p(q)
    )
    log_reach = math.log(config.reference_horizon_mpc) + (5.0 / 6.0) * (
        log_chirp_mass - math.log(config.reference_chirp_mass)
    )
    return np.asarray(observed["log_luminosity_distance"], dtype=float) <= log_reach


def _detected_events(
    rng,
    draw_batch: Callable[[object, int], Mapping[str, np.ndarray]],
    model,
    config: SyntheticSurveyConfig,
    hp: Mapping[str, float],
    *,
    what: str = "synthetic events",
):
    """Draw population batches until ``config.n_events`` systems are detected.

    ``truth_centered`` applies ``detection_mask`` to the true parameters and
    consumes the generator exactly as before. ``noisy_observation`` gives every
    drawn system one noise realisation and detects on the observed data.
    Returns ``(truths, observations)``; observations are ``None`` for
    ``truth_centered``.
    """
    noisy = config.observation_model == OBSERVATION_MODEL_NOISY
    bounds = _detector_prior_bounds(model, config, hp) if noisy else None
    truth_pieces: dict[str, list[np.ndarray]] = {}
    observed_pieces: dict[str, list[np.ndarray]] = {}
    n_have = 0
    attempts = 0

    while n_have < config.n_events:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError(f"failed to draw enough detected {what}")
        draw = draw_batch(
            rng,
            max(config.population_batch_size, 2 * (config.n_events - n_have)),
        )
        if noisy:
            observed = _observe(rng, draw, config)
            keep = observed_detection_mask(observed, config, bounds)
        else:
            observed = None
            keep = detection_mask(draw, config)
        if not np.any(keep):
            continue

        for name, values in draw.items():
            truth_pieces.setdefault(name, []).append(np.asarray(values)[keep])
        if observed is not None:
            for name, values in observed.items():
                observed_pieces.setdefault(name, []).append(values[keep])
        n_have += int(keep.sum())

    truths = {
        name: np.concatenate(chunks)[: config.n_events]
        for name, chunks in truth_pieces.items()
    }
    if not noisy:
        return truths, None
    observations = {
        name: np.concatenate(chunks)[: config.n_events]
        for name, chunks in observed_pieces.items()
    }
    return truths, observations


def _detected_population_truths(rng, hp, model, config):
    """Detected event truths of the baseline population (legacy helper)."""
    truths, _ = _detected_events(
        rng,
        lambda generator, n: _draw_population(generator, n, hp, model, config),
        model,
        config,
        hp,
    )
    return truths


def _truncated_normal_draw(rng, mean, sigma, low, high, size):
    a = (low - mean) / sigma
    b = (high - mean) / sigma
    return truncnorm.rvs(
        a,
        b,
        loc=mean,
        scale=sigma,
        size=size,
        random_state=rng,
    )


def _detector_prior_bounds(model, config, hp):
    """PE prior (and uniform-injection) box in the gwcat detector basis.

    ``truth_centered`` keeps the legacy upper mass edge
    ``max(m1_detector_max, 1.1 mmax_truth (1+zmax))``. ``noisy_observation``
    replaces the truth's mmax by ``max(mmax_truth, sup(prior mmax))`` so the
    box, and hence every event's PE prior, contains the detector-frame support
    of every Phase-3 hyperprior population and does not depend on the truth.
    """
    d_l_max = float(model.cosmology.dL_of_z(model.zmax))
    mmax = float(hp["mmax"])
    if config.observation_model == OBSERVATION_MODEL_NOISY:
        mmax = max(mmax, _HYPERPRIOR_MMAX_SUP)
    population_mass_max = mmax * (1.0 + model.zmax)
    m1_max = max(config.m1_detector_max, 1.1 * population_mass_max)
    return {
        "m1_min": config.m1_detector_min,
        "m1_max": m1_max,
        "q_min": model.q_floor,
        "q_max": 1.0,
        "d_l_min": 0.0,
        "d_l_max": d_l_max,
        "chi_min": -1.0,
        "chi_max": 1.0,
    }


def _uniform_detector_log_density(bounds):
    return -(
        np.log(bounds["m1_max"] - bounds["m1_min"])
        + np.log(bounds["q_max"] - bounds["q_min"])
        + np.log(bounds["d_l_max"] - bounds["d_l_min"])
        + np.log(4.0 * np.pi)
        + np.log(bounds["chi_max"] - bounds["chi_min"])
    )


def _noisy_posterior_draws(rng, observed_i, config, bounds, n_sample):
    """Exact posterior draws for one event under the uniform detector-box prior.

    With ``L(d | theta)`` Gaussian in (ln m1, q, ln d_L, chi_eff) and a prior
    uniform in (m1, q, d_L, chi_eff) on the box, the posterior factorizes:
    ``ln m1 ~ N(x_m + s_m^2, s_m)`` (the ``+ s_m^2`` is the Jacobian of the
    uniform-in-m1 prior) truncated to ``[ln m1_min, ln m1_max]``,
    ``q ~ N(x_q, s_q)`` on ``[q_min, q_max]``, ``ln d_L ~ N(x_d + s_d^2, s_d)``
    on ``(-inf, ln d_L_max]`` and ``chi_eff ~ N(x_chi, s_chi)`` on [-1, 1].
    Values are clipped onto the box only to absorb exp/log round-off.
    """
    s_m, s_q, s_d, s_c = _observation_sigmas(config)
    x_m = float(observed_i["log_m1_detector"])
    x_q = float(observed_i["q"])
    x_d = float(observed_i["log_luminosity_distance"])
    x_c = float(observed_i["chi_eff"])

    log_m1 = _truncated_normal_draw(
        rng,
        x_m + s_m**2,
        s_m,
        math.log(bounds["m1_min"]),
        math.log(bounds["m1_max"]),
        n_sample,
    )
    q = _truncated_normal_draw(
        rng,
        x_q,
        s_q,
        bounds["q_min"],
        bounds["q_max"],
        n_sample,
    )
    log_d_l = _truncated_normal_draw(
        rng,
        x_d + s_d**2,
        s_d,
        -np.inf,
        math.log(bounds["d_l_max"]),
        n_sample,
    )
    chi = _truncated_normal_draw(
        rng,
        x_c,
        s_c,
        bounds["chi_min"],
        bounds["chi_max"],
        n_sample,
    )
    return {
        "m1_detector": np.clip(np.exp(log_m1), bounds["m1_min"], bounds["m1_max"]),
        "q": q,
        "luminosity_distance": np.minimum(np.exp(log_d_l), bounds["d_l_max"]),
        "chi_eff": chi,
    }


def _make_posterior_catalog(rng, truths, model, config, hp, observations=None):
    noisy = config.observation_model == OBSERVATION_MODEL_NOISY
    if noisy and observations is None:
        raise ValueError(
            "noisy_observation PE is drawn given each event's observed data; "
            "pass the event observations"
        )
    if not noisy and observations is not None:
        raise ValueError(
            "event observations are only used by observation_model="
            f"{OBSERVATION_MODEL_NOISY!r}"
        )
    n_event = config.n_events
    n_sample = config.posterior_samples_per_event
    n_total = n_event * n_sample
    bounds = _detector_prior_bounds(model, config, hp)

    samples = {
        "m1_detector": np.empty(n_total),
        "q": np.empty(n_total),
        "luminosity_distance": np.empty(n_total),
        "ra": np.empty(n_total),
        "dec": np.empty(n_total),
        "chi_eff": np.empty(n_total),
    }

    for i in range(n_event):
        sl = slice(i * n_sample, (i + 1) * n_sample)
        if noisy:
            draws = _noisy_posterior_draws(
                rng,
                {name: observations[name][i] for name in _OBSERVED_FIELDS},
                config,
                bounds,
                n_sample,
            )
            for name, values in draws.items():
                samples[name][sl] = values
        else:
            m1 = truths["m1_detector"][i]
            q = truths["q"][i]
            d_l = truths["luminosity_distance"][i]
            chi = truths["chi_eff"][i]

            samples["m1_detector"][sl] = _truncated_normal_draw(
                rng,
                m1,
                max(config.pe_m1_fractional_sigma * m1, 0.1),
                bounds["m1_min"],
                bounds["m1_max"],
                n_sample,
            )
            samples["q"][sl] = _truncated_normal_draw(
                rng,
                q,
                config.pe_q_sigma,
                bounds["q_min"],
                bounds["q_max"],
                n_sample,
            )
            samples["luminosity_distance"][sl] = _truncated_normal_draw(
                rng,
                d_l,
                max(config.pe_d_l_fractional_sigma * d_l, 1.0),
                max(bounds["d_l_min"], 1e-6),
                bounds["d_l_max"],
                n_sample,
            )
            samples["chi_eff"][sl] = _truncated_normal_draw(
                rng,
                chi,
                config.pe_chi_eff_sigma,
                bounds["chi_min"],
                bounds["chi_max"],
                n_sample,
            )

        # The sky likelihood is taken to be delta-like for this validation mock.
        # Because both source population and PE prior are isotropic in dOmega,
        # the sky ratio is still represented correctly.
        samples["ra"][sl] = truths["ra"][i]
        samples["dec"][sl] = truths["dec"][i]

    log_ref = np.full(
        n_total,
        _uniform_detector_log_density(bounds),
        dtype=float,
    )
    offsets = np.arange(n_event + 1, dtype=np.int64) * n_sample
    names = tuple(f"SYNTH_{i:04d}" for i in range(n_event))
    metadata: dict[str, object] = {
        "fixture": "phase3-synthetic-pe",
        "reference_prior": "uniform detector basis",
    }
    if noisy:
        s_m, s_q, s_d, s_c = _observation_sigmas(config)
        metadata["observation_model"] = OBSERVATION_MODEL_NOISY
        metadata["observation_noise_sigma"] = {
            "log_m1_detector": s_m,
            "q": s_q,
            "log_luminosity_distance": s_d,
            "chi_eff": s_c,
        }

    return PosteriorCatalog(
        event_names=names,
        offsets=offsets,
        samples=samples,
        log_ref_density=log_ref,
        basis=gwcat_v2_basis_for_spin("chieff"),
        metadata=metadata,
    )


def pe_truth_quantiles(posterior: PosteriorCatalog, truths: Mapping[str, np.ndarray], name: str):
    """Quantile of each event's true ``name`` within its own PE samples.

    For PE drawn from the posterior given noisy data, the quantile of the truth
    is uniform across events for any coordinate that detection does not
    depend on and whose prior edges are far from the posterior (chi_eff here).
    Zero-noise PE centred on the truth puts it near 0.5 in every event.
    """
    values = np.asarray(posterior.samples[name], dtype=float)
    truth = np.asarray(truths[name], dtype=float)
    if truth.shape != (posterior.n_events,):
        raise ValueError("truths must hold one value per posterior event")
    out = np.empty(posterior.n_events, dtype=float)
    for i in range(posterior.n_events):
        start, stop = posterior.offsets[i], posterior.offsets[i + 1]
        out[i] = float(np.mean(values[start:stop] < truth[i]))
    return out


def _baseline_density_model(model) -> GwcatChiEffBBHModel:
    """The baseline density with the caller's model context.

    ``_draw_population`` always samples the baseline (power law + peak) family
    using only ``model``'s cosmology, zmax and q_floor, so the draw density must
    be the baseline class with that context, whatever ``model`` itself is.
    """
    if type(model) is GwcatChiEffBBHModel:
        return model
    try:
        return GwcatChiEffBBHModel(
            cosmology=model.cosmology,
            zmax=float(model.zmax),
            q_floor=float(model.q_floor),
            redshift_quadrature_order=int(model.redshift_quadrature_order),
        )
    except AttributeError as exc:
        raise TypeError(
            "population_proxy needs a model exposing cosmology, zmax, q_floor "
            "and redshift_quadrature_order"
        ) from exc


def _defensive_active(m1_source) -> np.ndarray:
    return np.asarray(m1_source, dtype=float) < POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX


def _defensive_u_max(m1_source, proxy: Mapping[str, float], q_floor: float):
    m1 = np.asarray(m1_source, dtype=float)
    return 1.0 - np.maximum(float(q_floor), float(proxy["mmin"]) / m1)


def _defensive_log_q_density(q, m1_source, proxy: Mapping[str, float], q_floor: float):
    """log g(q | m1): u = 1 - q log-uniform on [r u_max(m1), u_max(m1)]."""
    q = np.asarray(q, dtype=float)
    u = 1.0 - q
    u_max = _defensive_u_max(m1_source, proxy, q_floor)
    r = POPULATION_PROXY_DEFENSIVE_U_RATIO
    inside = (
        _defensive_active(m1_source)
        & (u_max > 0.0)
        & (u >= r * u_max)
        & (u <= u_max)
    )
    safe_u = np.where(inside, u, 1.0)
    return np.where(inside, -np.log(safe_u) - math.log(math.log(1.0 / r)), -np.inf)


def _apply_defensive_pairing(rng, draw, proxy: Mapping[str, float], q_floor: float, fraction: float):
    """Replace q by the defensive component for a Bernoulli(fraction) subset.

    Consumes two ``rng.random(n)`` blocks (selection mask, then the log-uniform
    variate). Only rows with ``m1_source < POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX``
    are eligible. ``fraction == 0`` consumes nothing and returns ``draw``.
    """
    if fraction == 0.0:
        return draw
    m1 = np.asarray(draw["m1_source"], dtype=float)
    n = m1.size
    use = (rng.random(n) < fraction) & _defensive_active(m1)
    v = rng.random(n)
    u = _defensive_u_max(m1, proxy, q_floor) * np.power(
        POPULATION_PROXY_DEFENSIVE_U_RATIO, v
    )
    out = dict(draw)
    out["q"] = np.where(use, 1.0 - u, np.asarray(draw["q"], dtype=float))
    return out


def population_proxy_log_draw_density(model, samples, config: SyntheticSurveyConfig):
    """Exact log draw density of ``config``'s population_proxy injections.

    The density is evaluated with the baseline class carrying ``model``'s
    cosmology, zmax, q_floor and quadrature order (``_draw_population`` samples
    the baseline family whatever ``model`` is). It is the baseline's
    normalized detector-basis log density at the proxy hyperparameters, with
    the pairing factor ``p_q(q | m1)`` replaced by the defensive mixture
    ``(1 - eps) p_q + eps g`` wherever ``m1_source`` is below
    ``POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX``; ``eps = 0`` returns exactly
    ``baseline(samples, proxy)``. Evaluated in 64-bit precision. Zero-support
    rows are returned as -inf; callers decide whether that is an error.
    """
    if config.injection_draw != INJECTION_DRAW_POPULATION_PROXY:
        raise ValueError(
            "population_proxy_log_draw_density needs a population_proxy config"
        )
    try:
        from jax import enable_x64
    except ImportError:  # pragma: no cover - JAX releases before the config API
        from jax.experimental import enable_x64
    import jax.numpy as jnp

    baseline = _baseline_density_model(model)
    proxy = {name: float(value) for name, value in config.injection_draw_hyperparameters.items()}
    fraction = float(config.injection_draw_defensive_fraction)
    columns = {
        name: np.asarray(samples[name], dtype=float) for name in _SELECTION_FIELDS
    }
    with enable_x64(True):
        log_model = np.array(baseline(columns, proxy), dtype=np.float64)
        if fraction > 0.0:
            # Same operations as the model's own detector -> source transform.
            z = baseline.cosmology.z_of_dL(jnp.asarray(columns["luminosity_distance"]))
            m1_source_jax = jnp.asarray(columns["m1_detector"]) / (1.0 + z)
            log_q_proxy = np.array(
                mass_ratio_logpdf(
                    jnp.asarray(columns["q"]),
                    m1_source_jax,
                    beta=proxy["beta_q"],
                    mmin=proxy["mmin"],
                    q_floor=baseline.q_floor,
                ),
                dtype=np.float64,
            )
            m1_source = np.array(m1_source_jax, dtype=np.float64)
    if log_model.shape != columns["m1_detector"].shape:
        raise RuntimeError(
            f"proxy draw density has shape {log_model.shape}; expected "
            f"{columns['m1_detector'].shape}"
        )
    out = log_model
    if fraction > 0.0:
        active = _defensive_active(m1_source) & np.isfinite(log_model)
        log_g = _defensive_log_q_density(columns["q"], m1_source, proxy, baseline.q_floor)
        safe_log_q = np.where(active, log_q_proxy, 0.0)
        mixture = np.logaddexp(
            math.log1p(-fraction) + safe_log_q,
            math.log(fraction) + np.where(active, log_g, -np.inf),
        )
        out = np.where(active, log_model - safe_log_q + mixture, log_model)
    if np.isnan(out).any() or np.isposinf(out).any():
        raise RuntimeError("proxy draw density contains NaN/+inf values")
    return out


def _selection_detection(rng, samples, model, config, hp):
    """Detection mask of freshly drawn injections for the configured observation."""
    if config.observation_model == OBSERVATION_MODEL_NOISY:
        bounds = _detector_prior_bounds(model, config, hp)
        return observed_detection_mask(_observe(rng, samples, config), config, bounds)
    return detection_mask(samples, config)


def _selection_observation_metadata(config) -> dict[str, object]:
    if config.observation_model != OBSERVATION_MODEL_NOISY:
        return {}
    s_m, s_q, s_d, s_c = _observation_sigmas(config)
    return {
        "observation_model": OBSERVATION_MODEL_NOISY,
        "observation_noise_sigma": {
            "log_m1_detector": s_m,
            "q": s_q,
            "log_luminosity_distance": s_d,
            "chi_eff": s_c,
        },
    }


def _campaign_metadata(config, **extra) -> dict[str, object]:
    metadata: dict[str, object] = {"detection_rule": "chirp_mass_scaled_reach"}
    if config.observation_model == OBSERVATION_MODEL_NOISY:
        metadata["detection_applied_to"] = "observed_data"
    metadata.update(extra)
    return metadata


def _make_population_proxy_selection_catalog(rng, model, config, hp):
    baseline = _baseline_density_model(model)
    proxy = {name: float(value) for name, value in config.injection_draw_hyperparameters.items()}
    fraction = float(config.injection_draw_defensive_fraction)
    n = int(config.n_injections)
    draw = _draw_population(rng, n, proxy, baseline, config)
    draw = _apply_defensive_pairing(rng, draw, proxy, baseline.q_floor, fraction)
    samples = {name: np.asarray(draw[name], dtype=float) for name in _SELECTION_FIELDS}
    keep = _selection_detection(rng, samples, baseline, config, hp)
    if not np.any(keep):
        raise RuntimeError("synthetic selection campaign produced no detections")

    retained = {name: values[keep] for name, values in samples.items()}
    log_draw = population_proxy_log_draw_density(baseline, retained, config)
    outside = ~np.isfinite(log_draw)
    if outside.any():
        # A proxy draw outside the proxy's own support can only come from a
        # floating-point round trip at an exact support edge. Never floor it.
        raise RuntimeError(
            f"{int(outside.sum())} population_proxy injection(s) have zero proxy "
            "density after the detector-frame round trip; first retained rows="
            f"{np.flatnonzero(outside)[:8].tolist()}"
        )
    n_detected = int(keep.sum())

    return SelectionCatalog(
        samples=retained,
        log_draw_density=log_draw,
        campaign_id=np.asarray(["SYNTH"] * n_detected),
        campaigns=(
            Campaign(
                "SYNTH",
                n_draw=n,
                observing_time_yr=config.observing_time_yr,
                metadata=_campaign_metadata(
                    config, injection_draw=INJECTION_DRAW_POPULATION_PROXY
                ),
            ),
        ),
        basis=gwcat_v2_basis_for_spin("chieff"),
        mode=SelectionMode.RAW_DRAW,
        metadata={
            "fixture": "phase3-synthetic-selection",
            "n_detected": n_detected,
            "injection_draw": INJECTION_DRAW_POPULATION_PROXY,
            "injection_draw_hyperparameters": proxy,
            "injection_draw_defensive_fraction": fraction,
            "defensive_pairing": {
                "u_ratio": POPULATION_PROXY_DEFENSIVE_U_RATIO,
                "m1_source_max": POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX,
            },
            "draw_density_model": baseline.to_config(),
            **_selection_observation_metadata(config),
        },
    )


def _make_selection_catalog(rng, model, config, hp):
    if config.injection_draw == INJECTION_DRAW_POPULATION_PROXY:
        return _make_population_proxy_selection_catalog(rng, model, config, hp)

    bounds = _detector_prior_bounds(model, config, hp)
    n = config.n_injections

    samples = {
        "m1_detector": rng.uniform(bounds["m1_min"], bounds["m1_max"], n),
        "q": rng.uniform(bounds["q_min"], bounds["q_max"], n),
        "luminosity_distance": rng.uniform(
            bounds["d_l_min"],
            bounds["d_l_max"],
            n,
        ),
        "ra": rng.uniform(0.0, 2.0 * np.pi, n),
        "dec": np.arcsin(rng.uniform(-1.0, 1.0, n)),
        "chi_eff": rng.uniform(bounds["chi_min"], bounds["chi_max"], n),
    }
    keep = _selection_detection(rng, samples, model, config, hp)
    if not np.any(keep):
        raise RuntimeError("synthetic selection campaign produced no detections")

    retained = {name: values[keep] for name, values in samples.items()}
    log_draw = np.full(
        int(keep.sum()),
        _uniform_detector_log_density(bounds),
        dtype=float,
    )

    return SelectionCatalog(
        samples=retained,
        log_draw_density=log_draw,
        campaign_id=np.asarray(["SYNTH"] * int(keep.sum())),
        campaigns=(
            Campaign(
                "SYNTH",
                n_draw=n,
                observing_time_yr=config.observing_time_yr,
                metadata=_campaign_metadata(config),
            ),
        ),
        basis=gwcat_v2_basis_for_spin("chieff"),
        mode=SelectionMode.RAW_DRAW,
        metadata={
            "fixture": "phase3-synthetic-selection",
            "n_detected": int(keep.sum()),
            **_selection_observation_metadata(config),
        },
    )


def _require_truth_inside_proxy(config: SyntheticSurveyConfig, hp: Mapping[str, float]) -> None:
    if config.injection_draw != INJECTION_DRAW_POPULATION_PROXY:
        return
    problems = population_proxy_point_violations(config.injection_draw_hyperparameters, hp)
    if problems:
        raise ValueError(
            "population_proxy injections do not cover the synthetic truth "
            "population: " + "; ".join(problems)
        )


def generate_baseline_synthetic_dataset(
    *,
    seed: int = 20260917,
    model: GwcatChiEffBBHModel | None = None,
    hyperparameters: Mapping[str, float] | None = None,
    config: SyntheticSurveyConfig | None = None,
) -> SyntheticDataset:
    """Generate a closed PE+selection mock for baseline HBI recovery.

    Events are always drawn from the baseline family; ``model`` only supplies
    the cosmology, zmax, q_floor and required fields.
    """
    model = GwcatChiEffBBHModel() if model is None else model
    config = SyntheticSurveyConfig() if config is None else config
    hp = dict(
        DEFAULT_BASELINE_HYPERPARAMETERS
        if hyperparameters is None
        else hyperparameters
    )
    missing = set(DEFAULT_BASELINE_HYPERPARAMETERS) - set(hp)
    if missing:
        raise ValueError(f"synthetic truth is missing hyperparameter(s) {sorted(missing)}")
    _require_truth_inside_proxy(config, hp)

    rng = np.random.default_rng(int(seed))
    truths, observations = _detected_events(
        rng,
        lambda generator, n: _draw_population(generator, n, hp, model, config),
        model,
        config,
        hp,
    )
    posterior = _make_posterior_catalog(rng, truths, model, config, hp, observations)
    selection = _make_selection_catalog(rng, model, config, hp)
    validate_pair(posterior, selection, model.required_fields)

    return SyntheticDataset(
        posterior=posterior,
        selection=selection,
        event_truths=truths,
        truth_hyperparameters={name: float(value) for name, value in hp.items()},
        config=config,
        seed=int(seed),
        event_observations=observations,
    )
