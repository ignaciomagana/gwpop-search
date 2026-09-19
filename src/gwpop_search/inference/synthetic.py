"""Closed synthetic BBH data for Phase-3 end-to-end recovery tests.

This is deliberately a simple validation survey, not an astrophysical detector
simulation. It produces PE samples and a raw-draw selection campaign in the
exact gwcat-v2 chi_eff density basis so the common HBI engine can be tested
without release-specific data.

Selection injections can be drawn in two ways (``SyntheticSurveyConfig.injection_draw``):

``uniform_detector_box`` (default, unchanged legacy behavior)
    Uniform in (m1_detector, q, luminosity_distance, chi_eff) over the detector
    prior box, isotropic sky. Simple, but the population occupies a tiny corner
    of the box, so the selection effective sample size is only ~1-2% of the
    detected injections.

``population_proxy``
    Injections are drawn from the baseline population itself at fixed proxy
    hyperparameters (``injection_draw_hyperparameters``) with
    ``_draw_population``. The stored raw-draw density is exactly the population
    model's own normalized detector-basis log density at the proxy point,
    ``model(samples, proxy)``, including the source-to-detector Jacobian and the
    1/(4 pi) sky factor. The proxy's support contains the support of every
    population inside the Phase-3 hyperprior, so the raw-draw estimator
    ``A = T/N_draw * sum_detected p_pop/p_draw`` stays unbiased for every trial
    hyperparameter point while concentrating injections where detected systems
    of plausible populations live.

In both modes the detection rule is the same deterministic chirp-mass-scaled
reach, the campaign stores the true total number of draws ``n_draw`` and only
detected rows are retained.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.stats import truncnorm

from ..data import Campaign, PosteriorCatalog, SelectionCatalog, SelectionMode, validate_pair
from ..data.adapters import gwcat_v2_basis_for_spin
from ..models import DEFAULT_BASELINE_HYPERPARAMETERS, GwcatChiEffBBHModel
from .priors import BASELINE_SYNTHETIC_PRIORS, PriorSpec

INJECTION_DRAW_UNIFORM_DETECTOR_BOX = "uniform_detector_box"
INJECTION_DRAW_POPULATION_PROXY = "population_proxy"
INJECTION_DRAWS = (
    INJECTION_DRAW_UNIFORM_DETECTOR_BOX,
    INJECTION_DRAW_POPULATION_PROXY,
)

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
# width), pairing tilted toward q=1 (beta_q 2 vs 1; this also samples the
# q -> 1 pile-up of target populations whose mmin exceeds 2, where p(q|m1)
# diverges as m1 -> mmin) and a redshift density weighted to low z (kappa -1.5
# vs 2). chi_eff does not affect detection; its proxy keeps the truth mean and
# is wider (0.25 vs 0.20) so chi weights stay bounded over the whole chi_sigma
# prior (efficiency factor >= 0.025 for chi_sigma up to 0.5 and 0.93 at the
# truth). Selected by Monte-Carlo scans of the selection effective sample size
# per draw at the truth, at the hyperprior median, over posterior-like clouds
# and over random hyperprior draws; see docs/phase3_recovery.md.
DEFAULT_POPULATION_PROXY_HYPERPARAMETERS: dict[str, float] = {
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

_SELECTION_FIELDS = (
    "m1_detector",
    "q",
    "luminosity_distance",
    "ra",
    "dec",
    "chi_eff",
)


def _prior_bounds(spec: PriorSpec) -> tuple[float, float] | None:
    if spec.family in {"uniform", "log_uniform"}:
        return float(spec.low), float(spec.high)
    return None


def population_proxy_support_violations(
    proxy_hyperparameters: Mapping[str, float],
    priors: Mapping[str, PriorSpec] = BASELINE_SYNTHETIC_PRIORS,
) -> tuple[str, ...]:
    """Reasons a baseline proxy draw fails to cover every population in ``priors``.

    The baseline population support is m1_source in [mmin, mmax],
    q in [max(q_floor, mmin/m1_source), 1], z in (0, zmax], chi_eff in [-1, 1]
    and the full sky. Every proxy value other than mmin/mmax gives a strictly
    positive density on that support, so coverage of all trial populations
    reduces to ``mmin_proxy <= inf(prior mmin)`` and
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

    ``injection_draw`` selects the selection-injection distribution (see the
    module docstring). ``injection_draw_hyperparameters`` is only accepted for
    ``population_proxy``; ``None`` there resolves to
    ``DEFAULT_POPULATION_PROXY_HYPERPARAMETERS`` so the resolved proxy is always
    recorded in manifests. A proxy must cover the Phase-3 hyperprior
    (``population_proxy_support_violations``), otherwise it is rejected.
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

        draw = str(self.injection_draw)
        if draw not in INJECTION_DRAWS:
            raise ValueError(
                f"injection_draw must be one of {INJECTION_DRAWS}; got {draw!r}"
            )
        object.__setattr__(self, "injection_draw", draw)
        if draw == INJECTION_DRAW_UNIFORM_DETECTOR_BOX:
            if self.injection_draw_hyperparameters is not None:
                raise ValueError(
                    "injection_draw_hyperparameters is only used by "
                    f"injection_draw={INJECTION_DRAW_POPULATION_PROXY!r}"
                )
            return
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
        object.__setattr__(self, "injection_draw_hyperparameters", proxy)

    def to_dict(self) -> dict[str, object]:
        """JSON manifest payload.

        The default ``uniform_detector_box`` draw omits both injection-draw
        fields, so manifests written before the option existed stay
        byte-identical; ``population_proxy`` records the resolved proxy.
        """
        payload = asdict(self)
        if self.injection_draw == INJECTION_DRAW_UNIFORM_DETECTOR_BOX:
            payload.pop("injection_draw")
            payload.pop("injection_draw_hyperparameters")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SyntheticSurveyConfig:
        return cls(**dict(payload))


@dataclass(frozen=True)
class SyntheticDataset:
    posterior: PosteriorCatalog
    selection: SelectionCatalog
    event_truths: Mapping[str, np.ndarray]
    truth_hyperparameters: Mapping[str, float]
    config: SyntheticSurveyConfig
    seed: int


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
    """Deterministic synthetic detection cut based on a chirp-mass-scaled reach."""
    m1 = np.asarray(samples["m1_detector"], dtype=float)
    q = np.asarray(samples["q"], dtype=float)
    d_l = np.asarray(samples["luminosity_distance"], dtype=float)

    chirp_mass = m1 * np.power(q, 3.0 / 5.0) / np.power(1.0 + q, 1.0 / 5.0)
    reach = config.reference_horizon_mpc * np.power(
        chirp_mass / config.reference_chirp_mass,
        5.0 / 6.0,
    )
    return d_l <= reach


def _detected_population_truths(rng, hp, model, config):
    pieces: dict[str, list[np.ndarray]] = {}
    n_have = 0
    attempts = 0

    while n_have < config.n_events:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("failed to draw enough detected synthetic events")
        draw = _draw_population(
            rng,
            max(config.population_batch_size, 2 * (config.n_events - n_have)),
            hp,
            model,
            config,
        )
        keep = detection_mask(draw, config)
        if not np.any(keep):
            continue

        for name, values in draw.items():
            pieces.setdefault(name, []).append(np.asarray(values)[keep])
        n_have += int(keep.sum())

    return {
        name: np.concatenate(chunks)[: config.n_events]
        for name, chunks in pieces.items()
    }


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
    d_l_max = float(model.cosmology.dL_of_z(model.zmax))
    population_mass_max = float(hp["mmax"]) * (1.0 + model.zmax)
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


def _make_posterior_catalog(rng, truths, model, config, hp):
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

    return PosteriorCatalog(
        event_names=names,
        offsets=offsets,
        samples=samples,
        log_ref_density=log_ref,
        basis=gwcat_v2_basis_for_spin("chieff"),
        metadata={
            "fixture": "phase3-synthetic-pe",
            "reference_prior": "uniform detector basis",
        },
    )


def population_proxy_log_draw_density(model, samples, proxy_hyperparameters):
    """Exact log draw density of ``population_proxy`` injections.

    ``model`` must be the population model whose ``_draw_population`` sampler
    generated the injections (same cosmology, zmax and q_floor). The result is
    that model's own normalized log density in the gwcat detector basis
    (m1_detector, q, luminosity_distance, sky, chi_eff) at the proxy
    hyperparameters, evaluated in 64-bit precision. Zero-support rows are
    returned as -inf; callers decide whether that is an error.
    """
    try:
        from jax import enable_x64
    except ImportError:  # pragma: no cover - JAX releases before the config API
        from jax.experimental import enable_x64

    columns = {
        name: np.asarray(samples[name], dtype=float) for name in _SELECTION_FIELDS
    }
    with enable_x64(True):
        values = model(columns, dict(proxy_hyperparameters))
        out = np.array(values, dtype=np.float64)
    if out.shape != columns["m1_detector"].shape:
        raise RuntimeError(
            f"proxy draw density has shape {out.shape}; expected "
            f"{columns['m1_detector'].shape}"
        )
    if np.isnan(out).any() or np.isposinf(out).any():
        raise RuntimeError("proxy draw density contains NaN/+inf values")
    return out


def _make_population_proxy_selection_catalog(rng, model, config):
    proxy = dict(config.injection_draw_hyperparameters)
    n = int(config.n_injections)
    draw = _draw_population(rng, n, proxy, model, config)
    samples = {name: np.asarray(draw[name], dtype=float) for name in _SELECTION_FIELDS}
    keep = detection_mask(samples, config)
    if not np.any(keep):
        raise RuntimeError("synthetic selection campaign produced no detections")

    retained = {name: values[keep] for name, values in samples.items()}
    log_draw = population_proxy_log_draw_density(model, retained, proxy)
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
                metadata={
                    "detection_rule": "chirp_mass_scaled_reach",
                    "injection_draw": INJECTION_DRAW_POPULATION_PROXY,
                },
            ),
        ),
        basis=gwcat_v2_basis_for_spin("chieff"),
        mode=SelectionMode.RAW_DRAW,
        metadata={
            "fixture": "phase3-synthetic-selection",
            "n_detected": n_detected,
            "injection_draw": INJECTION_DRAW_POPULATION_PROXY,
            "injection_draw_hyperparameters": proxy,
            "draw_density_model": model.to_config(),
        },
    )


def _make_selection_catalog(rng, model, config, hp):
    if config.injection_draw == INJECTION_DRAW_POPULATION_PROXY:
        return _make_population_proxy_selection_catalog(rng, model, config)

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
    keep = detection_mask(samples, config)
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
                metadata={"detection_rule": "chirp_mass_scaled_reach"},
            ),
        ),
        basis=gwcat_v2_basis_for_spin("chieff"),
        mode=SelectionMode.RAW_DRAW,
        metadata={
            "fixture": "phase3-synthetic-selection",
            "n_detected": int(keep.sum()),
        },
    )


def generate_baseline_synthetic_dataset(
    *,
    seed: int = 20260917,
    model: GwcatChiEffBBHModel | None = None,
    hyperparameters: Mapping[str, float] | None = None,
    config: SyntheticSurveyConfig | None = None,
) -> SyntheticDataset:
    """Generate a closed PE+selection mock for baseline HBI recovery."""
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

    rng = np.random.default_rng(int(seed))
    truths = _detected_population_truths(rng, hp, model, config)
    posterior = _make_posterior_catalog(rng, truths, model, config, hp)
    selection = _make_selection_catalog(rng, model, config, hp)
    validate_pair(posterior, selection, model.required_fields)

    return SyntheticDataset(
        posterior=posterior,
        selection=selection,
        event_truths=truths,
        truth_hyperparameters={name: float(value) for name, value in hp.items()},
        config=config,
        seed=int(seed),
    )
