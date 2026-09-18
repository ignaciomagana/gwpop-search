"""Closed synthetic BBH data for Phase-3 end-to-end recovery tests.

This is deliberately a simple validation survey, not an astrophysical detector
simulation. It produces PE samples and a raw-draw selection campaign in the
exact gwcat-v2 chi_eff density basis so the common HBI engine can be tested
without release-specific data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import truncnorm

from ..data import Campaign, PosteriorCatalog, SelectionCatalog, SelectionMode, validate_pair
from ..data.adapters import gwcat_v2_basis_for_spin
from ..models import DEFAULT_BASELINE_HYPERPARAMETERS, GwcatChiEffBBHModel


@dataclass(frozen=True)
class SyntheticSurveyConfig:
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


def _make_selection_catalog(rng, model, config, hp):
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
