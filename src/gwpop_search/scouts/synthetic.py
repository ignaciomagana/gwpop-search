"""Structured synthetic catalogs for validating flexible HSGP scouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import truncnorm

from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference.synthetic import (
    SyntheticDataset,
    SyntheticSurveyConfig,
    _make_posterior_catalog,
    _make_selection_catalog,
    _sample_chi_eff,
    _sample_primary_mass,
    _sample_q,
    _sample_redshift,
    detection_mask,
)
from gwpop_search.models import (
    DEFAULT_BASELINE_HYPERPARAMETERS,
    GwcatChiEffBBHModel,
)
from gwpop_search.data import validate_pair


_SUPPORTED = {
    "null",
    "pairing.beta.linear_m1",
    "chieff.mean.linear_q",
    "chieff.width.linear_q",
}


@dataclass(frozen=True)
class StructuredScoutInjection:
    mutation_id: str
    strength: float
    label: str | None = None

    def __post_init__(self) -> None:
        if self.mutation_id not in _SUPPORTED:
            raise ValueError(
                f"unsupported structured scout injection {self.mutation_id!r}"
            )
        if not np.isfinite(self.strength):
            raise ValueError("injection strength must be finite")
        if self.mutation_id == "null" and self.strength != 0.0:
            raise ValueError("null injection strength must be exactly zero")

    @property
    def injection_label(self) -> str:
        return self.label or f"{self.mutation_id}:{self.strength:+.6g}"


def _sample_q_with_beta(
    rng,
    m1_source,
    *,
    beta,
    mmin: float,
    q_floor: float,
):
    m1_source = np.asarray(m1_source, dtype=float)
    beta = np.broadcast_to(np.asarray(beta, dtype=float), m1_source.shape)
    qmin = np.maximum(float(q_floor), float(mmin) / m1_source)
    exponent = beta + 1.0
    u = rng.random(m1_source.size)

    near_log = np.abs(exponent) < 1e-10
    safe_exponent = np.where(near_log, 1.0, exponent)
    powered = np.power(
        np.power(qmin, safe_exponent)
        + u * (1.0 - np.power(qmin, safe_exponent)),
        1.0 / safe_exponent,
    )
    log_draw = qmin * np.power(1.0 / qmin, u)
    return np.where(near_log, log_draw, powered)


def _sample_chi_with_mu_sigma(rng, mu, sigma):
    mu = np.asarray(mu, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), mu.shape)
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("structured chi_eff sigma must be finite and positive")
    a = (-1.0 - mu) / sigma
    b = (1.0 - mu) / sigma
    return truncnorm.rvs(
        a,
        b,
        loc=mu,
        scale=sigma,
        random_state=rng,
    )


def _draw_structured_population(
    rng,
    n: int,
    hp: Mapping[str, float],
    model: GwcatChiEffBBHModel,
    survey: SyntheticSurveyConfig,
    injection: StructuredScoutInjection,
):
    m1_source = _sample_primary_mass(rng, n, hp)
    z = _sample_redshift(
        rng,
        n,
        hp,
        model,
        survey.redshift_sampling_grid,
    )

    spec = baseline_model_spec()
    if injection.mutation_id == "pairing.beta.linear_m1":
        pivot = float(spec.pairing.options["m1_pivot"])
        beta = float(hp["beta_q"]) + injection.strength * (
            m1_source - pivot
        )
        q = _sample_q_with_beta(
            rng,
            m1_source,
            beta=beta,
            mmin=float(hp["mmin"]),
            q_floor=model.q_floor,
        )
    else:
        q = _sample_q(rng, m1_source, hp, model.q_floor)

    if injection.mutation_id == "chieff.mean.linear_q":
        pivot = float(spec.chieff.options["q_pivot"])
        mu = float(hp["chi_mu"]) + injection.strength * (q - pivot)
        sigma = np.full(n, float(hp["chi_sigma"]))
        chi_eff = _sample_chi_with_mu_sigma(rng, mu, sigma)
    elif injection.mutation_id == "chieff.width.linear_q":
        pivot = float(spec.chieff.options["q_pivot"])
        mu = np.full(n, float(hp["chi_mu"]))
        sigma = float(hp["chi_sigma"]) * np.exp(
            injection.strength * (q - pivot)
        )
        chi_eff = _sample_chi_with_mu_sigma(rng, mu, sigma)
    else:
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


def _detected_structured_truths(
    rng,
    hp,
    model,
    survey,
    injection,
):
    pieces: dict[str, list[np.ndarray]] = {}
    n_have = 0
    attempts = 0

    while n_have < survey.n_events:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError(
                "failed to draw enough detected structured synthetic events"
            )
        draw = _draw_structured_population(
            rng,
            max(
                survey.population_batch_size,
                2 * (survey.n_events - n_have),
            ),
            hp,
            model,
            survey,
            injection,
        )
        keep = detection_mask(draw, survey)
        if not np.any(keep):
            continue
        for name, values in draw.items():
            pieces.setdefault(name, []).append(np.asarray(values)[keep])
        n_have += int(keep.sum())

    return {
        name: np.concatenate(chunks)[: survey.n_events]
        for name, chunks in pieces.items()
    }


def generate_structured_scout_dataset(
    *,
    seed: int,
    injection: StructuredScoutInjection,
    survey_config: SyntheticSurveyConfig | None = None,
    hyperparameters: Mapping[str, float] | None = None,
    model: GwcatChiEffBBHModel | None = None,
) -> SyntheticDataset:
    """Generate PE+selection data with one known grammar-matched dependence."""
    survey = (
        SyntheticSurveyConfig()
        if survey_config is None
        else survey_config
    )
    model = GwcatChiEffBBHModel() if model is None else model
    hp = dict(
        DEFAULT_BASELINE_HYPERPARAMETERS
        if hyperparameters is None
        else hyperparameters
    )
    missing = set(DEFAULT_BASELINE_HYPERPARAMETERS) - set(hp)
    if missing:
        raise ValueError(
            f"structured synthetic truth is missing {sorted(missing)}"
        )

    rng = np.random.default_rng(int(seed))
    truths = _detected_structured_truths(
        rng,
        hp,
        model,
        survey,
        injection,
    )
    posterior = _make_posterior_catalog(
        rng,
        truths,
        model,
        survey,
        hp,
    )
    selection = _make_selection_catalog(
        rng,
        model,
        survey,
        hp,
    )
    validate_pair(posterior, selection, model.required_fields)

    truth = {name: float(value) for name, value in hp.items()}
    if injection.mutation_id == "pairing.beta.linear_m1":
        truth["beta_q_m1_slope"] = float(injection.strength)
    elif injection.mutation_id == "chieff.mean.linear_q":
        truth["chi_mu_q_slope"] = float(injection.strength)
    elif injection.mutation_id == "chieff.width.linear_q":
        truth["log_chi_sigma_q_slope"] = float(injection.strength)

    return SyntheticDataset(
        posterior=posterior,
        selection=selection,
        event_truths=truths,
        truth_hyperparameters=truth,
        config=survey,
        seed=int(seed),
    )
