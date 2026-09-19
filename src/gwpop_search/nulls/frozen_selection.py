"""Null catalogs drawn from the frozen estimator-ready selection measure.

Truths
    The null event truths are ``n_events`` rows of the frozen production
    selection drawn with probability ``p_pop(theta | Lambda_null) /
    p_draw(theta)`` (normalized over the selected rows). Because the rows are
    detected injections, this is a draw from the detected-population
    distribution ``p_pop(theta) P_det(theta) / A`` that the null HBI
    likelihood assumes, with the real detection process.

PE (declared noisy-observation approximation)
    Zero-noise, truth-centred PE is disqualifying for null calibration: it
    puts every truth at the same quantile of its own posterior and biases
    population widths (Essick & Fishbach 2023, arXiv:2310.02017). Each null
    event instead receives exactly one synthetic observation
    ``d = (ln m1_det, q, ln d_L, chi_eff)_true + N(0, diag(s^2))`` with the
    survey's measurement-noise scales ``s``, and its PE samples are exact
    draws from ``p(theta | d)`` under the stored uniform detector-box PE
    prior (sky delta-like at the injection truth). This mirrors
    ``observation_model="noisy_observation"`` of
    :mod:`gwpop_search.inference.synthetic` (same noise model, same box
    prior, same exact posterior sampler).

Remaining declared approximation
    The frozen injection was detected by the real search with its own noise
    realisation, which is not linked to the synthetic observation ``d``: the
    synthetic PE is not conditioned on the detection statistic that selected
    the row. The metadata records this explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from gwpop_search.data import SelectionMode, validate_pair
from gwpop_search.grammar import ModelSpec, baseline_model_spec
from gwpop_search.inference.synthetic import (
    INJECTION_DRAW_UNIFORM_DETECTOR_BOX,
    OBSERVATION_MODEL_NOISY,
    SyntheticSurveyConfig,
    _detector_prior_bounds,
    _make_posterior_catalog,
    _observe,
)
from gwpop_search.models import compile_model_spec

FROZEN_SELECTION_NULL_FORMAT_VERSION = "gwpop-search-frozen-selection-null-2.0"

_TRUTH_FIELDS = (
    "m1_detector",
    "q",
    "luminosity_distance",
    "ra",
    "dec",
    "chi_eff",
)


@dataclass(frozen=True)
class FrozenSelectionNullDataset:
    posterior: object
    selection: object
    truth_rows: np.ndarray
    truth_hyperparameters: Mapping[str, float]
    metadata: Mapping[str, object]
    event_observations: Mapping[str, np.ndarray] | None = None


def frozen_selection_resampling_probabilities(
    selection,
    model_spec: ModelSpec,
    hyperparameters: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Normalize the selected-injection contributions p_pop/pdraw."""
    if selection.mode is not SelectionMode.ESTIMATOR_READY:
        raise ValueError(
            "frozen-selection null generation requires estimator_ready selection"
        )
    population_model = compile_model_spec(model_spec)
    selection.require(population_model.required_fields)

    log_population = np.asarray(
        population_model(selection.samples, hyperparameters),
        dtype=float,
    )
    if log_population.shape != (selection.n_selected,):
        raise ValueError("population density returned the wrong selection shape")
    if np.any(np.isnan(log_population) | np.isposinf(log_population)):
        raise ValueError("population density contains NaN or +inf on selection rows")

    log_weight = log_population - np.asarray(
        selection.log_draw_density,
        dtype=float,
    )
    finite = np.isfinite(log_weight)
    if not np.any(finite):
        raise ValueError(
            "null population has zero support on every selected injection"
        )

    log_norm = float(logsumexp(log_weight[finite]))
    probabilities = np.zeros(selection.n_selected, dtype=float)
    probabilities[finite] = np.exp(log_weight[finite] - log_norm)
    probabilities /= probabilities.sum()

    ess = float(1.0 / np.sum(probabilities**2))
    diagnostics = {
        "n_selected": int(selection.n_selected),
        "n_positive_weight": int(np.count_nonzero(probabilities > 0.0)),
        "resampling_ess": ess,
        "max_resampling_probability": float(probabilities.max()),
    }
    return probabilities, diagnostics


def require_frozen_selection_survey(survey_config: SyntheticSurveyConfig) -> None:
    """The survey settings a frozen-selection null may use.

    ``observation_model`` must be ``noisy_observation`` (truth-centred PE is
    refused) and the injection options must stay at their defaults (the
    frozen production selection is reused, nothing is injected).
    """
    if survey_config.observation_model != OBSERVATION_MODEL_NOISY:
        raise ValueError(
            "frozen-selection nulls require observation_model='noisy_observation': "
            "zero-noise truth-centred PE is disqualifying for null calibration "
            "(Essick & Fishbach 2023)"
        )
    if survey_config.injection_draw != INJECTION_DRAW_UNIFORM_DETECTOR_BOX:
        raise ValueError(
            "frozen-selection nulls reuse the frozen production selection; survey "
            "injection_draw options do not apply"
        )


def generate_frozen_selection_null_dataset(
    *,
    seed: int,
    observed_posterior,
    selection,
    truth_hyperparameters: Mapping[str, float],
    survey_config: SyntheticSurveyConfig,
    min_resampling_ess: float,
    model_spec: ModelSpec | None = None,
) -> FrozenSelectionNullDataset:
    """Generate one null catalog on the exact frozen production selection.

    ``survey_config`` supplies the event count (must equal the observed
    catalog), the PE samples per event and the measurement-noise scales of
    the declared noisy-observation PE approximation (see the module
    docstring). Generator order: truth rows, then one observation per event,
    then the PE draws.
    """
    model_spec = baseline_model_spec() if model_spec is None else model_spec
    if observed_posterior.basis.identity != selection.basis.identity:
        raise ValueError("observed PE and frozen selection basis identities differ")
    if selection.basis.spin_parameterization != "chieff":
        raise ValueError(
            "frozen-selection null PE approximation currently requires chieff basis"
        )
    if int(survey_config.n_events) != int(observed_posterior.n_events):
        raise ValueError(
            "frozen-selection null event count must equal the observed catalog: "
            f"config={survey_config.n_events}, observed={observed_posterior.n_events}"
        )
    if not np.isfinite(min_resampling_ess) or min_resampling_ess <= 0.0:
        raise ValueError("min_resampling_ess must be finite and positive")
    require_frozen_selection_survey(survey_config)

    probabilities, diagnostics = frozen_selection_resampling_probabilities(
        selection,
        model_spec,
        truth_hyperparameters,
    )
    if diagnostics["resampling_ess"] < float(min_resampling_ess):
        raise ValueError(
            "frozen-selection null resampling ESS is below the declared gate: "
            f"{diagnostics['resampling_ess']:.6g} < {float(min_resampling_ess):.6g}"
        )

    rng = np.random.default_rng(int(seed))
    truth_rows = rng.choice(
        selection.n_selected,
        size=observed_posterior.n_events,
        replace=True,
        p=probabilities,
    )
    selection.require(_TRUTH_FIELDS)
    truths = {
        name: np.asarray(selection.samples[name], dtype=float)[truth_rows]
        for name in _TRUTH_FIELDS
    }

    population_model = compile_model_spec(model_spec)
    observations = _observe(rng, truths, survey_config)
    posterior = _make_posterior_catalog(
        rng,
        truths,
        population_model,
        survey_config,
        truth_hyperparameters,
        observations,
    )
    validate_pair(
        posterior,
        selection,
        population_model.required_fields,
    )
    bounds = _detector_prior_bounds(population_model, survey_config, truth_hyperparameters)

    metadata = {
        "format_version": FROZEN_SELECTION_NULL_FORMAT_VERSION,
        "seed": int(seed),
        "model_hash": model_spec.model_hash,
        "selection_basis_identity": selection.basis.identity,
        "selection_mode": selection.mode.value,
        "observed_n_events": int(observed_posterior.n_events),
        "posterior_samples_per_event": int(
            survey_config.posterior_samples_per_event
        ),
        "min_resampling_ess": float(min_resampling_ess),
        **diagnostics,
        "n_unique_truth_rows": int(np.unique(truth_rows).size),
        "pe_approximation": {
            "observation_model": OBSERVATION_MODEL_NOISY,
            "observation": (
                "one synthetic observation per null event: "
                "(ln m1_detector, q, ln d_L, chi_eff)_true + N(0, diag(sigma^2))"
            ),
            "observation_noise_sigma": {
                "log_m1_detector": float(survey_config.pe_m1_fractional_sigma),
                "q": float(survey_config.pe_q_sigma),
                "log_luminosity_distance": float(survey_config.pe_d_l_fractional_sigma),
                "chi_eff": float(survey_config.pe_chi_eff_sigma),
            },
            "pe": (
                "exact posterior draws given the synthetic observation under the "
                "stored uniform detector-box PE prior"
            ),
            "reference_prior": "uniform_detector_basis",
            "prior_box": {name: float(value) for name, value in bounds.items()},
            "sky": "delta_like_at_selected_injection_truth",
            "detection_noise_link": "not_linked",
            "declared_approximation": (
                "the frozen injection's detection by the real search used its own noise "
                "realisation, which is independent of the synthetic PE observation; the "
                "null PE is therefore not conditioned on the detection statistic that "
                "selected the row"
            ),
        },
    }
    return FrozenSelectionNullDataset(
        posterior=posterior,
        selection=selection,
        truth_rows=np.asarray(truth_rows, dtype=np.int64),
        truth_hyperparameters={
            str(name): float(value)
            for name, value in truth_hyperparameters.items()
        },
        metadata=metadata,
        event_observations={name: np.asarray(v) for name, v in observations.items()},
    )
