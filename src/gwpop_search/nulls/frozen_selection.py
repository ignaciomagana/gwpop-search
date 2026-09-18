"""Null catalogs drawn from the frozen estimator-ready selection measure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from gwpop_search.data import SelectionMode, validate_pair
from gwpop_search.grammar import ModelSpec, baseline_model_spec
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    _make_posterior_catalog,
)
from gwpop_search.models import compile_model_spec


@dataclass(frozen=True)
class FrozenSelectionNullDataset:
    posterior: object
    selection: object
    truth_rows: np.ndarray
    truth_hyperparameters: Mapping[str, float]
    metadata: Mapping[str, object]


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
    """Generate null PE while reusing the exact frozen production selection."""
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
    required_truth = (
        "m1_detector",
        "q",
        "luminosity_distance",
        "ra",
        "dec",
        "chi_eff",
    )
    selection.require(required_truth)
    truths = {
        name: np.asarray(selection.samples[name], dtype=float)[truth_rows]
        for name in required_truth
    }

    population_model = compile_model_spec(model_spec)
    posterior = _make_posterior_catalog(
        rng,
        truths,
        population_model,
        survey_config,
        truth_hyperparameters,
    )
    validate_pair(
        posterior,
        selection,
        population_model.required_fields,
    )

    metadata = {
        "format_version": "gwpop-search-frozen-selection-null-1.0",
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
            "m1_fractional_sigma": survey_config.pe_m1_fractional_sigma,
            "q_sigma": survey_config.pe_q_sigma,
            "d_l_fractional_sigma": survey_config.pe_d_l_fractional_sigma,
            "chi_eff_sigma": survey_config.pe_chi_eff_sigma,
            "sky": "delta_like_at_selected_injection_truth",
            "reference_prior": "uniform_detector_basis",
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
    )
