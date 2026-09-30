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
    measurement-noise scales ``s``, and its PE samples are exact draws from
    ``p(theta | d)`` under the stored uniform detector-box PE prior (sky
    delta-like at the injection truth). This mirrors
    ``observation_model="noisy_observation"`` of
    :mod:`gwpop_search.inference.synthetic` (same noise model, same box
    prior, same exact posterior sampler). The box's upper mass edge follows
    the *searched* root hyperprior (production: ``gwtc5-v1``), not a fixed
    module constant.

PE precision (``pe_scale_policy``)
    Under ``match_observed`` (default) ``s`` and the per-event sample count
    come from the frozen observed catalog (:mod:`gwpop_search.nulls.pe_matching`),
    so the null events are measured like the real ones and the F3
    Monte-Carlo regime of a null is the regime of the observed run (D4).
    Under ``declared_fixed`` the configured scalar scales are used and the
    measured mismatch is recorded.

Declared approximations (all recorded in ``metadata["pe_approximation"]``)
    1. The frozen injection was detected by the real search with its own noise
       realisation, which is not linked to the synthetic observation ``d``:
       the synthetic PE is not conditioned on the detection statistic that
       selected the row (``detection_noise_link="not_linked"``).
    2. Only the four marginal PE widths are reproduced; the real posteriors'
       shapes and parameter correlations are not.
    3. Under ``declared_fixed``, the null PE precision and PE sample count
       differ from the observed catalog by the recorded ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from gwpop_search.data import SelectionMode, validate_pair
from gwpop_search.grammar import ModelSpec, baseline_model_spec
from gwpop_search.inference.model_spec import prior_specs_from_model_spec
from gwpop_search.inference.synthetic import (
    INJECTION_DRAW_UNIFORM_DETECTOR_BOX,
    OBSERVATION_MODEL_NOISY,
    SyntheticSurveyConfig,
    _detector_prior_bounds,
    _make_posterior_catalog,
    _observe,
)
from gwpop_search.models import compile_model_spec

from .pe_matching import (
    PE_SCALE_POLICY_DECLARED_FIXED,
    PE_SCALE_POLICY_MATCH_OBSERVED,
    measure_observed_pe_scales,
    pe_scale_comparison,
    rank_matched_event_sigmas,
    require_pe_scale_policy,
)

# 2.1 adds the PE-scale policy block (declared vs applied measurement scales,
# the measured comparison with the observed catalog) and the catalog-scaled
# resampling gate to the dataset metadata.
FROZEN_SELECTION_NULL_FORMAT_VERSION = "gwpop-search-frozen-selection-null-2.1"
DEFAULT_MIN_RESAMPLING_ESS_PER_EVENT = 10.0

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


def root_hyperprior_mmax_supremum(model_spec: ModelSpec) -> float:
    """``sup`` of the root's ``mmax`` hyperprior: the PE prior box's mass edge.

    The null PE prior must contain the detector-frame support of every trial
    population the search scores, which is set by the hyperprior of the graph
    root actually being searched (``gwtc5-v1``: ``mmax ~ U(60, 200)``), not by
    the Phase-3 synthetic constant. An unbounded ``mmax`` prior has no finite
    box and is refused.
    """
    priors = prior_specs_from_model_spec(model_spec)
    if "mmax" not in priors:
        raise ValueError(
            "the null root model has no 'mmax' hyperprior; the PE prior box "
            "cannot be derived from it"
        )
    spec = priors["mmax"]
    if spec.family not in {"uniform", "log_uniform"} or spec.high is None:
        raise ValueError(
            f"the null root 'mmax' hyperprior is {spec.family!r} (unbounded); no "
            "finite PE prior box covers every trial population"
        )
    return float(spec.high)


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


def required_resampling_ess(
    n_events: int,
    *,
    min_resampling_ess: float,
    min_resampling_ess_per_event: float,
) -> float:
    """The ESS a null catalog's resampling weights must reach.

    The null truths are ``n_events`` iid draws from a discrete distribution
    over the selected injections, so an absolute floor alone is not a gate: an
    ESS of a few hundred lets a 259-event catalog draw more events than there
    are effectively distinct truths, and all nulls then share that small atom
    set. The gate therefore scales with the catalog,
    ``max(min_resampling_ess, min_resampling_ess_per_event * n_events)``.
    """
    for name, value in (
        ("min_resampling_ess", min_resampling_ess),
        ("min_resampling_ess_per_event", min_resampling_ess_per_event),
    ):
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return max(
        float(min_resampling_ess),
        float(min_resampling_ess_per_event) * int(n_events),
    )


def generate_frozen_selection_null_dataset(
    *,
    seed: int,
    observed_posterior,
    selection,
    truth_hyperparameters: Mapping[str, float],
    survey_config: SyntheticSurveyConfig,
    min_resampling_ess: float,
    min_resampling_ess_per_event: float = DEFAULT_MIN_RESAMPLING_ESS_PER_EVENT,
    pe_scale_policy: str = PE_SCALE_POLICY_MATCH_OBSERVED,
    model_spec: ModelSpec | None = None,
) -> FrozenSelectionNullDataset:
    """Generate one null catalog on the exact frozen production selection.

    ``survey_config`` supplies the event count (must equal the observed
    catalog) and, under ``pe_scale_policy="declared_fixed"``, the PE samples
    per event and the measurement-noise scales of the declared
    noisy-observation PE approximation. Under ``match_observed`` (default)
    both are taken from ``observed_posterior`` instead (see the module
    docstring). Generator order: truth rows, then one observation per event,
    then the PE draws.

    ``model_spec`` is the searched graph root. Its population *density* at
    ``truth_hyperparameters`` defines the resampling weights, and the
    supremum of its ``mmax`` hyperprior sets the PE prior box (so the box
    follows the profile the search actually scores).
    """
    model_spec = baseline_model_spec() if model_spec is None else model_spec
    pe_scale_policy = require_pe_scale_policy(pe_scale_policy)
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
    required_ess = required_resampling_ess(
        int(observed_posterior.n_events),
        min_resampling_ess=min_resampling_ess,
        min_resampling_ess_per_event=min_resampling_ess_per_event,
    )
    require_frozen_selection_survey(survey_config)
    observed_scales = measure_observed_pe_scales(observed_posterior)
    declared_comparison = pe_scale_comparison(survey_config, observed_scales)
    if pe_scale_policy == PE_SCALE_POLICY_MATCH_OBSERVED:
        observed_samples = observed_scales.uniform_samples_per_event()
        if int(survey_config.posterior_samples_per_event) != observed_samples:
            raise ValueError(
                "pe_scale_policy='match_observed' draws the observed number of PE "
                "samples per null event so the Monte-Carlo regime of a null is the "
                "regime of the observed run; freeze "
                f"posterior_samples_per_event={observed_samples}, got "
                f"{int(survey_config.posterior_samples_per_event)}"
            )

    probabilities, diagnostics = frozen_selection_resampling_probabilities(
        selection,
        model_spec,
        truth_hyperparameters,
    )
    if diagnostics["resampling_ess"] < required_ess:
        raise ValueError(
            "frozen-selection null resampling ESS is below the declared gate: "
            f"{diagnostics['resampling_ess']:.6g} < {required_ess:.6g} "
            f"(max of the absolute floor {float(min_resampling_ess):.6g} and "
            f"{float(min_resampling_ess_per_event):.6g} x "
            f"{int(observed_posterior.n_events)} events)"
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
    mmax_sup = root_hyperprior_mmax_supremum(model_spec)
    event_sigmas = None
    scale_provenance: dict[str, object] = {
        "policy": PE_SCALE_POLICY_DECLARED_FIXED,
        "declared_scales": {
            "log_m1_detector": float(survey_config.pe_m1_fractional_sigma),
            "q": float(survey_config.pe_q_sigma),
            "log_luminosity_distance": float(survey_config.pe_d_l_fractional_sigma),
            "chi_eff": float(survey_config.pe_chi_eff_sigma),
        },
        "observed_scales": observed_scales.summary(),
    }
    if pe_scale_policy == PE_SCALE_POLICY_MATCH_OBSERVED:
        event_sigmas, scale_provenance = rank_matched_event_sigmas(
            observed_scales,
            truths,
        )
    observations = _observe(rng, truths, survey_config, event_sigmas)
    posterior = _make_posterior_catalog(
        rng,
        truths,
        population_model,
        survey_config,
        truth_hyperparameters,
        observations,
        event_sigmas,
        mmax_sup,
    )
    validate_pair(
        posterior,
        selection,
        population_model.required_fields,
    )
    bounds = _detector_prior_bounds(
        population_model,
        survey_config,
        truth_hyperparameters,
        mmax_sup,
    )

    declared_approximations = [
        {
            "name": "detection_noise_not_linked",
            "detail": (
                "the frozen injection's detection by the real search used its own noise "
                "realisation, which is independent of the synthetic PE observation; the "
                "null PE is therefore not conditioned on the detection statistic that "
                "selected the row"
            ),
        },
        {
            "name": "marginal_pe_widths_only",
            "detail": (
                "the null PE reproduces the four marginal measurement scales "
                "(ln m1_detector, q, ln d_L, chi_eff) of the observed catalog, not the "
                "shapes or parameter correlations of the real posteriors"
            ),
        },
    ]
    if pe_scale_policy == PE_SCALE_POLICY_DECLARED_FIXED:
        declared_approximations.append(
            {
                "name": "pe_precision_mismatch",
                "detail": (
                    "pe_scale_policy='declared_fixed': the null PE precision and PE "
                    "sample count are the configured fixed values, not the observed "
                    "catalog's; the measured ratios are recorded in "
                    "pe_scales.declared_vs_observed (observed/declared width ratios up "
                    f"to {declared_comparison['max_width_ratio']:.3g}, PE sample ratio "
                    f"{declared_comparison['posterior_samples_per_event']['ratio_observed_over_declared']:.3g})"
                ),
            }
        )
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
        "min_resampling_ess_per_event": float(min_resampling_ess_per_event),
        "required_resampling_ess": float(required_ess),
        **diagnostics,
        "n_unique_truth_rows": int(np.unique(truth_rows).size),
        "unique_truth_fraction": float(
            np.unique(truth_rows).size / int(observed_posterior.n_events)
        ),
        "pe_approximation": {
            "observation_model": OBSERVATION_MODEL_NOISY,
            "observation": (
                "one synthetic observation per null event: "
                "(ln m1_detector, q, ln d_L, chi_eff)_true + N(0, diag(sigma^2))"
            ),
            "observation_noise_sigma": (
                dict(scale_provenance["declared_scales"])
                if pe_scale_policy == PE_SCALE_POLICY_DECLARED_FIXED
                else dict(scale_provenance["applied_scales"])
            ),
            "pe": (
                "exact posterior draws given the synthetic observation under the "
                "stored uniform detector-box PE prior"
            ),
            "reference_prior": "uniform_detector_basis",
            "prior_box": {name: float(value) for name, value in bounds.items()},
            "prior_box_mmax_supremum": float(mmax_sup),
            "sky": "delta_like_at_selected_injection_truth",
            "detection_noise_link": "not_linked",
            "pe_scale_policy": pe_scale_policy,
            "pe_scales": {
                **scale_provenance,
                "declared_vs_observed": declared_comparison,
            },
            "declared_approximations": declared_approximations,
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
