import numpy as np
import pytest

from gwpop_search.data import Campaign, SelectionCatalog, SelectionMode, validate_pair
from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS, compile_model_spec
from gwpop_search.nulls import (
    frozen_selection_resampling_probabilities,
    generate_frozen_selection_null_dataset,
)


def _estimator_ready_selection(raw):
    return SelectionCatalog(
        samples=raw.samples,
        log_draw_density=raw.log_draw_density,
        campaign_id=np.asarray(["combined"] * raw.n_selected),
        campaigns=(
            Campaign(
                "combined",
                n_draw=sum(c.n_draw for c in raw.campaigns),
                observing_time_yr=sum(
                    c.observing_time_yr for c in raw.campaigns
                ),
            ),
        ),
        basis=raw.basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics="test estimator-ready denominator",
        metadata={"fixture": "frozen-selection-null-test"},
    )


def _fixture(seed=1):
    config = SyntheticSurveyConfig(
        n_events=5,
        posterior_samples_per_event=12,
        n_injections=2000,
        population_batch_size=128,
        redshift_sampling_grid=512,
    )
    dataset = generate_baseline_synthetic_dataset(
        seed=seed,
        config=config,
    )
    selection = _estimator_ready_selection(dataset.selection)
    return dataset.posterior, selection, config


def test_frozen_selection_probabilities_are_normalized_p_pop_over_pdraw():
    _, selection, _ = _fixture()
    model_spec = baseline_model_spec()
    model = compile_model_spec(model_spec)

    probabilities, diagnostics = frozen_selection_resampling_probabilities(
        selection,
        model_spec,
        DEFAULT_BASELINE_HYPERPARAMETERS,
    )
    log_pop = np.asarray(
        model(selection.samples, DEFAULT_BASELINE_HYPERPARAMETERS),
        dtype=float,
    )
    log_weight = log_pop - selection.log_draw_density
    direct = np.zeros_like(log_weight)
    finite = np.isfinite(log_weight)
    direct[finite] = np.exp(
        log_weight[finite] - np.max(log_weight[finite])
    )
    direct /= direct.sum()

    np.testing.assert_allclose(probabilities, direct, rtol=1e-12)
    assert np.isclose(probabilities.sum(), 1.0)
    assert diagnostics["resampling_ess"] > 1.0
    assert diagnostics["n_positive_weight"] > 0


def test_frozen_selection_null_reuses_exact_selection_and_event_count():
    observed, selection, config = _fixture(seed=2)
    null = generate_frozen_selection_null_dataset(
        seed=13,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
    )

    assert null.selection is selection
    assert null.posterior.n_events == observed.n_events
    assert null.metadata["selection_mode"] == "estimator_ready"
    assert null.metadata["resampling_ess"] >= 1.0
    assert len(null.truth_rows) == observed.n_events
    validate_pair(
        null.posterior,
        selection,
        compile_model_spec(baseline_model_spec()).required_fields,
    )


def test_frozen_selection_null_truth_rows_are_seed_deterministic():
    observed, selection, config = _fixture(seed=3)
    first = generate_frozen_selection_null_dataset(
        seed=17,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
    )
    second = generate_frozen_selection_null_dataset(
        seed=17,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
    )
    np.testing.assert_array_equal(first.truth_rows, second.truth_rows)


def test_frozen_selection_null_rejects_low_resampling_ess():
    observed, selection, config = _fixture(seed=4)
    with pytest.raises(ValueError, match="resampling ESS"):
        generate_frozen_selection_null_dataset(
            seed=18,
            observed_posterior=observed,
            selection=selection,
            truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
            survey_config=config,
            min_resampling_ess=1e12,
        )


def test_frozen_selection_null_requires_matching_observed_event_count():
    observed, selection, config = _fixture(seed=5)
    bad = SyntheticSurveyConfig(
        n_events=observed.n_events + 1,
        posterior_samples_per_event=config.posterior_samples_per_event,
        n_injections=config.n_injections,
    )
    with pytest.raises(ValueError, match="event count"):
        generate_frozen_selection_null_dataset(
            seed=19,
            observed_posterior=observed,
            selection=selection,
            truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
            survey_config=bad,
            min_resampling_ess=1.0,
        )
