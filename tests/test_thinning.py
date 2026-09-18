import numpy as np

from gwpop_search.data import (
    thin_posterior_catalog,
    thin_selection_catalog,
)
from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.hbi import evaluate_selection


def _constant_log_density(samples, hyperparameters=None):
    n = next(iter(samples.values())).size
    return np.zeros(n, dtype=float)


def test_posterior_thinning_is_deterministic_and_preserves_events():
    posterior = make_toy_posterior_catalog()
    a = thin_posterior_catalog(
        posterior,
        max_samples_per_event=4,
        seed=123,
    )
    b = thin_posterior_catalog(
        posterior,
        max_samples_per_event=4,
        seed=123,
    )
    c = thin_posterior_catalog(
        posterior,
        max_samples_per_event=4,
        seed=124,
    )

    assert a.event_names == posterior.event_names
    assert all(a.sample_count(i) <= 4 for i in range(a.n_events))
    np.testing.assert_array_equal(a.offsets, b.offsets)
    for name in a.samples:
        np.testing.assert_array_equal(a.samples[name], b.samples[name])
    assert any(
        not np.array_equal(a.samples[name], c.samples[name])
        for name in a.samples
    )


def test_selection_thinning_preserves_constant_weight_exposure_exactly():
    selection = make_toy_selection_catalog()
    full = evaluate_selection(selection, _constant_log_density)
    reduced = thin_selection_catalog(
        selection,
        max_selected_per_campaign=4,
        seed=99,
    )
    thinned = evaluate_selection(reduced, _constant_log_density)

    assert reduced.n_selected == 8
    assert reduced.campaigns == selection.campaigns
    assert np.isclose(thinned.log_exposure, full.log_exposure, atol=1e-12)


def test_selection_thinning_is_deterministic_and_records_correction():
    selection = make_toy_selection_catalog()
    a = thin_selection_catalog(
        selection,
        max_selected_per_campaign=5,
        seed=7,
    )
    b = thin_selection_catalog(
        selection,
        max_selected_per_campaign=5,
        seed=7,
    )

    np.testing.assert_array_equal(a.campaign_id, b.campaign_id)
    np.testing.assert_allclose(a.log_draw_density, b.log_draw_density)
    assert a.metadata["screening_thin"]["inclusion_probability_corrected"] is True
    assert a.metadata["screening_thin"]["source_n_selected"] == selection.n_selected
