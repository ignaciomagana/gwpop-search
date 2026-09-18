import numpy as np

from gwpop_search.data import validate_pair
from gwpop_search.data.adapters import gwcat_v2_basis_for_spin
from gwpop_search.hbi import shape_log_likelihood
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    detection_mask,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import GwcatChiEffBBHModel


def _small_config():
    return SyntheticSurveyConfig(
        n_events=12,
        posterior_samples_per_event=32,
        n_injections=2_000,
        population_batch_size=512,
        redshift_sampling_grid=2048,
    )


def test_synthetic_dataset_uses_exact_gwcat_chieff_basis_and_raw_selection():
    dataset = generate_baseline_synthetic_dataset(
        seed=1234,
        config=_small_config(),
    )
    basis = gwcat_v2_basis_for_spin("chieff")

    assert dataset.posterior.basis.identity == basis.identity
    assert dataset.selection.basis.identity == basis.identity
    validate_pair(
        dataset.posterior,
        dataset.selection,
        GwcatChiEffBBHModel.required_fields,
    )

    assert dataset.posterior.n_events == 12
    assert dataset.posterior.n_samples_total == 12 * 32
    assert dataset.selection.campaigns[0].n_draw == 2_000
    assert dataset.selection.campaigns[0].observing_time_yr == 1.0
    assert dataset.selection.n_selected > 0
    assert np.all(np.isfinite(dataset.posterior.log_ref_density))
    assert np.all(np.isfinite(dataset.selection.log_draw_density))


def test_synthetic_truths_satisfy_the_same_detection_rule_as_selection():
    config = _small_config()
    dataset = generate_baseline_synthetic_dataset(
        seed=91,
        config=config,
    )
    assert np.all(detection_mask(dataset.event_truths, config))


def test_synthetic_reference_and_draw_densities_are_explicit_uniform_constants():
    dataset = generate_baseline_synthetic_dataset(
        seed=8,
        config=_small_config(),
    )
    assert np.unique(dataset.posterior.log_ref_density).size == 1
    assert np.unique(dataset.selection.log_draw_density).size == 1


def test_synthetic_generation_is_deterministic_for_fixed_seed():
    config = _small_config()
    a = generate_baseline_synthetic_dataset(seed=55, config=config)
    b = generate_baseline_synthetic_dataset(seed=55, config=config)

    for field in a.posterior.field_names:
        np.testing.assert_allclose(a.posterior.samples[field], b.posterior.samples[field])
    for field in a.selection.field_names:
        np.testing.assert_allclose(a.selection.samples[field], b.selection.samples[field])
    np.testing.assert_allclose(a.posterior.log_ref_density, b.posterior.log_ref_density)
    np.testing.assert_allclose(a.selection.log_draw_density, b.selection.log_draw_density)


def test_injected_baseline_has_finite_end_to_end_hbi_likelihood():
    dataset = generate_baseline_synthetic_dataset(
        seed=2026,
        config=_small_config(),
    )
    model = GwcatChiEffBBHModel()
    result = shape_log_likelihood(
        dataset.posterior,
        dataset.selection,
        model,
        dataset.truth_hyperparameters,
    )

    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.terms.selection.log_exposure)
    assert result.terms.selection.exposure > 0.0
    assert all(diag.ess > 0.0 for diag in result.terms.events.diagnostics)
    assert result.terms.selection.diagnostics.ess > 0.0
