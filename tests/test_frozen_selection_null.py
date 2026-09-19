import numpy as np
import pytest

from gwpop_search.data import (
    Campaign,
    PosteriorCatalog,
    SelectionCatalog,
    SelectionMode,
    validate_pair,
)
from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
    pe_truth_quantiles,
)
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS, compile_model_spec
from gwpop_search.nulls import (
    frozen_selection_resampling_probabilities,
    generate_frozen_selection_null_dataset,
    measure_observed_pe_scales,
    rank_matched_event_sigmas,
    required_resampling_ess,
    root_hyperprior_mmax_supremum,
)
from gwpop_search.nulls.pe_matching import detection_statistic_proxy


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


def _fixture(seed=1, *, n_events=5):
    source = SyntheticSurveyConfig(
        n_events=n_events,
        posterior_samples_per_event=12,
        n_injections=2000,
        population_batch_size=128,
        redshift_sampling_grid=512,
    )
    dataset = generate_baseline_synthetic_dataset(
        seed=seed,
        config=source,
    )
    selection = _estimator_ready_selection(dataset.selection)
    # Null PE: the declared noisy-observation approximation (truth-centred PE is refused).
    config = SyntheticSurveyConfig(
        n_events=n_events,
        posterior_samples_per_event=12,
        n_injections=2000,
        population_batch_size=128,
        redshift_sampling_grid=512,
        observation_model="noisy_observation",
    )
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
        min_resampling_ess_per_event=1.0,
        pe_scale_policy="declared_fixed",
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
        min_resampling_ess_per_event=1.0,
        pe_scale_policy="declared_fixed",
    )
    second = generate_frozen_selection_null_dataset(
        seed=17,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
        min_resampling_ess_per_event=1.0,
        pe_scale_policy="declared_fixed",
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
            min_resampling_ess_per_event=1.0,
            pe_scale_policy="declared_fixed",
        )


def test_frozen_selection_null_requires_matching_observed_event_count():
    observed, selection, config = _fixture(seed=5)
    bad = SyntheticSurveyConfig(
        n_events=observed.n_events + 1,
        posterior_samples_per_event=config.posterior_samples_per_event,
        n_injections=config.n_injections,
        observation_model="noisy_observation",
    )
    with pytest.raises(ValueError, match="event count"):
        generate_frozen_selection_null_dataset(
            seed=19,
            observed_posterior=observed,
            selection=selection,
            truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
            survey_config=bad,
            min_resampling_ess=1.0,
            min_resampling_ess_per_event=1.0,
            pe_scale_policy="declared_fixed",
        )


def test_frozen_selection_null_refuses_truth_centred_pe_and_injection_options():
    observed, selection, config = _fixture(seed=6)
    for survey, match in (
        (
            SyntheticSurveyConfig(
                n_events=config.n_events,
                posterior_samples_per_event=config.posterior_samples_per_event,
            ),
            "noisy_observation",
        ),
        (
            SyntheticSurveyConfig(
                n_events=config.n_events,
                posterior_samples_per_event=config.posterior_samples_per_event,
                observation_model="noisy_observation",
                injection_draw="population_proxy",
            ),
            "do not apply",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            generate_frozen_selection_null_dataset(
                seed=21,
                observed_posterior=observed,
                selection=selection,
                truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
                survey_config=survey,
                min_resampling_ess=1.0,
                min_resampling_ess_per_event=1.0,
                pe_scale_policy="declared_fixed",
            )


def test_frozen_selection_null_pe_is_drawn_given_one_noisy_observation():
    observed, selection, config = _fixture(seed=7, n_events=40)
    null = generate_frozen_selection_null_dataset(
        seed=23,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
        min_resampling_ess_per_event=0.5,
        pe_scale_policy="declared_fixed",
    )
    approximation = null.metadata["pe_approximation"]
    assert null.metadata["format_version"] == "gwpop-search-frozen-selection-null-2.1"
    assert approximation["observation_model"] == "noisy_observation"
    assert approximation["detection_noise_link"] == "not_linked"
    declared = {item["name"]: item["detail"] for item in approximation["declared_approximations"]}
    assert "not conditioned on the detection statistic" in declared["detection_noise_not_linked"]
    assert approximation["observation_noise_sigma"]["chi_eff"] == config.pe_chi_eff_sigma

    truths = {
        name: np.asarray(selection.samples[name])[null.truth_rows]
        for name in ("chi_eff", "q")
    }
    # One observation per event; PE is centred on it, not on the truth: the truth's
    # quantile inside its own PE varies across events (zero-noise PE puts it at ~0.5).
    assert set(null.event_observations) == {
        "log_m1_detector", "q", "log_luminosity_distance", "chi_eff",
    }
    assert null.event_observations["chi_eff"].shape == (40,)
    assert not np.allclose(null.event_observations["chi_eff"], truths["chi_eff"])
    quantiles = pe_truth_quantiles(null.posterior, truths, "chi_eff")
    assert np.std(quantiles) > 0.15
    assert np.mean((quantiles > 0.1) & (quantiles < 0.9)) < 0.95
    for i in range(3):
        sl = null.posterior.event_slice(i)
        pe = np.asarray(null.posterior.samples["chi_eff"])[sl]
        assert abs(np.mean(pe) - null.event_observations["chi_eff"][i]) < 0.1
    # The PE prior (reference density) is the stored uniform detector box.
    assert np.all(np.isfinite(null.posterior.log_ref_density))
    assert np.unique(null.posterior.log_ref_density).size == 1


# ---------------------------------------------------------------------------
# Null PE precision tied to the frozen observed catalog (review finding 1)
# ---------------------------------------------------------------------------


def _event_widths(posterior):
    """Per-event marginal widths in the four measurement coordinates."""
    out = {name: [] for name in ("log_m1_detector", "q", "log_luminosity_distance", "chi_eff")}
    for index in range(posterior.n_events):
        sl = posterior.event_slice(index)
        out["log_m1_detector"].append(
            np.std(np.log(np.asarray(posterior.samples["m1_detector"])[sl]), ddof=1)
        )
        out["q"].append(np.std(np.asarray(posterior.samples["q"])[sl], ddof=1))
        out["log_luminosity_distance"].append(
            np.std(
                np.log(np.asarray(posterior.samples["luminosity_distance"])[sl]),
                ddof=1,
            )
        )
        out["chi_eff"].append(
            np.std(np.asarray(posterior.samples["chi_eff"])[sl], ddof=1)
        )
    return {name: np.asarray(values) for name, values in out.items()}


def test_observed_pe_scales_measure_each_event_and_refuse_ragged_counts():
    observed, _, _ = _fixture(seed=31, n_events=6)
    scales = measure_observed_pe_scales(observed)

    assert scales.n_events == observed.n_events
    assert scales.uniform_samples_per_event() == 12
    measured = _event_widths(observed)
    for name in ("log_m1_detector", "q", "log_luminosity_distance", "chi_eff"):
        np.testing.assert_allclose(scales.sigmas[name], measured[name], rtol=1e-12)

    ragged = PosteriorCatalog(
        event_names=observed.event_names,
        offsets=np.concatenate(
            [observed.offsets[:1], observed.offsets[1:] - np.arange(observed.n_events)]
        ),
        samples={
            name: values[: int(observed.offsets[-1]) - observed.n_events + 1]
            for name, values in observed.samples.items()
        },
        log_ref_density=observed.log_ref_density[
            : int(observed.offsets[-1]) - observed.n_events + 1
        ],
        basis=observed.basis,
    )
    with pytest.raises(ValueError, match="ragged"):
        measure_observed_pe_scales(ragged).uniform_samples_per_event()


def test_null_pe_precision_matches_the_frozen_observed_catalog():
    """match_observed reproduces the observed width multiset, declared_fixed does not."""
    observed, selection, config = _fixture(seed=32, n_events=24)
    observed_widths = _event_widths(observed)

    matched = generate_frozen_selection_null_dataset(
        seed=41,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
        min_resampling_ess_per_event=0.5,
        pe_scale_policy="match_observed",
    )
    approximation = matched.metadata["pe_approximation"]
    assert approximation["pe_scale_policy"] == "match_observed"
    applied = approximation["pe_scales"]["applied_scales"]
    # The applied per-event scales are exactly the observed widths, reordered.
    for name in ("log_m1_detector", "q", "log_luminosity_distance", "chi_eff"):
        observed_median = float(np.median(observed_widths[name]))
        assert applied[name]["median"] == pytest.approx(observed_median, rel=1e-12)
        assert applied[name]["min"] == pytest.approx(
            float(observed_widths[name].min()), rel=1e-12
        )
        assert applied[name]["max"] == pytest.approx(
            float(observed_widths[name].max()), rel=1e-12
        )
    # ... and the drawn null PE really is that much wider than the fixed default.
    null_widths = _event_widths(matched.posterior)
    for name in ("log_m1_detector", "q", "log_luminosity_distance"):
        ratio = float(np.median(null_widths[name]) / np.median(observed_widths[name]))
        assert 0.5 < ratio < 2.0

    fixed = generate_frozen_selection_null_dataset(
        seed=41,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
        min_resampling_ess_per_event=0.5,
        pe_scale_policy="declared_fixed",
    )
    fixed_approximation = fixed.metadata["pe_approximation"]
    assert fixed_approximation["pe_scale_policy"] == "declared_fixed"
    assert fixed_approximation["observation_noise_sigma"]["q"] == config.pe_q_sigma
    # The mismatch is measured and declared, never silent.
    comparison = fixed_approximation["pe_scales"]["declared_vs_observed"]
    assert comparison["posterior_width"]["q"]["declared"] == config.pe_q_sigma
    assert comparison["posterior_width"]["q"]["observed_median"] == pytest.approx(
        float(np.median(observed_widths["q"])), rel=1e-12
    )
    declared = {item["name"] for item in fixed_approximation["declared_approximations"]}
    assert "pe_precision_mismatch" in declared
    assert "marginal_pe_widths_only" in declared
    assert "pe_precision_mismatch" not in {
        item["name"] for item in approximation["declared_approximations"]
    }


def test_matched_null_pe_precision_tracks_event_loudness():
    """The matched widths follow the detection-statistic proxy, as in the data."""
    observed, selection, _ = _fixture(seed=33, n_events=24)
    scales = measure_observed_pe_scales(observed)
    truths = {
        "m1_detector": np.asarray(selection.samples["m1_detector"])[:24],
        "q": np.asarray(selection.samples["q"])[:24],
        "luminosity_distance": np.asarray(selection.samples["luminosity_distance"])[:24],
    }
    sigmas, provenance = rank_matched_event_sigmas(scales, truths)

    null_loudness = detection_statistic_proxy(
        truths["m1_detector"], truths["q"], truths["luminosity_distance"]
    )
    for name in sigmas:
        np.testing.assert_array_equal(
            np.sort(sigmas[name]), np.sort(scales.sigmas[name])
        )
        # Rank correlation of the assignment is exactly that of the observed data.
        null_rank = np.argsort(np.argsort(null_loudness))
        observed_rank = np.argsort(np.argsort(scales.loudness))
        assert np.corrcoef(
            null_rank, np.argsort(np.argsort(sigmas[name]))
        )[0, 1] == pytest.approx(
            np.corrcoef(observed_rank, np.argsort(np.argsort(scales.sigmas[name])))[0, 1],
            abs=1e-12,
        )
    assert provenance["policy"] == "match_observed"


def test_matched_null_pe_requires_the_observed_sample_count():
    observed, selection, config = _fixture(seed=34, n_events=6)
    thin = SyntheticSurveyConfig(
        n_events=config.n_events,
        posterior_samples_per_event=config.posterior_samples_per_event // 2,
        n_injections=config.n_injections,
        population_batch_size=config.population_batch_size,
        redshift_sampling_grid=config.redshift_sampling_grid,
        observation_model="noisy_observation",
    )
    with pytest.raises(ValueError, match="posterior_samples_per_event=12"):
        generate_frozen_selection_null_dataset(
            seed=42,
            observed_posterior=observed,
            selection=selection,
            truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
            survey_config=thin,
            min_resampling_ess=1.0,
            min_resampling_ess_per_event=0.5,
            pe_scale_policy="match_observed",
        )


# ---------------------------------------------------------------------------
# Catalog-scaled resampling gate (review finding 2)
# ---------------------------------------------------------------------------


def test_resampling_ess_gate_scales_with_the_catalog():
    assert required_resampling_ess(
        259, min_resampling_ess=200.0, min_resampling_ess_per_event=10.0
    ) == 2590.0
    assert required_resampling_ess(
        5, min_resampling_ess=200.0, min_resampling_ess_per_event=10.0
    ) == 200.0

    observed, selection, config = _fixture(seed=35, n_events=6)
    # An absolute floor below the measured ESS is not enough on its own: the
    # catalog-scaled part of the gate still refuses a pool smaller than 10x
    # the catalog, which is what the old absolute-only gate let through.
    _, diagnostics = frozen_selection_resampling_probabilities(
        selection, baseline_model_spec(), DEFAULT_BASELINE_HYPERPARAMETERS
    )
    assert diagnostics["resampling_ess"] > 6.0
    with pytest.raises(ValueError, match="resampling ESS is below the declared gate"):
        generate_frozen_selection_null_dataset(
            seed=43,
            observed_posterior=observed,
            selection=selection,
            truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
            survey_config=config,
            min_resampling_ess=1.0,
            min_resampling_ess_per_event=10.0,
            pe_scale_policy="declared_fixed",
        )


def test_null_metadata_records_the_truth_pool():
    observed, selection, config = _fixture(seed=36, n_events=8)
    null = generate_frozen_selection_null_dataset(
        seed=44,
        observed_posterior=observed,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
        min_resampling_ess_per_event=1.0,
        pe_scale_policy="declared_fixed",
    )
    assert null.metadata["min_resampling_ess_per_event"] == 1.0
    assert null.metadata["required_resampling_ess"] == 8.0
    assert null.metadata["unique_truth_fraction"] == pytest.approx(
        null.metadata["n_unique_truth_rows"] / observed.n_events
    )


# ---------------------------------------------------------------------------
# PE prior box follows the searched root hyperprior (review finding 8)
# ---------------------------------------------------------------------------


def test_null_pe_prior_box_follows_the_root_hyperprior_profile():
    observed, selection, config = _fixture(seed=37, n_events=6)
    boxes = {}
    for profile in ("phase3", "gwtc5-v1"):
        spec = baseline_model_spec(profile)
        null = generate_frozen_selection_null_dataset(
            seed=45,
            observed_posterior=observed,
            selection=selection,
            truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
            survey_config=config,
            min_resampling_ess=1.0,
            min_resampling_ess_per_event=1.0,
            pe_scale_policy="declared_fixed",
            model_spec=spec,
        )
        approximation = null.metadata["pe_approximation"]
        boxes[profile] = approximation["prior_box"]["m1_max"]
        expected_sup = root_hyperprior_mmax_supremum(spec)
        assert approximation["prior_box_mmax_supremum"] == expected_sup
        model = compile_model_spec(spec)
        assert approximation["prior_box"]["m1_max"] >= (
            1.1 * expected_sup * (1.0 + model.zmax)
        )
    # gwtc5-v1 (mmax ~ U(60, 200)) needs a strictly larger box than phase3.
    assert boxes["gwtc5-v1"] > boxes["phase3"]


def test_root_hyperprior_mmax_supremum_refuses_an_unbounded_prior():
    from dataclasses import replace as dataclass_replace

    from gwpop_search.grammar import PriorConfig

    spec = baseline_model_spec()
    priors = dict(spec.priors)
    priors["mmax"] = PriorConfig(family="normal", parameters={"loc": 90.0, "scale": 10.0})
    unbounded = dataclass_replace(spec, priors=priors)
    with pytest.raises(ValueError, match="unbounded"):
        root_hyperprior_mmax_supremum(unbounded)


def _widened_observed(posterior, factors):
    """Copy of ``posterior`` whose event ``i`` is ``factors[i]`` times wider.

    Masses and distances are widened in the log, q and chi_eff linearly, so the
    catalog stays positive and inside its coordinate ranges while carrying a
    per-event PE precision that the fixed synthetic noise model does not have.
    """
    samples = {name: np.array(values, dtype=float) for name, values in posterior.samples.items()}
    for index, factor in enumerate(factors):
        sl = posterior.event_slice(index)
        for name in ("m1_detector", "luminosity_distance"):
            x = np.log(samples[name][sl])
            samples[name][sl] = np.exp(x.mean() + factor * (x - x.mean()))
        for name, (low, high) in (("q", (1e-2, 1.0)), ("chi_eff", (-1.0, 1.0))):
            x = samples[name][sl]
            samples[name][sl] = np.clip(x.mean() + factor * (x - x.mean()), low, high)
    return PosteriorCatalog(
        event_names=posterior.event_names,
        offsets=posterior.offsets,
        samples=samples,
        log_ref_density=posterior.log_ref_density,
        basis=posterior.basis,
    )


def test_matched_null_pe_reproduces_a_heteroscedastic_observed_catalog():
    """The real catalog is 1.6-2.9x wider than the fixed synthetic scales.

    With a deliberately widened observed catalog, ``match_observed`` reproduces
    its measurement precision while ``declared_fixed`` keeps the configured
    scales: exactly the mismatch the null calibration must not have.
    """
    observed, selection, config = _fixture(seed=38, n_events=24)
    factors = np.linspace(1.6, 2.9, observed.n_events)
    widened = _widened_observed(observed, factors)
    observed_widths = _event_widths(widened)

    shared = dict(
        observed_posterior=widened,
        selection=selection,
        truth_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        survey_config=config,
        min_resampling_ess=1.0,
        min_resampling_ess_per_event=0.5,
    )
    matched = generate_frozen_selection_null_dataset(
        seed=46, pe_scale_policy="match_observed", **shared
    )
    fixed = generate_frozen_selection_null_dataset(
        seed=46, pe_scale_policy="declared_fixed", **shared
    )
    matched_widths = _event_widths(matched.posterior)
    fixed_widths = _event_widths(fixed.posterior)

    applied = matched.metadata["pe_approximation"]["pe_scales"]["applied_scales"]
    for name in ("log_m1_detector", "q", "log_luminosity_distance", "chi_eff"):
        observed_median = float(np.median(observed_widths[name]))
        # The applied measurement scales are the observed widths exactly ...
        assert applied[name]["median"] == pytest.approx(observed_median, rel=1e-12)
        # ... and the drawn PE reproduces them up to the declared truncation of
        # the exact posterior at the PE prior box (q and chi_eff are bounded).
        matched_ratio = float(np.median(matched_widths[name])) / observed_median
        fixed_ratio = float(np.median(fixed_widths[name])) / observed_median
        assert 0.7 < matched_ratio < 1.3
        # The fixed synthetic scales are far too sharp, as on the real catalog.
        assert fixed_ratio < 1.0 / 1.4
        assert abs(np.log(matched_ratio)) < abs(np.log(fixed_ratio))

    comparison = fixed.metadata["pe_approximation"]["pe_scales"]["declared_vs_observed"]
    assert comparison["max_width_ratio"] > 1.4
