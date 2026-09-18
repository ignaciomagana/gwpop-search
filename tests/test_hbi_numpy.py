import numpy as np
import pytest
from scipy.special import logsumexp

from gwpop_search.data import Campaign, CoordinateBasis, PosteriorCatalog, SelectionCatalog, SelectionMode
from gwpop_search.data.fixtures import make_toy_posterior_catalog, make_toy_selection_catalog
from gwpop_search.hbi import (
    HBIConfig,
    PopulationDensityError,
    SelectionSupportError,
    evaluate_events,
    evaluate_selection,
    poisson_log_likelihood,
    shape_log_likelihood,
)


def constant_log_density(samples, hp):
    return np.full_like(next(iter(samples.values())), float(hp), dtype=float)


def test_event_importance_average_constant_ratio():
    pe = make_toy_posterior_catalog()
    out = evaluate_events(pe, constant_log_density, np.log(2.5))
    np.testing.assert_allclose(out.log_likelihoods, np.log(2.5), rtol=0, atol=1e-14)
    assert all(np.isclose(d.ess, pe.sample_count(i)) for i, d in enumerate(out.diagnostics))
    assert all(np.isclose(d.max_weight_fraction, 1 / pe.sample_count(i)) for i, d in enumerate(out.diagnostics))


def test_raw_selection_exposure_uses_time_and_ndraw():
    sel = make_toy_selection_catalog()
    logc = np.log(3.0)
    out = evaluate_selection(sel, constant_log_density, logc)
    expected = 3.0 * (0.8 * 9 / 1000 + 1.2 * 13 / 1600)
    np.testing.assert_allclose(np.exp(out.log_exposure), expected, rtol=1e-14)
    assert len(out.campaigns) == 2
    np.testing.assert_allclose(np.exp(out.campaigns[0].log_efficiency), 3.0 * 9 / 1000)
    np.testing.assert_allclose(np.exp(out.campaigns[1].log_efficiency), 3.0 * 13 / 1600)


def test_raw_selection_can_explicitly_disable_time_weighting():
    sel = make_toy_selection_catalog()
    out = evaluate_selection(sel, constant_log_density, 0.0, raw_use_observing_time=False)
    expected = 9 / 1000 + 13 / 1600
    np.testing.assert_allclose(np.exp(out.log_exposure), expected, rtol=1e-14)


def test_estimator_ready_uses_sum_weights_without_second_ndraw_factor():
    base = make_toy_selection_catalog()
    pdraw = np.linspace(0.2, 1.1, base.n_selected)
    sel = SelectionCatalog(
        samples=base.samples,
        log_draw_density=np.log(pdraw),
        campaign_id=np.asarray(["combined"] * base.n_selected),
        campaigns=(Campaign("combined", n_draw=5000, observing_time_yr=1.5),),
        basis=base.basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics="already normalized pdraw",
    )
    pop = np.linspace(0.3, 0.9, base.n_selected)
    def density(samples, hp):
        return np.log(hp)
    out = evaluate_selection(sel, density, pop)
    direct = np.sum(pop / pdraw)
    np.testing.assert_allclose(np.exp(out.log_exposure), direct, rtol=1e-14)
    assert not np.isclose(np.exp(out.log_exposure), direct / 5000)


def test_shape_and_poisson_formulas():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    terms_events = evaluate_events(pe, constant_log_density, 0.0)
    terms_sel = evaluate_selection(sel, constant_log_density, 0.0)
    shape = shape_log_likelihood(pe, sel, constant_log_density, 0.0)
    expected_shape = terms_events.log_likelihood - pe.n_events * terms_sel.log_exposure
    np.testing.assert_allclose(shape.log_likelihood, expected_shape, rtol=0, atol=1e-14)

    rate = 4.2
    ppp = poisson_log_likelihood(pe, sel, constant_log_density, 0.0, rate=rate)
    expected_ppp = terms_events.log_likelihood + pe.n_events * np.log(rate) - rate * np.exp(terms_sel.log_exposure)
    np.testing.assert_allclose(ppp.log_likelihood, expected_ppp, rtol=0, atol=1e-14)


def test_selection_chunking_is_invariant():
    sel = make_toy_selection_catalog()
    def density(samples, hp):
        return -0.03 * samples["m1_source"] + hp * samples["chi_eff"]
    a = evaluate_selection(sel, density, 0.2, chunk_size=None)
    for size in (1, 4, 7, 100):
        b = evaluate_selection(sel, density, 0.2, chunk_size=size)
        np.testing.assert_allclose(b.log_exposure, a.log_exposure, rtol=0, atol=1e-13)


def test_permutation_invariance_within_events_and_selection():
    rng = np.random.default_rng(123)
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    def density(samples, hp):
        return -0.01 * samples["m1_source"] + 0.4 * samples["q"] + hp * samples["chi_eff"]
    base = shape_log_likelihood(pe, sel, density, 0.3).log_likelihood

    pe_samples = {k: v.copy() for k, v in pe.samples.items()}
    pe_ref = pe.log_ref_density.copy()
    for i in range(pe.n_events):
        sl = pe.event_slice(i)
        perm = rng.permutation(pe.sample_count(i))
        for k in pe_samples:
            pe_samples[k][sl] = pe_samples[k][sl][perm]
        pe_ref[sl] = pe_ref[sl][perm]
    pe2 = PosteriorCatalog(pe.event_names, pe.offsets, pe_samples, pe_ref, pe.basis)

    perm = rng.permutation(sel.n_selected)
    sel2 = SelectionCatalog(
        samples={k: v[perm] for k, v in sel.samples.items()},
        log_draw_density=sel.log_draw_density[perm],
        campaign_id=sel.campaign_id[perm],
        campaigns=sel.campaigns,
        basis=sel.basis,
        mode=sel.mode,
    )
    shuffled = shape_log_likelihood(pe2, sel2, density, 0.3).log_likelihood
    np.testing.assert_allclose(shuffled, base, rtol=0, atol=1e-13)


def test_bad_population_values_fail_but_minus_inf_support_is_allowed():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    def bad(samples, hp):
        out = np.zeros_like(next(iter(samples.values())))
        out.flat[0] = np.nan
        return out
    with pytest.raises(PopulationDensityError):
        evaluate_events(pe, bad)

    def zero(samples, hp):
        return np.full_like(next(iter(samples.values())), -np.inf)
    events = evaluate_events(pe, zero)
    assert np.all(np.isneginf(events.log_likelihoods))
    with pytest.raises(SelectionSupportError):
        evaluate_selection(sel, zero)


def test_likelihood_variance_diagnostics_are_finite_and_nonnegative():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    def density(samples, hp):
        return hp * samples["chi_eff"] - 0.01 * samples["m1_source"]
    out = shape_log_likelihood(pe, sel, density, 0.5)
    v = out.terms.variance
    assert np.isfinite(v.event_variance) and v.event_variance >= 0
    assert np.isfinite(v.selection_variance) and v.selection_variance >= 0
    assert np.isclose(v.shape_log_likelihood_variance, v.event_variance + pe.n_events**2 * v.selection_variance)


def test_discrete_toy_recovers_known_population_probability():
    rng = np.random.default_rng(44)
    truth = 0.72
    x = rng.binomial(1, truth, 120).astype(float)
    basis = CoordinateBasis("binary", ("x",), "none", "none", "counting", "1")
    pe = PosteriorCatalog(
        event_names=tuple(f"E{i}" for i in range(x.size)),
        offsets=np.arange(x.size + 1),
        samples={"x": x},
        log_ref_density=np.full(x.size, np.log(0.5)),
        basis=basis,
    )
    sel = SelectionCatalog(
        samples={"x": np.array([0.0, 1.0])},
        log_draw_density=np.full(2, np.log(0.5)),
        campaign_id=np.array(["all", "all"]),
        campaigns=(Campaign("all", n_draw=2, observing_time_yr=1.0),),
        basis=basis,
        mode=SelectionMode.RAW_DRAW,
    )
    def bernoulli(samples, p):
        return np.where(samples["x"] == 1.0, np.log(p), np.log1p(-p))
    grid = np.linspace(0.55, 0.85, 61)
    ll = np.array([shape_log_likelihood(pe, sel, bernoulli, p).log_likelihood for p in grid])
    p_hat = grid[np.argmax(ll)]
    assert abs(p_hat - x.mean()) <= 0.006
    assert abs(p_hat - truth) < 0.08
