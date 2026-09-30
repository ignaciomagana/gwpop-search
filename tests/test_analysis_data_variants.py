"""Importance reweighting to data variants (OD-3b: 249 events, SNR 9 / SNR 11)."""

import math

import numpy as np
import pytest
from scipy.special import logsumexp

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from gwpop_search.analysis import toys  # noqa: E402
from gwpop_search.analysis._common import (  # noqa: E402
    AnalysisInputError,
    WeightedPosterior,
    posterior_from_equal_weight_draws,
)
from gwpop_search.analysis.data_variants import (  # noqa: E402
    RERUN_POLICY,
    ReweightCriteria,
    event_run_is_pre_o3,
    make_data_variant,
    o1_o2_event_names,
    reweight_to_variant,
    subset_selection_rows,
    variant_edge_log_bayes_factor,
)
from gwpop_search.analysis.terms import BatchedCatalogTerms, pad_catalog, CatalogWeightEvaluator  # noqa: E402
from gwpop_search.hbi import HBIConfig  # noqa: E402

HBI = HBIConfig(selection_chunk_size=None)
NAMES = ("mu",)
MODEL = toys.GaussianToy(fixed={"sigma": 1.0})
EVENT_NAMES = (
    "GW150914_095045", "GW151226_033853", "GW170104_101158", "GW170814_103043",
) + tuple(f"GW19{m:02d}{d:02d}_000000" for m in range(5, 13) for d in (1, 15))


def _data(seed=0, d_th=0.5):
    rng = np.random.default_rng(seed)
    obs = toys.ToyObservation(sigma_obs=0.5, d_th=d_th, s_draw=2.0)
    d = toys.toy_observed_data(rng, lambda r, n: r.normal(0.3, 1.0, n), len(EVENT_NAMES), obs)
    pe = toys.toy_posterior_catalog(rng, d, 200, obs, names=EVENT_NAMES)
    sel = toys.toy_selection_catalog(rng, 20000, obs)
    return pe, sel, rng


def _grid_posterior(pe, sel, lo=-1.5, hi=2.0, n=1500):
    grid = np.linspace(lo, hi, n)[:, None]
    ll = BatchedCatalogTerms(pe, sel, MODEL, NAMES, hbi_config=HBI, batch_size=256).log_likelihood(grid)
    logw = ll - logsumexp(ll)
    keep = np.exp(logw) > 0
    w = np.exp(logw[keep])
    sample = WeightedPosterior(
        names=NAMES, points=grid[keep], weights=w / w.sum(), run_index=np.zeros(int(keep.sum()), dtype=int),
        n_runs=1, source="grid",
    )
    return sample, grid[keep], ll[keep]


def test_event_run_classification():
    assert event_run_is_pre_o3("GW170817_124104") and event_run_is_pre_o3("GW150914")
    assert not event_run_is_pre_o3("GW190412_053044") and not event_run_is_pre_o3("GW190401_000000")
    assert o1_o2_event_names(EVENT_NAMES) == EVENT_NAMES[:4]
    with pytest.raises(ValueError):
        event_run_is_pre_o3("S230518h")


def test_row_subset_keeps_campaign_normalisation():
    _, sel, _ = _data()
    mask = np.arange(sel.n_selected) % 3 != 0
    sub = subset_selection_rows(sel, mask, reason="test")
    assert sub.n_selected == int(mask.sum())
    assert sub.campaigns == sel.campaigns  # n_draw and T unchanged: rows dropped only
    assert sub.metadata["row_subset"]["renormalised"] is False
    with pytest.raises(ValueError):
        subset_selection_rows(sel, np.zeros(sel.n_selected, dtype=bool), reason="x")


def test_reweighting_reproduces_the_variant_posterior_and_evidence_exactly():
    pe, sel, rng = _data()
    sel_snr = toys.toy_selection_catalog(rng, 20000, toys.ToyObservation(sigma_obs=0.5, d_th=0.4, s_draw=2.0))
    variant = make_data_variant("o3o4b_249", pe, sel, drop_o1o2_events=True, selection_override=sel_snr)
    assert variant.derivation["dropped_events"] == list(EVENT_NAMES[:4])
    assert variant.posterior.n_events == len(EVENT_NAMES) - 4
    sample, grid, ll_p = _grid_posterior(pe, sel)
    out = reweight_to_variant(sample, pe, sel, MODEL, variant, hbi_config=HBI, verify_identity=False, batch_size=64,
                              criteria=ReweightCriteria(min_ess=10, borderline_ess=20))
    ll_v = BatchedCatalogTerms(variant.posterior, sel_snr, MODEL, NAMES, hbi_config=HBI, batch_size=256).log_likelihood(grid)
    # the reweighted posterior is the variant posterior on the same points (uniform prior)
    w_v = np.exp(ll_v - logsumexp(ll_v))
    mean_v = float(np.sum(w_v * grid[:, 0]))
    assert out["posterior_variant"]["mu"]["mean"] == pytest.approx(mean_v, abs=1e-10)
    # and the evidence shift is exact: ln Z' - ln Z on the grid quadrature
    expected = float(logsumexp(ll_v) - logsumexp(ll_p))
    assert out["delta_log_evidence"] == pytest.approx(expected, abs=1e-9)
    assert out["n_events_variant"] == len(EVENT_NAMES) - 4
    assert out["rerun_policy"] == RERUN_POLICY


def test_low_ess_is_flagged_and_never_rerun():
    pe, sel, rng = _data()
    grid_sample, _, _ = _grid_posterior(pe, sel)
    draws = grid_sample.equal_weight_draws(3000, seed=1)
    sample = posterior_from_equal_weight_draws(NAMES, draws)
    # a much harder detection threshold moves the posterior far from the draws
    hard = toys.toy_selection_catalog(rng, 20000, toys.ToyObservation(sigma_obs=0.5, d_th=2.0, s_draw=2.0))
    variant = make_data_variant("snr11", pe, sel, selection_override=hard)
    out = reweight_to_variant(sample, pe, sel, MODEL, variant, hbi_config=HBI, verify_identity=False)
    assert out["ess_variant"] < 1000.0
    assert out["rerun_recommended"] is True
    assert "ess_below_min" in out["rerun_reasons"]
    assert out["status"] == "rerun_recommended_operator_gated"
    # a mild variant from 3000 draws: reliable (ESS well above 1000, small k-hat)
    mild = make_data_variant("o3o4b_249", pe, sel, drop_events=EVENT_NAMES[:1])
    out = reweight_to_variant(sample, pe, sel, MODEL, mild, hbi_config=HBI, verify_identity=False)
    assert out["ess_variant"] > 1500.0 and out["pareto_khat"] < 0.5, out
    assert out["rerun_recommended"] is False
    # an unchanged dataset reweights with r = 1: no shift, full ESS
    same = make_data_variant("same", pe, sel)
    out = reweight_to_variant(sample, pe, sel, MODEL, same, hbi_config=HBI, verify_identity=False,
                              criteria=ReweightCriteria(min_ess=10, borderline_ess=20))
    assert out["delta_log_evidence"] == pytest.approx(0.0, abs=1e-12)
    assert out["ess_variant"] == pytest.approx(out["ess_primary"], rel=1e-12)
    assert out["rerun_recommended"] is False and out["status"] == "reweighting_reliable"


def test_taper_is_applied_to_both_likelihoods_and_required_when_declared():
    pe, sel, _ = _data()
    sample, grid, _ = _grid_posterior(pe, sel)
    variant = make_data_variant("snr11", pe, sel, selection_row_mask=np.arange(sel.n_selected) % 2 == 0)

    def log_taper(v):
        return -0.3 * np.asarray(v)

    out = reweight_to_variant(sample, pe, sel, MODEL, variant, hbi_config=HBI, verify_identity=False,
                              log_taper=log_taper)
    ev_p = CatalogWeightEvaluator(pad_catalog(pe, sel, MODEL, hbi_config=HBI), MODEL, NAMES)
    ev_v = CatalogWeightEvaluator(pad_catalog(variant.posterior, variant.selection, MODEL, hbi_config=HBI), MODEL, NAMES)
    mp = ev_p.moments(grid, sample.weights)
    mv = ev_v.moments(grid, np.zeros(grid.shape[0]))
    lr = (mv["log_likelihood"] - 0.3 * mv["variance"]) - (mp["log_likelihood"] - 0.3 * mp["variance"])
    expected = float(logsumexp(np.log(sample.weights) + lr))
    assert out["taper_applied"] is True
    assert out["delta_log_evidence"] == pytest.approx(expected, abs=1e-9)
    tapered = WeightedPosterior(
        names=sample.names, points=sample.points, weights=sample.weights, run_index=sample.run_index,
        n_runs=1, source="grid", likelihood_identity={"hbi_config": {"variance_taper_threshold": 1.0}},
    )
    with pytest.raises(AnalysisInputError, match="variance taper"):
        reweight_to_variant(tapered, pe, sel, MODEL, variant, hbi_config=HBI, verify_identity=False)
    with pytest.raises(AnalysisInputError, match="no likelihood identity"):
        reweight_to_variant(sample, pe, sel, MODEL, variant, hbi_config=HBI)


def test_edge_variant_bayes_factor_and_borderline_flags():
    base = {"variant_id": "snr9", "delta_log_evidence_mc_se": 0.01, "rerun_recommended": False}
    child = dict(base, model_hash="c", delta_log_evidence=-1.0)
    parent = dict(base, model_hash="p", delta_log_evidence=-0.2)
    row = variant_edge_log_bayes_factor(child, parent, 5.0)
    assert row["log_bayes_factor_variant"] == pytest.approx(4.2)
    assert row["rerun_recommended"] is False
    row = variant_edge_log_bayes_factor(child, parent, 3.5)
    assert row["log_bayes_factor_variant"] == pytest.approx(2.7)
    assert row["rerun_reasons"] == ["near_decision_threshold"]
    row = variant_edge_log_bayes_factor(dict(child, delta_log_evidence=-3.0), parent, 1.0)
    assert "sign_change" in row["rerun_reasons"]
    row = variant_edge_log_bayes_factor(dict(child, rerun_recommended=True), parent, 8.0)
    assert row["rerun_reasons"] == ["child_reweighting_flagged"]
    with pytest.raises(AnalysisInputError):
        variant_edge_log_bayes_factor(dict(child, variant_id="snr11"), parent, 1.0)
    assert math.isclose(row["reweighting_mc_se"], math.hypot(0.01, 0.01))
