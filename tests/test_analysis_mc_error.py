import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gwpop_search.analysis import toys  # noqa: E402
from gwpop_search.analysis._common import (  # noqa: E402
    AnalysisInputError,
    WeightedPosterior,
    identity_for,
    posterior_from_equal_weight_draws,
)
from gwpop_search.analysis.edge_mc_error import (  # noqa: E402
    InsufficientReweightingESSError,
    bootstrap_counts,
    bootstrap_edge_mc_error,
    compute_model_mc_weights,
    edge_mc_error,
    load_model_mc_weights,
    mc_covariance,
    mc_covariance_matrix,
    save_model_mc_weights,
)
from gwpop_search.analysis.terms import (  # noqa: E402
    BatchedCatalogTerms,
    CatalogWeightEvaluator,
    pad_catalog,
)
from gwpop_search.data import Campaign, SelectionCatalog, SelectionMode  # noqa: E402
from gwpop_search.data.fixtures import (  # noqa: E402
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.hbi import (  # noqa: E402
    HBIConfig,
    PopulationDensityError,
    evaluate_catalog_terms,
    shape_log_likelihood,
)
from gwpop_search.inference import build_batched_log_likelihood  # noqa: E402

NAMES = ("a", "b", "mcut")
HBI = HBIConfig(selection_chunk_size=None)


class ToyDensity:
    """Unnormalized toy density with a hard m1 cut (zero support above mcut)."""

    required_fields = ("m1_source", "q", "z", "chi_eff")

    def __init__(self, scale=1.0, nan_above=None):
        self.scale = scale
        self.nan_above = nan_above

    def __call__(self, samples, hp):
        xp = jnp if any(isinstance(v, jax.Array) for v in (hp["a"], samples["q"])) else np
        value = (
            -0.02 * samples["m1_source"]
            + self.scale * hp["a"] * samples["q"]
            + hp["b"] * samples["chi_eff"]
            - 0.1 * samples["z"]
        )
        value = xp.where(samples["m1_source"] <= hp["mcut"], value, -xp.inf)
        if self.nan_above is not None:
            value = xp.where(hp["a"] > self.nan_above, xp.nan, value)
        return value

    def to_config(self):
        return {"class": "ToyDensity", "scale": self.scale, "nan_above": self.nan_above}


def estimator_ready_selection():
    raw = make_toy_selection_catalog()
    return SelectionCatalog(
        samples=raw.samples,
        log_draw_density=raw.log_draw_density + np.log(1000.0),
        campaign_id=raw.campaign_id,
        campaigns=(Campaign("O3", n_draw=None), Campaign("O4", n_draw=900)),
        basis=raw.basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics="toy-estimator-ready",
    )


POINTS = np.array(
    [[0.4, -0.25, 100.0], [0.1, 0.35, 100.0], [2.0, 1.5, 45.0], [-1.0, 0.8, 60.0], [0.7, -2.0, 70.0]]
)


def test_batched_terms_reproduce_the_dynesty_likelihood_and_numpy_reference():
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()
    model = ToyDensity()
    terms = BatchedCatalogTerms(pe, sel, model, NAMES, hbi_config=HBI, batch_size=3)
    events, exposure = terms(POINTS)
    assert events.shape == (5, 3) and exposure.shape == (5,)
    loglike = build_batched_log_likelihood(pe, sel, model, NAMES, hbi_config=HBI, batch_size=4)
    np.testing.assert_allclose(terms.log_likelihood(POINTS), loglike(POINTS), rtol=0, atol=1e-11)
    for row, ev, ex in zip(POINTS, events, exposure):
        ref = evaluate_catalog_terms(pe, sel, model, dict(zip(NAMES, row)), config=HBI)
        np.testing.assert_allclose(ev, ref.events.log_likelihoods, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ex, ref.selection.log_exposure, rtol=0, atol=1e-12)
    # zero support stays -inf, NaN raises
    ev, _ = terms(np.array([[0.4, -0.25, 12.0]]))
    assert np.isneginf(ev).any()
    bad = BatchedCatalogTerms(pe, sel, ToyDensity(nan_above=5.0), NAMES, hbi_config=HBI)
    with pytest.raises(PopulationDensityError):
        bad(np.array([[6.0, 0.0, 100.0]]))


@pytest.mark.parametrize("selection_kind", ["raw_draw", "estimator_ready"])
def test_weight_evaluator_jax_matches_numpy_and_the_repository_variance(selection_kind):
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog() if selection_kind == "raw_draw" else estimator_ready_selection()
    model = ToyDensity()
    cat = pad_catalog(pe, sel, model, hbi_config=HBI, selection_capacity=40, pe_capacity=9)
    W = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
    jax_out = CatalogWeightEvaluator(cat, model, NAMES, batch_size=2).moments(POINTS, W)
    np_out = CatalogWeightEvaluator(cat, model, NAMES, backend="numpy").moments(POINTS, W)
    for key in jax_out:
        np.testing.assert_allclose(np.asarray(jax_out[key], dtype=float), np.asarray(np_out[key], dtype=float),
                                   rtol=1e-11, atol=1e-13)
    # pointwise V is the repository's shape-likelihood variance diagnostic
    for k, row in enumerate(POINTS):
        ref = evaluate_catalog_terms(pe, sel, model, dict(zip(NAMES, row)), config=HBI)
        np.testing.assert_allclose(jax_out["variance"][k], ref.variance.shape_log_likelihood_variance, rtol=1e-10)
        np.testing.assert_allclose(
            jax_out["log_likelihood"][k],
            shape_log_likelihood(pe, sel, model, dict(zip(NAMES, row)), config=HBI).log_likelihood,
            rtol=0, atol=1e-11,
        )


def _per_draw_omegas(cat, model, X):
    """Explicit per-draw normalized weights (NumPy reference)."""
    ev = CatalogWeightEvaluator(cat, model, NAMES, backend="numpy")
    lw_e, lu, _ = ev._numpy_log_weights(X)
    om_e = np.exp(lw_e - np.logaddexp.reduce(lw_e, axis=2, keepdims=True))
    om_s = np.exp(lu - np.logaddexp.reduce(lu, axis=1, keepdims=True))
    frac = om_s @ cat.campaign_onehot
    return np.nan_to_num(om_e), np.nan_to_num(om_s), frac


def _brute_covariance(cat, om_ea, om_sa, fa, om_eb, om_sb, fb):
    n = cat.n_events
    events = np.sum(np.sum(om_ea * om_eb, axis=1) - 1.0 / cat.pe_counts)
    sel = np.sum(om_sa * om_sb) - np.sum(fa * fb / cat.campaign_n_draw)
    return events + n**2 * sel


def test_mc_formulas_equal_brute_force_double_sums():
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()
    model_a, model_b = ToyDensity(), ToyDensity(scale=0.3)
    cat = pad_catalog(pe, sel, model_a, hbi_config=HBI)
    rng = np.random.default_rng(1)
    Xa = np.column_stack([rng.uniform(-1, 1, 7), rng.uniform(-1, 1, 7), rng.uniform(60, 100, 7)])
    Xb = np.column_stack([rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6), rng.uniform(60, 100, 6)])
    Wa = rng.dirichlet(np.ones(7))
    Wb = rng.dirichlet(np.ones(6))
    ident_a = identity_for(pe, sel, model_a, NAMES, HBI)
    ident_b = identity_for(pe, sel, model_b, NAMES, HBI)
    sa = WeightedPosterior(NAMES, Xa, Wa, np.zeros(7, int), 1, "test", ident_a)
    sb = WeightedPosterior(NAMES, Xb, Wb, np.zeros(6, int), 1, "test", ident_b)
    wa = compute_model_mc_weights(sa, pe, sel, model_a, label="a", backend="numpy")
    wb = compute_model_mc_weights(sb, pe, sel, model_b, label="b")
    ea, sa_, fa = _per_draw_omegas(cat, model_a, Xa)
    eb, sb_, fb = _per_draw_omegas(cat, model_b, Xb)
    # U-statistic C_PP: weighted average over s != s'
    num = den = 0.0
    for s in range(7):
        for t in range(7):
            if s != t:
                c = _brute_covariance(cat, ea[s], sa_[s], fa[s], ea[t], sa_[t], fa[t])
                num += Wa[s] * Wa[t] * c
                den += Wa[s] * Wa[t]
    assert wa.c_pp() == pytest.approx(num / den, rel=1e-10)
    plug = sum(Wa[s] * Wa[t] * _brute_covariance(cat, ea[s], sa_[s], fa[s], ea[t], sa_[t], fa[t])
               for s in range(7) for t in range(7))
    assert wa.c_pp(u_statistic=False) == pytest.approx(plug, rel=1e-10)
    cross = sum(Wa[s] * Wb[t] * _brute_covariance(cat, ea[s], sa_[s], fa[s], eb[t], sb_[t], fb[t])
                for s in range(7) for t in range(6))
    assert mc_covariance(wa, wb) == pytest.approx(cross, rel=1e-10)
    edge = edge_mc_error(wa, wb)
    assert edge.variance == pytest.approx(wa.c_pp() + wb.c_pp() - 2 * cross, rel=1e-10)
    # E_P[sigma_A^2] and the predicted bias
    sigma_a2 = [np.sum(sa_[s] ** 2) - np.sum(fa[s] ** 2 / cat.campaign_n_draw) for s in range(7)]
    n = cat.n_events
    assert wa.expected_selection_sigma2 == pytest.approx(np.dot(Wa, sigma_a2), rel=1e-10)
    assert wa.predicted_log_evidence_bias() == pytest.approx(
        0.5 * n * (n + 1) * np.dot(Wa, sigma_a2) - 0.5 * wa.c_pp(), rel=1e-10
    )
    assert edge.bias == pytest.approx(wa.predicted_log_evidence_bias() - wb.predicted_log_evidence_bias())
    matrix = mc_covariance_matrix([wa, wb])
    assert matrix[0, 1] == matrix[1, 0] == pytest.approx(cross)
    assert matrix[0, 0] == pytest.approx(wa.c_pp())


def test_mc_weights_round_trip_and_refuse_different_data(tmp_path):
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()
    model = ToyDensity()
    X = np.array([[0.1, 0.2, 90.0], [0.3, -0.1, 80.0], [0.0, 0.0, 70.0]])
    sample = posterior_from_equal_weight_draws(NAMES, X, likelihood_identity=identity_for(pe, sel, model, NAMES, HBI))
    weights = compute_model_mc_weights(sample, pe, sel, model, label="m")
    save_model_mc_weights(tmp_path / "w.npz", weights)
    loaded = load_model_mc_weights(tmp_path / "w.npz")
    assert loaded.c_pp() == pytest.approx(weights.c_pp(), rel=1e-12)
    assert loaded.summary()["predicted_log_evidence_bias"] == pytest.approx(
        weights.summary()["predicted_log_evidence_bias"]
    )
    other_pe = make_toy_posterior_catalog(seed=8)
    other = posterior_from_equal_weight_draws(
        NAMES, X, likelihood_identity=identity_for(other_pe, sel, model, NAMES, HBI)
    )
    with pytest.raises(AnalysisInputError, match="differing keys"):
        compute_model_mc_weights(other, pe, sel, model)
    other_weights = compute_model_mc_weights(other, other_pe, sel, model)
    with pytest.raises(AnalysisInputError, match="same data"):
        edge_mc_error(weights, other_weights)


def test_bootstrap_counts_resample_samples_and_all_draws():
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()
    cat = pad_catalog(pe, sel, ToyDensity(), hbi_config=HBI, selection_capacity=30)
    counts_e, counts_s = bootstrap_counts(cat, 400, seed=3)
    assert counts_e.shape == (400, 3, cat.n_max) and counts_s.shape == (400, 30)
    np.testing.assert_array_equal(counts_e.sum(axis=2), np.broadcast_to(cat.pe_counts, (400, 3)))
    assert np.all(counts_e[:, ~cat.pe_mask] == 0) and np.all(counts_s[:, cat.n_selected:] == 0)
    for rows, n_draw in zip(cat.selection_rows_per_campaign, cat.campaign_n_draw):
        per_row = counts_s[:, rows].mean(axis=0)
        np.testing.assert_allclose(per_row.mean(), 1.0, atol=0.1)
        assert counts_s[:, rows].sum(axis=1).max() <= n_draw
    again_e, again_s = bootstrap_counts(cat, 3, seed=3)
    np.testing.assert_array_equal(again_e, counts_e[:3])
    np.testing.assert_array_equal(again_s, counts_s[:3])


def test_bootstrap_refuses_when_the_reweighting_ess_is_too_small():
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()
    model = ToyDensity()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-0.5, 0.5, 20), rng.uniform(-0.5, 0.5, 20), rng.uniform(70, 100, 20)])
    ident = identity_for(pe, sel, model, NAMES, HBI)
    sample = posterior_from_equal_weight_draws(NAMES, X, likelihood_identity=ident)
    result = bootstrap_edge_mc_error(sample, sample, pe, sel, model, model, n_replicates=8, min_ess=1.0)
    assert result.n_replicates == 8
    np.testing.assert_allclose(result.log_bayes_factor_shifts, 0.0, atol=1e-12)
    with pytest.raises(InsufficientReweightingESSError, match="reweighting ESS"):
        bootstrap_edge_mc_error(sample, sample, pe, sel, model, model, n_replicates=8, min_ess=1e6)


# ---------------------------------------------------------------------------
# Toy with a known answer
# ---------------------------------------------------------------------------


def _toy_realization_setup(n_events=40):
    obs = toys.ToyObservation()
    rng = np.random.default_rng(20260919)
    d = toys.toy_observed_data(rng, lambda r, n: r.normal(0.0, 1.0, n), n_events, obs)
    models = {"a": toys.GaussianToy(), "b": toys.GaussianToy(delta=0.6)}
    names = ("mu", "sigma")
    prior_box = ((-3.0, 3.0), (0.1, 3.0))
    grids, exact = {}, {}
    for key, model in models.items():
        fine = toys.grid_2d(names, prior_box, prior_box, 201)
        hp = {"mu": fine.points[:, 0], "sigma": fine.points[:, 1]}
        ll = model.exact_event_log_likelihoods(d, obs.sigma_obs, hp).sum(-1) - n_events * model.exact_log_exposure(
            obs.d_th, obs.sigma_obs, hp
        )
        keep = ll > ll.max() - 40
        lo0, hi0 = fine.points[keep, 0].min(), fine.points[keep, 0].max()
        lo1, hi1 = fine.points[keep, 1].min(), fine.points[keep, 1].max()
        box = ((lo0 - 0.1, hi0 + 0.1), (max(0.1, lo1 - 0.1), hi1 + 0.1))
        grid = toys.grid_2d(names, box, prior_box, 31)
        hp = {"mu": grid.points[:, 0], "sigma": grid.points[:, 1]}
        ll = model.exact_event_log_likelihoods(d, obs.sigma_obs, hp).sum(-1) - n_events * model.exact_log_exposure(
            obs.d_th, obs.sigma_obs, hp
        )
        grids[key], exact[key] = grid, toys.grid_log_evidence(ll, grid)
    return obs, rng, d, models, names, grids, exact


def _toy_model_posteriors(pe, sel, models, names, grids, n_draw):
    samples, cats, lnz, mcw = {}, {}, {}, {}
    for key, model in models.items():
        grid = grids[key]
        cat = pad_catalog(pe, sel, model, hbi_config=HBI, selection_capacity=n_draw)
        ev = CatalogWeightEvaluator(cat, model, names, batch_size=128)
        base = ev.bootstrap(grid.points, cat.pe_mask[None].astype(float), cat.sel_mask[None].astype(float))[0]
        lnz[key] = toys.grid_log_evidence(base, grid)
        W = toys.grid_posterior_weights(base, grid)
        keep = W > 0
        samples[key] = WeightedPosterior(
            names, grid.points[keep], W[keep] / W[keep].sum(), np.zeros(keep.sum(), int), 1, "grid",
            identity_for(pe, sel, model, names, HBI),
        )
        cats[key] = cat
        mcw[key] = compute_model_mc_weights(samples[key], pe, sel, model, label=key, catalog=cat, batch_size=128)
    return samples, cats, lnz, mcw


def test_mc_error_predictions_track_a_toy_with_known_evidences():
    """Formula predictions vs the realized spread of ln Zhat and ln BFhat.

    Each realization redraws PE samples and injections for fixed data; exact
    evidences come from the analytic likelihood on the same grid. The full
    validation (hundreds of realizations, two regimes, with the bootstrap) is
    recorded under gwpop-search-data/validation/analysis_estimators/.
    """
    obs, rng, d, models, names, grids, exact = _toy_realization_setup()
    n_real, n_draw, n_pe = 32, 8000, 64
    rows = []
    for _ in range(n_real):
        pe = toys.toy_posterior_catalog(rng, d, n_pe, obs)
        sel = toys.toy_selection_catalog(rng, n_draw, obs)
        _, _, lnz, mcw = _toy_model_posteriors(pe, sel, models, names, grids, n_draw)
        edge = edge_mc_error(mcw["a"], mcw["b"])
        rows.append((lnz["a"] - exact["a"], (lnz["a"] - lnz["b"]) - (exact["a"] - exact["b"]),
                     mcw["a"].c_pp(), edge.variance, edge.bias))
    rows = np.asarray(rows)
    emp_var_a, emp_var_bf = rows[:, 0].var(ddof=1), rows[:, 1].var(ddof=1)
    pred_var_a, pred_var_bf = rows[:, 2].mean(), rows[:, 3].mean()
    # per-model: C_PP tracks Var(ln Zhat) (relative SE of the empirical variance ~ 0.23)
    assert 0.55 < pred_var_a / emp_var_a < 1.8
    # common random numbers: the edge error is far below the naive quadrature sum
    assert pred_var_bf < 0.05 * 2 * pred_var_a
    # the edge prediction has the right size (it is conservative when V is large)
    assert 0.3 < pred_var_bf / emp_var_bf < 4.0
    # no significant ln BF bias in this regime, and the prediction is small too
    se = rows[:, 1].std(ddof=1) / math.sqrt(n_real)
    assert abs(rows[:, 1].mean()) < 4 * se + 0.01
    assert abs(rows[:, 4].mean()) < 0.05


def test_bootstrap_matches_the_formula_within_one_realization():
    obs, rng, d, models, names, grids, _ = _toy_realization_setup()
    n_draw = 8000
    pe = toys.toy_posterior_catalog(rng, d, 64, obs)
    sel = toys.toy_selection_catalog(rng, n_draw, obs)
    samples, cats, _, mcw = _toy_model_posteriors(pe, sel, models, names, grids, n_draw)
    boot = bootstrap_edge_mc_error(
        samples["a"], samples["b"], pe, sel, models["a"], models["b"], n_replicates=80, seed=5,
        min_ess=5.0, catalogs=(cats["a"], cats["b"]), batch_size=128,
    )
    info = boot.to_dict()
    for key in ("a", "b"):
        boot_var = info[f"model_{key}"]["sd"] ** 2
        assert 0.6 < boot_var / mcw[key].c_pp() < 1.6
        assert info[f"model_{key}"]["min_reweighting_ess"] >= 5.0
    edge = edge_mc_error(mcw["a"], mcw["b"])
    # the common-random-number edge error is small in both estimators
    assert boot.sigma < 0.2 * edge.naive_independent_sigma
    assert edge.sigma < 0.2 * edge.naive_independent_sigma
    assert 0.15 < boot.sigma**2 / edge.variance < 4.0


def test_bootstrap_likelihoods_match_numpy_and_do_not_depend_on_chunking():
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()
    model = ToyDensity()
    cat = pad_catalog(pe, sel, model, hbi_config=HBI, selection_capacity=30)
    counts_e, counts_s = bootstrap_counts(cat, 5, seed=2)
    fast = CatalogWeightEvaluator(cat, model, NAMES, batch_size=2)
    ref = CatalogWeightEvaluator(cat, model, NAMES, backend="numpy").bootstrap(POINTS, counts_e, counts_s)
    for chunk in (1, 2, 16):
        np.testing.assert_allclose(fast.bootstrap(POINTS, counts_e, counts_s, replicate_chunk=chunk), ref,
                                   rtol=1e-12, atol=1e-12)
    ones = fast.bootstrap(POINTS, cat.pe_mask[None].astype(int), cat.sel_mask[None].astype(int))[0]
    loglike = build_batched_log_likelihood(pe, sel, model, NAMES, hbi_config=HBI)
    np.testing.assert_allclose(ones, loglike(POINTS), rtol=0, atol=1e-11)
    with pytest.raises(ValueError, match="multiplicities"):
        fast.bootstrap(POINTS, counts_e + 0.5, counts_s)


# ---------------------------------------------------------------------------
# Review finding: the catalog must be padded with the sampled HBI configuration
# ---------------------------------------------------------------------------


def test_selection_weights_depend_on_the_hbi_configuration_of_the_catalog():
    """A catalog padded with the wrong HBI config silently changes sigma_MC.

    ``raw_selection_use_observing_time`` sets the per-campaign log(T_k / N_k)
    in ``sel_log_factor``, hence the self-normalized selection weights that
    C_PP, E[sigma_A^2] and the edge sigma_MC are built from. The likelihood
    identity check cannot see the mismatch, because it is in the catalog and
    not in the config used for the identity.
    """
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()
    model = ToyDensity()
    with_time = HBIConfig(selection_chunk_size=None, raw_selection_use_observing_time=True)
    without = HBIConfig(selection_chunk_size=None, raw_selection_use_observing_time=False)
    cat_time = pad_catalog(pe, sel, model, hbi_config=with_time)
    cat_plain = pad_catalog(pe, sel, model, hbi_config=without)
    assert cat_time.raw_selection_use_observing_time is True
    assert cat_plain.raw_selection_use_observing_time is False
    assert not np.allclose(cat_time.sel_log_factor, cat_plain.sel_log_factor)

    rng = np.random.default_rng(3)
    X = np.column_stack([rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6), rng.uniform(60, 100, 6)])
    W = rng.dirichlet(np.ones(6))
    ident = identity_for(pe, sel, model, NAMES, without)
    sample = WeightedPosterior(NAMES, X, W, np.zeros(6, int), 1, "test", ident)

    right = compute_model_mc_weights(sample, pe, sel, model, label="m", catalog=cat_plain)
    # the quantities the edge error budget is built from move materially
    derived = compute_model_mc_weights(sample, pe, sel, model, label="m")
    assert derived.c_pp() == pytest.approx(right.c_pp())
    assert derived.min_selection_ess == pytest.approx(right.min_selection_ess)

    # the wrong catalog is refused rather than silently producing other numbers
    with pytest.raises(AnalysisInputError, match="raw_selection_use_observing_time"):
        compute_model_mc_weights(sample, pe, sel, model, label="m", catalog=cat_time)
    with pytest.raises(AnalysisInputError, match="raw_selection_use_observing_time"):
        bootstrap_edge_mc_error(sample, sample, pe, sel, model, model, n_replicates=2,
                                min_ess=1.0, catalogs=(cat_time, cat_time))

    # and they really would have been different numbers
    wrong = compute_model_mc_weights(sample, pe, sel, model, label="m", catalog=cat_time,
                                     hbi_config=with_time, verify_identity=False)
    assert wrong.c_pp() != pytest.approx(right.c_pp(), rel=1e-3)
