"""Variance taper of the shape log-likelihood (GWTC-5 / Callister & Farr 2024)."""
import json
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gwpop_search.data import Campaign, SelectionCatalog, SelectionMode  # noqa: E402
from gwpop_search.data.fixtures import (  # noqa: E402
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.hbi import HBIConfig, RateTreatment, shape_log_likelihood  # noqa: E402
from gwpop_search.hbi.jax_backend import (  # noqa: E402
    build_shape_log_likelihood,
    build_shape_log_likelihood_components,
    build_terms_and_variance_function,
    build_terms_function,
)
from gwpop_search.hbi.numpy_backend import taper_variance  # noqa: E402
from gwpop_search.hbi.taper import (  # noqa: E402
    CALLISTER_FARR_EXPONENT,
    VarianceTaper,
    log_taper_jax,
    log_taper_numpy,
    taper_region_summary,
)
from gwpop_search.inference import (  # noqa: E402
    DynestyConfig,
    PriorSpec,
    build_batched_log_likelihood,
    build_importance_diagnostics,
    prior_transform_for,
    run_dynesty,
)
from gwpop_search.inference.dynesty_backend import (  # noqa: E402
    build_likelihood_identity,
    posterior_taper_mass,
)

SMOOTH = VarianceTaper()
SHARP = VarianceTaper(kind="sharp")


# ---------------------------------------------------------------------------
# The taper function
# ---------------------------------------------------------------------------


def test_taper_configuration_validates_and_round_trips():
    assert SMOOTH.kind == "smooth" and SMOOTH.threshold == 1.0
    assert SMOOTH.exponent == CALLISTER_FARR_EXPONENT == 30.0
    for taper in (SMOOTH, SHARP, VarianceTaper(threshold=2, exponent=10, region_suppression=0.05)):
        assert VarianceTaper.from_dict(json.loads(json.dumps(taper.to_dict()))) == taper
    with pytest.raises(ValueError, match="kind"):
        VarianceTaper(kind="soft")
    with pytest.raises(ValueError, match="positive"):
        VarianceTaper(threshold=0.0)
    with pytest.raises(ValueError, match=">= 1"):
        VarianceTaper(exponent=0.5)
    with pytest.raises(ValueError, match="region_suppression"):
        VarianceTaper(region_suppression=1.0)
    with pytest.raises(TypeError):
        VarianceTaper(threshold=True)
    with pytest.raises(ValueError, match="unknown"):
        VarianceTaper.from_dict({"kind": "smooth", "cut": 1})


@pytest.mark.parametrize("threshold", [1.0, 2.0])
def test_smooth_taper_limits(threshold):
    taper = VarianceTaper(threshold=threshold)
    p = taper.exponent
    # Far below the threshold: no suppression; ln T ~ -x^p.
    assert taper.log_taper(0.0) == 0.0
    small = 0.3 * threshold
    np.testing.assert_allclose(taper.log_taper(small), -(0.3**p), rtol=1e-12)
    assert taper.log_taper(0.5 * threshold) > -1e-8
    # At the threshold T = 1/2 exactly.
    np.testing.assert_allclose(taper.log_taper(threshold), -math.log(2.0), rtol=1e-15)
    # At twice the threshold: T = 1 / (1 + 2^30) = 9.3e-10.
    np.testing.assert_allclose(np.exp(taper.log_taper(2 * threshold)), 1 / (1 + 2.0**30), rtol=1e-12)
    # Far above: power-law suppression ln T -> -p ln x.
    big = 50.0 * threshold
    np.testing.assert_allclose(taper.log_taper(big), -p * math.log(50.0), rtol=1e-12)
    # Degenerate inputs.
    assert np.isneginf(taper.log_taper(np.inf))
    assert np.isneginf(taper.log_taper(np.nan))
    assert np.isneginf(taper.log_taper(-1.0))
    # Region onset: T(onset) = 1 - region_suppression.
    np.testing.assert_allclose(
        np.exp(taper.log_taper(taper.region_onset)), 1 - taper.region_suppression, rtol=1e-12
    )
    np.testing.assert_allclose(taper.region_onset, threshold * (0.01 / 0.99) ** (1 / 30), rtol=1e-14)


def test_smooth_taper_approaches_the_sharp_cut():
    v = np.array([0.0, 0.2, 0.9, 0.99, 1.01, 1.1, 3.0, np.inf])
    sharp = SHARP.log_taper(v)
    np.testing.assert_array_equal(sharp, [0, 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf])
    steep = VarianceTaper(exponent=1e5).log_taper(v)
    np.testing.assert_allclose(steep[:4], 0.0, atol=1e-200)
    assert np.all(steep[4:] < -900)
    # T at the threshold stays 1/2 for every steepness (the sharp cut keeps T = 1 there).
    assert SHARP.log_taper(1.0) == 0.0
    np.testing.assert_allclose(VarianceTaper(exponent=1e5).log_taper(1.0), -math.log(2))


def test_taper_is_continuous_monotone_and_numpy_matches_jax():
    v = np.concatenate([[0.0], np.geomspace(1e-6, 1e3, 20001)])
    for taper in (SMOOTH, VarianceTaper(threshold=2.0), VarianceTaper(exponent=4.0)):
        ln_np = log_taper_numpy(v, taper)
        ln_jax = np.asarray(log_taper_jax(jnp.asarray(v), taper))
        np.testing.assert_allclose(ln_jax, ln_np, rtol=1e-14, atol=1e-300)
        assert np.all(np.diff(ln_np) <= 0.0)  # monotone non-increasing
        # Continuity: each step is bounded by |d ln T / d v| dv (no jumps).
        with np.errstate(divide="ignore"):
            slope = taper.exponent / v[:-1]  # |d ln T/dv| <= p / v on [v_i, v_i+1]
        assert np.all(np.abs(np.diff(ln_np)) <= slope * np.diff(v) * 1.0000001 + 1e-300)
        T = np.exp(ln_np)
        assert T[0] == 1.0 and 0.0 <= T[-1] < 1e-10
    np.testing.assert_array_equal(
        np.asarray(log_taper_jax(jnp.asarray([np.inf, np.nan, -2.0]), SMOOTH)), [-np.inf] * 3
    )


def test_taper_is_differentiable_with_the_analytic_gradient():
    for taper in (SMOOTH, VarianceTaper(threshold=2.0)):
        g = jax.grad(lambda v: log_taper_jax(v, taper))
        h = jax.grad(g)
        p, c = taper.exponent, taper.threshold
        for v in (0.05, 0.5, 0.9, 1.0, 1.3, 2.0, 7.0):
            x = (v / c) ** p
            analytic = -p * x / (v * (1.0 + x))
            np.testing.assert_allclose(float(g(v)), analytic, rtol=1e-12, atol=1e-300)
            eps = 1e-6 * v
            fd = (log_taper_numpy(v + eps, taper) - log_taper_numpy(v - eps, taper)) / (2 * eps)
            np.testing.assert_allclose(float(g(v)), fd, rtol=1e-6, atol=1e-12)
            assert np.isfinite(float(h(v)))
        # At sigma^2 = 0 the value and its gradient are finite (no NaN from log 0).
        assert float(log_taper_jax(0.0, taper)) == 0.0
        assert float(g(0.0)) == 0.0
    # The second derivative is continuous across the threshold (C^2 there).
    h = jax.grad(jax.grad(lambda v: log_taper_jax(v, SMOOTH)))
    left, right = float(h(1.0 - 1e-7)), float(h(1.0 + 1e-7))
    np.testing.assert_allclose(left, right, rtol=1e-4)


def test_taper_region_summary_with_known_weights():
    v = np.array([0.1, 0.5, 0.9, 1.5, 4.0])
    w = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    out = taper_region_summary(v, w, SMOOTH)
    # onset 0.858: 0.9, 1.5, 4.0 are inside (weights 2 + 3 + 3 of 10).
    np.testing.assert_allclose(out["posterior_mass_in_taper_region"], 0.8)
    np.testing.assert_allclose(out["posterior_mass_above_threshold"], 0.6)
    expected_t = np.sum(w / w.sum() * np.exp(SMOOTH.log_taper(v)))
    np.testing.assert_allclose(out["posterior_mean_taper"], expected_t)
    assert out["variance_quantiles"]["q0.5"] == 1.5
    assert out["variance_quantiles"]["max"] == 4.0
    sharp = taper_region_summary(v, w, SHARP)
    np.testing.assert_allclose(sharp["posterior_mass_in_taper_region"], 0.6)
    with pytest.raises(ValueError):
        taper_region_summary(v, np.zeros(5), SMOOTH)
    with pytest.raises(ValueError):
        taper_region_summary(v, w[:3], SMOOTH)


# ---------------------------------------------------------------------------
# HBI configuration
# ---------------------------------------------------------------------------


def test_hbi_config_carries_the_taper_only_when_configured():
    plain = HBIConfig()
    assert "variance_taper" not in plain.to_dict()
    assert HBIConfig.from_dict(plain.to_dict()) == plain
    tapered = HBIConfig(variance_taper={"kind": "smooth", "threshold": 2.0})
    assert tapered.variance_taper == VarianceTaper(threshold=2.0)
    payload = tapered.to_dict()
    assert payload["variance_taper"]["threshold"] == 2.0
    assert HBIConfig.from_dict(json.loads(json.dumps(payload))) == tapered
    with pytest.raises(ValueError, match="shape likelihood"):
        HBIConfig(rate_treatment=RateTreatment.POISSON, variance_taper=SMOOTH)
    with pytest.raises(TypeError):
        HBIConfig(variance_taper=1.0)
    with pytest.raises(ValueError, match="unknown"):
        HBIConfig.from_dict({**plain.to_dict(), "taper": None})


# ---------------------------------------------------------------------------
# Variance and taper inside the likelihood
# ---------------------------------------------------------------------------

NAMES = ("a", "b", "mcut")
POINTS = np.array(
    [
        [0.4, -0.25, 100.0],
        [0.1, 0.35, 100.0],
        [2.0, 1.5, 45.0],
        [-1.0, 0.8, 30.0],  # an event loses all PE support
        [0.7, -2.0, 60.0],
    ]
)


def density_np(samples, hp):
    value = hp["a"] * samples["q"] + hp["b"] * samples["chi_eff"] - 0.02 * samples["m1_source"]
    return np.where(samples["m1_source"] <= hp["mcut"], value, -np.inf)


def density_jax(samples, hp):
    value = hp["a"] * samples["q"] + hp["b"] * samples["chi_eff"] - 0.02 * samples["m1_source"]
    return jnp.where(samples["m1_source"] <= hp["mcut"], value, -jnp.inf)


def estimator_ready_selection():
    base = make_toy_selection_catalog()
    return SelectionCatalog(
        samples=base.samples,
        log_draw_density=np.log(np.linspace(0.2, 1.1, base.n_selected)),
        campaign_id=np.asarray(["combined"] * base.n_selected),
        campaigns=(Campaign("combined", n_draw=None, observing_time_yr=1.5),),
        basis=base.basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics="already normalized pdraw",
    )


@pytest.mark.parametrize("selection_kind", ["raw_draw", "estimator_ready"])
@pytest.mark.parametrize("chunk", [None, 5])
def test_likelihood_variance_matches_the_importance_diagnostics(selection_kind, chunk):
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog() if selection_kind == "raw_draw" else estimator_ready_selection()
    taper = VarianceTaper(threshold=0.6)
    cfg = HBIConfig(selection_chunk_size=chunk, variance_taper=taper)
    loglike = build_batched_log_likelihood(pe, sel, density_jax, NAMES, hbi_config=cfg, batch_size=3)
    comps = loglike.components(POINTS)
    diag = build_importance_diagnostics(pe, sel, density_jax, NAMES, hbi_config=cfg, batch_size=2)(
        POINTS
    )
    untapered_cfg = HBIConfig(selection_chunk_size=chunk)
    untapered = build_batched_log_likelihood(
        pe, sel, density_jax, NAMES, hbi_config=untapered_cfg, batch_size=3
    )(POINTS)
    single = build_shape_log_likelihood(pe, sel, density_jax, config=cfg)
    for k, row in enumerate(POINTS):
        hp = dict(zip(NAMES, row))
        ref = shape_log_likelihood(pe, sel, density_np, hp, config=cfg)
        if not np.isfinite(ref.log_likelihood_untapered):
            assert np.isneginf(comps["log_likelihood"][k])
            assert np.isinf(comps["variance"][k])
            continue
        # sigma^2: likelihood pass == NumPy reference == jitted diagnostics.
        np.testing.assert_allclose(comps["variance"][k], ref.taper_variance, rtol=1e-12)
        np.testing.assert_allclose(diag.taper_variance[k], ref.taper_variance, rtol=1e-12)
        np.testing.assert_allclose(
            diag.shape_log_likelihood_variance[k], ref.terms.variance.shape_log_likelihood_variance,
            rtol=1e-12,
        )
        np.testing.assert_allclose(comps["log_taper"][k], ref.log_taper, rtol=1e-12)
        # Tapered likelihood = untapered + ln T, identical across the builders.
        np.testing.assert_allclose(comps["log_likelihood"][k], ref.log_likelihood, rtol=0, atol=1e-11)
        np.testing.assert_allclose(diag.log_likelihood[k], ref.log_likelihood, rtol=0, atol=1e-11)
        np.testing.assert_allclose(float(single(hp)), ref.log_likelihood, rtol=0, atol=1e-11)
        # The untapered part is the pre-taper likelihood (up to rounding: XLA
        # fuses the extra variance reductions into the same graph).
        np.testing.assert_allclose(
            comps["log_likelihood_untapered"][k], untapered[k], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            comps["log_likelihood"][k] - comps["log_likelihood_untapered"][k],
            comps["log_taper"][k],
            rtol=0,
            atol=1e-12,
        )
    stats = loglike.stats()
    supported = np.isfinite(comps["log_likelihood_untapered"])
    assert stats["n_in_taper_region"] == int(np.sum(supported & taper.in_region(comps["variance"])))
    assert stats["n_above_taper_threshold"] == int(
        np.sum(supported & (comps["variance"] > taper.threshold))
    )
    assert stats["n_in_taper_region"] > 0


def test_untapered_likelihood_is_unchanged_by_the_variance_pass():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    for chunk in (None, 4):
        cfg = HBIConfig(selection_chunk_size=chunk)
        plain = build_terms_function(pe, sel, density_jax, config=cfg)
        with_var = build_terms_and_variance_function(pe, sel, density_jax, config=cfg)
        for row in POINTS:
            hp = dict(zip(NAMES, row))
            e0, a0 = plain(hp)
            e1, a1, _, _ = with_var(hp)
            np.testing.assert_allclose(np.asarray(e1), np.asarray(e0), rtol=1e-14, atol=0)
            np.testing.assert_allclose(float(a1), float(a0), rtol=1e-14, atol=0)
        # with_variance=True without a taper: values identical, ln T = 0.
        base = build_batched_log_likelihood(pe, sel, density_jax, NAMES, hbi_config=cfg)
        var = build_batched_log_likelihood(pe, sel, density_jax, NAMES, hbi_config=cfg, with_variance=True)
        comps = var.components(POINTS)
        np.testing.assert_allclose(comps["log_likelihood"], base(POINTS), rtol=1e-14, atol=0)
        np.testing.assert_array_equal(comps["log_taper"][np.isfinite(comps["variance"])], 0.0)
        with pytest.raises(ValueError, match="variance"):
            base.components(POINTS)
    # The likelihood identity records the taper (so a tapered run cannot be
    # diagnosed as an untapered one); without a taper it is unchanged.
    plain_id = build_likelihood_identity(pe, sel, density_jax, NAMES, HBIConfig())
    taper_id = build_likelihood_identity(
        pe, sel, density_jax, NAMES, HBIConfig(variance_taper=SMOOTH)
    )
    assert "variance_taper" not in plain_id["hbi_config"]
    assert taper_id["hbi_config"]["variance_taper"] == SMOOTH.to_dict()


def test_tapered_likelihood_is_differentiable():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    cfg = HBIConfig(variance_taper=VarianceTaper(threshold=0.5))
    components = build_shape_log_likelihood_components(pe, sel, density_jax, config=cfg, jit=False)

    def f(ab):
        return components({"a": ab[0], "b": ab[1], "mcut": 100.0})[0]

    grad = jax.jit(jax.grad(f))
    for ab in ([0.4, -0.25], [0.1, 0.35], [0.0, 0.0]):
        x = jnp.asarray(ab)
        g = np.asarray(grad(x))
        assert np.all(np.isfinite(g))
        for i in range(2):
            e = np.zeros(2)
            e[i] = 1e-6
            fd = (float(f(x + e)) - float(f(x - e))) / 2e-6
            np.testing.assert_allclose(g[i], fd, rtol=1e-5, atol=1e-7)
        # The taper contributes to the gradient where it is active.
        ln_t = components({"a": x[0], "b": x[1], "mcut": 100.0})[3]
        assert float(ln_t) < -1e-6


def test_analysis_evaluators_refuse_a_tapered_configuration():
    from gwpop_search.analysis.terms import BatchedCatalogTerms, pad_catalog

    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    cfg = HBIConfig(selection_chunk_size=None, variance_taper=SMOOTH)
    with pytest.raises(NotImplementedError, match="untapered"):
        BatchedCatalogTerms(pe, sel, density_jax, NAMES, hbi_config=cfg)
    with pytest.raises(NotImplementedError, match="untapered"):
        pad_catalog(pe, sel, density_jax, hbi_config=cfg)


# ---------------------------------------------------------------------------
# Evidence with the taper = integral over the tapered likelihood
# ---------------------------------------------------------------------------

A_RANGE = (-6.0, 6.0)
B_RANGE = (-4.0, 4.0)
TOY_TAPER = VarianceTaper(threshold=0.8)


def density_ab_jax(samples, hp):
    return hp["a"] * samples["q"] + hp["b"] * samples["chi_eff"] - 0.02 * samples["m1_source"]


def _grid_log_evidence(components, n_a=241, n_b=161):
    """ln of (1/area) int int exp(f) da db by the trapezoid rule on a grid."""
    a = np.linspace(*A_RANGE, n_a)
    b = np.linspace(*B_RANGE, n_b)
    A, B = np.meshgrid(a, b, indexing="ij")
    values = components(np.column_stack([A.ravel(), B.ravel()]))
    out = {}
    for key in ("log_likelihood", "log_likelihood_untapered"):
        f = values[key].reshape(A.shape)
        peak = np.max(f)
        integral = np.trapezoid(np.trapezoid(np.exp(f - peak), b, axis=1), a)
        area = (A_RANGE[1] - A_RANGE[0]) * (B_RANGE[1] - B_RANGE[0])
        out[key] = float(peak + np.log(integral / area))
    return out, values


def test_evidence_with_the_taper_is_the_integral_of_the_tapered_likelihood():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    cfg = HBIConfig(variance_taper=TOY_TAPER)
    loglike = build_batched_log_likelihood(
        pe, sel, density_ab_jax, ("a", "b"), hbi_config=cfg, batch_size=16
    )
    truth, grid = _grid_log_evidence(loglike.components)
    # The taper matters on this problem: a large part of the prior is suppressed.
    in_region = TOY_TAPER.in_region(grid["variance"])
    assert 0.2 < np.mean(in_region) < 0.95
    assert truth["log_likelihood_untapered"] - truth["log_likelihood"] > 0.9

    names, transform = prior_transform_for(
        {
            "a": PriorSpec("uniform", low=A_RANGE[0], high=A_RANGE[1]),
            "b": PriorSpec("uniform", low=B_RANGE[0], high=B_RANGE[1]),
        }
    )
    config = DynestyConfig(nlive=300, batch_size=16, dlogz=0.05, num_posterior_samples=1000)
    result = run_dynesty(loglike, transform, 2, seed=11, config=config, names=names)
    error = abs(result.log_evidence - truth["log_likelihood"])
    assert error < max(3.0 * result.log_evidence_error, 0.15), (
        result.log_evidence,
        truth,
        result.log_evidence_error,
    )
    # ... and not the untapered integral.
    assert abs(result.log_evidence - truth["log_likelihood_untapered"]) > 5 * max(
        result.log_evidence_error, 0.1
    )
    # The evaluation tallies travel with the result.
    cumulative = result.diagnostics["cumulative"]
    assert 0 < cumulative["n_in_taper_region"] <= result.n_likelihood_evaluations
    assert cumulative["n_above_taper_threshold"] <= cumulative["n_in_taper_region"]


def test_posterior_taper_mass_on_a_tapered_run():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    cfg = HBIConfig(variance_taper=VarianceTaper(threshold=0.6))
    names = ("a", "b")
    loglike = build_batched_log_likelihood(pe, sel, density_ab_jax, names, hbi_config=cfg, batch_size=16)
    _, transform = prior_transform_for(
        {
            "a": PriorSpec("uniform", low=A_RANGE[0], high=A_RANGE[1]),
            "b": PriorSpec("uniform", low=B_RANGE[0], high=B_RANGE[1]),
        }
    )
    config = DynestyConfig(nlive=100, batch_size=16, dlogz=0.5, num_posterior_samples=200)
    identity = {"likelihood": loglike.likelihood_identity()}
    runs = [
        run_dynesty(loglike, transform, 2, seed=seed, config=config, names=names)
        for seed in (1, 2)
    ]
    for run in runs:
        assert run.likelihood_identity == identity["likelihood"]
    report = posterior_taper_mass(runs, loglike)
    assert report["max_relative_log_likelihood_mismatch"] <= 1e-9
    for block in (*report["runs"], report["pooled"]):
        assert 0.0 <= block["posterior_mass_above_threshold"] <= block["posterior_mass_in_taper_region"] <= 1.0
        assert 0.0 < block["posterior_mean_taper"] <= 1.0
    # Direct check of one run.
    comps = loglike.components(runs[0].samples)
    w = runs[0].weights
    expected = float(np.sum(w[VarianceTaper(threshold=0.6).in_region(comps["variance"])]))
    np.testing.assert_allclose(report["runs"][0]["posterior_mass_in_taper_region"], expected)
    pooled = 0.5 * (
        report["runs"][0]["posterior_mass_in_taper_region"]
        + report["runs"][1]["posterior_mass_in_taper_region"]
    )
    np.testing.assert_allclose(report["pooled"]["posterior_mass_in_taper_region"], pooled)
    # A different taper is a different likelihood.
    other = build_batched_log_likelihood(
        pe, sel, density_ab_jax, names, hbi_config=HBIConfig(variance_taper=SMOOTH), batch_size=16
    )
    with pytest.raises(ValueError, match="different likelihood"):
        posterior_taper_mass(runs, other)
    untapered = build_batched_log_likelihood(pe, sel, density_ab_jax, names, batch_size=16)
    with pytest.raises(ValueError, match="variance taper"):
        posterior_taper_mass(runs, untapered)
