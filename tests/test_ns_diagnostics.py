"""Pure-NumPy nested-sampling diagnostics for repeated dynesty runs."""

from types import SimpleNamespace

import numpy as np
import pytest

from gwpop_search.inference.ns_diagnostics import (
    cross_run_rhat,
    insertion_index_ranks,
    insertion_index_test,
    kish_ess,
    max_pairwise_z,
    normalized_weights,
    pooled_weighted_samples,
    prior_edge_mass,
    rank_normalized_split_rhat,
    run_termination,
    weighted_quantiles,
)
from gwpop_search.inference.priors import PriorSpec


def _run(samples, log_weights, names=("x", "y")):
    return SimpleNamespace(
        names=tuple(names),
        samples=np.asarray(samples, dtype=float),
        log_weights=np.asarray(log_weights, dtype=float),
    )


def test_kish_ess_and_normalized_weights_handle_plateau_points():
    log_w = np.array([0.0, 0.0, -np.inf, 0.0, 0.0])
    w = normalized_weights(log_w)
    np.testing.assert_allclose(w, [0.25, 0.25, 0.0, 0.25, 0.25])
    assert kish_ess(log_w) == pytest.approx(4.0)
    assert kish_ess(np.log([1.0, 3.0])) == pytest.approx(16.0 / 10.0)
    with pytest.raises(ValueError, match="NaN or \\+inf"):
        normalized_weights([0.0, np.nan])
    with pytest.raises(ValueError, match="no finite"):
        normalized_weights([-np.inf, -np.inf])


def test_pooled_samples_are_an_equal_mixture_of_separately_normalized_runs():
    # Run 0 has 2 points and run 1 has 4 points with very different evidences:
    # the pooled posterior still gives each run total mass 1/2.
    a = _run([[0.0, 0.0], [1.0, 1.0]], [np.log(1.0), np.log(3.0)])
    b = _run(np.arange(8.0).reshape(4, 2), np.full(4, 100.0))
    samples, weights = pooled_weighted_samples([a, b])
    assert samples.shape == (6, 2)
    np.testing.assert_allclose(weights, [0.125, 0.375, 0.125, 0.125, 0.125, 0.125])
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="different parameter names"):
        pooled_weighted_samples([a, _run([[0.0, 0.0]], [0.0], names=("x", "z"))])


def test_weighted_quantiles_match_unweighted_inverted_cdf_for_equal_weights():
    rng = np.random.default_rng(3)
    values = rng.normal(size=101)
    weights = np.ones_like(values)
    for q in (0.05, 0.5, 0.95):
        assert weighted_quantiles(values, weights, [q])[0] == np.quantile(
            values, q, method="inverted_cdf"
        )
    # Point masses: all weight on one value.
    assert weighted_quantiles([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.1, 0.9]).tolist() == [2.0, 2.0]


def test_rank_normalized_split_rhat_matches_arviz():
    arviz = pytest.importorskip("arviz")
    rng = np.random.default_rng(11)
    good = rng.normal(size=(2, 400))
    shifted = good.copy()
    shifted[1] += 1.0
    heavy = rng.standard_t(df=2, size=(3, 301))
    for chains in (good, shifted, heavy):
        np.testing.assert_allclose(
            rank_normalized_split_rhat(chains), float(arviz.rhat(chains, method="rank")), rtol=1e-10
        )
    assert rank_normalized_split_rhat(good) < 1.01
    assert rank_normalized_split_rhat(shifted) > 1.05
    with pytest.raises(ValueError, match="at least 2 chains"):
        rank_normalized_split_rhat(good[:1])


def test_cross_run_rhat_detects_runs_that_disagree():
    rng = np.random.default_rng(5)

    def weighted_run(mean, n=4000):
        # Uniform proposal reweighted to N(mean, 1): valid weighted posterior samples.
        x = rng.uniform(-6.0, 6.0, size=(n, 1))
        log_w = -0.5 * (x[:, 0] - mean) ** 2
        return _run(x, log_w, names=("x",))

    agree = cross_run_rhat([weighted_run(0.0), weighted_run(0.0)], max_draws=500, seed=1)
    assert agree["n_runs"] == 2 and agree["draws_per_run"] == 500
    assert agree["max"] < 1.02 and agree["worst_parameter"] == "x"
    disagree = cross_run_rhat([weighted_run(0.0), weighted_run(0.6)], max_draws=500, seed=1)
    assert disagree["max"] > 1.05
    # Deterministic for a fixed seed; the draw count is capped by the smallest Kish ESS.
    again = cross_run_rhat([weighted_run(0.0), weighted_run(0.0)], max_draws=500, seed=1)
    assert again["draws_per_run"] == 500
    tiny = _run(np.arange(12.0).reshape(12, 1), np.log(np.r_[np.ones(6), np.full(6, 1e-6)]), ("x",))
    capped = cross_run_rhat([tiny, tiny], max_draws=500, seed=2)
    assert capped["draws_per_run"] == 6
    with pytest.raises(ValueError, match="at least two"):
        cross_run_rhat([tiny], seed=1)


def test_max_pairwise_z():
    assert max_pairwise_z([1.0], [0.1]) is None
    # Pairs: 0.3/hypot(0.1, 0.1) = 2.12, 0.1/hypot(0.1, 0.2), 0.4/hypot(0.1, 0.2) = 1.79.
    assert max_pairwise_z([-10.0, -10.3, -9.9], [0.1, 0.1, 0.2]) == pytest.approx(
        0.3 / np.hypot(0.1, 0.1)
    )
    assert max_pairwise_z([1.0, 1.0], [0.0, 0.0]) == 0.0
    assert max_pairwise_z([1.0, 2.0], [0.0, 0.0]) == np.inf


def test_run_termination_reads_the_backend_convergence_flag():
    assert run_termination(SimpleNamespace(diagnostics={"converged": True})) == "dlogz"
    assert run_termination(SimpleNamespace(diagnostics={"converged": False})) == "budget"
    assert run_termination(SimpleNamespace(diagnostics={})) == "unknown"


def test_prior_edge_mass_uses_linear_and_log_widths():
    names = ("u", "l", "n")
    priors = {
        "u": PriorSpec("uniform", low=0.0, high=10.0),
        "l": PriorSpec("log_uniform", low=1.0, high=100.0),
        "n": PriorSpec("normal", loc=0.0, scale=1.0),
    }
    samples = np.array([[0.05, 1.01, 0.0], [5.0, 10.0, 1.0], [9.99, 99.0, 2.0]])
    weights = np.array([0.2, 0.5, 0.3])
    edge = prior_edge_mass(samples, weights, names, priors)
    assert set(edge) == {"u", "l"}
    assert edge["u"] == {"lower": 0.2, "upper": 0.3}
    assert edge["l"] == {"lower": 0.2, "upper": 0.3}


def test_insertion_ranks_of_an_exact_nested_sampler_are_uniform():
    """Simulate exact constrained-prior sampling and check the ported reconstruction."""
    rng = np.random.default_rng(7)
    nlive, niter = 20, 400
    live_logl = list(rng.random(nlive))
    live_born = [0] * nlive
    dead_logl, dead_born = [], []
    true_ranks = []
    for it in range(1, niter + 1):
        worst = int(np.argmin(live_logl))
        threshold = live_logl[worst]
        dead_logl.append(live_logl.pop(worst))
        dead_born.append(live_born.pop(worst))
        new = threshold + (1.0 - threshold) * rng.random()  # exact: uniform above L*
        true_ranks.append(int(np.sum(np.asarray(live_logl) < new)))
        live_logl.append(new)
        live_born.append(it)
    order = np.argsort(live_logl)
    logl = np.r_[dead_logl, np.asarray(live_logl)[order]]
    born = np.r_[dead_born, np.asarray(live_born)[order]]
    ranks, excluded = insertion_index_ranks(logl, born, niter=niter, nlive=nlive)
    assert excluded == 0
    # The last insertion (iteration niter) replaced row niter - 1 and is reconstructed too.
    assert ranks.tolist() == true_ranks
    result = insertion_index_test(ranks, nlive)
    assert result["n_insertions"] == niter
    assert result["p_value"] > 0.01
    assert np.isnan(insertion_index_test([], nlive)["p_value"])
