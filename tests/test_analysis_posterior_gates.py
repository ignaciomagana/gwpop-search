import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from gwpop_search.analysis import toys  # noqa: E402
from gwpop_search.analysis._common import pool_dynesty_results  # noqa: E402
from gwpop_search.analysis.posterior_gates import (  # noqa: E402
    PosteriorGateCriteria,
    cross_run_rhat,
    evaluate_posterior_gates,
    rank_normalized_split_rhat,
)
from gwpop_search.data.fixtures import make_toy_posterior_catalog, make_toy_selection_catalog  # noqa: E402
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference import DynestyConfig, PriorSpec, run_dynesty_population  # noqa: E402
from gwpop_search.validation.holdout import (  # noqa: E402
    _reference_heldout_log_predictive,
    heldout_detected_log_predictive,
)

HBI = HBIConfig(selection_chunk_size=None)


def test_rank_normalized_rhat_matches_arviz():
    array_stats = pytest.importorskip("arviz_stats.base").array_stats
    rng = np.random.default_rng(0)
    for shift in (0.0, 0.3):
        chains = rng.normal(size=(3, 400))
        chains[0] += shift
        assert rank_normalized_split_rhat(chains) == pytest.approx(
            float(array_stats.rhat(chains, method="rank")), rel=1e-10
        )


@pytest.fixture(scope="module")
def toy_runs(tmp_path_factory):
    obs = toys.ToyObservation()
    rng = np.random.default_rng(21)
    d = toys.toy_observed_data(rng, lambda r, n: r.normal(0.0, 1.0, n), 12, obs)
    pe = toys.toy_posterior_catalog(rng, d, 120, obs)
    sel = toys.toy_selection_catalog(rng, 12000, obs)
    model = toys.GaussianToy()
    priors = {"mu": PriorSpec("uniform", low=-2.0, high=2.0), "sigma": PriorSpec("uniform", low=0.3, high=3.0)}
    cfg = DynestyConfig(nlive=250, sample="unif", dlogz=0.1, batch_size=64, num_posterior_samples=1500)
    root = tmp_path_factory.mktemp("gates")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = [
            run_dynesty_population(pe, sel, model, priors, seed=s, config=cfg, hbi_config=HBI, run_dir=root / f"r{s}")
            for s in (1, 2)
        ]
    return pe, sel, model, runs


def test_posterior_gates_on_repeated_runs(toy_runs):
    pe, sel, model, runs = toy_runs
    rhat = cross_run_rhat(runs)
    assert rhat["max"] < 1.05 and set(rhat["per_parameter"]) == {"mu", "sigma"}
    lenient = PosteriorGateCriteria(
        max_cross_run_rhat=1.05, min_kish_ess_per_run=200.0, min_event_ess=5.0, min_selection_ess=50.0,
        max_event_weight_fraction=0.5, max_selection_weight_fraction=0.25, max_shape_log_likelihood_variance=4.0,
        n_draws=64,
    )
    out = evaluate_posterior_gates(runs, pe, sel, model, criteria=lenient, seed=3)
    assert out["passed"], [c for c in out["checks"] if not c["passed"]]
    names = {c["name"] for c in out["checks"]}
    for metric in ("min_event_ess", "selection_ess", "max_event_max_weight", "selection_max_weight",
                   "shape_log_likelihood_variance"):
        assert {f"{metric}_at_posterior_median", f"{metric}_median_over_posterior"} <= names
    strict = PosteriorGateCriteria.f4()
    failed = evaluate_posterior_gates(runs, pe, sel, model, criteria=strict, seed=3)
    assert not failed["passed"]
    kish = [c for c in failed["checks"] if c["name"].startswith("kish_ess")]
    assert kish and not any(c["passed"] for c in kish)  # nlive 250 cannot give Kish >= 2000
    single = evaluate_posterior_gates(runs[:1], pe, sel, model, criteria=lenient, seed=3)
    assert not single["passed"]  # cross-run R-hat needs two repeats


def test_vectorized_heldout_predictive_matches_the_numpy_reference(toy_runs):
    pe, sel, model, runs = toy_runs
    pooled = pool_dynesty_results(runs, n_draws=60, seed=4)
    samples = {name: pooled.points[:, k] for k, name in enumerate(pooled.names)}
    heldout = pe.event_names[:3]
    fast = heldout_detected_log_predictive(pe, sel, model, samples, heldout_events=heldout, config=HBI,
                                           log_weights=np.log(pooled.weights), batch_size=16)
    ref = _reference_heldout_log_predictive(pe, sel, model, samples, heldout_events=heldout, config=HBI,
                                            log_weights=np.log(pooled.weights))
    for name in heldout:
        assert fast[name] == pytest.approx(ref[name], abs=1e-10)
    # weighted points equal their resampled expansion
    counts = np.rint(pooled.weights * 60).astype(int)
    expanded = {k: np.repeat(v, counts) for k, v in samples.items()}
    equal = heldout_detected_log_predictive(pe, sel, model, expanded, heldout_events=heldout, config=HBI)
    for name in heldout:
        assert equal[name] == pytest.approx(fast[name], abs=1e-10)
    with pytest.raises(ValueError, match="unknown held-out"):
        heldout_detected_log_predictive(pe, sel, model, samples, heldout_events=("nope",))


def test_heldout_predictive_on_multicampaign_toy_catalogs():
    pe, sel = make_toy_posterior_catalog(), make_toy_selection_catalog()

    def density(samples, hp):
        import jax.numpy as jnp

        return jnp.asarray(-0.02 * samples["m1_source"] + hp["a"] * samples["q"] + hp["b"] * samples["chi_eff"])

    samples = {"a": np.array([0.1, 0.4, -0.2]), "b": np.array([0.3, -0.1, 0.0])}
    fast = heldout_detected_log_predictive(pe, sel, density, samples, heldout_events=("GWTOY_B",))
    ref = _reference_heldout_log_predictive(pe, sel, density, samples, heldout_events=("GWTOY_B",),
                                            config=HBI)
    assert fast["GWTOY_B"] == pytest.approx(ref["GWTOY_B"], abs=1e-11)
