import json
import pickle
import threading

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

dynesty = pytest.importorskip("dynesty")
dynesty_utils = pytest.importorskip("dynesty.utils")

from gwpop_search.data import Campaign, SelectionCatalog, SelectionMode  # noqa: E402
from gwpop_search.data.fixtures import (  # noqa: E402
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import baseline_model_spec  # noqa: E402
from gwpop_search.hbi import (  # noqa: E402
    HBIConfig,
    PopulationDensityError,
    evaluate_catalog_terms,
    shape_log_likelihood,
)
from gwpop_search.hbi.jax_backend import build_shape_log_likelihood  # noqa: E402
from gwpop_search.inference import (  # noqa: E402
    DynestyConfig,
    PriorSpec,
    ThreadBatchPool,
    build_batched_log_likelihood,
    build_importance_diagnostics,
    equal_weight_resample,
    importance_diagnostics_over_posterior,
    load_dynesty_result,
    prior_specs_from_model_spec,
    prior_transform_for,
    run_dynesty,
    run_dynesty_population,
    summarize_evidence_repeats,
)
from gwpop_search.inference.dynesty_backend import _new_sampler  # noqa: E402
from gwpop_search.inference.synthetic import (  # noqa: E402
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import compile_model_spec  # noqa: E402


# ---------------------------------------------------------------------------
# Analytic problems (top-level classes so dynesty checkpoints can pickle them)
# ---------------------------------------------------------------------------

BOX = 5.0
MU = np.array([0.3, -0.7, 1.1])
SIGMA = 0.5


class GaussianLikelihood:
    """Normalized isotropic Gaussian likelihood, evaluated for a block of rows."""

    def __init__(self, mu=MU, sigma=SIGMA):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = float(sigma)

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        d = self.mu.size
        return -0.5 * np.sum(((X - self.mu) / self.sigma) ** 2, axis=1) - d * np.log(
            self.sigma * np.sqrt(2.0 * np.pi)
        )


def gaussian_problem(mu=MU):
    mu = np.asarray(mu, dtype=float)
    names, transform = prior_transform_for(
        {f"x{i}": PriorSpec("uniform", low=-BOX, high=BOX) for i in range(mu.size)}
    )
    per_dim = np.log(ndtr((BOX - mu) / SIGMA) - ndtr((-BOX - mu) / SIGMA)) - np.log(2 * BOX)
    return names, transform, float(per_dim.sum())


class SimulatedKill(Exception):
    """Stands in for a job being killed: the process state is lost."""


class KillAfter:
    """Batched likelihood that 'kills the job' after a fixed number of device calls."""

    def __init__(self, inner, n_calls):
        self.inner = inner
        self.n_calls = n_calls
        self.calls = 0

    def __call__(self, X):
        self.calls += 1
        if self.calls > self.n_calls:
            raise SimulatedKill(f"killed after {self.n_calls} batched calls")
        return self.inner(X)


def assert_results_identical(a, b):
    assert a.names == b.names
    assert a.log_evidence == b.log_evidence
    assert a.log_evidence_error == b.log_evidence_error
    assert a.information == b.information
    assert a.niter == b.niter
    assert a.ncall == b.ncall
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.log_weights, b.log_weights)
    np.testing.assert_array_equal(a.log_likelihoods, b.log_likelihoods)
    np.testing.assert_array_equal(a.log_volumes, b.log_volumes)
    np.testing.assert_array_equal(a.posterior_samples, b.posterior_samples)


# ---------------------------------------------------------------------------
# Prior transform and configuration
# ---------------------------------------------------------------------------


def test_prior_transform_maps_unit_cube_for_every_family():
    priors = {
        "b": PriorSpec("log_uniform", low=0.1, high=10.0),
        "a": PriorSpec("uniform", low=-2.0, high=6.0),
        "c": PriorSpec("normal", loc=1.5, scale=0.3),
    }
    names, transform = prior_transform_for(priors)
    assert names == ("a", "b", "c")

    theta = transform(np.array([0.25, 0.5, 0.975]))
    np.testing.assert_allclose(
        theta, [-2.0 + 0.25 * 8.0, 0.1 * 100.0**0.5, 1.5 + 0.3 * ndtri(0.975)], rtol=1e-14
    )
    np.testing.assert_allclose(transform(np.array([0.0, 0.0, 0.5]))[:2], [-2.0, 0.1])
    np.testing.assert_allclose(transform(np.array([1.0, 1.0, 0.5]))[:2], [6.0, 10.0])

    u = np.random.default_rng(3).random((4000, 3))
    block = transform(u)
    np.testing.assert_array_equal(block[:5], np.stack([transform(row) for row in u[:5]]))
    # Distribution checks: flat in a, flat in log b, normal in c.
    assert abs(np.mean(block[:, 0]) - 2.0) < 0.15
    assert abs(np.mean(np.log10(block[:, 1]))) < 0.05
    assert abs(np.mean(block[:, 2]) - 1.5) < 0.02 and abs(np.std(block[:, 2]) - 0.3) < 0.02

    clone = pickle.loads(pickle.dumps(transform))
    assert clone == transform
    np.testing.assert_array_equal(clone(u), block)
    with pytest.raises(ValueError):
        transform(np.zeros(2))
    with pytest.raises(ValueError):
        prior_transform_for({})


def test_dynesty_config_validates_and_round_trips():
    cfg = DynestyConfig(
        nlive=123, sample="rwalk", walks=30, update_interval=0.5, maxcall=1000, batch_size=7
    )
    assert DynestyConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    as_calls = DynestyConfig(update_interval=250)
    restored = DynestyConfig.from_dict(json.loads(json.dumps(as_calls.to_dict())))
    assert restored == as_calls and isinstance(restored.update_interval, int)

    bad = [
        dict(nlive=1),
        dict(bound="ellipsoids"),
        dict(sample="auto"),
        dict(sample="rwalk", slices=4),
        dict(walks=10),
        dict(bootstrap=1),
        dict(enlarge=0.9),
        dict(enlarge=1.5, bootstrap=5),
        dict(dlogz=0.0),
        dict(maxiter=0),
        dict(batch_size=0),
        dict(checkpoint_every=0.0),
        dict(num_posterior_samples=0),
        dict(update_interval=True),
    ]
    for kwargs in bad:
        with pytest.raises((ValueError, TypeError)):
            DynestyConfig(**kwargs)
    with pytest.raises(TypeError):
        DynestyConfig(nlive=10.5)
    with pytest.raises(ValueError, match="unknown"):
        DynestyConfig.from_dict({"nlive": 10, "queue_size": 3})


# ---------------------------------------------------------------------------
# ThreadBatchPool
# ---------------------------------------------------------------------------


class RecordingLikelihood:
    """Analytic batched likelihood that records the size of every batch it sees."""

    def __init__(self, fail_on=None, nan_on=None):
        self.batches = []
        self.fail_on = fail_on
        self.nan_on = nan_on
        self._lock = threading.Lock()

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        with self._lock:
            self.batches.append(X.shape[0])
        values = -0.5 * np.sum(X**2, axis=1) + np.sin(3.0 * X[:, 0])
        if self.fail_on is not None and np.any(np.isclose(X[:, 0], self.fail_on)):
            raise RuntimeError("boom in batched likelihood")
        if self.nan_on is not None:
            values = np.where(np.isclose(X[:, 0], self.nan_on), np.nan, values)
        return values


def chained_calls(loglike, item):
    """A task whose sequential likelihood inputs depend on the previous outputs."""
    x = np.array([0.1 * item, 1.0, -0.5])
    total = 0.0
    for k in range(1 + item % 5):
        value = loglike(x)
        total += value
        x = x + 0.01 * (k + 1) + 1e-3 * value
    return total, x


def call_with_timeout(fn, timeout=30.0):
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"ThreadBatchPool call did not finish within {timeout} s (deadlock?)")
    if "error" in box:
        raise box["error"]
    return box["value"]


def sequential_reference(fn, items):
    return [chained_calls(lambda x: float(fn(x[None, :])[0]), item) for item in items]


def test_thread_batch_pool_matches_sequential_evaluation_and_batches_requests():
    fn = RecordingLikelihood()
    items = list(range(20))
    with ThreadBatchPool(fn, 8) as pool:
        got = call_with_timeout(
            lambda: pool.map(lambda item: chained_calls(pool.point_loglikelihood, item), items)
        )
        stats = pool.stats()
    reference = sequential_reference(RecordingLikelihood(), items)
    for (total, x), (ref_total, ref_x) in zip(got, reference):
        assert total == ref_total
        np.testing.assert_array_equal(x, ref_x)
    n_evaluations = sum(1 + item % 5 for item in items)
    assert stats["n_evaluations"] == n_evaluations
    assert max(fn.batches) <= 8
    assert stats["n_batches"] < n_evaluations  # requests were actually batched


@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("n_items", [1, 5, 23])
@pytest.mark.parametrize("n_workers", [None, 3])
def test_thread_batch_pool_has_no_deadlock_for_uneven_sizes(batch_size, n_items, n_workers):
    fn = RecordingLikelihood()
    items = list(range(n_items))
    with ThreadBatchPool(fn, batch_size, n_workers=n_workers) as pool:
        for _ in range(2):  # the pool is reusable across maps
            got = call_with_timeout(
                lambda: pool.map(lambda item: chained_calls(pool.point_loglikelihood, item), items)
            )
            assert [g[0] for g in got] == [r[0] for r in sequential_reference(fn, items)]
    assert max(fn.batches) <= batch_size


def test_thread_batch_pool_propagates_exceptions_and_stays_usable():
    fn = RecordingLikelihood(fail_on=0.3)
    with ThreadBatchPool(fn, 4) as pool:

        def task(item):
            if item == 6:
                raise ValueError("task 6 failed")
            return chained_calls(pool.point_loglikelihood, item)

        with pytest.raises(ValueError, match="task 6 failed"):
            call_with_timeout(lambda: pool.map(task, range(4, 16)))
        # Item 3 starts at x0 = 0.3: the batched likelihood itself raises, and
        # the first failing item (in item order) is reported.
        with pytest.raises(RuntimeError, match="boom in batched likelihood"):
            call_with_timeout(
                lambda: pool.map(lambda i: chained_calls(pool.point_loglikelihood, i), range(8))
            )
        # The pool keeps working after failures, including outside of map().
        good = call_with_timeout(
            lambda: pool.map(lambda i: chained_calls(pool.point_loglikelihood, i), [0, 1, 2])
        )
        assert len(good) == 3
        assert np.isfinite(pool.evaluate(np.array([0.5, 0.5, 0.5])))
        with pytest.raises(RuntimeError, match="own worker threads"):
            call_with_timeout(lambda: pool.map(lambda i: pool.map(abs, [i]), [1]))

    with ThreadBatchPool(RecordingLikelihood(nan_on=0.2), 4) as pool:
        with pytest.raises(PopulationDensityError):
            call_with_timeout(
                lambda: pool.map(lambda i: chained_calls(pool.point_loglikelihood, i), range(5))
            )
    with pytest.raises(RuntimeError, match="closed"):
        pool.map(abs, [1])
    with pytest.raises(TypeError):
        pickle.dumps(pool)


def test_pooled_likelihood_proxy_pickles_unbound_and_close_never_strands_map():
    with ThreadBatchPool(RecordingLikelihood(), 2) as pool:
        proxy = pickle.loads(pickle.dumps(pool.point_loglikelihood))
        with pytest.raises(RuntimeError, match="not bound"):
            proxy(np.zeros(3))
        proxy.bind(pool)
        assert np.isfinite(proxy(np.zeros(3)))

    pool = ThreadBatchPool(RecordingLikelihood(), 2, n_workers=1)
    started = threading.Event()
    release = threading.Event()

    def slow(item):
        started.set()
        release.wait(10.0)
        return item

    box = {}

    def run_map():
        try:
            box["value"] = pool.map(slow, range(4))
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=run_map, daemon=True)
    thread.start()
    assert started.wait(10.0)
    closer = threading.Thread(target=pool.close, daemon=True)
    closer.start()
    release.set()
    thread.join(10.0)
    closer.join(10.0)
    assert not thread.is_alive(), "map() hung after close()"
    assert isinstance(box.get("error"), RuntimeError)


def test_dynesty_initial_live_points_are_evaluated_in_full_batches():
    _, transform, _ = gaussian_problem(mu=[0.2, -0.4])
    fn = RecordingLikelihood()
    cfg = DynestyConfig(nlive=40, batch_size=8)
    with ThreadBatchPool(fn, cfg.batch_size) as pool:
        sampler = _new_sampler(dynesty, pool, transform, 2, seed=1, config=cfg)
        stats = pool.stats()
    assert sampler.ncall == stats["n_evaluations"] == 40
    assert fn.batches == [8, 8, 8, 8, 8]


def test_equal_weight_resample_reproduces_dynesty_resample_equal():
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(200, 2))
    weights = rng.random(200) ** 3
    weights /= weights.sum()
    ours = equal_weight_resample(samples, weights, 200, np.random.default_rng(9))
    reference = dynesty_utils.resample_equal(samples, weights, rstate=np.random.default_rng(9))
    np.testing.assert_array_equal(ours, reference)

    index = equal_weight_resample(np.arange(200), weights, 1000, np.random.default_rng(1))
    counts = np.bincount(index, minlength=200)
    assert index.shape == (1000,)
    assert np.all(np.abs(counts - 1000 * weights) < 1.0)


# ---------------------------------------------------------------------------
# run_dynesty on analytic likelihoods
# ---------------------------------------------------------------------------


def test_run_dynesty_recovers_analytic_gaussian_evidence():
    names, transform, truth = gaussian_problem()
    cfg = DynestyConfig(nlive=250, batch_size=8, dlogz=0.1, num_posterior_samples=2000)
    result = run_dynesty(GaussianLikelihood(), transform, 3, seed=2024, config=cfg, names=names)

    error = abs(result.log_evidence - truth)
    assert error < 3.0 * result.log_evidence_error
    assert error < 0.3
    assert result.diagnostics["converged"]
    assert result.posterior_samples.shape == (2000, 3)
    np.testing.assert_allclose(result.posterior_samples.mean(axis=0), MU, atol=0.1)
    np.testing.assert_allclose(result.posterior_samples.std(axis=0), SIGMA, rtol=0.2)
    assert result.ncall > result.niter > 0
    assert 0 < result.kish_ess <= result.samples.shape[0]
    assert set(result.versions) == {"dynesty", "jax", "numpy"}
    pool = result.diagnostics["session_pool"]
    assert pool["n_evaluations"] >= result.ncall - cfg.nlive  # every call went through the pool
    assert pool["mean_batch_fill"] > 1.0


class QuadrantGaussianLikelihood:
    """Gaussian likelihood with zero support (-inf) outside the positive quadrant.

    Three quarters of the prior box have L = 0, mimicking the hard support
    edges of the population likelihood (most prior draws give -inf).
    """

    mu = np.array([1.0, 1.5])

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        value = -0.5 * np.sum(((X - self.mu) / SIGMA) ** 2, axis=1) - 2 * np.log(
            SIGMA * np.sqrt(2.0 * np.pi)
        )
        return np.where(np.all(X > 0.0, axis=1), value, -np.inf)


def test_run_dynesty_evidence_with_a_zero_likelihood_region():
    names, transform, _ = gaussian_problem(mu=QuadrantGaussianLikelihood.mu)
    mu = QuadrantGaussianLikelihood.mu
    truth = float(
        np.sum(np.log(ndtr((BOX - mu) / SIGMA) - ndtr((0.0 - mu) / SIGMA))) - 2 * np.log(2 * BOX)
    )
    cfg = DynestyConfig(nlive=200, batch_size=8, dlogz=0.1, num_posterior_samples=1000)
    result = run_dynesty(
        QuadrantGaussianLikelihood(), transform, 2, seed=77, config=cfg, names=names
    )
    error = abs(result.log_evidence - truth)
    assert error < 3.0 * result.log_evidence_error
    assert error < 0.3
    # Initial live points drawn in the L = 0 region are reported with -inf.
    assert result.diagnostics["n_zero_likelihood_points"] > 0
    zero = np.isneginf(result.log_likelihoods)
    assert np.all(np.isneginf(result.log_weights[zero]))
    assert np.all(result.posterior_samples > 0.0)


def test_run_dynesty_is_deterministic_for_a_fixed_seed():
    _, transform, _ = gaussian_problem(mu=[0.2, -0.4])
    cfg = DynestyConfig(nlive=60, batch_size=8, dlogz=0.3, num_posterior_samples=300)
    first = run_dynesty(GaussianLikelihood(mu=[0.2, -0.4]), transform, 2, seed=5, config=cfg)
    second = run_dynesty(GaussianLikelihood(mu=[0.2, -0.4]), transform, 2, seed=5, config=cfg)
    assert_results_identical(first, second)
    other = run_dynesty(GaussianLikelihood(mu=[0.2, -0.4]), transform, 2, seed=6, config=cfg)
    assert other.log_evidence != first.log_evidence
    # Results plug into the existing repeated-evidence summary used for scoring.
    summary = summarize_evidence_repeats([first, other])
    assert summary.n_repeats == 2
    assert summary.log_evidence_mean == pytest.approx(
        0.5 * (first.log_evidence + other.log_evidence)
    )


def test_checkpoint_resume_after_a_kill_reproduces_the_uninterrupted_run(tmp_path):
    """Kill mid-run, resume from dynesty's last checkpoint, compare bit for bit.

    dynesty pickles the complete sampler state (generator, live points,
    bounds, queued proposals) between iterations, so resuming continues the
    identical trajectory and the result equals the uninterrupted run.
    """
    names, transform, _ = gaussian_problem(mu=[0.2, -0.4])
    loglike = GaussianLikelihood(mu=[0.2, -0.4])
    cfg = DynestyConfig(
        nlive=60, batch_size=8, dlogz=0.2, checkpoint_every=1e-6, num_posterior_samples=300
    )
    reference = run_dynesty(loglike, transform, 2, seed=11, config=cfg, names=names)
    n_batches = reference.diagnostics["session_pool"]["n_batches"]

    checkpoint = tmp_path / "gauss.pkl"
    with pytest.raises(SimulatedKill):
        run_dynesty(
            KillAfter(loglike, n_batches // 2),
            transform,
            2,
            seed=11,
            config=cfg,
            checkpoint_file=checkpoint,
            names=names,
        )
    with pytest.warns(UserWarning, match="queue_size"):
        restored = dynesty.NestedSampler.restore(str(checkpoint))  # no pool: inspection only
    assert not restored.added_live and 1 < restored.it < reference.niter

    with pytest.raises(FileExistsError):
        run_dynesty(loglike, transform, 2, seed=11, config=cfg, checkpoint_file=checkpoint)
    with pytest.raises(ValueError, match="seed"):
        run_dynesty(
            loglike, transform, 2, seed=12, config=cfg, checkpoint_file=checkpoint,
            resume=True, names=names,
        )
    resumed = run_dynesty(
        loglike,
        transform,
        2,
        seed=11,
        config=cfg,
        checkpoint_file=checkpoint,
        resume=True,
        names=names,
    )
    assert resumed.diagnostics["resumed"]
    assert_results_identical(resumed, reference)

    # The final checkpoint holds the finished run: resuming only rebuilds the result.
    again = run_dynesty(
        loglike, transform, 2, seed=11, config=cfg, checkpoint_file=checkpoint, resume=True
    )
    assert again.diagnostics["finished_in_checkpoint"]
    assert again.diagnostics["session_pool"]["n_evaluations"] == 0
    assert again.log_evidence == reference.log_evidence


def test_checkpoint_resume_after_a_maxiter_interruption(tmp_path):
    """Interrupt with dynesty's maxiter (add_live=False), then resume.

    Stopping on maxiter makes dynesty recompute logz/logzvar/H with
    ``compute_integrals`` before writing the checkpoint, which rounds
    differently from the incremental update it would otherwise carry. Those
    quantities enter the continuation only through the dlogz stopping test
    (and they are recomputed from logl/logvol at the end); the sampling state
    (generator, live points, bounds, queue) is untouched. The resumed run is
    therefore identical unless the stopping test falls within rounding of
    dlogz, which does not happen here; the result is compared bit for bit.
    Note that ``add_live=True`` (dynesty's default) would instead finalize the
    run and make it non-resumable.
    """
    names, transform, _ = gaussian_problem(mu=[0.2, -0.4])
    loglike = GaussianLikelihood(mu=[0.2, -0.4])
    cfg = DynestyConfig(nlive=60, batch_size=8, dlogz=0.2, num_posterior_samples=300)
    reference = run_dynesty(loglike, transform, 2, seed=13, config=cfg, names=names)

    checkpoint = tmp_path / "partial.pkl"
    with ThreadBatchPool(loglike, cfg.batch_size) as pool:
        sampler = _new_sampler(dynesty, pool, transform, 2, seed=13, config=cfg, names=names)
        with pytest.warns(UserWarning, match="stopped short"):
            sampler.run_nested(
                maxiter=reference.niter // 2,
                dlogz=cfg.dlogz,
                add_live=False,
                print_progress=False,
                save_bounds=False,
                checkpoint_file=str(checkpoint),
            )
    resumed = run_dynesty(
        loglike,
        transform,
        2,
        seed=13,
        config=cfg,
        checkpoint_file=checkpoint,
        resume=True,
        names=names,
    )
    assert_results_identical(resumed, reference)


@pytest.mark.parametrize("budget", [{"maxiter": 120}, {"maxcall": 600}])
def test_budgets_are_totals_across_resumes(tmp_path, budget):
    _, transform, _ = gaussian_problem(mu=[0.2, -0.4])
    loglike = GaussianLikelihood(mu=[0.2, -0.4])
    cfg = DynestyConfig(
        nlive=40, batch_size=4, checkpoint_every=1e-6, num_posterior_samples=100, **budget
    )
    with pytest.warns(UserWarning, match="stopped short"):
        reference = run_dynesty(loglike, transform, 2, seed=17, config=cfg)
    assert not reference.diagnostics["converged"]  # the budget, not dlogz, stopped the run
    if cfg.maxiter is not None:
        assert reference.niter == cfg.maxiter + 1  # dynesty semantics
    else:
        assert reference.ncall - cfg.nlive > cfg.maxcall  # stops once the budget is exceeded

    checkpoint = tmp_path / "budget.pkl"
    with pytest.raises(SimulatedKill):
        run_dynesty(
            KillAfter(loglike, reference.diagnostics["session_pool"]["n_batches"] // 2),
            transform,
            2,
            seed=17,
            config=cfg,
            checkpoint_file=checkpoint,
        )
    with pytest.warns(UserWarning, match="stopped short"):
        resumed = run_dynesty(
            loglike, transform, 2, seed=17, config=cfg, checkpoint_file=checkpoint, resume=True
        )
    assert_results_identical(resumed, reference)


# ---------------------------------------------------------------------------
# Batched HBI likelihood
# ---------------------------------------------------------------------------

TOY_NAMES = ("a", "b", "mcut")


def toy_density_np(samples, hp):
    value = (
        -0.02 * samples["m1_source"]
        + hp["a"] * samples["q"]
        + hp["b"] * samples["chi_eff"]
        - 0.1 * samples["z"]
    )
    return np.where(samples["m1_source"] <= hp["mcut"], value, -np.inf)


def toy_density_jax(samples, hp):
    value = (
        -0.02 * samples["m1_source"]
        + hp["a"] * samples["q"]
        + hp["b"] * samples["chi_eff"]
        - 0.1 * samples["z"]
    )
    value = jnp.where(samples["m1_source"] <= hp["mcut"], value, -jnp.inf)
    # A deliberate bug for a > 5: NaN instead of a density.
    return jnp.where(hp["a"] > 5.0, jnp.nan, value)


TOY_POINTS = np.array(
    [
        [0.4, -0.25, 100.0],
        [0.1, 0.35, 100.0],
        [2.0, 1.5, 45.0],
        [-1.0, 0.8, 30.0],
        [0.7, -2.0, 60.0],
    ]
)


def test_batched_likelihood_matches_numpy_reference_and_jax_builder():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    loglike = build_batched_log_likelihood(pe, sel, toy_density_jax, TOY_NAMES, batch_size=3)
    values = loglike(TOY_POINTS)
    assert values.shape == (5,) and values.dtype == np.float64

    single = build_shape_log_likelihood(pe, sel, toy_density_jax)
    for row, value in zip(TOY_POINTS, values):
        hp = dict(zip(TOY_NAMES, row))
        reference = shape_log_likelihood(pe, sel, toy_density_np, hp).log_likelihood
        np.testing.assert_allclose(value, reference, rtol=0, atol=1e-11)
        np.testing.assert_allclose(value, float(single(hp)), rtol=0, atol=1e-12)

    chunked = build_batched_log_likelihood(
        pe, sel, toy_density_jax, TOY_NAMES, hbi_config=HBIConfig(selection_chunk_size=4)
    )
    np.testing.assert_allclose(chunked(TOY_POINTS), values, rtol=0, atol=1e-12)


def test_batched_likelihood_keeps_zero_support_and_raises_on_nan():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    loglike = build_batched_log_likelihood(pe, sel, toy_density_jax, TOY_NAMES, batch_size=4)
    # mcut below every PE sample of at least one event: genuine zero support.
    values = loglike(np.array([[0.4, -0.25, 12.0], [0.4, -0.25, 100.0]]))
    assert np.isneginf(values[0]) and np.isfinite(values[1])
    assert loglike.stats()["n_zero_support"] == 1

    with pytest.raises(PopulationDensityError, match="a=6"):
        loglike(np.array([[0.4, -0.25, 100.0], [6.0, 0.0, 100.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        loglike(np.array([[np.nan, 0.0, 100.0]]))
    with pytest.raises(ValueError, match="shape"):
        loglike(np.zeros((2, 2)))


def test_batched_likelihood_rows_do_not_depend_on_batch_composition():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    loglike = build_batched_log_likelihood(pe, sel, toy_density_jax, TOY_NAMES, batch_size=8)
    rng = np.random.default_rng(0)
    target = TOY_POINTS[1]
    reference = loglike(target[None, :])[0]
    for position in range(8):
        companions = np.column_stack(
            [rng.uniform(-1, 2, 8), rng.uniform(-2, 2, 8), rng.uniform(20, 100, 8)]
        )
        companions[position] = target
        assert loglike(companions)[position] == reference


# ---------------------------------------------------------------------------
# Population runs and importance diagnostics on a tiny synthetic survey
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_survey():
    config = SyntheticSurveyConfig(
        n_events=6,
        posterior_samples_per_event=32,
        n_injections=3_000,
        population_batch_size=512,
        redshift_sampling_grid=1024,
    )
    dataset = generate_baseline_synthetic_dataset(seed=5, config=config)
    spec = baseline_model_spec()
    return dataset, compile_model_spec(spec), prior_specs_from_model_spec(spec)


@pytest.fixture(scope="module")
def tiny_population_run(tiny_survey, tmp_path_factory):
    dataset, model, priors = tiny_survey
    run_dir = tmp_path_factory.mktemp("dynesty-population") / "run"
    cfg = DynestyConfig(nlive=50, batch_size=8, maxiter=150, num_posterior_samples=200)
    with pytest.warns(UserWarning, match="stopped short"):
        result = run_dynesty_population(
            dataset.posterior,
            dataset.selection,
            model,
            priors,
            seed=3,
            config=cfg,
            hbi_config=HBIConfig(selection_chunk_size=None),
            run_dir=run_dir,
        )
    return result, run_dir, cfg


def test_population_run_on_tiny_synthetic_survey_is_finite_and_resumable(
    tiny_survey, tiny_population_run
):
    dataset, model, priors = tiny_survey
    result, run_dir, cfg = tiny_population_run
    names = tuple(sorted(priors))
    assert result.names == names
    assert np.isfinite(result.log_evidence) and np.isfinite(result.log_evidence_error)
    assert result.niter == cfg.maxiter + 1
    assert result.posterior_samples.shape == (200, len(names))
    for k, name in enumerate(names):
        spec = priors[name]
        assert np.all(result.posterior_samples[:, k] >= spec.low)
        assert np.all(result.posterior_samples[:, k] <= spec.high)
    assert np.all(np.isfinite(result.log_likelihoods[-cfg.nlive :]))
    assert (run_dir / "manifest.json").exists() and (run_dir / "checkpoint.pkl").exists()

    loaded = load_dynesty_result(run_dir / "result.npz")
    assert_results_identical(loaded, result)
    assert loaded.config == cfg and loaded.seed == 3

    again = run_dynesty_population(
        dataset.posterior,
        dataset.selection,
        model,
        priors,
        seed=3,
        config=cfg,
        hbi_config=HBIConfig(selection_chunk_size=None),
        run_dir=run_dir,
    )
    assert_results_identical(again, result)
    with pytest.raises(ValueError, match="root_seed"):
        run_dynesty_population(
            dataset.posterior,
            dataset.selection,
            model,
            priors,
            seed=4,
            config=cfg,
            hbi_config=HBIConfig(selection_chunk_size=None),
            run_dir=run_dir,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["data"]["event_names"] == list(dataset.posterior.event_names)
    assert manifest["dynesty_config"] == cfg.to_dict()


def estimator_ready_toy_selection():
    base = make_toy_selection_catalog()
    pdraw = np.linspace(0.2, 1.1, base.n_selected)
    return SelectionCatalog(
        samples=base.samples,
        log_draw_density=np.log(pdraw),
        campaign_id=np.asarray(["combined"] * base.n_selected),
        campaigns=(Campaign("combined", n_draw=5000, observing_time_yr=1.5),),
        basis=base.basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics="already normalized pdraw",
    )


@pytest.mark.parametrize("selection_kind", ["raw_draw", "estimator_ready"])
@pytest.mark.parametrize("chunk", [None, 5])
def test_jax_importance_diagnostics_match_numpy_reference(selection_kind, chunk):
    pe = make_toy_posterior_catalog()
    if selection_kind == "raw_draw":
        sel = make_toy_selection_catalog()
    else:
        sel = estimator_ready_toy_selection()
    fn = build_importance_diagnostics(
        pe, sel, toy_density_jax, TOY_NAMES, batch_size=2, selection_chunk_size=chunk
    )
    # The last point removes all PE support of at least one event (ESS = 0).
    points = np.vstack([TOY_POINTS, [[0.4, -0.25, 16.0]]])
    batch = fn(points)
    assert batch.campaign_ids == tuple(c.campaign_id for c in sel.campaigns)

    rtol = 1e-10
    for k, row in enumerate(points):
        ref = evaluate_catalog_terms(pe, sel, toy_density_np, dict(zip(TOY_NAMES, row)))
        events = ref.events
        diags = events.diagnostics
        np.testing.assert_allclose(
            batch.event_log_likelihoods[k], events.log_likelihoods, rtol=rtol
        )
        np.testing.assert_allclose(batch.event_ess[k], [d.ess for d in diags], rtol=rtol)
        np.testing.assert_allclose(
            batch.event_ess_fraction[k], [d.ess_fraction_of_draws for d in diags], rtol=rtol
        )
        np.testing.assert_allclose(
            batch.event_max_weight[k], [d.max_weight_fraction for d in diags], rtol=rtol
        )
        np.testing.assert_allclose(
            batch.event_variance[k],
            [d.variance_log_estimate for d in events.diagnostics],
            rtol=rtol,
            atol=1e-15,
        )
        np.testing.assert_array_equal(
            batch.event_n_zero_weight[k], [d.n_zero_weight for d in events.diagnostics]
        )
        sel_ref = ref.selection
        np.testing.assert_allclose(batch.log_exposure[k], sel_ref.log_exposure, rtol=rtol)
        np.testing.assert_allclose(batch.selection_ess[k], sel_ref.diagnostics.ess, rtol=rtol)
        np.testing.assert_allclose(
            batch.selection_ess_fraction[k], sel_ref.diagnostics.ess_fraction_of_draws, rtol=rtol
        )
        np.testing.assert_allclose(
            batch.selection_max_weight[k], sel_ref.diagnostics.max_weight_fraction, rtol=rtol
        )
        assert batch.selection_n_zero_weight[k] == sel_ref.diagnostics.n_zero_weight
        for c, campaign in enumerate(sel_ref.campaigns):
            np.testing.assert_allclose(
                batch.campaign_log_exposure[k, c], campaign.log_exposure, rtol=rtol
            )
            np.testing.assert_allclose(
                batch.campaign_ess[k, c], campaign.diagnostics.ess, rtol=rtol
            )
            np.testing.assert_allclose(
                batch.campaign_max_weight[k, c], campaign.diagnostics.max_weight_fraction, rtol=rtol
            )
            np.testing.assert_allclose(
                batch.campaign_variance[k, c],
                campaign.diagnostics.variance_log_estimate,
                rtol=rtol,
                atol=1e-15,
            )
        np.testing.assert_allclose(
            batch.selection_variance[k], sel_ref.variance_log_exposure, rtol=rtol, atol=1e-15
        )
        np.testing.assert_allclose(
            batch.event_variance_total[k], ref.variance.event_variance, rtol=rtol, atol=1e-15
        )
        np.testing.assert_allclose(
            batch.shape_log_likelihood_variance[k],
            ref.variance.shape_log_likelihood_variance,
            rtol=rtol,
            atol=1e-15,
        )
        expected = ref.events.log_likelihood - pe.n_events * sel_ref.log_exposure
        np.testing.assert_allclose(batch.log_likelihood[k], expected, rtol=rtol)

    # No PE support for an event: ESS = 0 and infinite Var[log ell_i]. This point
    # also leaves a campaign without support, so (as in the reference) Var[log A]
    # and therefore Var[log L] are NaN rather than a finite number.
    assert np.min(batch.event_ess[-1]) == 0.0
    assert np.isinf(np.max(batch.event_variance[-1]))
    assert not np.isfinite(batch.shape_log_likelihood_variance[-1])
    with pytest.raises(PopulationDensityError):
        fn(np.array([[6.0, 0.0, 100.0]]))


def test_importance_diagnostics_over_posterior(tiny_survey, tiny_population_run):
    dataset, model, _ = tiny_survey
    result, _, _ = tiny_population_run
    summary = importance_diagnostics_over_posterior(
        result,
        dataset.posterior,
        dataset.selection,
        model,
        n_draws=40,
        quantiles=(0.5, 0.9, 0.99),
        batch_size=8,
    )
    assert summary.n_draws == 40
    assert summary.batch.hyperparameters.shape == (41, len(result.names))
    assert set(summary.median_hyperparameters) == set(result.names)
    stats = summary.at_median
    assert stats["worst_event"] in dataset.posterior.event_names
    assert np.isfinite(stats["log_likelihood"])
    assert stats["shape_log_likelihood_variance"] >= 0.0
    for name in ("shape_log_likelihood_variance", "min_event_ess", "selection_ess"):
        assert set(summary.over_posterior[name]) == {"q0.5", "q0.9", "q0.99"}
        q = summary.over_posterior[name]
        assert q["q0.5"] <= q["q0.9"] <= q["q0.99"]
    json.dumps(summary.to_dict())

    # The jitted diagnostics reproduce the batched likelihood on the same points.
    loglike = build_batched_log_likelihood(
        dataset.posterior, dataset.selection, model, result.names, batch_size=8
    )
    np.testing.assert_allclose(
        summary.batch.log_likelihood,
        loglike(summary.batch.hyperparameters),
        rtol=0,
        atol=1e-8,
    )
