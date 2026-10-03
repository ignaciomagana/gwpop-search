"""dynesty initialization under -inf regions: kept-volume error and the stall limit."""
import math

import numpy as np
import pytest

pytest.importorskip("dynesty")

from gwpop_search.inference.dynesty_backend import (  # noqa: E402
    DynestyConfig,
    InsufficientFiniteSupportError,
    initial_volume_uncertainty,
    run_dynesty,
)

KEPT = 0.05  # the likelihood is finite only on x0 < KEPT of the unit square
SIGMA = 0.1


def _unit(u):
    return np.asarray(u, dtype=float)


def _cut_gaussian(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    out = -0.5 * ((x[:, 1] - 0.5) / SIGMA) ** 2
    return np.where(x[:, 0] < KEPT, out, -np.inf)


def test_initial_volume_uncertainty_is_the_binomial_relative_error():
    full = initial_volume_uncertainty(0.0, n_finite_initial=500, nlive=500)
    assert full["n_init_batches"] == 1 and full["log_evidence_error"] == 0.0
    cut = initial_volume_uncertainty(-math.log(40), n_finite_initial=100, nlive=500)
    assert cut["n_init_batches"] == 40
    np.testing.assert_allclose(cut["log_evidence_error"], math.sqrt(1 / 100 - 1 / 20000))
    np.testing.assert_allclose(cut["kept_fraction_estimate"], 100 / 20000)
    with pytest.raises(ValueError):
        initial_volume_uncertainty(0.0, n_finite_initial=0, nlive=500)


def test_evidence_over_the_full_prior_carries_the_initial_volume_error():
    config = DynestyConfig(nlive=200, batch_size=16, dlogz=0.05, num_posterior_samples=200)
    result = run_dynesty(_cut_gaussian, _unit, 2, seed=3, config=config)
    init = result.diagnostics["initial_volume"]
    # 200 draws per batch keep ~10 points: dynesty needs ~10 batches for 100
    assert init["n_init_batches"] >= 5 and init["n_finite_initial"] >= 100
    assert init["log_evidence_error"] > 0.08
    np.testing.assert_allclose(
        result.log_evidence_error, math.hypot(init["dynesty_logzerr"], init["log_evidence_error"])
    )
    # ln Z over the FULL prior: KEPT * integral of the Gaussian in x1 over [0, 1]
    from scipy.stats import norm

    truth = math.log(KEPT * SIGMA * math.sqrt(2 * math.pi) * (norm.cdf(5.0) - norm.cdf(-5.0)))
    assert abs(result.log_evidence - truth) < 4.0 * result.log_evidence_error


def test_initialization_stall_is_a_typed_failure_not_an_endless_loop():
    calls = {"n": 0}

    def three_finite_points(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = np.full(x.shape[0], -np.inf)
        for i in range(x.shape[0]):
            if calls["n"] < 3:
                out[i] = 0.0
            calls["n"] += 1
        return out

    config = DynestyConfig(nlive=30, batch_size=30, dlogz=0.5, num_posterior_samples=50)
    with pytest.raises(InsufficientFiniteSupportError, match="initialization stalled"):
        run_dynesty(three_finite_points, _unit, 2, seed=1, config=config)
    assert calls["n"] == 1000 * 30  # dynesty's 1000 batches, then the error


def test_evidence_layer_maps_the_stall_to_no_finite_support(monkeypatch, tmp_path):
    from gwpop_search.inference import evidence

    def stall(*args, **kwargs):
        raise InsufficientFiniteSupportError("dynesty initialization stalled")

    monkeypatch.setattr(evidence, "run_dynesty_population", stall)
    with pytest.raises(evidence.NoFiniteSupportError, match="stalled"):
        evidence.run_hbi_evidence(
            None, None, None, {}, seed=1, config=DynestyConfig(), hbi_config=None, run_dir=tmp_path
        )
