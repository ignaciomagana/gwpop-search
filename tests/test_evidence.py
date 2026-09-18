import numpy as np
import pytest
from scipy.stats import norm

from gwpop_search.inference import (
    EvidenceResult,
    NestedSamplingConfig,
    load_evidence_result,
    run_numpyro_nested_model,
    save_evidence_result,
    summarize_evidence_repeats,
)


def _fake_result(logz, error, seed):
    return EvidenceResult(
        log_evidence=logz,
        log_evidence_error=error,
        posterior_samples={"x": np.arange(5, dtype=float)},
        diagnostics={"nested_ess": 20.0},
        seed=seed,
        backend="numpyro-jaxns",
        config=NestedSamplingConfig(
            num_live_points=50,
            max_samples=1000,
            dlogz=0.1,
            num_posterior_samples=5,
        ),
    )


def test_evidence_result_roundtrip(tmp_path):
    result = _fake_result(-3.2, 0.15, 4)
    path = tmp_path / "evidence.npz"
    save_evidence_result(path, result)
    restored = load_evidence_result(path)

    assert restored.log_evidence == result.log_evidence
    assert restored.log_evidence_error == result.log_evidence_error
    assert restored.config == result.config
    np.testing.assert_array_equal(
        restored.posterior_samples["x"],
        result.posterior_samples["x"],
    )


def test_repeat_summary_keeps_between_run_scatter_explicit():
    results = [
        _fake_result(-10.0, 0.1, 1),
        _fake_result(-9.8, 0.15, 2),
        _fake_result(-10.2, 0.12, 3),
    ]
    summary = summarize_evidence_repeats(results)

    assert np.isclose(summary.log_evidence_mean, -10.0)
    assert summary.repeat_std > 0.0
    assert summary.max_reported_error == 0.15
    assert summary.conservative_error >= summary.repeat_std
    assert summary.n_repeats == 3


def test_nested_sampling_recovers_analytic_normal_evidence():
    pytest.importorskip("jaxns")
    numpyro = pytest.importorskip("numpyro")
    import numpyro.distributions as dist

    observed = 0.3

    def model():
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(mu, 1.0), obs=observed)

    result = run_numpyro_nested_model(
        model,
        seed=123,
        config=NestedSamplingConfig(
            num_live_points=60,
            max_samples=4000,
            dlogz=0.1,
            num_posterior_samples=200,
        ),
    )
    expected = norm.logpdf(observed, loc=0.0, scale=np.sqrt(2.0))

    assert np.isfinite(result.log_evidence)
    assert np.isfinite(result.log_evidence_error)
    assert result.log_evidence_error >= 0.0
    assert abs(result.log_evidence - expected) < 0.35
    assert result.posterior_samples["mu"].shape == (200,)
    assert result.diagnostics["total_num_likelihood_evaluations"] > 0
