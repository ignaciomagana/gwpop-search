import numpy as np
import pytest

pytest.importorskip("numpyro")

from gwpop_search.inference.recovery import chain_diagnostics, posterior_summary


def test_recovery_summary_reports_truth_interval_and_chain_diagnostics():
    rng = np.random.default_rng(42)
    samples = {
        "alpha": rng.normal(3.0, 0.2, size=(4, 200)),
        "beta_q": rng.normal(1.0, 0.3, size=(4, 200)),
    }
    truth = {"alpha": 3.0, "beta_q": 1.0}

    posterior = posterior_summary(samples, truth)
    diagnostics = chain_diagnostics(samples)

    assert posterior["alpha"]["truth_in_90pct_interval"]
    assert posterior["beta_q"]["truth_in_90pct_interval"]
    assert diagnostics["max_r_hat"] is not None
    assert np.isfinite(diagnostics["max_r_hat"])
    assert diagnostics["min_n_eff"] is not None
    assert diagnostics["min_n_eff"] > 0.0
    assert set(diagnostics["per_parameter"]) == set(samples)
