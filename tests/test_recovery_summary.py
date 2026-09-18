import numpy as np
import pytest

pytest.importorskip("numpyro")

from gwpop_search.hbi import HBIConfig
from gwpop_search.inference.recovery import (
    chain_diagnostics,
    importance_diagnostics,
    posterior_summary,
)
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import GwcatChiEffBBHModel


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



def test_recovery_importance_summary_is_finite_at_synthetic_truth():
    model = GwcatChiEffBBHModel(redshift_quadrature_order=32)
    dataset = generate_baseline_synthetic_dataset(
        seed=818,
        model=model,
        config=SyntheticSurveyConfig(
            n_events=4,
            posterior_samples_per_event=16,
            n_injections=500,
            population_batch_size=128,
            redshift_sampling_grid=1024,
        ),
    )
    diagnostics = importance_diagnostics(
        dataset,
        model,
        dataset.truth_hyperparameters,
        HBIConfig(selection_chunk_size=128),
    )

    assert np.isfinite(diagnostics["log_likelihood"])
    assert diagnostics["min_event_ess"] > 0.0
    assert diagnostics["selection_ess"] > 0.0
    assert diagnostics["shape_log_likelihood_variance"] >= 0.0
