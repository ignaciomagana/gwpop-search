import numpy as np
from types import SimpleNamespace

from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference.numpyro import NUTSConfig, NUTSResult
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS
from gwpop_search.scouts import (
    ConditionalHSGPConfig,
    ConditionalHSGPResidualModel,
    ConditionalMomentSummaryConfig,
    ConditionalScoutRunConfig,
    HSGPAxis,
    ScoutNumericalCriteria,
    run_conditional_hsgp_scout,
    summarize_conditional_hsgp,
)


def _identity_catalogs():
    basis = SimpleNamespace(identity="test-basis")
    posterior = SimpleNamespace(
        basis=basis,
        event_names=("A", "B"),
        n_events=2,
        n_samples_total=40,
    )
    selection = SimpleNamespace(
        basis=basis,
        n_selected=100,
    )
    return posterior, selection


def _q_m1_model():
    config = ConditionalHSGPConfig(
        target="q",
        covariate="m1_source",
        target_axis=HSGPAxis("q", 0.05, 1.0, modes=4),
        covariate_axis=HSGPAxis("m1_source", 5.0, 90.0, modes=3),
        amplitude=1.0,
        target_length_scale=0.18,
        covariate_length_scale=20.0,
        quadrature_order=48,
    )
    return ConditionalHSGPResidualModel(
        baseline_model_spec(),
        DEFAULT_BASELINE_HYPERPARAMETERS,
        config,
    )


def test_hsgp_summary_turns_mean_structure_into_registered_pairing_proposal():
    rng = np.random.default_rng(123)
    model = _q_m1_model()
    n = 80
    samples = {
        name: 0.01 * rng.normal(size=n)
        for name in model.coefficient_names
    }
    # target mode 2 x first covariate mode is antisymmetric in q and changes
    # the conditional mean as the m1 basis amplitude changes.
    samples[model.coefficient_names[3]] = 2.0 + 0.02 * rng.normal(size=n)

    summary = summarize_conditional_hsgp(
        model,
        samples,
        config=ConditionalMomentSummaryConfig(
            covariate_grid_size=11,
            target_grid_size=201,
            max_posterior_draws=80,
            proposal_minimum_abs_z=2.0,
        ),
    )

    dependence = summary["dependence_summaries"][0]
    assert dependence["target"] == "pairing"
    assert dependence["covariate"] == "m1"
    assert abs(dependence["z_score"]) > 2.0
    assert any(
        proposal["mutation_id"] == "pairing.beta.linear_m1"
        for proposal in summary["proposals"]
    )


def test_hsgp_scout_numerical_failure_suppresses_raw_proposals(
    monkeypatch,
    tmp_path,
):
    model = _q_m1_model()
    rng = np.random.default_rng(4)
    samples = {
        name: rng.normal(size=(2, 20))
        for name in model.coefficient_names
    }
    fake_result = NUTSResult(
        samples=samples,
        extra_fields={"diverging": np.zeros((2, 20), dtype=bool)},
        seed=77,
        config=NUTSConfig(
            num_warmup=10,
            num_samples=20,
            num_chains=2,
            progress_bar=False,
        ),
    )
    raw = {
        "format_version": "test",
        "proposals": [
            {
                "proposal_id": "p1",
                "mutation_id": "pairing.beta.linear_m1",
                "target": "pairing",
                "covariate": "m1",
                "score": 10.0,
                "evidence": {},
                "status": "proposed",
            }
        ],
    }

    monkeypatch.setattr(
        "gwpop_search.scouts.inference.run_resumable_chains",
        lambda *args, **kwargs: fake_result,
    )
    monkeypatch.setattr(
        "gwpop_search.scouts.inference.chain_diagnostics",
        lambda samples: {
            "max_r_hat": 1.4,
            "min_n_eff": 10.0,
            "per_parameter": {},
        },
    )
    monkeypatch.setattr(
        "gwpop_search.scouts.inference._importance_diagnostics",
        lambda *args, **kwargs: {
            "log_likelihood": -10.0,
            "min_event_ess": 100.0,
            "max_event_weight_fraction": 0.05,
            "selection_ess": 1000.0,
            "selection_max_weight_fraction": 0.01,
            "shape_log_likelihood_variance": 0.2,
        },
    )
    monkeypatch.setattr(
        "gwpop_search.scouts.inference.summarize_conditional_hsgp",
        lambda *args, **kwargs: raw,
    )

    posterior, selection = _identity_catalogs()
    result, summary = run_conditional_hsgp_scout(
        tmp_path,
        posterior,
        selection,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        hsgp_config=model.config,
        seed=77,
        config=ConditionalScoutRunConfig(
            nuts=fake_result.config,
            numerical=ScoutNumericalCriteria(
                max_r_hat=1.05,
                min_mcmc_n_eff=20.0,
                max_divergences=0,
                min_event_ess=10.0,
                min_selection_ess=100.0,
                max_event_weight_fraction=0.35,
                max_selection_weight_fraction=0.15,
                max_shape_log_likelihood_variance=2.0,
            ),
        ),
    )

    assert result is fake_result
    assert not summary["numerical"]["passed"]
    assert summary["raw_proposals"]
    assert summary["validated_proposals"] == []
    assert summary["proposal_firewall"] == "suppressed_due_to_numerical_failure"
    assert (tmp_path / "scout_summary.json").exists()


def test_hsgp_scout_numerical_pass_allows_proposals(
    monkeypatch,
    tmp_path,
):
    model = _q_m1_model()
    samples = {
        name: np.zeros((2, 30))
        for name in model.coefficient_names
    }
    fake_result = NUTSResult(
        samples=samples,
        extra_fields={"diverging": np.zeros((2, 30), dtype=bool)},
        seed=88,
        config=NUTSConfig(
            num_warmup=10,
            num_samples=30,
            num_chains=2,
            progress_bar=False,
        ),
    )
    raw = {
        "format_version": "test",
        "proposals": [
            {
                "proposal_id": "test-pairing-m1",
                "mutation_id": "pairing.beta.linear_m1",
                "target": "pairing",
                "covariate": "m1",
                "score": 5.0,
                "evidence": {
                    "z_score": 5.0,
                    "slope": 0.1,
                    "slope_error": 0.02,
                    "n_effective": 60.0,
                    "method": "test",
                },
                "status": "proposed",
            }
        ],
    }

    monkeypatch.setattr(
        "gwpop_search.scouts.inference.run_resumable_chains",
        lambda *args, **kwargs: fake_result,
    )
    monkeypatch.setattr(
        "gwpop_search.scouts.inference.chain_diagnostics",
        lambda samples: {
            "max_r_hat": 1.001,
            "min_n_eff": 500.0,
            "per_parameter": {},
        },
    )
    monkeypatch.setattr(
        "gwpop_search.scouts.inference._importance_diagnostics",
        lambda *args, **kwargs: {
            "log_likelihood": -10.0,
            "min_event_ess": 100.0,
            "max_event_weight_fraction": 0.05,
            "selection_ess": 1000.0,
            "selection_max_weight_fraction": 0.01,
            "shape_log_likelihood_variance": 0.2,
        },
    )
    monkeypatch.setattr(
        "gwpop_search.scouts.inference.summarize_conditional_hsgp",
        lambda *args, **kwargs: raw,
    )

    posterior, selection = _identity_catalogs()
    _, summary = run_conditional_hsgp_scout(
        tmp_path,
        posterior,
        selection,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        hsgp_config=model.config,
        seed=88,
        config=ConditionalScoutRunConfig(
            nuts=fake_result.config,
            numerical=ScoutNumericalCriteria(
                max_r_hat=1.05,
                min_mcmc_n_eff=20.0,
                max_divergences=0,
                min_event_ess=10.0,
                min_selection_ess=100.0,
                max_event_weight_fraction=0.35,
                max_selection_weight_fraction=0.15,
                max_shape_log_likelihood_variance=2.0,
            ),
        ),
    )

    assert summary["numerical"]["passed"]
    assert summary["validated_proposals"] == raw["proposals"]
    assert summary["proposal_firewall"] == "numerical_pass_required"
    assert len(summary["validated_descendants"]) == 1
    descendant = summary["validated_descendants"][0]
    assert descendant["mutation_id"] == "pairing.beta.linear_m1"
    assert descendant["parent_model_hash"] == baseline_model_spec().model_hash
    assert descendant["child_model_hash"] != descendant["parent_model_hash"]
