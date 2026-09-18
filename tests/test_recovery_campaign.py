import json

import pytest

from gwpop_search.inference.campaign import (
    RecoveryAcceptanceCriteria,
    aggregate_recovery_summaries,
    assess_recovery_campaign,
    assess_recovery_summary,
    build_campaign_plan,
    checkpoint_fingerprints,
    recovery_seed_pairs,
)
from gwpop_search.inference.numpyro import NUTSConfig
from gwpop_search.inference.synthetic import SyntheticSurveyConfig


def _summary(data_seed=1, sampler_seed=2, *, fail=None):
    diagnostics = {
        "n_divergent": 0,
        "max_r_hat": 1.005,
        "min_n_eff": 500.0,
    }
    importance = {
        "posterior_median": {
            "min_event_ess": 100.0,
            "selection_ess": 1000.0,
            "max_event_weight_fraction": 0.05,
            "selection_max_weight_fraction": 0.01,
            "shape_log_likelihood_variance": 0.2,
        }
    }
    if fail == "rhat":
        diagnostics["max_r_hat"] = 1.2
    if fail == "selection":
        importance["posterior_median"]["selection_ess"] = 2.0
    if fail == "variance":
        importance["posterior_median"]["shape_log_likelihood_variance"] = 4.0

    return {
        "data_seed": data_seed,
        "sampler_seed": sampler_seed,
        "diagnostics": diagnostics,
        "importance_diagnostics": importance,
        "posterior": {
            "alpha": {
                "truth": 3.0,
                "median": 3.1,
                "std": 0.2,
                "truth_in_90pct_interval": True,
            },
            "beta_q": {
                "truth": 1.0,
                "median": 0.9,
                "std": 0.3,
                "truth_in_90pct_interval": True,
            },
        },
    }


def test_single_recovery_acceptance_checks_all_numerical_gates():
    result = assess_recovery_summary(_summary())
    assert result["passed"]
    assert all(item["passed"] for item in result["checks"])

    failed = assess_recovery_summary(_summary(fail="rhat"))
    assert not failed["passed"]
    assert any(
        item["name"] == "max_r_hat" and not item["passed"]
        for item in failed["checks"]
    )


def test_campaign_requires_minimum_runs_and_all_numerical_pass():
    criteria = RecoveryAcceptanceCriteria(min_runs=4)
    three = aggregate_recovery_summaries(
        [_summary(i, i + 100) for i in range(3)],
        criteria,
    )
    assert not three["phase3_numerical_gate_passed"]
    assert not three["enough_runs"]

    four = aggregate_recovery_summaries(
        [_summary(i, i + 100) for i in range(4)],
        criteria,
    )
    assert four["phase3_numerical_gate_passed"]
    assert four["coverage"]["alpha"]["central_90pct_coverage_fraction"] == 1.0

    one_bad = aggregate_recovery_summaries(
        [_summary(i, i + 100, fail="selection" if i == 2 else None) for i in range(4)],
        criteria,
    )
    assert not one_bad["phase3_numerical_gate_passed"]
    assert one_bad["n_numerical_fail"] == 1


def test_seed_pairs_are_deterministic_distinct_and_root_dependent():
    a = recovery_seed_pairs(6, root_seed=123)
    b = recovery_seed_pairs(6, root_seed=123)
    c = recovery_seed_pairs(6, root_seed=124)

    assert a == b
    assert a != c
    assert len({item[0] for item in a}) == 6
    assert len({item[1] for item in a}) == 6
    assert all(data != sampler for data, sampler in a)


def test_campaign_plan_serializes_all_phase3_acceptance_configuration():
    plan = build_campaign_plan(
        n_runs=4,
        root_seed=77,
        survey_config=SyntheticSurveyConfig(n_events=12, n_injections=1000),
        nuts_config=NUTSConfig(
            num_warmup=20,
            num_samples=30,
            num_chains=4,
            progress_bar=False,
        ),
        selection_chunk_size=256,
        criteria=RecoveryAcceptanceCriteria(min_runs=4),
    )

    assert plan["n_runs"] == 4
    assert len(plan["seed_pairs"]) == 4
    assert plan["survey_config"]["n_events"] == 12
    assert plan["nuts_config"]["num_chains"] == 4
    assert plan["criteria"]["max_r_hat"] == 1.01
    assert plan["selection_chunk_size"] == 256


def test_assess_existing_campaign_writes_machine_readable_summary(tmp_path):
    for index in range(4):
        run = tmp_path / f"run_{index:03d}"
        run.mkdir()
        (run / "recovery_summary.json").write_text(
            json.dumps(_summary(index, index + 10))
        )

    result = assess_recovery_campaign(tmp_path)
    assert result["phase3_numerical_gate_passed"]
    written = json.loads((tmp_path / "campaign_summary.json").read_text())
    assert written == result


def test_checkpoint_fingerprints_cover_chain_payload_and_metadata(tmp_path):
    chains = tmp_path / "chains"
    chains.mkdir()
    (chains / "chain_000.npz").write_bytes(b"payload")
    (chains / "chain_000.npz.json").write_text('{"seed": 1}')

    before = checkpoint_fingerprints(tmp_path)
    after = checkpoint_fingerprints(tmp_path)
    assert before == after
    assert set(before) == {"chain_000.npz", "chain_000.npz.json"}

    (chains / "chain_000.npz").write_bytes(b"changed")
    assert checkpoint_fingerprints(tmp_path)["chain_000.npz"] != before["chain_000.npz"]


def test_acceptance_criteria_reject_invalid_thresholds():
    with pytest.raises(ValueError):
        RecoveryAcceptanceCriteria(max_r_hat=1.0)
    with pytest.raises(ValueError):
        RecoveryAcceptanceCriteria(min_runs=0)
    with pytest.raises(ValueError):
        RecoveryAcceptanceCriteria(max_event_weight_fraction=1.1)
