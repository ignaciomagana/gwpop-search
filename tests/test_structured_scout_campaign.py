import json

import numpy as np
import pytest

from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS, GwcatChiEffBBHModel
from gwpop_search.scouts import (
    StructuredScoutAcceptanceCriteria,
    StructuredScoutInjection,
    assess_structured_scout_campaign,
    build_structured_scout_campaign_plan,
    default_scout_campaign_config,
    generate_structured_scout_dataset,
    reachable_mutation_ids,
)
from gwpop_search.scouts.synthetic import (
    _sample_chi_with_mu_sigma,
    _sample_q_with_beta,
)


def _criteria(min_runs=1):
    return StructuredScoutAcceptanceCriteria(min_runs=min_runs)


def test_variable_beta_q_sampler_responds_to_injected_slope():
    rng = np.random.default_rng(1)
    n = 50_000
    m1 = np.full(n, 40.0)
    low_beta = _sample_q_with_beta(
        rng,
        m1,
        beta=np.full(n, -2.0),
        mmin=5.0,
        q_floor=0.05,
    )
    high_beta = _sample_q_with_beta(
        rng,
        m1,
        beta=np.full(n, 6.0),
        mmin=5.0,
        q_floor=0.05,
    )
    assert np.mean(high_beta) > np.mean(low_beta) + 0.25
    assert np.all((low_beta >= 0.125) & (low_beta <= 1.0))
    assert np.all((high_beta >= 0.125) & (high_beta <= 1.0))


def test_variable_chieff_sampler_tracks_mean_and_width():
    rng = np.random.default_rng(2)
    n = 40_000

    low = _sample_chi_with_mu_sigma(
        rng,
        np.full(n, -0.15),
        np.full(n, 0.08),
    )
    high = _sample_chi_with_mu_sigma(
        rng,
        np.full(n, 0.20),
        np.full(n, 0.22),
    )
    assert np.mean(high) > np.mean(low) + 0.30
    assert np.std(high) > np.std(low) * 2.0
    assert np.all(np.abs(low) <= 1.0)
    assert np.all(np.abs(high) <= 1.0)


def test_structured_dataset_uses_canonical_pe_selection_contract():
    injection = StructuredScoutInjection(
        "chieff.mean.linear_q",
        0.4,
    )
    dataset = generate_structured_scout_dataset(
        seed=8,
        injection=injection,
        survey_config=SyntheticSurveyConfig(
            n_events=6,
            posterior_samples_per_event=12,
            n_injections=500,
            population_batch_size=128,
            redshift_sampling_grid=512,
        ),
    )

    assert dataset.posterior.n_events == 6
    assert dataset.posterior.basis.identity == dataset.selection.basis.identity
    assert dataset.truth_hyperparameters["chi_mu_q_slope"] == 0.4
    assert dataset.posterior.n_samples_total == 72
    assert dataset.selection.n_selected > 0


def test_structured_campaign_plan_has_deterministic_independent_seeds():
    scout = default_scout_campaign_config("chi_eff", "q")
    plan_a = build_structured_scout_campaign_plan(
        n_runs=4,
        root_seed=99,
        injection=StructuredScoutInjection(
            "chieff.mean.linear_q",
            0.4,
        ),
        survey_config=SyntheticSurveyConfig(
            n_events=8,
            posterior_samples_per_event=16,
            n_injections=500,
        ),
        scout_config=scout,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        criteria=_criteria(),
    )
    plan_b = build_structured_scout_campaign_plan(
        n_runs=4,
        root_seed=99,
        injection=StructuredScoutInjection(
            "chieff.mean.linear_q",
            0.4,
        ),
        survey_config=SyntheticSurveyConfig(
            n_events=8,
            posterior_samples_per_event=16,
            n_injections=500,
        ),
        scout_config=scout,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        criteria=_criteria(),
    )
    assert plan_a == plan_b
    pairs = {
        (item["data_seed"], item["sampler_seed"])
        for item in plan_a["runs"]
    }
    assert len(pairs) == 4
    assert all(a != b for a, b in pairs)


def _write_fake_summary(path, *, passed, mutations):
    path.mkdir(parents=True, exist_ok=True)
    (path / "scout_summary.json").write_text(
        json.dumps(
            {
                "numerical": {"passed": passed},
                "validated_proposals": [
                    {"mutation_id": mutation}
                    for mutation in mutations
                ],
            }
        )
    )


def test_structured_campaign_assessment_separates_expected_and_offtarget(tmp_path):
    scout = default_scout_campaign_config("chi_eff", "q")
    plan = build_structured_scout_campaign_plan(
        n_runs=3,
        root_seed=2,
        injection=StructuredScoutInjection(
            "chieff.mean.linear_q",
            0.4,
        ),
        survey_config=SyntheticSurveyConfig(),
        scout_config=scout,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        criteria=_criteria(),
    )
    (tmp_path / "campaign_plan.json").write_text(json.dumps(plan))

    _write_fake_summary(
        tmp_path / "run_000" / "scout",
        passed=True,
        mutations=["chieff.mean.linear_q"],
    )
    _write_fake_summary(
        tmp_path / "run_001" / "scout",
        passed=True,
        mutations=["chieff.width.linear_q"],
    )
    _write_fake_summary(
        tmp_path / "run_002" / "scout",
        passed=False,
        mutations=["chieff.mean.linear_q"],
    )

    summary = assess_structured_scout_campaign(tmp_path)
    assert summary["expected_mutation_reachable"]
    assert summary["n_numerical_pass"] == 2
    assert summary["n_expected_proposed"] == 1
    assert summary["expected_proposal_fraction_among_numerical_pass"] == 0.5
    assert summary["n_off_target_proposals"] == 1


def test_offtarget_control_is_marked_not_failed_recovery(tmp_path):
    scout = default_scout_campaign_config("chi_eff", "q")
    assert "pairing.beta.linear_m1" not in reachable_mutation_ids(scout)
    plan = build_structured_scout_campaign_plan(
        n_runs=1,
        root_seed=3,
        injection=StructuredScoutInjection(
            "pairing.beta.linear_m1",
            0.1,
        ),
        survey_config=SyntheticSurveyConfig(),
        scout_config=scout,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        criteria=_criteria(),
    )
    (tmp_path / "campaign_plan.json").write_text(json.dumps(plan))
    _write_fake_summary(
        tmp_path / "run_000" / "scout",
        passed=True,
        mutations=[],
    )

    summary = assess_structured_scout_campaign(tmp_path)
    assert not summary["expected_mutation_reachable"]
    assert summary["expected_proposal_fraction_among_numerical_pass"] is None
    assert summary["interpretation"] == "off_target_control"



@pytest.mark.parametrize(
    "mutation_id,strength,truth_key",
    [
        ("pairing.beta.linear_m1", 0.10, "beta_q_m1_slope"),
        ("chieff.mean.linear_m1", 0.01, "chi_mu_m1_slope"),
        ("chieff.mean.linear_q", 0.40, "chi_mu_q_slope"),
        ("chieff.mean.linear_z", 0.20, "chi_mu_z_slope"),
        ("chieff.width.linear_m1", 0.02, "log_chi_sigma_m1_slope"),
        ("chieff.width.linear_q", 1.00, "log_chi_sigma_q_slope"),
        ("chieff.width.linear_z", 0.50, "log_chi_sigma_z_slope"),
    ],
)
def test_structured_generator_supports_every_registered_linear_scout_axis(
    mutation_id,
    strength,
    truth_key,
):
    dataset = generate_structured_scout_dataset(
        seed=13,
        injection=StructuredScoutInjection(mutation_id, strength),
        survey_config=SyntheticSurveyConfig(
            n_events=3,
            posterior_samples_per_event=8,
            n_injections=300,
            population_batch_size=96,
            redshift_sampling_grid=256,
        ),
    )
    assert dataset.posterior.n_events == 3
    assert dataset.truth_hyperparameters[truth_key] == strength
    assert np.all(np.isfinite(dataset.event_truths["chi_eff"]))


@pytest.mark.parametrize(
    "mutation_id,strength",
    [
        ("pairing.beta.linear_m1", 0.31),
        ("chieff.mean.linear_m1", 0.021),
        ("chieff.mean.linear_q", 0.61),
        ("chieff.mean.linear_z", 0.41),
        ("chieff.width.linear_m1", 0.051),
        ("chieff.width.linear_q", 2.01),
        ("chieff.width.linear_z", 1.01),
    ],
)
def test_structured_injection_rejects_strength_outside_child_prior(
    mutation_id,
    strength,
):
    with pytest.raises(ValueError, match="outside registered prior support"):
        StructuredScoutInjection(mutation_id, strength)



def test_structured_scout_engineering_gate_passes_recovery_at_declared_threshold(
    tmp_path,
):
    scout = default_scout_campaign_config("chi_eff", "q")
    criteria = StructuredScoutAcceptanceCriteria(min_runs=8)
    plan = build_structured_scout_campaign_plan(
        n_runs=8,
        root_seed=91,
        injection=StructuredScoutInjection(
            "chieff.mean.linear_q",
            0.4,
        ),
        survey_config=SyntheticSurveyConfig(),
        scout_config=scout,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        criteria=criteria,
    )
    (tmp_path / "campaign_plan.json").write_text(json.dumps(plan))

    for index in range(8):
        mutations = (
            ["chieff.mean.linear_q"]
            if index < 6
            else []
        )
        _write_fake_summary(
            tmp_path / f"run_{index:03d}" / "scout",
            passed=True,
            mutations=mutations,
        )

    summary = assess_structured_scout_campaign(tmp_path)
    assert summary["expected_proposal_fraction_among_numerical_pass"] == 0.75
    assert summary["engineering_acceptance_passed"]
    assert summary["interpretation"] == "engineering_gate_passed"


def test_structured_scout_engineering_gate_fails_excess_null_proposals(
    tmp_path,
):
    scout = default_scout_campaign_config("chi_eff", "q")
    criteria = StructuredScoutAcceptanceCriteria(min_runs=8)
    plan = build_structured_scout_campaign_plan(
        n_runs=8,
        root_seed=92,
        injection=StructuredScoutInjection("null", 0.0),
        survey_config=SyntheticSurveyConfig(),
        scout_config=scout,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        criteria=criteria,
    )
    (tmp_path / "campaign_plan.json").write_text(json.dumps(plan))

    for index in range(8):
        mutations = (
            ["chieff.mean.linear_q"]
            if index < 3
            else []
        )
        _write_fake_summary(
            tmp_path / f"run_{index:03d}" / "scout",
            passed=True,
            mutations=mutations,
        )

    summary = assess_structured_scout_campaign(tmp_path)
    assert summary["any_proposal_fraction_among_numerical_pass"] == 0.375
    assert not summary["engineering_acceptance_passed"]
    assert summary["interpretation"] == "engineering_gate_failed"


def test_structured_scout_plan_rejects_too_few_runs_for_frozen_gate():
    with pytest.raises(ValueError, match="below the frozen scout acceptance"):
        build_structured_scout_campaign_plan(
            n_runs=4,
            root_seed=93,
            injection=StructuredScoutInjection(
                "chieff.mean.linear_q",
                0.4,
            ),
            survey_config=SyntheticSurveyConfig(),
            scout_config=default_scout_campaign_config("chi_eff", "q"),
            base_spec=baseline_model_spec(),
            base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
            criteria=StructuredScoutAcceptanceCriteria(min_runs=8),
        )
