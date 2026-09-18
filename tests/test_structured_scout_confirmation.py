import json

import pytest

from gwpop_search.grammar import baseline_model_spec, load_model_spec
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS
from gwpop_search.scouts import (
    StructuredScoutAcceptanceCriteria,
    StructuredScoutInjection,
    build_structured_scout_campaign_plan,
    confirm_structured_scout_descendant,
    default_scout_campaign_config,
    generate_structured_scout_dataset,
)


def _write_campaign(root, *, mutation_id="chieff.mean.linear_q"):
    scout = default_scout_campaign_config("chi_eff", "q")
    injection = StructuredScoutInjection(
        mutation_id,
        0.4 if mutation_id != "null" else 0.0,
    )
    survey = SyntheticSurveyConfig(
        n_events=3,
        posterior_samples_per_event=8,
        n_injections=300,
        population_batch_size=96,
        redshift_sampling_grid=256,
    )
    plan = build_structured_scout_campaign_plan(
        n_runs=1,
        root_seed=77,
        injection=injection,
        survey_config=survey,
        scout_config=scout,
        base_spec=baseline_model_spec(),
        base_hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
        criteria=StructuredScoutAcceptanceCriteria(min_runs=1),
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign_plan.json").write_text(json.dumps(plan))

    run = plan["runs"][0]
    run_dir = root / "run_000"
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_structured_scout_dataset(
        seed=int(run["data_seed"]),
        injection=injection,
        survey_config=survey,
        hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS,
    )
    dataset.posterior.to_hdf5(run_dir / "pe.h5")
    dataset.selection.to_hdf5(run_dir / "selection.h5")
    (run_dir / "scout").mkdir(parents=True, exist_ok=True)

    if mutation_id == "null":
        validated = []
        expected = False
    else:
        validated = [
            {
                "proposal_id": "injected-proposal",
                "mutation_id": mutation_id,
                "target": "chieff_mean",
                "covariate": "q",
                "score": 5.0,
                "evidence": {
                    "slope": 0.4,
                    "slope_error": 0.08,
                    "z_score": 5.0,
                    "n_effective": 100.0,
                    "method": "test",
                },
                "status": "proposed",
            }
        ]
        expected = True

    scout_summary = {
        "base_model_hash": baseline_model_spec().model_hash,
        "numerical": {"passed": True},
        "validated_proposals": validated,
    }
    (run_dir / "scout" / "scout_summary.json").write_text(
        json.dumps(scout_summary)
    )
    campaign_summary = {
        "expected_mutation_id": mutation_id,
        "engineering_acceptance_passed": mutation_id != "null",
        "runs": [
            {
                "run_index": 0,
                "complete": True,
                "numerical_pass": True,
                "proposal_mutation_ids": [mutation_id] if expected else [],
                "expected_proposed": expected,
            }
        ],
    }
    (root / "campaign_summary.json").write_text(json.dumps(campaign_summary))
    return plan


def test_confirm_structured_descendant_materializes_expected_child(
    monkeypatch,
    tmp_path,
):
    campaign_root = tmp_path / "campaign"
    _write_campaign(campaign_root)
    output = tmp_path / "confirmation"

    def fake_compare(
        root,
        posterior,
        selection,
        fidelity_config,
        **kwargs,
    ):
        return {
            "both_numerically_valid": True,
            "log_bayes_factor_child_over_parent": 4.2,
            "parent_model_hash": kwargs["parent"].model_hash,
            "child_model_hash": kwargs["child"].model_hash,
        }

    monkeypatch.setattr(
        "gwpop_search.scouts.structured_confirmation."
        "compare_scout_descendant_evidence_config",
        fake_compare,
    )

    result = confirm_structured_scout_descendant(
        campaign_root,
        output,
        FidelityRunConfig(),
    )

    assert result["run_index"] == 0
    assert result["expected_mutation_id"] == "chieff.mean.linear_q"
    assert result["engineering_confirmation_passed"]
    assert result["comparison"]["log_bayes_factor_child_over_parent"] == 4.2

    parent = load_model_spec(output / "parent.json")
    child = load_model_spec(output / "child.json")
    assert parent.model_hash == baseline_model_spec().model_hash
    assert child.model_hash != parent.model_hash
    assert child.chieff.options["mean_dependence"] == "linear_q"

    review = json.loads((output / "review.json").read_text())
    assert review["decision"] == "accepted"
    assert review["mutation_id"] == "chieff.mean.linear_q"


def test_confirm_structured_descendant_rejects_null_campaign(tmp_path):
    campaign_root = tmp_path / "null-campaign"
    _write_campaign(campaign_root, mutation_id="null")

    with pytest.raises(ValueError, match="no injected descendant"):
        confirm_structured_scout_descendant(
            campaign_root,
            tmp_path / "confirmation",
            FidelityRunConfig(),
        )


def test_confirm_structured_descendant_rejects_ineligible_requested_run(
    tmp_path,
):
    campaign_root = tmp_path / "campaign"
    _write_campaign(campaign_root)
    summary = json.loads((campaign_root / "campaign_summary.json").read_text())
    summary["runs"][0]["expected_proposed"] = False
    (campaign_root / "campaign_summary.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="not an eligible"):
        confirm_structured_scout_descendant(
            campaign_root,
            tmp_path / "confirmation",
            FidelityRunConfig(),
            run_index=0,
        )



def test_confirm_structured_descendant_requires_campaign_gate(
    tmp_path,
):
    campaign_root = tmp_path / "campaign"
    _write_campaign(campaign_root)
    summary = json.loads((campaign_root / "campaign_summary.json").read_text())
    summary["engineering_acceptance_passed"] = False
    (campaign_root / "campaign_summary.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="frozen engineering gate"):
        confirm_structured_scout_descendant(
            campaign_root,
            tmp_path / "confirmation",
            FidelityRunConfig(),
        )


def test_negative_child_bayes_factor_does_not_pass_confirmation(
    monkeypatch,
    tmp_path,
):
    campaign_root = tmp_path / "campaign"
    _write_campaign(campaign_root)

    def fake_compare(
        root,
        posterior,
        selection,
        fidelity_config,
        **kwargs,
    ):
        return {
            "both_numerically_valid": True,
            "log_bayes_factor_child_over_parent": -0.5,
            "parent_model_hash": kwargs["parent"].model_hash,
            "child_model_hash": kwargs["child"].model_hash,
        }

    monkeypatch.setattr(
        "gwpop_search.scouts.structured_confirmation."
        "compare_scout_descendant_evidence_config",
        fake_compare,
    )

    result = confirm_structured_scout_descendant(
        campaign_root,
        tmp_path / "confirmation",
        FidelityRunConfig(),
    )
    assert not result["engineering_confirmation_passed"]
