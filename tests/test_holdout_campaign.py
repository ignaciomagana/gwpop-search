import numpy as np
import pytest

from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference import NUTSResult
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
)
from gwpop_search.search import SchedulerConfig
from gwpop_search.validation import HoldoutCampaignConfig, run_holdout_campaign


def _campaign():
    return ProductionCampaignConfig(
        campaign_id="holdout-test",
        dataset_manifest_hash="a" * 64,
        model_graph_hash="b" * 64,
        model_graph_root_hash="c" * 64,
        git_commit="d" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=31),
        budget=SearchBudget(100.0, 10, 4, 20),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )


def _fixed_folds(event_names, *, n_folds, seed):
    assert n_folds == 2
    return {
        "GWTOY_A": 0,
        "GWTOY_B": 1,
        "GWTOY_C": 0,
    }


def _fake_run(
    run_dir,
    posterior,
    selection,
    population_model,
    priors,
    *,
    seed,
    config,
    hbi_config,
):
    return NUTSResult(
        samples={
            name: np.full(8, float(index + 1))
            for index, name in enumerate(priors)
        },
        extra_fields={"diverging": np.zeros(8, dtype=bool)},
        seed=seed,
        config=config,
    )


def _pass_summary(*args, **kwargs):
    return {
        "passed": True,
        "checks": [],
        "chain": {"max_r_hat": 1.0, "min_n_eff": 1000.0},
        "n_divergent": 0,
        "posterior_median": {},
        "importance": {},
    }


def _predictive(
    posterior,
    selection,
    population_model,
    hyperposterior_samples,
    *,
    heldout_events,
    config,
):
    values = {"GWTOY_A": 1.0, "GWTOY_B": 2.0, "GWTOY_C": 3.0}
    return {name: values[name] for name in heldout_events}


def test_holdout_campaign_refits_training_folds_and_scores_each_event(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.deterministic_event_folds",
        _fixed_folds,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.run_resumable_chains",
        _fake_run,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.summarize_nuts_fit",
        _pass_summary,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.heldout_detected_log_predictive",
        _predictive,
    )

    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    model = baseline_model_spec()
    summary = run_holdout_campaign(
        tmp_path,
        posterior,
        selection,
        _campaign(),
        dataset_identity="dataset",
        models=(model,),
        config=HoldoutCampaignConfig(n_folds=2, seed=7),
    )

    assert summary["n_models"] == 1
    assert summary["n_models_complete"] == 1
    assert summary["model_total_log_predictive"][model.model_hash] == 6.0
    model_summary = summary["models"][0]
    assert set(model_summary["event_scores"]) == set(posterior.event_names)
    folds = model_summary["folds"]
    assert set(folds[0]["training_events"]).isdisjoint(
        folds[0]["heldout_events"]
    )
    assert set(folds[1]["training_events"]).isdisjoint(
        folds[1]["heldout_events"]
    )
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / model.model_hash / "fold_00" / "posterior.npz").exists()


def test_holdout_campaign_blocks_model_total_when_one_fold_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.deterministic_event_folds",
        _fixed_folds,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.run_resumable_chains",
        _fake_run,
    )
    calls = {"n": 0}

    def summary(*args, **kwargs):
        calls["n"] += 1
        payload = _pass_summary()
        payload["passed"] = calls["n"] != 2
        return payload

    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.summarize_nuts_fit",
        summary,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.heldout_detected_log_predictive",
        _predictive,
    )

    model = baseline_model_spec()
    result = run_holdout_campaign(
        tmp_path,
        make_toy_posterior_catalog(),
        make_toy_selection_catalog(),
        _campaign(),
        dataset_identity="dataset",
        models=(model,),
        config=HoldoutCampaignConfig(n_folds=2, seed=7),
    )
    assert result["n_models_complete"] == 0
    assert result["model_total_log_predictive"] == {}
    assert not result["models"][0]["all_folds_numerically_valid"]


def test_holdout_manifest_prevents_changed_resume_contract(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.deterministic_event_folds",
        _fixed_folds,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.run_resumable_chains",
        _fake_run,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.summarize_nuts_fit",
        _pass_summary,
    )
    monkeypatch.setattr(
        "gwpop_search.validation.holdout_campaign.heldout_detected_log_predictive",
        _predictive,
    )

    model = baseline_model_spec()
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    run_holdout_campaign(
        tmp_path,
        posterior,
        selection,
        _campaign(),
        dataset_identity="dataset",
        models=(model,),
        config=HoldoutCampaignConfig(n_folds=2, seed=7),
    )

    with pytest.raises(ValueError, match="manifest does not match"):
        run_holdout_campaign(
            tmp_path,
            posterior,
            selection,
            _campaign(),
            dataset_identity="different-dataset",
            models=(model,),
            config=HoldoutCampaignConfig(n_folds=2, seed=7),
        )
