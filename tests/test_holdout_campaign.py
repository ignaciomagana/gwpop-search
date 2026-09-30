import json

import numpy as np
import pytest

from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference import DynestyConfig, DynestyResult
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
)
from gwpop_search.search import SchedulerConfig
from gwpop_search.validation import HoldoutCampaignConfig, run_holdout_campaign
from gwpop_search.validation.holdout_campaign import CAMPAIGN_FORMAT, MANIFEST_FORMAT


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


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, posterior, selection, population_model, priors, *, seed, config, hbi_config, run_dir):
        self.calls.append({"events": tuple(posterior.event_names), "seed": seed, "config": config,
                           "hbi": hbi_config, "run_dir": run_dir})
        names = tuple(sorted(priors))
        n = 6
        samples = np.tile(np.arange(1.0, len(names) + 1.0), (n, 1)) + 0.01 * np.arange(n)[:, None]
        log_weights = np.log(np.full(n, 1.0 / n))
        return DynestyResult(
            names=names,
            log_evidence=-10.0 - 0.01 * len(self.calls),
            log_evidence_error=0.1,
            information=2.0,
            niter=n,
            ncall=100,
            efficiency=10.0,
            samples=samples,
            log_weights=log_weights,
            log_likelihoods=np.zeros(n),
            log_volumes=-np.arange(1.0, n + 1.0),
            posterior_samples=samples,
            kish_ess=float(n),
            elapsed_seconds=0.0,
            config=config,
            seed=seed,
            versions={},
        )


def _pass_gates(results, *args, **kwargs):
    return {"passed": True, "checks": [], "posterior_median": {}, "importance": {}}


def _predictive(posterior, selection, population_model, hyperposterior_samples, *, heldout_events, config,
                log_weights):
    assert np.isclose(np.exp(np.logaddexp.reduce(log_weights)), 1.0)
    values = {"GWTOY_A": 1.0, "GWTOY_B": 2.0, "GWTOY_C": 3.0}
    return {name: values[name] for name in heldout_events}


def _patch(monkeypatch, recorder, gates=_pass_gates):
    module = "gwpop_search.validation.holdout_campaign"
    monkeypatch.setattr(f"{module}.deterministic_event_folds", _fixed_folds)
    monkeypatch.setattr(f"{module}.run_dynesty_population", recorder)
    monkeypatch.setattr(f"{module}.evaluate_posterior_gates", gates)
    monkeypatch.setattr(f"{module}.heldout_detected_log_predictive", _predictive)


def test_holdout_campaign_refits_training_folds_with_dynesty_repeats(monkeypatch, tmp_path):
    recorder = Recorder()
    _patch(monkeypatch, recorder)
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    model = baseline_model_spec()
    summary = run_holdout_campaign(
        tmp_path, posterior, selection, _campaign(), dataset_identity="dataset", models=(model,),
        config=HoldoutCampaignConfig(n_folds=2, seed=7),
    )

    assert summary["format_version"] == "gwpop-search-holdout-summary-2.0"
    assert summary["n_models_complete"] == 1
    assert summary["model_total_log_predictive"][model.model_hash] == 6.0
    # 2 folds x 2 repeats, every run on the training events of its fold
    assert len(recorder.calls) == 4
    assert {call["events"] for call in recorder.calls} == {("GWTOY_B",), ("GWTOY_A", "GWTOY_C")}
    for call in recorder.calls:
        cfg = call["config"]
        assert (cfg.nlive, cfg.bound, cfg.sample, cfg.slices) == (1000, "multi", "rslice", 2 * (3 + 10))
        assert call["hbi"].selection_chunk_size is None
    assert len({call["seed"] for call in recorder.calls}) == 4
    assert (tmp_path / model.model_hash / "fold_00").is_dir()
    assert str(recorder.calls[1]["run_dir"]).endswith("fold_00/repeat_001")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format_version"] == MANIFEST_FORMAT
    assert manifest["holdout_config"]["criteria"]["max_cross_run_rhat"] == 1.01
    assert manifest["hbi_config"]["selection_chunk_size"] is None
    folds = summary["models"][0]["folds"]
    assert set(folds[0]["training_events"]).isdisjoint(folds[0]["heldout_events"])
    assert folds[0]["training_log_evidence"]["interpretation"].startswith("advisory")


def test_holdout_campaign_blocks_model_total_when_one_fold_fails(monkeypatch, tmp_path):
    calls = {"n": 0}

    def gates(*args, **kwargs):
        calls["n"] += 1
        return {"passed": calls["n"] != 2, "checks": []}

    _patch(monkeypatch, Recorder(), gates)
    model = baseline_model_spec()
    result = run_holdout_campaign(
        tmp_path, make_toy_posterior_catalog(), make_toy_selection_catalog(), _campaign(),
        dataset_identity="dataset", models=(model,), config=HoldoutCampaignConfig(n_folds=2, seed=7),
    )
    assert result["n_models_complete"] == 0
    assert result["model_total_log_predictive"] == {}
    assert not result["models"][0]["all_folds_numerically_valid"]


def test_holdout_manifest_prevents_changed_resume_contract(monkeypatch, tmp_path):
    _patch(monkeypatch, Recorder())
    model = baseline_model_spec()
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    config = HoldoutCampaignConfig(n_folds=2, seed=7)
    run_holdout_campaign(tmp_path, posterior, selection, _campaign(), dataset_identity="dataset",
                         models=(model,), config=config)
    # the checkpoint cadence does not change the computation and is accepted on resume
    cadence = HoldoutCampaignConfig(
        n_folds=2, seed=7, posterior_config=DynestyConfig(nlive=1000, sample="rslice", checkpoint_every=30.0)
    )
    run_holdout_campaign(tmp_path, posterior, selection, _campaign(), dataset_identity="dataset",
                         models=(model,), config=cadence)
    with pytest.raises(ValueError, match="manifest does not match"):
        run_holdout_campaign(tmp_path, posterior, selection, _campaign(), dataset_identity="different-dataset",
                             models=(model,), config=config)
    with pytest.raises(ValueError, match="manifest does not match"):
        run_holdout_campaign(tmp_path, posterior, selection, _campaign(), dataset_identity="dataset",
                             models=(model,), config=HoldoutCampaignConfig(n_folds=2, seed=7, repeats=3))


def test_holdout_config_validation_and_round_trip():
    config = HoldoutCampaignConfig(n_folds=3, seed=5, predictive_draws=500)
    assert HoldoutCampaignConfig.from_dict(config.to_dict()) == config
    assert config.resolved_dynesty_config(7).slices == 20
    explicit = HoldoutCampaignConfig(posterior_config=DynestyConfig(sample="rslice", slices=9))
    assert explicit.resolved_dynesty_config(7).slices == 9
    with pytest.raises(ValueError, match="repeats"):
        HoldoutCampaignConfig(repeats=1)
    with pytest.raises(ValueError, match="1.0 campaigns"):
        HoldoutCampaignConfig(format_version="gwpop-search-holdout-campaign-1.0")
    assert config.format_version == CAMPAIGN_FORMAT


def test_holdout_campaign_end_to_end_with_real_dynesty_refits(tmp_path):
    pytest.importorskip("dynesty")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from gwpop_search.analysis.posterior_gates import PosteriorGateCriteria
    from gwpop_search.inference.synthetic import SyntheticSurveyConfig, generate_baseline_synthetic_dataset

    dataset = generate_baseline_synthetic_dataset(
        seed=5,
        config=SyntheticSurveyConfig(n_events=6, posterior_samples_per_event=32, n_injections=3_000,
                                     population_batch_size=512, redshift_sampling_grid=1024),
    )
    lenient = PosteriorGateCriteria(
        max_cross_run_rhat=5.0, min_kish_ess_per_run=5.0, require_dlogz_termination=False,
        min_event_ess=1.0, min_selection_ess=1.0, max_event_weight_fraction=1.0,
        max_selection_weight_fraction=1.0, max_shape_log_likelihood_variance=1e6, n_draws=16,
    )
    config = HoldoutCampaignConfig(
        n_folds=2, seed=3, criteria=lenient, predictive_draws=64,
        posterior_config=DynestyConfig(nlive=30, sample="rslice", batch_size=16, maxiter=60,
                                       num_posterior_samples=100),
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = run_holdout_campaign(
            tmp_path, dataset.posterior, dataset.selection, _campaign(), dataset_identity="synthetic",
            models=(baseline_model_spec(),), config=config,
        )
    model = summary["models"][0]
    # whichever folds pass their gates (tiny runs), the bookkeeping must be consistent
    for fold in model["folds"]:
        diagnostics = fold["diagnostics"]
        assert diagnostics["format_version"] == "gwpop-search-posterior-gates-1.0"
        assert diagnostics["passed"] == all(c["passed"] for c in diagnostics["checks"] if c["stage"] == "gate")
        assert fold["dynesty_config"]["slices"] == 2 * (3 + 10)
        if diagnostics["passed"]:
            assert set(fold["heldout_log_predictive"]) == set(fold["heldout_events"])
            assert all(np.isfinite(v) for v in fold["heldout_log_predictive"].values())
        else:
            assert fold["heldout_log_predictive"] == {}
    all_passed = all(fold["diagnostics"]["passed"] for fold in model["folds"])
    assert model["all_folds_numerically_valid"] == all_passed
    assert (baseline_model_spec().model_hash in summary["model_total_log_predictive"]) == all_passed
    assert (tmp_path / baseline_model_spec().model_hash / "fold_00" / "repeat_001" / "result.npz").exists()
