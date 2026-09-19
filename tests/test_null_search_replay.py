"""One exact-null replay under the f3_completion statistic (decision D4)."""

import json

import pytest

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
from gwpop_search.nulls import run_baseline_null_search_replay, search_statistics_from_evidence
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    model_graph_hash,
)
from gwpop_search.search import (
    ComplexityModelPrior,
    EvaluationRecord,
    Fidelity,
    ModelEvidence,
    SchedulerConfig,
    SearchExecutionConfig,
)


class _FakeEvaluator:
    calls = []
    log_evidence = {}

    def __init__(self, posterior, selection, config, dataset_identity):
        self.dataset_identity = dataset_identity

    def evaluate(self, model, fidelity, *, seed, run_dir):
        type(self).calls.append((model.model_hash, fidelity.value, seed))
        run_dir.mkdir(parents=True, exist_ok=True)
        logz = type(self).log_evidence[model.model_hash] if fidelity.value != "F0" else 0.0
        (run_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "format_version": "gwpop-search-fidelity-evaluation-2.0",
                    "model_hash": model.model_hash,
                    "fidelity": fidelity.value,
                    "diagnostics": {
                        "passed": True,
                        "checks": [],
                        "evidence": {"log_evidence_mean": logz, "conservative_error": 0.1},
                    },
                }
            )
        )
        return EvaluationRecord(model.model_hash, fidelity, True, logz, 0.01)


def test_f3_completion_null_replay_runs_f0_and_f3_only(monkeypatch, tmp_path):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=4)
    _FakeEvaluator.calls = []
    _FakeEvaluator.log_evidence = {
        node.model_hash: -100.0 + index for index, node in enumerate(graph.nodes)
    }
    monkeypatch.setattr(
        "gwpop_search.nulls.search_replay.DeterministicHBIEvaluator", _FakeEvaluator
    )
    monkeypatch.setattr(
        "gwpop_search.production.completion.DeterministicHBIEvaluator", _FakeEvaluator
    )
    campaign = ProductionCampaignConfig(
        campaign_id="null-replay",
        dataset_manifest_hash="a" * 64,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="b" * 40,
        model_prior={"version": "axis-complexity-v1", "penalty_per_axis": 0.5},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(beam_width=1, exploration_quota=0),
        seed_policy=SeedPolicy(root_seed=3),
        budget=SearchBudget(5.0, len(graph.nodes), 1, 10),
        artifact_root="runs",
        state_database="state.sqlite",
    )
    model_prior = ComplexityModelPrior(penalty_per_axis=0.5)
    result = run_baseline_null_search_replay(
        0,
        11,
        root=tmp_path,
        graph=graph,
        model_prior=model_prior,
        execution_config=SearchExecutionConfig(
            root_seed=7,
            scheduler=campaign.scheduler,
            stop_fidelity=Fidelity.F3_EVIDENCE,
            max_models_by_fidelity={"F3": len(graph.nodes)},
            max_total_compute_cost=5.0,
        ),
        fidelity_config=campaign.fidelity,
        survey_config=SyntheticSurveyConfig(
            n_events=4,
            posterior_samples_per_event=8,
            n_injections=300,
            population_batch_size=128,
            redshift_sampling_grid=512,
            observation_model="noisy_observation",
        ),
        completion_campaign=campaign,
        completion_seed_root=7,
        data_mode="synthetic_survey",
        statistic="f3_completion",
    )
    fidelities = [call[1] for call in _FakeEvaluator.calls]
    assert "F4" not in fidelities
    assert fidelities.count("F0") == len(graph.nodes)
    # F0 promotes every pass to F3, so completion finds every node already evaluated
    # with the identical F3 seed and re-runs nothing.
    assert fidelities.count("F3") == len(graph.nodes)
    expected = search_statistics_from_evidence(
        graph,
        {
            model_hash: ModelEvidence(model_hash, value, 0.1)
            for model_hash, value in _FakeEvaluator.log_evidence.items()
        },
        model_prior=model_prior,
    )
    assert result.max_log_bayes_factor == pytest.approx(expected["max_log_bayes_factor"])
    assert result.max_log_posterior_odds == pytest.approx(expected["max_log_posterior_odds"])
    assert result.metadata["statistic"] == "f3_completion"
    assert result.metadata["statistic_evidence_fidelities"] == ["F3"]
    assert result.metadata["execution"]["completed_fidelity"] == "F3"
    assert result.metadata["evidence_completion"]["n_reused"] == len(graph.nodes)
    with pytest.raises(ValueError, match="unsupported null statistic"):
        run_baseline_null_search_replay(
            1, 12, root=tmp_path, graph=graph, model_prior=model_prior,
            execution_config=SearchExecutionConfig(), fidelity_config=campaign.fidelity,
            statistic="best_available",
        )
