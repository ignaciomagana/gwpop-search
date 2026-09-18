import json
from dataclasses import asdict

import pytest

from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
)
from gwpop_search.scouts import (
    StructureProposal,
    compare_scout_descendant_evidence,
    review_scout_proposal,
    scout_comparison_seed_root,
)
from gwpop_search.search import EvaluationRecord, Fidelity, SchedulerConfig


def _proposal():
    return StructureProposal(
        proposal_id="hsgp-mean-q",
        mutation_id="chieff.mean.linear_q",
        target="chieff_mean",
        covariate="q",
        score=5.0,
        evidence={
            "slope": 0.2,
            "slope_error": 0.04,
            "z_score": 5.0,
            "n_effective": 100.0,
            "method": "hsgp",
        },
    )


def _summary(parent, *, numerical_pass=True):
    return {
        "base_model_hash": parent.model_hash,
        "numerical": {"passed": numerical_pass},
        "validated_proposals": [asdict(_proposal())],
    }


def _campaign():
    return ProductionCampaignConfig(
        campaign_id="scout-compare",
        dataset_manifest_hash="a" * 64,
        model_graph_hash="b" * 64,
        model_graph_root_hash="c" * 64,
        git_commit="d" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=123),
        budget=SearchBudget(100.0, 10, 4, 20),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )


def test_scout_proposal_requires_explicit_acceptance_to_materialize_child():
    parent = baseline_model_spec()
    review, child = review_scout_proposal(
        _summary(parent),
        parent,
        proposal_id="hsgp-mean-q",
        decision="accepted",
        note="reviewed",
    )

    assert review.decision == "accepted"
    assert review.parent_model_hash == parent.model_hash
    assert child is not None
    assert review.child_model_hash == child.model_hash
    assert child.model_hash != parent.model_hash
    assert child.chieff.options["mean_dependence"] == "linear_q"
    assert "chi_mu_q_slope" in child.priors


def test_rejected_scout_proposal_does_not_materialize_child():
    parent = baseline_model_spec()
    review, child = review_scout_proposal(
        _summary(parent),
        parent,
        proposal_id="hsgp-mean-q",
        decision="rejected",
    )
    assert review.decision == "rejected"
    assert review.child_model_hash is None
    assert child is None


def test_numerically_failed_scout_cannot_be_reviewed_as_validated():
    parent = baseline_model_spec()
    with pytest.raises(ValueError, match="numerical gate failed"):
        review_scout_proposal(
            _summary(parent, numerical_pass=False),
            parent,
            proposal_id="hsgp-mean-q",
            decision="accepted",
        )


def test_scout_comparison_seed_is_deterministic_and_pair_specific():
    parent = baseline_model_spec()
    _, child = review_scout_proposal(
        _summary(parent),
        parent,
        proposal_id="hsgp-mean-q",
        decision="accepted",
    )
    a = scout_comparison_seed_root(1, parent.model_hash, child.model_hash)
    b = scout_comparison_seed_root(1, parent.model_hash, child.model_hash)
    c = scout_comparison_seed_root(2, parent.model_hash, child.model_hash)
    assert a == b
    assert a != c


class _FakeEvidenceEvaluator:
    values = {}
    failed = set()

    def __init__(self, posterior, selection, config, dataset_identity):
        self.dataset_identity = dataset_identity

    def evaluate(self, model, fidelity, *, seed, run_dir):
        assert fidelity is Fidelity.F3_EVIDENCE
        value = float(type(self).values[model.model_hash])
        passed = model.model_hash not in type(self).failed
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "model_hash": model.model_hash,
                    "fidelity": "F3",
                    "diagnostics": {
                        "passed": passed,
                        "evidence": {
                            "log_evidence_mean": value,
                            "conservative_error": 0.1,
                        },
                    },
                }
            )
        )
        return EvaluationRecord(
            model_hash=model.model_hash,
            fidelity=Fidelity.F3_EVIDENCE,
            diagnostics_pass=passed,
            screen_value=value,
            compute_cost=0.1,
        )


def test_reviewed_descendant_is_independently_refit_and_compared(
    monkeypatch,
    tmp_path,
):
    parent = baseline_model_spec()
    _, child = review_scout_proposal(
        _summary(parent),
        parent,
        proposal_id="hsgp-mean-q",
        decision="accepted",
    )
    _FakeEvidenceEvaluator.values = {
        parent.model_hash: -100.0,
        child.model_hash: -96.5,
    }
    _FakeEvidenceEvaluator.failed = set()
    monkeypatch.setattr(
        "gwpop_search.scouts.comparison.DeterministicHBIEvaluator",
        _FakeEvidenceEvaluator,
    )

    result = compare_scout_descendant_evidence(
        tmp_path,
        object(),
        object(),
        _campaign(),
        dataset_identity="dataset",
        parent=parent,
        child=child,
        proposal_id="hsgp-mean-q",
    )

    assert result["both_numerically_valid"]
    assert result["log_bayes_factor_child_over_parent"] == 3.5
    assert result["interpretation"] == "independent_full_hbi_refit"
    assert (tmp_path / "parent" / "evaluation.json").exists()
    assert (tmp_path / "child" / "evaluation.json").exists()


def test_descendant_bayes_factor_is_blocked_by_numerical_failure(
    monkeypatch,
    tmp_path,
):
    parent = baseline_model_spec()
    _, child = review_scout_proposal(
        _summary(parent),
        parent,
        proposal_id="hsgp-mean-q",
        decision="accepted",
    )
    _FakeEvidenceEvaluator.values = {
        parent.model_hash: -100.0,
        child.model_hash: -96.5,
    }
    _FakeEvidenceEvaluator.failed = {child.model_hash}
    monkeypatch.setattr(
        "gwpop_search.scouts.comparison.DeterministicHBIEvaluator",
        _FakeEvidenceEvaluator,
    )

    result = compare_scout_descendant_evidence(
        tmp_path,
        object(),
        object(),
        _campaign(),
        dataset_identity="dataset",
        parent=parent,
        child=child,
        proposal_id="hsgp-mean-q",
    )
    assert not result["both_numerically_valid"]
    assert result["log_bayes_factor_child_over_parent"] is None
    assert result["interpretation"] == "comparison_blocked_by_numerical_failure"
