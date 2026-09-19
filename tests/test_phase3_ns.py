"""Phase-3 v2 dynesty recovery campaign: plan, criteria, R-hat, resume integrity."""

from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from gwpop_search.cli import build_parser  # noqa: E402
from gwpop_search.inference import dynesty_backend  # noqa: E402
from gwpop_search.inference.campaign import recovery_seed_pairs  # noqa: E402
from gwpop_search.inference.dynesty_backend import load_dynesty_result  # noqa: E402
from gwpop_search.inference.phase3_ns import (  # noqa: E402
    CAMPAIGN_PLAN_NAME,
    NSRecoveryAcceptanceCriteria,
    PLAN_FORMAT_VERSION,
    RECOVERY_SUMMARY_FORMAT_VERSION,
    aggregate_ns_recovery_summaries,
    assess_ns_fit,
    assess_ns_recovery_campaign,
    build_ns_campaign_plan,
    compare_ns_fingerprints,
    default_phase3_dynesty_config,
    default_phase3_survey_config,
    ns_run_fingerprints,
    phase3_hbi_config,
    rank_normalized_split_rhat,
    repeat_seeds,
    run_ns_recovery_campaign,
    slices_for_ndim,
    write_or_verify_plan,
)
from gwpop_search.inference.synthetic import SyntheticSurveyConfig  # noqa: E402

# Data seeds of the rejected NUTS campaign (docs/phase3_recovery.md).
V1_DATA_SEEDS = [1685370682, 1653124771, 610249415, 200179658]


def tiny_survey():
    return SyntheticSurveyConfig(
        n_events=6,
        posterior_samples_per_event=32,
        n_injections=3_000,
        population_batch_size=512,
        redshift_sampling_grid=4096,
        injection_draw="population_proxy",
        observation_model="noisy_observation",
    )


def tiny_dynesty(**overrides):
    payload = dict(nlive=30, batch_size=8, maxiter=100, dlogz=0.5, num_posterior_samples=200)
    payload.update(overrides)
    return default_phase3_dynesty_config(**payload)


TINY_CRITERIA = NSRecoveryAcceptanceCriteria(min_runs=1, min_repeats=2)
TINY_SETTINGS = dict(importance_draws=48, rhat_draws_per_run=100)


def run_tiny_campaign(root, *, n_runs=1, repeats=2, dynesty_config=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dirty-tree and stopped-short warnings
        return run_ns_recovery_campaign(
            root,
            n_runs=n_runs,
            repeats=repeats,
            survey_config=tiny_survey(),
            dynesty_config=tiny_dynesty() if dynesty_config is None else dynesty_config,
            criteria=TINY_CRITERIA,
            **TINY_SETTINGS,
        )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def default_plan():
    return build_ns_campaign_plan(
        n_runs=4,
        root_seed=20260917,
        repeats=4,
        survey_config=default_phase3_survey_config(),
        dynesty_config=default_phase3_dynesty_config(),
        hbi_config=phase3_hbi_config(),
        criteria=NSRecoveryAcceptanceCriteria(),
    )


def test_default_plan_pins_the_v2_decisions_and_reuses_the_v1_catalog_seeds():
    plan = default_plan()
    assert plan["format_version"] == PLAN_FORMAT_VERSION == "gwpop-search-phase3-campaign-plan-2.0"
    assert plan["n_runs"] == 4 and plan["repeats"] == 4 and plan["root_seed"] == 20260917
    v1 = recovery_seed_pairs(4, root_seed=20260917)
    assert [(p["data_seed"], p["sampler_seed"]) for p in plan["seed_pairs"]] == list(v1)
    assert [p["data_seed"] for p in plan["seed_pairs"]] == V1_DATA_SEEDS
    for pair in plan["seed_pairs"]:
        assert pair["repeat_seeds"] == list(repeat_seeds(pair["sampler_seed"], 4))
    all_repeat_seeds = [s for p in plan["seed_pairs"] for s in p["repeat_seeds"]]
    assert len(set(all_repeat_seeds)) == 16

    dynesty = plan["dynesty_config"]
    assert dynesty["nlive"] == 1000 and dynesty["sample"] == "rslice"
    assert dynesty["bound"] == "multi" and dynesty["dlogz"] == 0.1
    assert dynesty["slices"] == 26 == slices_for_ndim(10) and dynesty["batch_size"] == 64
    assert "checkpoint_every" not in dynesty

    survey = plan["survey_config"]
    assert survey["observation_model"] == "noisy_observation"
    assert survey["injection_draw"] == "population_proxy"
    assert survey["n_events"] == 48 and survey["posterior_samples_per_event"] == 1024
    assert survey["n_injections"] == 100_000
    assert plan["hbi_config"]["selection_chunk_size"] is None
    # The importance diagnostics must evaluate every pooled posterior draw of a
    # planned fit: otherwise the gated tail quantiles are Monte-Carlo estimates
    # whose verdict depends on the seed the plan happens to pin, and the plan
    # cannot be changed once a campaign root exists.
    pooled_draws = plan["repeats"] * dynesty["num_posterior_samples"]
    assert pooled_draws == 16000
    assert plan["diagnostics"]["importance_draws"] == 16384 >= pooled_draws

    criteria = plan["criteria"]
    assert criteria["format_version"] == NSRecoveryAcceptanceCriteria.FORMAT_VERSION
    assert NSRecoveryAcceptanceCriteria.from_dict(criteria) == NSRecoveryAcceptanceCriteria()


def test_repeat_seeds_are_deterministic_and_depend_on_the_sampler_seed():
    assert repeat_seeds(7, 4) == repeat_seeds(7, 4)
    assert repeat_seeds(7, 4)[:2] == repeat_seeds(7, 2)
    assert set(repeat_seeds(7, 4)).isdisjoint(repeat_seeds(8, 4))
    with pytest.raises(ValueError):
        repeat_seeds(7, 0)


def test_plan_refuses_a_different_campaign_but_not_a_new_checkpoint_cadence(tmp_path):
    path = tmp_path / CAMPAIGN_PLAN_NAME
    plan = default_plan()
    write_or_verify_plan(path, plan)
    write_or_verify_plan(path, default_plan())  # identical: accepted

    same_but_cadence = build_ns_campaign_plan(
        n_runs=4,
        root_seed=20260917,
        repeats=4,
        survey_config=default_phase3_survey_config(),
        dynesty_config=default_phase3_dynesty_config(checkpoint_every=5.0),
        hbi_config=phase3_hbi_config(),
        criteria=NSRecoveryAcceptanceCriteria(),
    )
    write_or_verify_plan(path, same_but_cadence)

    variants = {
        "dynesty_config": dict(dynesty_config=default_phase3_dynesty_config(nlive=500)),
        "repeats": dict(repeats=3),
        "criteria": dict(criteria=NSRecoveryAcceptanceCriteria(max_r_hat=1.05)),
        "survey_config": dict(survey_config=replace(default_phase3_survey_config(), n_events=40)),
        "root_seed": dict(root_seed=1),
    }
    base = dict(
        n_runs=4,
        root_seed=20260917,
        repeats=4,
        survey_config=default_phase3_survey_config(),
        dynesty_config=default_phase3_dynesty_config(),
        hbi_config=phase3_hbi_config(),
        criteria=NSRecoveryAcceptanceCriteria(),
    )
    for key, change in variants.items():
        with pytest.raises(ValueError, match="refusing to resume"):
            write_or_verify_plan(path, build_ns_campaign_plan(**{**base, **change}))
    assert json.loads(path.read_text()) == plan


def test_campaign_refuses_a_root_with_another_plan_before_any_computation(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    v1_plan = {"format_version": "gwpop-search-phase3-campaign-plan-1.0"}
    (root / CAMPAIGN_PLAN_NAME).write_text(json.dumps(v1_plan))
    with pytest.raises(ValueError, match="refusing to resume"):
        run_tiny_campaign(root)
    assert sorted(p.name for p in root.iterdir()) == [CAMPAIGN_PLAN_NAME]


# ---------------------------------------------------------------------------
# Rank-normalized split R-hat
# ---------------------------------------------------------------------------


def test_rank_normalized_split_rhat_matches_arviz():
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(1)
    for trial in range(40):
        n_chains = int(rng.integers(2, 6))
        n_draws = int(rng.integers(4, 400))
        chains = rng.normal(size=(n_chains, n_draws)) * rng.uniform(0.5, 2.0)
        chains += rng.normal(size=(n_chains, 1)) * rng.uniform(0.0, 0.5)
        if trial % 3 == 0:
            chains = np.round(chains, 1)  # ties
        ours = rank_normalized_split_rhat(chains)
        assert ours["rank"] == pytest.approx(float(az.rhat(chains, method="rank")), abs=1e-12)
        assert ours["bulk"] == pytest.approx(float(az.rhat(chains, method="z_scale")), abs=1e-12)
        assert ours["tail"] == pytest.approx(float(az.rhat(chains, method="folded")), abs=1e-12)


def test_rank_normalized_split_rhat_detects_disagreeing_chains_and_validates_input():
    rng = np.random.default_rng(2)
    same = rng.normal(size=(4, 2000))
    assert rank_normalized_split_rhat(same)["rank"] < 1.01
    shifted = same.copy()
    shifted[0] += 1.0  # one chain displaced by one posterior sigma: the bulk R-hat sees it
    result = rank_normalized_split_rhat(shifted)
    assert result["bulk"] > 1.05 and result["rank"] == result["bulk"]
    wider = same.copy()
    wider[1] *= 3.0  # same location, different scale: the folded (tail) R-hat sees it
    result = rank_normalized_split_rhat(wider)
    assert result["tail"] > 1.05 and result["rank"] == result["tail"]
    with pytest.raises(ValueError):
        rank_normalized_split_rhat(same[:1])
    with pytest.raises(ValueError):
        rank_normalized_split_rhat(same[:, :3])
    bad = same.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        rank_normalized_split_rhat(bad)
    assert np.isnan(rank_normalized_split_rhat(np.ones((2, 10)))["rank"])


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def good_fit(n_repeats=4):
    over = {
        "min_event_ess": {"q0.1": 60.0, "q0.5": 90.0},
        "selection_ess": {"q0.1": 500.0, "q0.5": 900.0},
        "max_event_weight_fraction": {"q0.5": 0.08, "q0.9": 0.12},
        "selection_max_weight_fraction": {"q0.5": 0.01, "q0.9": 0.02},
        "shape_log_likelihood_variance": {"q0.5": 0.3, "q0.9": 0.5},
    }
    return {
        "n_repeats": n_repeats,
        "totals": {
            "min_kish_ess": 3000.0,
            "n_nan_or_posinf_log_likelihoods": 0,
            "n_selection_unsupported": 0,
            "all_converged": True,
        },
        "evidence": {"repeat_std": 0.1, "max_reported_error": 0.15, "max_pairwise_z": 1.0},
        "convergence": {"max_r_hat": 1.004},
        "importance_diagnostics": {
            "posterior_median": {
                "min_event_ess": 100.0,
                "selection_ess": 1000.0,
                "max_event_weight_fraction": 0.05,
                "selection_max_weight_fraction": 0.01,
                "shape_log_likelihood_variance": 0.2,
            },
            "over_posterior": over,
        },
    }


def failed(result):
    return sorted(item["name"] for item in result["checks"] if not item["passed"])


def test_criteria_defaults_are_the_phase3_v2_values_and_round_trip():
    c = NSRecoveryAcceptanceCriteria()
    assert (c.max_r_hat, c.min_kish_ess_per_run, c.max_evidence_repeat_std) == (1.01, 1000.0, 0.2)
    assert (c.max_log_evidence_error, c.max_pairwise_repeat_z) == (0.2, 3.0)
    assert (c.min_event_ess, c.min_selection_ess) == (20.0, 200.0)
    assert (c.max_event_weight_fraction, c.max_selection_weight_fraction) == (0.25, 0.10)
    assert c.max_shape_log_likelihood_variance == 1.0
    assert (c.tail_ess_quantile, c.tail_weight_quantile) == (0.1, 0.9)
    assert c.min_runs == 4 and c.min_repeats == 4 and c.max_selection_unsupported_evaluations == 0
    f3 = NSRecoveryAcceptanceCriteria.f3_level()
    assert (f3.max_r_hat, f3.min_kish_ess_per_run, f3.max_pairwise_repeat_z) == (1.05, 500.0, 3.5)
    assert (f3.max_evidence_repeat_std, f3.max_log_evidence_error) == (0.5, 0.5)
    assert (f3.min_event_ess, f3.min_selection_ess) == (10.0, 100.0)
    assert f3.max_event_weight_fraction == 0.35
    assert (f3.max_selection_weight_fraction, f3.max_shape_log_likelihood_variance) == (0.15, 2.0)
    for criteria in (c, f3):
        payload = criteria.to_dict()
        assert payload["format_version"] == "gwpop-search-ns-acceptance-criteria-2.0"
        assert NSRecoveryAcceptanceCriteria.from_dict(json.loads(json.dumps(payload))) == criteria
    with pytest.raises(ValueError, match="format"):
        NSRecoveryAcceptanceCriteria.from_dict({**c.to_dict(), "format_version": "x"})
    for bad in (
        dict(max_r_hat=1.0),
        dict(min_kish_ess_per_run=0.0),
        dict(max_event_weight_fraction=1.5),
        dict(tail_ess_quantile=0.9),
        dict(tail_weight_quantile=0.3),
        dict(tail_ess_quantile=0.2),
        # The tail must lie strictly beyond the median: a tail at 0.5 would
        # serialize and validate while enforcing nothing the draw-median checks
        # do not already enforce.
        dict(tail_ess_quantile=0.5),
        dict(tail_weight_quantile=0.5),
        dict(tail_ess_quantile=0.5, tail_weight_quantile=0.5),
        dict(min_repeats=1),
        dict(min_runs=0),
        dict(max_selection_unsupported_evaluations=-1),
        dict(max_r_hat=float("nan")),
    ):
        with pytest.raises(ValueError):
            NSRecoveryAcceptanceCriteria(**bad)
    for bad in (
        dict(min_runs=4.0),
        dict(min_repeats=True),
        dict(max_r_hat="1.01"),
        dict(require_dlogz_termination=1),
    ):
        with pytest.raises(TypeError):
            NSRecoveryAcceptanceCriteria(**bad)
    coerced = NSRecoveryAcceptanceCriteria(min_event_ess=20, max_r_hat=np.float64(1.01))
    assert isinstance(coerced.min_event_ess, float) and coerced == NSRecoveryAcceptanceCriteria()


def test_assessment_enforces_importance_at_point_draw_median_and_tail():
    criteria = NSRecoveryAcceptanceCriteria()
    ok = assess_ns_fit(good_fit(), criteria)
    assert ok["passed"], failed(ok)
    names = {item["name"] for item in ok["checks"]}
    for where in ("point", "draw_median", "tail"):
        for key in (
            "min_event_ess",
            "selection_ess",
            "max_event_weight_fraction",
            "selection_max_weight_fraction",
            "shape_log_likelihood_variance",
        ):
            assert f"{where}.{key}" in names

    # Only the tail fails: the posterior-median point and the draw median pass.
    fit = good_fit()
    fit["importance_diagnostics"]["over_posterior"]["min_event_ess"]["q0.1"] = 12.0
    fit["importance_diagnostics"]["over_posterior"]["shape_log_likelihood_variance"]["q0.9"] = 1.4
    result = assess_ns_fit(fit, criteria)
    assert failed(result) == ["tail.min_event_ess", "tail.shape_log_likelihood_variance"]
    tail = next(item for item in result["checks"] if item["name"] == "tail.min_event_ess")
    assert tail["quantile"] == 0.1 and tail["value"] == 12.0

    fit = good_fit()
    fit["importance_diagnostics"]["over_posterior"]["selection_ess"]["q0.5"] = 150.0
    assert failed(assess_ns_fit(fit, criteria)) == ["draw_median.selection_ess"]

    fit = good_fit()
    fit["importance_diagnostics"]["posterior_median"]["selection_max_weight_fraction"] = 0.2
    assert failed(assess_ns_fit(fit, criteria)) == ["point.selection_max_weight_fraction"]

    # A missing or non-finite tail value is a failure, never a pass.
    fit = good_fit()
    del fit["importance_diagnostics"]["over_posterior"]["max_event_weight_fraction"]["q0.9"]
    fit["importance_diagnostics"]["over_posterior"]["selection_ess"]["q0.1"] = None
    assert failed(assess_ns_fit(fit, criteria)) == [
        "tail.max_event_weight_fraction",
        "tail.selection_ess",
    ]

    # The criteria choose the tail quantile.
    fit = good_fit()
    fit["importance_diagnostics"]["over_posterior"]["min_event_ess"]["q0.05"] = 5.0
    strict = NSRecoveryAcceptanceCriteria(tail_ess_quantile=0.05)
    assert "tail.min_event_ess" in failed(assess_ns_fit(fit, strict))


def test_assessment_sampler_and_evidence_gates():
    criteria = NSRecoveryAcceptanceCriteria()
    cases = {
        "max_cross_run_r_hat": ("convergence", "max_r_hat", 1.02),
        "min_kish_ess_per_run": ("totals", "min_kish_ess", 900.0),
        "evidence_repeat_std": ("evidence", "repeat_std", 0.25),
        "max_log_evidence_error": ("evidence", "max_reported_error", 0.21),
        "max_pairwise_repeat_z": ("evidence", "max_pairwise_z", 3.2),
        "n_selection_unsupported": ("totals", "n_selection_unsupported", 1),
        "n_nan_or_posinf_log_likelihoods": (
            "totals",
            "n_nan_or_posinf_log_likelihoods",
            2,
        ),
        "all_runs_terminated_by_dlogz": ("totals", "all_converged", False),
    }
    for name, (block, key, value) in cases.items():
        fit = good_fit()
        fit[block][key] = value
        assert failed(assess_ns_fit(fit, criteria)) == [name]
    # Unknown values fail.
    for block, key in (("convergence", "max_r_hat"), ("totals", "n_selection_unsupported")):
        fit = good_fit()
        fit[block][key] = None
        assert not assess_ns_fit(fit, criteria)["passed"]
    fit = good_fit(n_repeats=3)
    assert failed(assess_ns_fit(fit, criteria)) == ["n_repeats"]
    fit = good_fit()
    fit["totals"]["all_converged"] = False
    lenient = NSRecoveryAcceptanceCriteria(require_dlogz_termination=False)
    assert assess_ns_fit(fit, lenient)["passed"]
    fit["evidence"]["repeat_std"] = 0.45
    assert assess_ns_fit(fit, NSRecoveryAcceptanceCriteria.f3_level())["passed"] is False  # dlogz
    f3_lenient = NSRecoveryAcceptanceCriteria.f3_level(require_dlogz_termination=False)
    assert assess_ns_fit(fit, f3_lenient)["passed"]


def summary_for(data_seed, fit, *, truth=3.0, median=3.1, std=0.2):
    return {
        "format_version": RECOVERY_SUMMARY_FORMAT_VERSION,
        "data_seed": data_seed,
        "sampler_seed": data_seed + 1,
        "fit": fit,
        "posterior": {
            "alpha": {
                "truth": truth,
                "median": median,
                "std": std,
                "truth_in_90pct_interval": abs(median - truth) < 1.645 * std,
                "standardized_offset": (median - truth) / std,
            }
        },
    }


def test_campaign_gate_needs_enough_catalogs_and_every_catalog_passing():
    criteria = NSRecoveryAcceptanceCriteria()
    three = aggregate_ns_recovery_summaries(
        [summary_for(i, good_fit()) for i in range(3)], criteria
    )
    assert three["format_version"] == "gwpop-search-phase3-campaign-2.0"
    assert three["all_numerical_pass"] and not three["enough_runs"]
    assert not three["phase3_numerical_gate_passed"]

    four = aggregate_ns_recovery_summaries([summary_for(i, good_fit()) for i in range(4)], criteria)
    assert four["phase3_numerical_gate_passed"]
    coverage = four["coverage"]["alpha"]
    assert coverage["central_90pct_coverage_fraction"] == 1.0
    assert coverage["median_standardized_offset"] == pytest.approx(0.5)
    assert four["criteria"] == criteria.to_dict()

    bad = good_fit()
    bad["importance_diagnostics"]["over_posterior"]["selection_ess"]["q0.1"] = 10.0
    mixed = aggregate_ns_recovery_summaries(
        [summary_for(i, good_fit()) for i in range(3)] + [summary_for(9, bad)], criteria
    )
    assert mixed["n_numerical_fail"] == 1 and not mixed["phase3_numerical_gate_passed"]
    with pytest.raises(ValueError, match="phase3-recovery-2.0"):
        aggregate_ns_recovery_summaries([{**summary_for(1, good_fit()), "format_version": "x"}])


def test_campaign_gate_needs_every_planned_catalog_not_just_min_runs(tmp_path):
    """A subset of the planned catalogs is a selection effect, not a campaign."""
    plan = build_ns_campaign_plan(
        n_runs=6,
        root_seed=20260917,
        repeats=4,
        survey_config=default_phase3_survey_config(),
        dynesty_config=default_phase3_dynesty_config(),
        hbi_config=phase3_hbi_config(),
        criteria=NSRecoveryAcceptanceCriteria(),
    )
    write_or_verify_plan(tmp_path / CAMPAIGN_PLAN_NAME, plan)
    pairs = plan["seed_pairs"]
    for index, pair in enumerate(pairs[:4]):
        directory = tmp_path / f"run_{index:03d}"
        directory.mkdir()
        summary = summary_for(pair["data_seed"], good_fit())
        summary["sampler_seed"] = pair["sampler_seed"]
        (directory / "recovery_summary.json").write_text(json.dumps(summary))

    partial = assess_ns_recovery_campaign(tmp_path)
    assert partial["n_runs"] == 4 and partial["n_planned_runs"] == 6
    assert partial["all_numerical_pass"] and partial["enough_runs"]
    assert not partial["all_planned_catalogs_present"]
    assert not partial["phase3_numerical_gate_passed"]
    assert [item["data_seed"] for item in partial["missing_seed_pairs"]] == [
        pair["data_seed"] for pair in pairs[4:]
    ]

    for index, pair in enumerate(pairs[4:], start=4):
        directory = tmp_path / f"run_{index:03d}"
        directory.mkdir()
        summary = summary_for(pair["data_seed"], good_fit())
        summary["sampler_seed"] = pair["sampler_seed"]
        (directory / "recovery_summary.json").write_text(json.dumps(summary))
    complete = assess_ns_recovery_campaign(tmp_path)
    assert complete["missing_seed_pairs"] == [] and complete["n_planned_runs"] == 6
    assert complete["phase3_numerical_gate_passed"]


def test_checks_fail_closed_on_a_non_numeric_summary_field():
    """A corrupted summary is reported as a failure, never raised out of."""
    criteria = NSRecoveryAcceptanceCriteria()
    for value in ("not-a-number", [1, 2], {"a": 1}):
        fit = good_fit()
        fit["totals"]["min_kish_ess"] = value
        assessment = assess_ns_fit(fit, criteria)
        assert failed(assessment) == ["min_kish_ess_per_run"]
        record = next(
            item for item in assessment["checks"] if item["name"] == "min_kish_ess_per_run"
        )
        assert record["value"] is None and record["raw_value"] == repr(value)
    fit = good_fit()
    fit["importance_diagnostics"]["over_posterior"]["min_event_ess"]["q0.1"] = "nan"
    assert failed(assess_ns_fit(fit, criteria)) == ["tail.min_event_ess"]


def test_nan_or_posinf_criterion_is_a_measurement_of_the_stored_log_likelihoods():
    """``-inf`` is support; NaN and ``+inf`` are counted and gated."""
    from gwpop_search.inference.phase3_ns import n_nan_or_posinf_log_likelihoods

    class _Fake:
        def __init__(self, values):
            self.log_likelihoods = np.asarray(values, dtype=float)

    assert n_nan_or_posinf_log_likelihoods(_Fake([-1.0, -np.inf, -3.0])) == 0
    assert n_nan_or_posinf_log_likelihoods(_Fake([np.nan, -1.0])) == 1
    assert n_nan_or_posinf_log_likelihoods(_Fake([np.inf, np.nan, -np.inf])) == 2

    criteria = NSRecoveryAcceptanceCriteria()
    fit = good_fit()
    fit["totals"]["n_nan_or_posinf_log_likelihoods"] = 1
    assert failed(assess_ns_fit(fit, criteria)) == ["n_nan_or_posinf_log_likelihoods"]
    # The unmeasurable quantity is not gated behind a hard-coded literal.
    names = {item["name"] for item in assess_ns_fit(good_fit(), criteria)["checks"]}
    assert "n_nan_or_posinf_evaluations" not in names


# ---------------------------------------------------------------------------
# End to end on a tiny catalog: layout, summary, reuse, resume
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3-ns") / "campaign"
    summary = run_tiny_campaign(root)
    return root, summary


def test_tiny_campaign_writes_the_v2_summary(tiny_campaign):
    root, campaign = tiny_campaign
    assert campaign["format_version"] == "gwpop-search-phase3-campaign-2.0"
    assert campaign["n_runs"] == 1 and len(campaign["run_assessments"]) == 1
    saved = json.loads((root / "campaign_summary.json").read_text())
    assert saved == json.loads(json.dumps(campaign))

    summary = json.loads((root / "run_000" / "recovery_summary.json").read_text())
    plan = json.loads((root / CAMPAIGN_PLAN_NAME).read_text())
    assert summary["format_version"] == RECOVERY_SUMMARY_FORMAT_VERSION
    assert summary["data_seed"] == plan["seed_pairs"][0]["data_seed"] == V1_DATA_SEEDS[0]
    assert summary["repeat_seeds"] == plan["seed_pairs"][0]["repeat_seeds"]
    names = sorted(summary["truth_hyperparameters"])
    assert sorted(summary["posterior"]) == names
    for stats in summary["posterior"].values():
        expected = {"q05", "median", "q95", "mean", "std", "truth", "truth_in_90pct_interval"}
        assert expected <= set(stats)
        assert "standardized_offset" in stats

    fit = summary["fit"]
    assert fit["n_repeats"] == 2 and len(fit["runs"]) == 2
    assert fit["data_identity"]["n_events"] == 6 and len(fit["data_identity"]["pe_sha256"]) == 64
    for r, run in enumerate(fit["runs"]):
        assert run["run_dir"] == f"repeats/repeat_{r:03d}"
        for key in (
            "log_evidence",
            "log_evidence_error",
            "information",
            "niter",
            "ncall",
            "efficiency",
            "kish_ess",
            "elapsed_seconds",
        ):
            assert run[key] is not None
        result = load_dynesty_result(root / "run_000" / run["run_dir"] / "result.npz")
        assert result.seed == run["seed"] == summary["repeat_seeds"][r]
        assert result.log_evidence == run["log_evidence"]
        assert result.likelihood_identity["hbi_config"]["selection_chunk_size"] is None
    evidence = fit["evidence"]
    assert evidence["n_repeats"] == 2 and evidence["repeat_std"] is not None
    assert evidence["conservative_error"] == max(
        evidence["repeat_std"], evidence["max_reported_error"]
    )
    assert evidence["max_pairwise_z"] is not None
    convergence = fit["convergence"]
    assert convergence["n_chains"] == 2 and sorted(convergence["per_parameter"]) == names
    assert convergence["max_r_hat"] == max(v["rank"] for v in convergence["per_parameter"].values())

    importance = fit["importance_diagnostics"]
    assert importance["n_draws"] == TINY_SETTINGS["importance_draws"]
    assert importance["n_pooled_draws"] == 2 * 200
    assert importance["all_pooled_draws_evaluated"] is False
    assert importance["hbi_config"]["selection_chunk_size"] is None
    for block in ("posterior_median", "truth"):
        assert importance[block]["min_event_ess"] > 0.0
        assert importance[block]["worst_event"].startswith("SYNTH_")
    for key, distribution in importance["over_posterior"].items():
        assert {"q0.1", "q0.5", "q0.9", "min", "max", "n_nonfinite"} <= set(distribution)
        # This tiny fit deliberately subsamples, so the quantiles carry noise
        # and the reported standard error says so.
        errors = distribution["subsample_standard_error"]
        assert set(errors) == {f"q{q:g}" for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
                                                   0.99)}
    assert any(
        distribution["subsample_standard_error"]["q0.1"] > 0.0
        for distribution in importance["over_posterior"].values()
    )

    guarantees = fit["guarantees"]["nan_or_posinf_evaluations"]
    assert guarantees["count"] is None and guarantees["measured"] is False
    assert "PopulationDensityError" in guarantees["enforced_by"]
    assert fit["totals"]["n_nan_or_posinf_log_likelihoods"] == 0
    assert all(run["n_nan_or_posinf_log_likelihoods"] == 0 for run in fit["runs"])

    ranks = summary["truth_rank_diagnostics"]
    assert ranks["observation_model"] == "noisy_observation"
    assert len(ranks["chi_eff"]["quantiles"]) == 6

    pooled = np.load(root / "run_000" / "posterior_pooled.npz")
    assert list(pooled["names"]) == names
    assert pooled["samples"].shape == (2 * 200, len(names))
    assert np.array_equal(np.bincount(pooled["run_index"]), [200, 200])

    checks = {item["name"] for item in campaign["run_assessments"][0]["checks"]}
    assert {"max_cross_run_r_hat", "min_kish_ess_per_run", "tail.selection_ess"} <= checks
    # 100-iteration test runs never pass the Phase-3 gates.
    assert not campaign["phase3_numerical_gate_passed"]


def test_gated_importance_quantiles_are_seed_free_once_every_pooled_draw_is_used(tiny_campaign):
    """The tail gate must not depend on which draws the pinned seed happened to pick."""
    from gwpop_search.inference.phase3_ns import _generate_catalog, pooled_importance_diagnostics

    root, _ = tiny_campaign
    dataset, model = _generate_catalog(V1_DATA_SEEDS[0], tiny_survey())
    results = [
        load_dynesty_result(root / "run_000" / f"repeats/repeat_{r:03d}" / "result.npz")
        for r in (0, 1)
    ]

    def tail(seed, n_draws):
        payload = pooled_importance_diagnostics(
            results,
            dataset.posterior,
            dataset.selection,
            model,
            hbi_config=phase3_hbi_config(),
            n_draws=n_draws,
            seed=seed,
        )
        return payload, payload["over_posterior"]["min_event_ess"]["q0.1"]

    small = [tail(seed, 48) for seed in (1, 2, 3, 4)]
    assert all(payload["n_draws"] == 48 for payload, _ in small)
    assert all(payload["all_pooled_draws_evaluated"] is False for payload, _ in small)
    # Subsampling: the recorded tail moves with the seed, and the reported
    # standard error is positive.
    assert len({value for _, value in small}) > 1
    assert all(
        payload["over_posterior"]["min_event_ess"]["subsample_standard_error"]["q0.1"] > 0.0
        for payload, _ in small
    )

    full = [tail(seed, 16384) for seed in (1, 2, 3, 4)]
    assert all(payload["n_draws"] == payload["n_pooled_draws"] == 400 for payload, _ in full)
    assert all(payload["all_pooled_draws_evaluated"] is True for payload, _ in full)
    assert len({value for _, value in full}) == 1
    for payload, _ in full:
        for distribution in payload["over_posterior"].values():
            assert set(distribution["subsample_standard_error"].values()) == {0.0}


def test_rerunning_a_completed_campaign_reuses_every_run_untouched(tiny_campaign):
    root, campaign = tiny_campaign
    before = ns_run_fingerprints(root)
    summary_before = (root / "run_000" / "recovery_summary.json").read_text()
    assert {
        "campaign_plan.json",
        "run_000/repeats/repeat_000/manifest.json",
        "run_000/repeats/repeat_000/result.npz",
        "run_000/repeats/repeat_000/result.npz.json",
        "run_000/repeats/repeat_001/result.npz",
    } <= set(before)
    assert not any("checkpoint" in key for key in before)
    again = run_tiny_campaign(root)
    after = ns_run_fingerprints(root)
    comparison = compare_ns_fingerprints(before, after)
    assert comparison["passed"] and not comparison["added"] and after == before
    assert (root / "run_000" / "recovery_summary.json").read_text() == summary_before
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(campaign))
    assessed = assess_ns_recovery_campaign(root)
    assert assessed["criteria"] == TINY_CRITERIA.to_dict()  # read back from the plan


class SimulatedKill(Exception):
    pass


class KillAfter:
    def __init__(self, inner, n_calls):
        self.inner = inner
        self.n_calls = n_calls
        self.calls = 0

    def __call__(self, X):
        self.calls += 1
        if self.calls > self.n_calls:
            raise SimulatedKill
        return self.inner(X)

    def __getattr__(self, name):
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def result_arrays(path):
    result = load_dynesty_result(path)
    return (
        result.log_evidence,
        result.log_evidence_error,
        result.niter,
        result.samples,
        result.log_weights,
        result.posterior_samples,
    )


def assert_same_result(a, b):
    ra, rb = result_arrays(a), result_arrays(b)
    assert ra[:3] == rb[:3]
    for x, y in zip(ra[3:], rb[3:]):
        np.testing.assert_array_equal(x, y)


def test_campaign_resumes_an_interrupted_repeat_from_its_checkpoint(
    tmp_path, monkeypatch, tiny_campaign
):
    """An exception mid-run (in-process kill) leaves a checkpoint; the rerun resumes it."""
    control_root, _ = tiny_campaign
    root = tmp_path / "killed"
    config = tiny_dynesty(checkpoint_every=1e-6)
    real_builder = dynesty_backend.build_batched_log_likelihood
    state = {"built": 0}

    def builder(*args, **kwargs):
        state["built"] += 1
        likelihood = real_builder(*args, **kwargs)
        # The first repeat runs to completion; the second is killed mid-run.
        return likelihood if state["built"] == 1 else KillAfter(likelihood, 25)

    monkeypatch.setattr(dynesty_backend, "build_batched_log_likelihood", builder)
    with pytest.raises(SimulatedKill):
        run_tiny_campaign(root, dynesty_config=config)
    monkeypatch.setattr(dynesty_backend, "build_batched_log_likelihood", real_builder)
    repeat0 = root / "run_000" / "repeats" / "repeat_000"
    repeat1 = root / "run_000" / "repeats" / "repeat_001"
    assert (repeat0 / "result.npz").exists()
    assert (repeat1 / "checkpoint.pkl").exists() and not (repeat1 / "result.npz").exists()
    before = ns_run_fingerprints(root)

    run_tiny_campaign(root)  # default cadence: not part of the plan
    comparison = compare_ns_fingerprints(before, ns_run_fingerprints(root))
    assert comparison["passed"], comparison
    assert "run_000/repeats/repeat_001/result.npz" in comparison["added"]
    assert load_dynesty_result(repeat1 / "result.npz").diagnostics["resumed"]
    for r in (0, 1):
        assert_same_result(
            root / "run_000" / "repeats" / f"repeat_{r:03d}" / "result.npz",
            control_root / "run_000" / "repeats" / f"repeat_{r:03d}" / "result.npz",
        )


_KILL_SCRIPT = r"""
import json, sys, warnings
import jax
jax.config.update("jax_enable_x64", True)
from gwpop_search.inference.dynesty_backend import DynestyConfig
from gwpop_search.inference.phase3_ns import NSRecoveryAcceptanceCriteria, run_ns_recovery_campaign
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
payload = json.loads(sys.argv[1])
warnings.simplefilter("ignore")
run_ns_recovery_campaign(
    payload["root"],
    n_runs=1,
    repeats=2,
    survey_config=SyntheticSurveyConfig.from_dict(payload["survey"]),
    dynesty_config=DynestyConfig.from_dict(payload["dynesty"]),
    criteria=NSRecoveryAcceptanceCriteria.from_dict(payload["criteria"]),
    importance_draws=payload["importance_draws"],
    rhat_draws_per_run=payload["rhat_draws_per_run"],
)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs POSIX SIGKILL")
def test_campaign_survives_sigkill_mid_run(tmp_path, tiny_campaign):
    """SIGKILL a campaign process mid-way through its second repeat, then resume."""
    control_root, _ = tiny_campaign
    root = tmp_path / "sigkill"
    payload = {
        "root": str(root),
        "survey": tiny_survey().to_dict(),
        "dynesty": tiny_dynesty(checkpoint_every=1e-6).to_dict(),
        "criteria": TINY_CRITERIA.to_dict(),
        **TINY_SETTINGS,
    }
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.Popen(
        [sys.executable, "-c", _KILL_SCRIPT, json.dumps(payload)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    repeat0 = root / "run_000" / "repeats" / "repeat_000"
    repeat1 = root / "run_000" / "repeats" / "repeat_001"
    try:
        deadline = time.monotonic() + 300.0
        while not (repeat1 / "checkpoint.pkl").exists():
            if process.poll() is not None:
                raise AssertionError(
                    f"campaign process exited early: {process.stderr.read().decode()[-2000:]}"
                )
            if time.monotonic() > deadline:
                raise AssertionError("no checkpoint of the second repeat appeared")
            time.sleep(0.005)
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
    assert process.returncode == -signal.SIGKILL
    assert (repeat0 / "result.npz").exists()
    assert not (repeat1 / "result.npz").exists(), "the kill landed after the repeat finished"
    assert not (root / "run_000" / "recovery_summary.json").exists()
    before = ns_run_fingerprints(root)

    run_tiny_campaign(root)
    comparison = compare_ns_fingerprints(before, ns_run_fingerprints(root))
    assert comparison["passed"], comparison
    assert load_dynesty_result(repeat1 / "result.npz").diagnostics["resumed"]
    for r in (0, 1):
        assert_same_result(
            root / "run_000" / "repeats" / f"repeat_{r:03d}" / "result.npz",
            control_root / "run_000" / "repeats" / f"repeat_{r:03d}" / "result.npz",
        )
    control = json.loads((control_root / "run_000" / "recovery_summary.json").read_text())
    resumed = json.loads((root / "run_000" / "recovery_summary.json").read_text())
    assert resumed["posterior"] == control["posterior"]
    assert resumed["fit"]["evidence"] == control["fit"]["evidence"]
    assert resumed["fit"]["convergence"] == control["fit"]["convergence"]
    assert resumed["fit"]["importance_diagnostics"] == control["fit"]["importance_diagnostics"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_ns_commands_default_to_the_v2_plan(tmp_path):
    from gwpop_search.cli import (
        _evidence_check_dynesty,
        _ns_dynesty_config,
        _ns_survey_config,
    )

    parser = build_parser()
    args = parser.parse_args(["synthetic-campaign-ns", "--root", str(tmp_path)])
    assert args.func.__name__ == "_run_synthetic_campaign_ns"
    assert (args.n_runs, args.repeats, args.root_seed, args.min_runs, args.min_repeats) == (
        4,
        4,
        20260917,
        4,
        4,
    )
    assert _ns_survey_config(args) == default_phase3_survey_config()
    config = _ns_dynesty_config(args, slices=slices_for_ndim(10))
    assert config.identity_dict() == default_phase3_dynesty_config().identity_dict()
    assert args.importance_draws == 16384 and args.slices is None
    assert args.importance_draws >= args.repeats * args.num_posterior_samples

    evidence = parser.parse_args(["synthetic-evidence-check", "--root", str(tmp_path)])
    assert evidence.func.__name__ == "_run_synthetic_evidence_check"
    assert (evidence.n_catalogs, evidence.repeats, evidence.slices) == (4, 2, None)
    assert _ns_survey_config(evidence) == default_phase3_survey_config()
    base, rule = _evidence_check_dynesty(evidence)
    assert rule == "2*(3+ndim)" and base.slices is None and base.nlive == 1000
    fixed = parser.parse_args(["synthetic-evidence-check", "--root", "x", "--slices", "30"])
    assert _evidence_check_dynesty(fixed)[1] == "fixed"
    assert _evidence_check_dynesty(fixed)[0].slices == 30
    rwalk = parser.parse_args(["synthetic-evidence-check", "--root", "x", "--sample", "rwalk"])
    assert _evidence_check_dynesty(rwalk)[1] == "fixed"

    fingerprint = parser.parse_args(["fingerprint-ns-run", "--run-dir", str(tmp_path)])
    assert fingerprint.func.__name__ == "_fingerprint_ns_run"
    assess = parser.parse_args(["assess-synthetic-campaign-ns", "--root", str(tmp_path)])
    assert assess.min_runs is None and assess.min_repeats is None


def test_fingerprint_cli_writes_and_compares(tmp_path, tiny_campaign, capsys):
    root, _ = tiny_campaign
    parser = build_parser()
    out = tmp_path / "before.json"
    args = parser.parse_args(["fingerprint-ns-run", "--run-dir", str(root), "--output", str(out)])
    args.func(args)
    report = json.loads(out.read_text())
    assert report["format_version"] == "gwpop-search-ns-run-fingerprints-1.0"
    assert report["fingerprints"] == ns_run_fingerprints(root)
    capsys.readouterr()

    args = parser.parse_args(
        ["fingerprint-ns-run", "--run-dir", str(root), "--compare-to", str(out)]
    )
    args.func(args)
    assert json.loads(capsys.readouterr().out)["comparison"]["passed"]

    tampered = dict(report)
    key = "run_000/repeats/repeat_000/result.npz"
    tampered["fingerprints"] = {**report["fingerprints"], key: "0" * 64}
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered))
    args = parser.parse_args(
        ["fingerprint-ns-run", "--run-dir", str(root), "--compare-to", str(bad)]
    )
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert excinfo.value.code == 1
    assert json.loads(capsys.readouterr().out)["comparison"]["changed"] == [key]


def test_assess_cli_overrides_only_what_it_is_told(tiny_campaign, capsys):
    root, _ = tiny_campaign
    parser = build_parser()
    args = parser.parse_args(
        ["assess-synthetic-campaign-ns", "--root", str(root), "--min-runs", "2"]
    )
    args.func(args)
    assert "runs=1" in capsys.readouterr().out
    summary = json.loads((root / "campaign_summary.json").read_text())
    assert summary["criteria"]["min_runs"] == 2 and summary["criteria"]["min_repeats"] == 2
    assert not summary["enough_runs"]
    args = parser.parse_args(["assess-synthetic-campaign-ns", "--root", str(root)])
    args.func(args)
    summary = json.loads((root / "campaign_summary.json").read_text())
    assert summary["criteria"] == TINY_CRITERIA.to_dict()


def test_fingerprints_only_cover_completed_run_artifacts(tmp_path):
    run = tmp_path / "a" / "repeat_000"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    (run / "checkpoint.pkl").write_bytes(b"x")
    assert list(ns_run_fingerprints(tmp_path)) == ["a/repeat_000/manifest.json"]
    (run / "result.npz").write_bytes(b"y")
    (run / "result.npz.json").write_text("{}")
    assert list(ns_run_fingerprints(tmp_path)) == [
        "a/repeat_000/manifest.json",
        "a/repeat_000/result.npz",
        "a/repeat_000/result.npz.json",
    ]
    assert list(ns_run_fingerprints(run)) == ["manifest.json", "result.npz", "result.npz.json"]
    with pytest.raises(FileNotFoundError):
        ns_run_fingerprints(tmp_path / "missing")
    comparison = compare_ns_fingerprints({"a": "1", "b": "2"}, {"a": "1", "c": "3"})
    assert comparison == {
        "unchanged": ["a"],
        "changed": [],
        "missing": ["b"],
        "added": ["c"],
        "passed": False,
    }


# ---------------------------------------------------------------------------
# Launcher and cross-module contract
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_h100_launcher_runs_the_pinned_checkout_not_the_installed_package():
    """The environment unsets PYTHONPATH and installs the package from main."""
    script = (REPO_ROOT / "scripts/slurm/phase3_ns_h100.sbatch.example").read_text()
    assert 'export PYTHONPATH="${GWPOP_CODE}/src"' in script
    assert script.index("source \"${GWPOP_ENV}\"") < script.index("export PYTHONPATH")
    assert 'GWPOP="python -m gwpop_search.cli"' in script
    # The console script belongs to the environment's editable install of the
    # MAIN checkout; a bare invocation would silently run the wrong code (and,
    # before this branch is merged, would not know the subcommands at all).
    for command in ("synthetic-campaign-ns", "fingerprint-ns-run", "synthetic-evidence-check"):
        assert f"gwpop-search {command}" not in script
        assert f"${{GWPOP}} {command}" in script
    # The pre-flight block refuses a mismatched package before burning GPU time.
    assert "gwpop_search.__file__" in script or "import gwpop_search" in script
    assert "is_relative_to" in script


def test_private_helpers_this_module_imports_still_behave_as_assumed(tmp_path):
    """Track A's plan/manifest identity rests on private helpers of other tracks."""
    from gwpop_search.inference import campaign as campaign_module
    from gwpop_search.inference import numpyro as numpyro_module
    from gwpop_search.inference import phase3_evidence, phase3_ns

    for module, names in (
        (
            dynesty_backend,
            (
                "_atomic_write_text",
                "_json_ready",
                "_json_sha256",
                "_manifest_differences",
                "_without_chunk_size",
            ),
        ),
        (campaign_module, ("_derived_seed", "file_sha256")),
        (numpyro_module, ("_hbi_config_dict",)),
    ):
        for name in names:
            assert callable(getattr(module, name)), f"{module.__name__}.{name}"

    assert dynesty_backend._json_ready({"a": np.float64(1.5)}) == {"a": 1.5}
    digest = dynesty_backend._json_sha256({"a": 1})
    assert len(digest) == 64 and digest == dynesty_backend._json_sha256({"a": 1})
    assert dynesty_backend._manifest_differences({"a": 1}, {"a": 1}) == []
    assert dynesty_backend._manifest_differences({"a": 1}, {"a": 2}) == ["a"]
    chunked = {"hbi_config": {"selection_chunk_size": 4096, "log_selection_floor": None}}
    assert "selection_chunk_size" not in dynesty_backend._without_chunk_size(chunked)["hbi_config"]
    path = tmp_path / "x.json"
    dynesty_backend._atomic_write_text(path, "{}")
    assert path.read_text() == "{}"
    assert len(campaign_module.file_sha256(path)) == 64
    assert campaign_module._derived_seed(7, "tag", 0) == campaign_module._derived_seed(7, "tag", 0)
    assert numpyro_module._hbi_config_dict(phase3_hbi_config())["selection_chunk_size"] is None
    # The evidence check reuses the recovery module's private helpers too.
    assert phase3_evidence._finite_or_none is phase3_ns._finite_or_none
