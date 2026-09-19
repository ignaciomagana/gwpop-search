"""Fidelity ladder v2 (F0 -> F3 -> F4) on dynesty: configuration, gates, evaluator."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.special import logsumexp

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from gwpop_search.grammar import (  # noqa: E402
    PriorConfig,
    baseline_model_spec,
    enumerate_model_graph,
)
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference import (  # noqa: E402
    DynestyConfig,
    DynestyResult,
    NoFiniteSupportError,
    build_batched_log_likelihood,
    build_importance_diagnostics,
    build_likelihood_identity,
    equal_weight_resample,
    prior_specs_from_model_spec,
    prior_transform_for,
)
from gwpop_search.inference.evidence_campaign import EvidenceCampaignConfig  # noqa: E402
from gwpop_search.inference.fidelity import (  # noqa: E402
    EVALUATION_FORMAT_VERSION,
    FIDELITY_CONFIG_FORMAT_VERSION,
    DeterministicHBIEvaluator,
    F0SanityConfig,
    FidelityRunConfig,
    LegacyFidelityConfigError,
    NumericalCriteria,
    fidelity_run_config_from_dict,
    fidelity_run_config_to_dict,
    load_pooled_posterior,
    read_evaluation,
    summarize_dynesty_fit,
)
from gwpop_search.inference.synthetic import (  # noqa: E402
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import compile_model_spec  # noqa: E402
from gwpop_search.search.scheduler import Fidelity  # noqa: E402

HBI = HBIConfig(selection_chunk_size=None)


@pytest.fixture(scope="module")
def dataset():
    return generate_baseline_synthetic_dataset(
        seed=44,
        config=SyntheticSurveyConfig(
            n_events=5,
            posterior_samples_per_event=32,
            n_injections=2_000,
            population_batch_size=256,
            redshift_sampling_grid=1024,
        ),
    )


def _lenient(**overrides):
    values = dict(
        max_cross_run_r_hat=100.0,
        min_kish_ess_per_run=1.0,
        max_evidence_error=100.0,
        max_evidence_repeat_std=100.0,
        max_repeat_consistency_z=100.0,
        min_event_ess=1e-9,
        min_selection_ess=1e-9,
        max_event_weight_fraction=1.0,
        max_selection_weight_fraction=1.0,
        max_shape_log_likelihood_variance=1e12,
    )
    values.update(overrides)
    return NumericalCriteria(**values)


def _small_config(**overrides):
    dynesty = DynestyConfig(
        nlive=20, bound="multi", sample="rslice", dlogz=0.5, maxiter=30, batch_size=8,
        num_posterior_samples=60,
    )
    values = dict(
        f0=F0SanityConfig(
            prior_draws=128,
            batch_size=16,
            parity_pe_samples_per_event=16,
            parity_selected_per_campaign=128,
        ),
        f3_evidence=EvidenceCampaignConfig(repeats=2, dynesty=dynesty),
        f4_evidence=EvidenceCampaignConfig(repeats=2, dynesty=dynesty),
        f3_criteria=_lenient(),
        f4_criteria=_lenient(),
        posterior_draws=32,
        rhat_draws_per_run=50,
    )
    values.update(overrides)
    return FidelityRunConfig(**values)


_COMPILED = {}


def _compiled(dataset, spec, hbi):
    """Batched likelihood and identity per (spec, HBI config), compiled once per module."""
    key = (spec.model_hash, hbi)
    if key not in _COMPILED:
        model = compile_model_spec(spec)
        names, _ = prior_transform_for(prior_specs_from_model_spec(spec))
        _COMPILED[key] = (
            build_batched_log_likelihood(
                dataset.posterior, dataset.selection, model, names, hbi_config=hbi, batch_size=64
            ),
            build_likelihood_identity(dataset.posterior, dataset.selection, model, names, hbi),
        )
    return _COMPILED[key]


def _fake_runs(
    dataset,
    spec,
    *,
    n_runs=2,
    n=300,
    seed=0,
    log_evidences=None,
    errors=None,
    converged=None,
    unsupported=None,
    hbi=HBI,
):
    """Weighted-prior-sample stand-ins for dynesty runs (valid weighted posteriors)."""
    priors = prior_specs_from_model_spec(spec)
    names, transform = prior_transform_for(priors)
    loglike, identity = _compiled(dataset, spec, hbi)
    config = DynestyConfig(nlive=50, num_posterior_samples=80)
    results = []
    for r in range(n_runs):
        rng = np.random.default_rng([seed, r])
        theta = transform(rng.random((n, len(names))))
        logl = loglike(theta)
        log_w = np.where(np.isfinite(logl), logl - np.log(n), -np.inf)
        logz = float(logsumexp(log_w))
        weights = np.exp(log_w - logz)
        ok = True if converged is None else converged[r]
        results.append(
            DynestyResult(
                names=names,
                log_evidence=logz if log_evidences is None else float(log_evidences[r]),
                log_evidence_error=0.1 if errors is None else errors[r],
                information=2.0,
                niter=n,
                ncall=n,
                efficiency=100.0,
                samples=theta,
                log_weights=log_w,
                log_likelihoods=logl,
                log_volumes=np.linspace(0.0, -5.0, n),
                posterior_samples=equal_weight_resample(theta, weights, 80, rng),
                kish_ess=float(1.0 / np.sum(weights**2)),
                elapsed_seconds=0.5,
                config=config,
                seed=int(1000 + r),
                versions={"dynesty": "3.1.0"},
                diagnostics={
                    "converged": ok,
                    "final_delta_logz": 0.01 if ok else 3.0,
                    "n_zero_likelihood_points": int(np.count_nonzero(~np.isfinite(logl))),
                },
                n_likelihood_evaluations=n,
                n_selection_unsupported=0 if unsupported is None else unsupported[r],
                provenance={"likelihood": identity},
            )
        )
    return results


def _names(checks, *, failed_gates_only=False):
    return [
        item["name"]
        for item in checks
        if not failed_gates_only or (item["stage"] == "gate" and not item["passed"])
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_config_is_ladder_v2_with_d3_thresholds_and_round_trips():
    config = FidelityRunConfig()
    assert config.format_version == FIDELITY_CONFIG_FORMAT_VERSION
    assert config.hbi.selection_chunk_size is None
    assert config.f0.prior_draws == 4096 and config.f0.min_finite_fraction == 1e-3
    assert config.f0.max_zero_exposure_fraction == 0.0
    f3, f4 = config.f3_evidence, config.f4_evidence
    assert (f3.repeats, f3.dynesty.nlive, f3.dynesty.dlogz) == (2, 1000, 0.1)
    assert (f4.repeats, f4.dynesty.nlive, f4.dynesty.dlogz) == (3, 2000, 0.05)
    for rung in (f3, f4):
        assert rung.dynesty.bound == "multi" and rung.dynesty.sample == "rslice"
        assert rung.slices_multiplier == 2 and rung.dynesty_config_for(13).slices == 32
    c3, c4 = config.f3_criteria, config.f4_criteria
    assert (c3.max_cross_run_r_hat, c3.min_kish_ess_per_run) == (1.05, 500.0)
    assert (c3.max_evidence_repeat_std, c3.max_evidence_error, c3.max_repeat_consistency_z) == (
        0.5, 0.5, 3.5,
    )
    assert (
        c3.min_event_ess, c3.min_selection_ess, c3.max_event_weight_fraction,
        c3.max_selection_weight_fraction, c3.max_shape_log_likelihood_variance,
    ) == (10.0, 100.0, 0.35, 0.15, 2.0)
    assert (c4.max_cross_run_r_hat, c4.min_kish_ess_per_run) == (1.01, 2000.0)
    assert (c4.max_evidence_repeat_std, c4.max_evidence_error, c4.max_repeat_consistency_z) == (
        0.2, 0.2, 3.0,
    )
    assert (
        c4.min_event_ess, c4.min_selection_ess, c4.max_event_weight_fraction,
        c4.max_selection_weight_fraction, c4.max_shape_log_likelihood_variance,
    ) == (20.0, 200.0, 0.25, 0.10, 1.0)
    assert c3.importance_tail_quantile == c4.importance_tail_quantile == 0.1
    assert c3.gate_importance_over_posterior and c4.gate_importance_over_posterior
    assert c4.insertion_index_advisory and not c3.insertion_index_advisory
    assert config.posterior_draws == 512

    payload = json.loads(json.dumps(fidelity_run_config_to_dict(config)))
    assert payload["format_version"] == "gwpop-search-fidelity-config-2.0"
    assert payload["ladder_fidelities"] == ["F0", "F3", "F4"]
    assert "f4_nuts" not in payload and "f1_nuts" not in payload
    assert fidelity_run_config_from_dict(payload) == config


def test_nuts_jaxns_era_configs_are_refused():
    legacy = {
        "f0_pe_samples_per_event": 32,
        "f1_nuts": {"num_warmup": 300},
        "f3_evidence": {"repeats": 2, "nested_sampling": {"num_live_points": 250}},
    }
    with pytest.raises(LegacyFidelityConfigError, match="NUTS/JAXNS-era config; re-freeze"):
        fidelity_run_config_from_dict(legacy)
    payload = fidelity_run_config_to_dict(FidelityRunConfig())
    payload["format_version"] = "gwpop-search-fidelity-config-1.0"
    with pytest.raises(LegacyFidelityConfigError, match="re-freeze"):
        fidelity_run_config_from_dict(payload)
    criteria = FidelityRunConfig().f3_criteria.to_dict()
    criteria["max_r_hat"] = 1.05
    with pytest.raises(LegacyFidelityConfigError, match="MCMC-era"):
        NumericalCriteria.from_dict(criteria)


def test_config_enforces_rslice_multi_and_repeats_for_cross_run_gates():
    config = FidelityRunConfig()
    rwalk = EvidenceCampaignConfig(
        dynesty=DynestyConfig(nlive=100, sample="rwalk"), slices_multiplier=None
    )
    with pytest.raises(ValueError, match="rslice"):
        replace(config, f3_evidence=rwalk)
    single = EvidenceCampaignConfig(repeats=1)
    with pytest.raises(ValueError, match="repeats >= 2"):
        replace(config, f3_evidence=single)
    replace(config, f3_evidence=single, f3_criteria=NumericalCriteria())  # no cross-run gates
    with pytest.raises(ValueError, match="shape"):
        replace(config, hbi=HBIConfig(rate_treatment="poisson"))
    with pytest.raises(ValueError, match="exceed one"):
        NumericalCriteria(max_cross_run_r_hat=1.0)
    with pytest.raises(ValueError, match="tail_quantile"):
        NumericalCriteria(importance_tail_quantile=0.5)


# ---------------------------------------------------------------------------
# F0
# ---------------------------------------------------------------------------


def test_f0_scans_the_prior_on_full_data_with_parity(dataset, tmp_path):
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior,
        dataset.selection,
        config=_small_config(),
        dataset_identity="synthetic-test",
    )
    record = evaluator.evaluate(
        baseline_model_spec(), Fidelity.F0_SANITY, seed=123, run_dir=tmp_path / "f0"
    )
    assert record.status == "complete"
    assert record.diagnostics_pass
    assert record.screen_value == 0.0
    payload = read_evaluation(tmp_path / "f0" / "evaluation.json")
    assert payload["format_version"] == EVALUATION_FORMAT_VERSION
    assert payload["dataset_identity"] == "synthetic-test"
    assert payload["screen_value_semantics"] == "sanity_constant"
    assert payload["sampler_backend"]["name"] == "dynesty"
    diagnostics = payload["diagnostics"]
    assert all(item["passed"] for item in diagnostics["checks"])
    assert diagnostics["support"]["n_draws"] == 128
    assert 1e-3 <= diagnostics["support"]["finite_fraction"] <= 1.0
    assert diagnostics["support"]["zero_exposure_fraction"] == 0.0
    assert diagnostics["support"]["event_zero_support_fraction_top"]
    assert diagnostics["parity"]["n_points_compared"] == 2
    assert diagnostics["parity"]["max_rel_diff"] <= 1e-9
    assert diagnostics["throughput"]["batch_size"] == 16
    assert diagnostics["parameterization"]["kind"] == "identity"


def test_f0_fails_for_a_model_without_population_support(dataset, tmp_path):
    spec = baseline_model_spec()
    priors = dict(spec.priors)
    # Every draw has mmax below the observed masses (and mostly mmax < mmin).
    priors["mmax"] = PriorConfig("uniform", {"low": 2.5, "high": 3.0})
    empty = replace(spec, priors=priors)
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior, dataset.selection, config=_small_config()
    )
    record = evaluator.evaluate(empty, Fidelity.F0_SANITY, seed=3, run_dir=tmp_path / "f0")
    assert not record.diagnostics_pass
    payload = read_evaluation(tmp_path / "f0" / "evaluation.json")
    failed = _names(payload["diagnostics"]["checks"], failed_gates_only=True)
    assert "f0.finite_fraction" in failed
    assert "f0.zero_exposure_fraction" in failed
    assert "f0.jax_numpy_parity_rel_diff" in failed  # no finite point to compare
    assert payload["diagnostics"]["support"]["finite_fraction"] == 0.0


def test_f1_and_f2_are_not_rungs_of_ladder_v2(dataset, tmp_path):
    evaluator = DeterministicHBIEvaluator(dataset.posterior, dataset.selection)
    for fidelity in (Fidelity.F1_SCREEN, Fidelity.F2_INFERENCE):
        with pytest.raises(ValueError, match="not part of fidelity ladder v2"):
            evaluator.evaluate(
                baseline_model_spec(), fidelity, seed=1, run_dir=tmp_path / fidelity.value
            )
    assert DeterministicHBIEvaluator.supported_fidelities == ("F0", "F3", "F4")


# ---------------------------------------------------------------------------
# F3/F4 gates
# ---------------------------------------------------------------------------


def test_f3_evaluation_gates_pools_and_saves_the_posterior(monkeypatch, dataset, tmp_path):
    spec = baseline_model_spec()
    results = _fake_runs(dataset, spec, log_evidences=[-10.05, -9.95])
    seen = {}

    def fake_repeats(model_dir, model, posterior, selection, **kwargs):
        seen.update(kwargs)
        return results, {}

    monkeypatch.setattr("gwpop_search.inference.fidelity.run_model_evidence_repeats", fake_repeats)
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior, dataset.selection, config=_small_config(), dataset_identity="sha"
    )
    record = evaluator.evaluate(spec, Fidelity.F3_EVIDENCE, seed=9, run_dir=tmp_path / "f3")
    assert seen["dataset_identity"] == "sha" and seen["root_seed"] == 9
    assert seen["hbi_config"] == HBI
    assert record.diagnostics_pass
    expected_mean = np.mean([result.log_evidence for result in results])
    assert record.screen_value == pytest.approx(expected_mean)

    payload = read_evaluation(tmp_path / "f3" / "evaluation.json")
    assert payload["screen_value_semantics"] == "log_evidence_mean"
    diagnostics = payload["diagnostics"]
    assert diagnostics["failure"] is None
    names = _names(diagnostics["checks"])
    for expected in (
        "nested_sampling.all_runs_terminated_by_dlogz",
        "nested_sampling.selection_unsupported_evaluations",
        "nested_sampling.min_kish_ess_per_run",
        "nested_sampling.cross_run_r_hat",
        "evidence.conservative_error",
        "evidence.repeat_std",
        "evidence.max_pairwise_z",
    ):
        assert expected in names
    for metric in (
        "min_event_ess",
        "selection_ess",
        "max_event_weight_fraction",
        "selection_max_weight_fraction",
        "shape_log_likelihood_variance",
    ):
        for where in ("point", "draw_median", "draw_tail"):
            assert f"importance.{metric}.{where}" in names
    evidence = diagnostics["evidence"]
    assert evidence["log_evidence_mean"] == pytest.approx(expected_mean)
    estimates = [result.log_evidence for result in results]
    assert evidence["repeat_std"] == pytest.approx(np.std(estimates, ddof=1))
    assert evidence["conservative_error"] == pytest.approx(
        max(evidence["repeat_std"], evidence["max_reported_error"])
    )
    sampling = diagnostics["nested_sampling"]
    assert sampling["n_runs"] == 2 and len(sampling["runs"]) == 2
    assert sampling["resolved_dynesty_config"]["slices"] == 26
    assert sampling["cross_run_r_hat"]["n_runs"] == 2
    assert sampling["insertion_index"]["available"] is False
    posterior = diagnostics["posterior"]
    assert posterior["parameter_order"] == sorted(spec.priors)
    assert diagnostics["posterior_median"] == posterior["median"]
    for name in spec.priors:
        assert posterior["q05"][name] <= posterior["median"][name] <= posterior["q95"][name]
    artifact = posterior["pooled_equal_weight"]
    pooled = load_pooled_posterior(tmp_path / "f3" / artifact["artifact"])
    assert pooled["samples"].shape == (160, len(spec.priors))
    assert pooled["parameter_names"].tolist() == sorted(spec.priors)
    np.testing.assert_array_equal(pooled["run_index"], np.repeat([0, 1], 80))
    np.testing.assert_array_equal(
        pooled["samples"], np.vstack([r.posterior_samples for r in results])
    )
    importance = diagnostics["importance"]
    assert importance["hbi_config"]["selection_chunk_size"] is None
    assert importance["over_posterior"]["n_draws"] == 32
    point = importance["at_posterior_median"]
    assert point["hyperparameters"] == pytest.approx(posterior["median"])
    tail = importance["over_posterior"]["metrics"]["min_event_ess"]
    assert tail["tail_quantile"] == 0.1 and tail["tail"] <= tail["median"]
    assert importance["over_posterior"]["metrics"]["shape_log_likelihood_variance"][
        "tail_quantile"
    ] == 0.9


def test_gate_matrix_termination_consistency_kish_and_selection_support(dataset):
    spec = baseline_model_spec()
    model = compile_model_spec(spec)
    priors = prior_specs_from_model_spec(spec)

    diagnostics_fn = build_importance_diagnostics(
        dataset.posterior, dataset.selection, model, tuple(sorted(priors)), hbi_config=HBI
    )

    def summarize(results, criteria=None):
        return summarize_dynesty_fit(
            results,
            dataset.posterior,
            dataset.selection,
            model,
            hbi_config=HBI,
            criteria=_lenient() if criteria is None else criteria,
            priors=priors,
            n_draws=16,
            rhat_draws_per_run=40,
            diagnostics_fn=diagnostics_fn,
        )

    base = [-10.0, -10.05]
    good = summarize(_fake_runs(dataset, spec, log_evidences=base))
    assert good["passed"] and not _names(good["checks"], failed_gates_only=True)

    stopped = summarize(_fake_runs(dataset, spec, log_evidences=base, converged=[True, False]))
    assert _names(stopped["checks"], failed_gates_only=True) == [
        "nested_sampling.all_runs_terminated_by_dlogz"
    ]
    inconsistent = summarize(
        _fake_runs(dataset, spec, log_evidences=[-10.0, -8.0], errors=[0.1, 0.1]),
        _lenient(max_repeat_consistency_z=3.0),
    )
    assert _names(inconsistent["checks"], failed_gates_only=True) == ["evidence.max_pairwise_z"]
    noisy = summarize(
        _fake_runs(dataset, spec, log_evidences=[-10.0, -8.0], errors=[1.0, 1.0]),
        _lenient(max_repeat_consistency_z=3.0, max_evidence_repeat_std=0.5),
    )
    assert _names(noisy["checks"], failed_gates_only=True) == ["evidence.repeat_std"]
    kish = summarize(
        _fake_runs(dataset, spec, log_evidences=base), _lenient(min_kish_ess_per_run=1e6)
    )
    assert _names(kish["checks"], failed_gates_only=True) == [
        "nested_sampling.min_kish_ess_per_run"
    ]
    unsupported = summarize(_fake_runs(dataset, spec, log_evidences=base, unsupported=[0, 3]))
    assert _names(unsupported["checks"], failed_gates_only=True) == [
        "nested_sampling.selection_unsupported_evaluations"
    ]


def test_importance_tail_gate_catches_what_the_point_misses(dataset):
    spec = baseline_model_spec()
    model = compile_model_spec(spec)
    results = _fake_runs(dataset, spec, n=2000, log_evidences=[-10.0, -10.0])
    diagnostics_fn = build_importance_diagnostics(
        dataset.posterior, dataset.selection, model, results[0].names, hbi_config=HBI
    )

    def summarize(criteria):
        return summarize_dynesty_fit(
            results, dataset.posterior, dataset.selection, model, hbi_config=HBI,
            criteria=criteria, n_draws=64, rhat_draws_per_run=40,
            diagnostics_fn=diagnostics_fn,
        )

    reference = summarize(_lenient())
    point = reference["importance"]["at_posterior_median"]
    metrics = reference["importance"]["over_posterior"]["metrics"]
    # A metric whose posterior tail is worse than its value at the median point.
    options = {
        "shape_log_likelihood_variance": ("max_shape_log_likelihood_variance", 1.0),
        "max_event_weight_fraction": ("max_event_weight_fraction", 1.0),
        "min_event_ess": ("min_event_ess", -1.0),
        "selection_ess": ("min_selection_ess", -1.0),
    }
    chosen = None
    for metric, (attribute, sign) in options.items():
        if sign * (metrics[metric]["tail"] - point[metric]) > 0:
            chosen = metric, attribute
            break
    assert chosen is not None, (point, metrics)
    metric, attribute = chosen
    limit = 0.5 * (point[metric] + metrics[metric]["tail"])
    strict = summarize(_lenient(**{attribute: limit}))
    failed = _names(strict["checks"], failed_gates_only=True)
    assert f"importance.{metric}.draw_tail" in failed
    assert f"importance.{metric}.point" not in failed
    assert not strict["passed"]
    # Over-posterior checks become advisory only when explicitly requested.
    advisory = summarize(_lenient(**{attribute: limit, "gate_importance_over_posterior": False}))
    tail_check = [
        item for item in advisory["checks"] if item["name"] == f"importance.{metric}.draw_tail"
    ][0]
    assert tail_check["stage"] == "advisory" and not tail_check["passed"]
    assert advisory["passed"]


def test_diagnostics_refuse_runs_of_a_different_estimator(dataset):
    spec = baseline_model_spec()
    other = HBIConfig(selection_chunk_size=None, raw_selection_use_observing_time=False)
    results = _fake_runs(dataset, spec, hbi=other)
    with pytest.raises(ValueError, match="raw_selection_use_observing_time"):
        summarize_dynesty_fit(
            results, dataset.posterior, dataset.selection, compile_model_spec(spec),
            hbi_config=HBI, criteria=_lenient(), n_draws=8,
        )
    unverifiable = [replace(result, provenance={}) for result in _fake_runs(dataset, spec)]
    with pytest.raises(ValueError, match="no likelihood identity"):
        summarize_dynesty_fit(
            unverifiable, dataset.posterior, dataset.selection, compile_model_spec(spec),
            hbi_config=HBI, criteria=_lenient(), n_draws=8,
        )


def test_no_finite_support_completes_as_a_typed_failure(monkeypatch, dataset, tmp_path):
    def no_support(*args, **kwargs):
        raise NoFiniteSupportError("dynesty found no hyperparameter with finite log-likelihood")

    monkeypatch.setattr("gwpop_search.inference.fidelity.run_model_evidence_repeats", no_support)
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior, dataset.selection, config=_small_config()
    )
    record = evaluator.evaluate(
        baseline_model_spec(), Fidelity.F4_PRODUCTION, seed=1, run_dir=tmp_path / "f4"
    )
    assert record.status == "complete" and not record.diagnostics_pass
    assert record.screen_value is None
    payload = read_evaluation(tmp_path / "f4" / "evaluation.json")
    assert payload["screen_value"] is None
    assert payload["diagnostics"]["failure"]["type"] == "no_finite_support"


def test_real_tiny_f3_runs_repeats_writes_artifacts_and_resumes(monkeypatch, dataset, tmp_path):
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior, dataset.selection, config=_small_config(), dataset_identity="tiny"
    )
    spec = baseline_model_spec()
    with pytest.warns(UserWarning):
        record = evaluator.evaluate(spec, Fidelity.F3_EVIDENCE, seed=5, run_dir=tmp_path / "f3")
    payload = read_evaluation(tmp_path / "f3" / "evaluation.json")
    diagnostics = payload["diagnostics"]
    # maxiter stops these runs before dlogz: the termination gate must say so.
    assert not record.diagnostics_pass
    assert "nested_sampling.all_runs_terminated_by_dlogz" in _names(
        diagnostics["checks"], failed_gates_only=True
    )
    assert [row["termination"] for row in diagnostics["nested_sampling"]["runs"]] == [
        "budget", "budget",
    ]
    assert np.isfinite(record.screen_value)
    for repeat in ("repeat_000", "repeat_001"):
        assert (tmp_path / "f3" / "evidence" / repeat / "result.npz").exists()
    pooled = load_pooled_posterior(tmp_path / "f3" / "pooled_posterior.npz")
    assert pooled["samples"].shape == (120, len(spec.priors))

    def refuse(*args, **kwargs):
        raise AssertionError("completed repeats must be reused")

    monkeypatch.setattr("gwpop_search.inference.dynesty_backend.run_dynesty", refuse)
    again = evaluator.evaluate(spec, Fidelity.F3_EVIDENCE, seed=5, run_dir=tmp_path / "f3")
    assert again.screen_value == record.screen_value
    assert again.diagnostics_pass == record.diagnostics_pass


def test_mixture_model_is_evaluated_on_canonical_labels(dataset, tmp_path):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1)
    (mixture,) = [node for node in graph.nodes if node.chieff.family == "gaussian_mixture"]
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior, dataset.selection, config=_small_config()
    )
    f0 = evaluator.evaluate(mixture, Fidelity.F0_SANITY, seed=2, run_dir=tmp_path / "f0")
    assert f0.diagnostics_pass
    f0_payload = read_evaluation(tmp_path / "f0" / "evaluation.json")
    assert f0_payload["diagnostics"]["parameterization"]["kind"] == "ordered_exchangeable_pairs"
    with pytest.warns(UserWarning):
        evaluator.evaluate(mixture, Fidelity.F3_EVIDENCE, seed=2, run_dir=tmp_path / "f3")
    payload = read_evaluation(tmp_path / "f3" / "evaluation.json")
    sampling = payload["diagnostics"]["nested_sampling"]
    assert sampling["parameterization"]["kind"] == "ordered_exchangeable_pairs"
    pooled = load_pooled_posterior(tmp_path / "f3" / "pooled_posterior.npz")
    names = pooled["parameter_names"].tolist()
    lower, upper = names.index("chi_mu_1"), names.index("chi_mu_2")
    assert np.all(pooled["samples"][:, lower] <= pooled["samples"][:, upper])
    median = payload["diagnostics"]["posterior_median"]
    assert median["chi_mu_1"] <= median["chi_mu_2"]
    manifest = json.loads(
        (tmp_path / "f3" / "evidence" / "repeat_000" / "manifest.json").read_text()
    )
    assert manifest["parameterization"]["kind"] == "ordered_exchangeable_pairs"
