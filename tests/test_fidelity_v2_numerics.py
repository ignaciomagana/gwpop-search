"""Per-check binding flags, the taper-mass diagnostic in fidelity, and the v2 numerics."""
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from test_fidelity_evaluator import (  # noqa: E402
    _compiled,
    _fake_runs,
    _lenient,
    _names,
    _small_config,
    dataset,  # noqa: F401  (module fixture)
)

from gwpop_search.grammar import baseline_model_spec  # noqa: E402
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.hbi.taper import VarianceTaper  # noqa: E402
from gwpop_search.inference import build_importance_diagnostics, prior_specs_from_model_spec  # noqa: E402
from gwpop_search.inference.fidelity import (  # noqa: E402
    CHECK_FAMILIES,
    DeterministicHBIEvaluator,
    FidelityRunConfig,
    NumericalCriteria,
    apply_binding,
    check_family,
    fidelity_config_sha256,
    fidelity_run_config_from_dict,
    fidelity_run_config_to_dict,
    load_fidelity_run_config,
    read_evaluation,
    summarize_dynesty_fit,
)
from gwpop_search.inference.v2_numerics import (  # noqa: E402
    V2_NON_BINDING_FAMILIES,
    PilotSeedScatterRule,
    SecondSeedRule,
    SeedPolicy,
    v2_campaign_numerics,
    v2_fidelity_run_config,
    v2_variance_taper,
    write_v2_draft_configs,
)
from gwpop_search.models import compile_model_spec  # noqa: E402
from gwpop_search.search.scheduler import Fidelity  # noqa: E402

HBI = HBIConfig(selection_chunk_size=None)
TAPERED = HBIConfig(selection_chunk_size=None, variance_taper=VarianceTaper(threshold=1.0))


# ---------------------------------------------------------------------------
# Binding flags
# ---------------------------------------------------------------------------


def test_binding_flags_validate_and_serialize_backward_compatibly():
    plain = NumericalCriteria()
    payload = plain.to_dict()
    # Pre-binding configurations keep their exact serialization (and hash).
    assert "binding" not in payload and "max_posterior_taper_mass" not in payload
    assert NumericalCriteria.from_dict(payload) == plain
    flagged = NumericalCriteria(
        binding={"importance.selection_ess": False, "evidence.repeat_std": True},
        max_posterior_taper_mass=0.5,
    )
    round_trip = NumericalCriteria.from_dict(json.loads(json.dumps(flagged.to_dict())))
    assert round_trip == flagged
    assert not flagged.is_binding("importance.selection_ess.draw_tail")
    assert flagged.is_binding("importance.min_event_ess.point")
    assert flagged.is_binding("evidence.repeat_std")
    with pytest.raises(ValueError, match="unknown check families"):
        NumericalCriteria(binding={"importance.ess": False})
    with pytest.raises(TypeError, match="bool"):
        NumericalCriteria(binding={"evidence.repeat_std": 0})
    with pytest.raises(TypeError, match="mapping"):
        NumericalCriteria(binding=["evidence.repeat_std"])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NumericalCriteria(max_posterior_taper_mass=1.5)
    assert check_family("importance.selection_ess.point") == "importance.selection_ess"
    assert check_family("evidence.max_pairwise_z") == "evidence.max_pairwise_z"
    assert len(set(CHECK_FAMILIES)) == len(CHECK_FAMILIES)


def test_apply_binding_turns_non_binding_gates_into_advisories():
    criteria = NumericalCriteria(binding={"importance.selection_ess": False})
    checks = [
        {"name": "importance.selection_ess.point", "stage": "gate", "passed": False},
        {"name": "importance.selection_ess.draw_tail", "stage": "advisory", "passed": False},
        {"name": "evidence.conservative_error", "stage": "gate", "passed": True},
    ]
    out = apply_binding(checks, criteria)
    assert [(c["stage"], c["binding"]) for c in out] == [
        ("advisory", False),
        ("advisory", False),
        ("gate", True),
    ]
    assert checks[0]["stage"] == "gate"  # inputs are not mutated


def test_non_binding_checks_are_reported_but_do_not_fail_the_fit(dataset):  # noqa: F811
    spec = baseline_model_spec()
    model = compile_model_spec(spec)
    priors = prior_specs_from_model_spec(spec)
    results = _fake_runs(dataset, spec, log_evidences=[-10.0, -10.05])
    diagnostics_fn = build_importance_diagnostics(
        dataset.posterior, dataset.selection, model, results[0].names, hbi_config=HBI
    )

    def summarize(criteria):
        return summarize_dynesty_fit(
            results, dataset.posterior, dataset.selection, model, hbi_config=HBI,
            criteria=criteria, priors=priors, n_draws=16, rhat_draws_per_run=40,
            diagnostics_fn=diagnostics_fn,
        )

    strict = dict(min_kish_ess_per_run=1e6, max_shape_log_likelihood_variance=1e-9)
    binding = summarize(_lenient(**strict))
    failed = _names(binding["checks"], failed_gates_only=True)
    assert "nested_sampling.min_kish_ess_per_run" in failed
    assert "importance.shape_log_likelihood_variance.point" in failed
    assert not binding["passed"]
    assert all(item["binding"] for item in binding["checks"] if item["stage"] == "gate")
    assert "taper" not in binding  # untapered likelihood: no taper block

    relaxed = summarize(
        _lenient(
            **strict,
            binding={
                "nested_sampling.min_kish_ess_per_run": False,
                "importance.shape_log_likelihood_variance": False,
            },
        )
    )
    assert relaxed["passed"]
    assert not _names(relaxed["checks"], failed_gates_only=True)
    by_name = {item["name"]: item for item in relaxed["checks"]}
    for name in (
        "nested_sampling.min_kish_ess_per_run",
        "importance.shape_log_likelihood_variance.point",
        "importance.shape_log_likelihood_variance.draw_tail",
    ):
        assert by_name[name]["stage"] == "advisory"
        assert by_name[name]["binding"] is False
        assert by_name[name]["passed"] is False  # still reported as failing
    assert by_name["evidence.conservative_error"]["binding"] is True


# ---------------------------------------------------------------------------
# Taper-mass diagnostic in the fidelity summary
# ---------------------------------------------------------------------------


def test_taper_mass_is_reported_and_optionally_gated(dataset):  # noqa: F811
    spec = baseline_model_spec()
    model = compile_model_spec(spec)
    results = _fake_runs(dataset, spec, log_evidences=[-10.0, -10.05], hbi=TAPERED)
    loglike, identity = _compiled(dataset, spec, TAPERED)
    assert identity["hbi_config"]["variance_taper"]["threshold"] == 1.0
    diagnostics_fn = build_importance_diagnostics(
        dataset.posterior, dataset.selection, model, results[0].names, hbi_config=TAPERED
    )

    def summarize(criteria, hbi=TAPERED):
        return summarize_dynesty_fit(
            results, dataset.posterior, dataset.selection, model, hbi_config=hbi,
            criteria=criteria, n_draws=16, rhat_draws_per_run=40,
            diagnostics_fn=diagnostics_fn, taper_loglike=loglike,
        )

    reported = summarize(_lenient())
    taper = reported["taper"]
    assert taper["taper"] == TAPERED.variance_taper.to_dict()
    assert len(taper["runs"]) == 2
    mass = taper["pooled"]["posterior_mass_in_taper_region"]
    assert 0.0 <= mass <= 1.0
    assert taper["max_relative_log_likelihood_mismatch"] <= 1e-9
    assert "taper.posterior_mass_in_taper_region" not in _names(reported["checks"])
    # The importance log-likelihood is the tapered one the runs sampled.
    point = reported["importance"]["at_posterior_median"]
    assert np.isfinite(point["log_likelihood"])

    gated = summarize(_lenient(max_posterior_taper_mass=1.0))
    check = [c for c in gated["checks"] if c["name"] == "taper.posterior_mass_in_taper_region"]
    assert check and check[0]["passed"] and check[0]["binding"]
    if mass > 0.0:
        failing = summarize(_lenient(max_posterior_taper_mass=0.0))
        assert "taper.posterior_mass_in_taper_region" in _names(
            failing["checks"], failed_gates_only=True
        )
        advisory = summarize(
            _lenient(
                max_posterior_taper_mass=0.0,
                binding={"taper.posterior_mass_in_taper_region": False},
            )
        )
        assert advisory["passed"]
    with pytest.raises(ValueError, match="no variance taper"):
        summarize_dynesty_fit(
            _fake_runs(dataset, spec, log_evidences=[-10.0, -10.05]),
            dataset.posterior, dataset.selection, model, hbi_config=HBI,
            criteria=_lenient(max_posterior_taper_mass=0.5), n_draws=8,
        )
    # Untapered runs cannot be diagnosed as tapered ones.
    with pytest.raises(ValueError, match="variance_taper"):
        summarize_dynesty_fit(
            _fake_runs(dataset, spec, log_evidences=[-10.0, -10.05]),
            dataset.posterior, dataset.selection, model, hbi_config=TAPERED,
            criteria=_lenient(), n_draws=8,
        )


def test_f0_parity_holds_for_the_tapered_likelihood(dataset, tmp_path):  # noqa: F811
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior,
        dataset.selection,
        config=_small_config(hbi=TAPERED),
        dataset_identity="synthetic-test",
    )
    record = evaluator.evaluate(
        baseline_model_spec(), Fidelity.F0_SANITY, seed=123, run_dir=tmp_path / "f0"
    )
    assert record.status == "complete" and record.diagnostics_pass
    diagnostics = read_evaluation(tmp_path / "f0" / "evaluation.json")["diagnostics"]
    assert diagnostics["parity"]["n_points_compared"] == 2
    assert diagnostics["parity"]["max_rel_diff"] <= 1e-9
    assert diagnostics["support"]["diagnostics_likelihood_max_rel_diff"] <= 1e-9
    # Sharp cut: the support gate is the untapered support; parity also covers sigma^2
    # and the cut itself (thinning inflates sigma^2, so the thinned points are all cut).
    taper = diagnostics["support"]["taper"]
    assert taper["taper"]["kind"] == "sharp"
    assert taper["n_finite_tapered"] <= diagnostics["support"]["n_finite"]
    for point in diagnostics["parity"]["points"]:
        assert point["variance_rel_diff"] <= 1e-9 and point["tapered_rel_diff"] == 0.0


def _f0_with_taper_gate(dataset, tmp_path, taper, *, min_kept, max_draws, name):  # noqa: F811
    from gwpop_search.inference.fidelity import F0SanityConfig

    f0 = F0SanityConfig(
        prior_draws=128, batch_size=16, parity_pe_samples_per_event=16,
        parity_selected_per_campaign=128, min_taper_finite_fraction=min_kept,
        taper_support_max_draws=max_draws,
    )
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior,
        dataset.selection,
        config=_small_config(hbi=HBIConfig(selection_chunk_size=None, variance_taper=taper), f0=f0),
        dataset_identity="synthetic-test",
    )
    record = evaluator.evaluate(baseline_model_spec(), Fidelity.F0_SANITY, seed=123, run_dir=tmp_path / name)
    return record, read_evaluation(tmp_path / name / "evaluation.json")["diagnostics"]


def test_f0_gates_the_fraction_of_the_prior_a_sharp_cut_keeps(dataset, tmp_path):  # noqa: F811
    from gwpop_search.inference.fidelity import F0_TAPER_SUPPORT_TARGET_FINITE, F0SanityConfig

    # a cut no prior draw passes: population support is fine, but dynesty could
    # never start -> the scan extends to its maximum and the gate fails
    record, diag = _f0_with_taper_gate(
        dataset, tmp_path, VarianceTaper(kind="sharp", threshold=1e-12),
        min_kept=1e-3, max_draws=512, name="cut_all",
    )
    scan = diag["support"]["taper"]["kept_fraction_scan"]
    assert scan["n_kept"] == 0 and scan["n_draws"] == 512 and scan["extension_rounds"] == 3
    assert diag["support"]["finite_fraction"] > 0.0  # untapered support is unaffected
    gate = [c for c in diag["checks"] if c["name"] == "f0.taper_kept_fraction"]
    assert gate and not gate[0]["passed"] and not record.diagnostics_pass
    # a generous cut: enough kept draws in the first scan, no extension, gate passes
    record, diag = _f0_with_taper_gate(
        dataset, tmp_path, VarianceTaper(kind="sharp", threshold=1e6),
        min_kept=1e-3, max_draws=512, name="keep_all",
    )
    scan = diag["support"]["taper"]["kept_fraction_scan"]
    assert scan["n_kept"] >= F0_TAPER_SUPPORT_TARGET_FINITE and scan["extension_rounds"] == 0
    assert scan["n_draws"] == 128
    assert [c for c in diag["checks"] if c["name"] == "f0.taper_kept_fraction"][0]["passed"]
    # the gate is off unless configured, and its fields stay out of pre-v2 hashes
    assert "min_taper_finite_fraction" not in F0SanityConfig().to_dict()
    with pytest.raises(ValueError, match="together"):
        F0SanityConfig(min_taper_finite_fraction=1e-3)
    with pytest.raises(ValueError, match=">= prior_draws"):
        F0SanityConfig(prior_draws=128, min_taper_finite_fraction=1e-3, taper_support_max_draws=64)
    v2 = v2_fidelity_run_config().f0
    assert v2.min_taper_finite_fraction == 4.0e-4 and v2.taper_support_max_draws == 65536


# ---------------------------------------------------------------------------
# v2 seed rules and fidelity configuration
# ---------------------------------------------------------------------------


def test_second_seed_rule_is_the_two_sigma_window_around_plus_minus_three():
    rule = SecondSeedRule()
    assert rule.requires_second_seed(3.0, 0.1)
    assert rule.requires_second_seed(-3.0, 0.1)
    assert rule.requires_second_seed(3.99, 0.5)  # |3.99 - 3| = 0.99 <= 2 * 0.5
    assert not rule.requires_second_seed(4.01, 0.5)
    assert rule.requires_second_seed(-1.81, 0.6)  # |1.81 - 3| = 1.19 <= 1.2
    assert not rule.requires_second_seed(-1.7, 0.6)
    assert not rule.requires_second_seed(0.0, 0.7)
    assert not rule.requires_second_seed(13.4, 0.7)
    assert rule.requires_second_seed(13.4, 0.7, claimed=True)
    assessed = rule.assess(2.5, 0.4)
    assert assessed["second_seed"] and assessed["reason"].startswith("|ln BF| within")
    np.testing.assert_allclose(assessed["distance_to_threshold"], 0.5)
    with pytest.raises(ValueError):
        rule.assess(np.nan, 0.1)
    with pytest.raises(ValueError):
        rule.assess(1.0, -0.1)


def test_pilot_seed_scatter_rule_falls_back_above_one_and_a_half_times_dynesty():
    rule = PilotSeedScatterRule()
    lnz = np.array([-100.0, -100.2, -99.8])  # std 0.2
    calm = rule.assess(lnz, [0.2, 0.2, 0.2])
    np.testing.assert_allclose(calm["measured_scatter"], 0.2)
    np.testing.assert_allclose(calm["ratio"], 1.0)
    assert not calm["fallback_to_two_seeds_everywhere"] and calm["seeds_per_model"] == 1
    edge = rule.assess(lnz, [0.2 / 1.49] * 3)
    assert not edge["fallback_to_two_seeds_everywhere"]  # 1.49x: within the rule
    assert rule.assess(lnz, [0.2 / 1.51] * 3)["fallback_to_two_seeds_everywhere"]
    noisy = rule.assess(lnz, [0.12] * 3)
    assert noisy["fallback_to_two_seeds_everywhere"] and noisy["seeds_per_model"] == 2
    with pytest.raises(ValueError, match="3 seeds"):
        rule.assess([-1.0, -1.1], [0.1, 0.1])
    with pytest.raises(ValueError):
        rule.assess(lnz, [0.2, 0.0, 0.2])
    policy = SeedPolicy()
    assert policy.seeds_for_edge(0.5, 0.6) == 1
    assert policy.seeds_for_edge(2.5, 0.6) == 2
    assert policy.seeds_for_edge(0.5, 0.6, pilot_fallback=True) == 2
    assert policy.seeds_for_edge(9.0, 0.6, claimed=True) == 2


def test_v2_fidelity_config_is_valid_and_round_trips(tmp_path):
    config = v2_fidelity_run_config()
    assert isinstance(config, FidelityRunConfig)
    f3, f4 = config.f3_evidence, config.f4_evidence
    assert (f3.repeats, f3.dynesty.nlive, f3.dynesty.dlogz) == (1, 500, 0.1)
    assert (f4.repeats, f4.dynesty.nlive, f4.dynesty.dlogz) == (2, 500, 0.1)
    for rung in (f3, f4):
        assert rung.dynesty.sample == "rslice" and rung.dynesty.bound == "multi"
    assert config.hbi.variance_taper == VarianceTaper(kind="sharp", threshold=1.0)  # the LVK cut
    assert config.hbi.variance_taper.region_onset == pytest.approx(0.95)
    assert v2_variance_taper(kind="smooth") == VarianceTaper(kind="smooth", threshold=1.0, exponent=30.0)
    assert config.f3_criteria.max_cross_run_r_hat is None
    assert config.f4_criteria.max_repeat_consistency_z == 3.0
    for criteria in (config.f3_criteria, config.f4_criteria):
        for family in V2_NON_BINDING_FAMILIES:
            assert not criteria.is_binding(family + ".point")
        assert criteria.is_binding("evidence.conservative_error")
        assert criteria.is_binding("nested_sampling.all_runs_terminated_by_dlogz")
    payload = json.loads(json.dumps(fidelity_run_config_to_dict(config)))
    assert payload["hbi"]["variance_taper"]["threshold"] == 1.0
    assert fidelity_run_config_from_dict(payload) == config
    sensitivity = v2_fidelity_run_config(2.0)
    assert sensitivity.hbi.variance_taper.threshold == 2.0
    assert fidelity_config_sha256(sensitivity) != fidelity_config_sha256(config)

    files = write_v2_draft_configs(tmp_path)
    assert load_fidelity_run_config(tmp_path / files["primary"]) == config
    assert load_fidelity_run_config(tmp_path / files["taper_sensitivity"]) == sensitivity
    campaign = json.loads((tmp_path / files["campaign"]).read_text())
    assert campaign["status"].startswith("DRAFT")
    assert campaign["fidelity"]["primary"]["sha256"] == fidelity_config_sha256(config)
    assert campaign["seed_policy"]["second_seed"]["n_sigma"] == 2.0
    assert campaign["seed_policy"]["pilot"]["max_scatter_ratio"] == 1.5
    assert campaign == json.loads(json.dumps(v2_campaign_numerics(fidelity_files={
        "primary": files["primary"], "taper_sensitivity": files["taper_sensitivity"],
    })))
