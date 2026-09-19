"""Phase-3c evidence check: root equivalence, injections, Savage-Dickey, pass rule."""

import json
import math
import warnings

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import kstest, truncnorm

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

import gwpop_search.models as models_module  # noqa: E402
from gwpop_search.grammar import baseline_model_spec  # noqa: E402
from gwpop_search.inference.dynesty_backend import (  # noqa: E402
    DynestyConfig,
    prior_transform_for,
    run_dynesty,
)
from gwpop_search.inference.phase3_evidence import (  # noqa: E402
    ATOM_SLOPE_PARAMETERS,
    EVIDENCE_CHECK_ATOMS,
    EVIDENCE_CHECK_PLAN_NAME,
    EvidenceCheckPassRule,
    atom_slope_prior,
    default_injection_strengths,
    evaluate_evidence_check_rule,
    evidence_check_model_specs,
    generate_injected_catalog,
    root_equivalence_check,
    run_evidence_check,
    savage_dickey_ln_bf,
)
from gwpop_search.inference.phase3_ns import (  # noqa: E402
    NSRecoveryAcceptanceCriteria,
    default_phase3_dynesty_config,
    ns_run_fingerprints,
)
from gwpop_search.inference.priors import BASELINE_SYNTHETIC_PRIORS, PriorSpec  # noqa: E402
from gwpop_search.inference.synthetic import (  # noqa: E402
    SyntheticSurveyConfig,
    _detector_prior_bounds,
    generate_baseline_synthetic_dataset,
    observed_detection_mask,
)
from gwpop_search.models import (  # noqa: E402
    DEFAULT_BASELINE_HYPERPARAMETERS,
    GwcatChiEffBBHModel,
)
from gwpop_search.models.declarative import (  # noqa: E402
    chieff_logpdf_from_spec,
    pairing_logpdf_from_spec,
)
from gwpop_search.scouts.synthetic import (  # noqa: E402
    StructuredScoutInjection,
    _draw_structured_population,
)


def tiny_survey(**overrides):
    payload = dict(
        n_events=6,
        posterior_samples_per_event=32,
        n_injections=3_000,
        population_batch_size=512,
        redshift_sampling_grid=4096,
        injection_draw="population_proxy",
        observation_model="noisy_observation",
    )
    payload.update(overrides)
    return SyntheticSurveyConfig(**payload)


# ---------------------------------------------------------------------------
# Models and injection strengths
# ---------------------------------------------------------------------------


def test_models_are_the_registered_nested_one_slope_atoms():
    specs = evidence_check_model_specs()
    root = specs["root"]
    assert root == baseline_model_spec()
    assert list(specs) == ["root", *EVIDENCE_CHECK_ATOMS]
    for atom in EVIDENCE_CHECK_ATOMS:
        child = specs[atom]
        slope = ATOM_SLOPE_PARAMETERS[atom]
        assert child.model_hash != root.model_hash
        assert set(child.priors) == set(root.priors) | {slope}
        assert all(child.priors[name] == root.priors[name] for name in root.priors)
    assert atom_slope_prior("chieff.mean.linear_q") == PriorSpec("uniform", low=-0.6, high=0.6)
    assert atom_slope_prior("pairing.beta.linear_m1") == PriorSpec("uniform", low=-0.3, high=0.3)
    strengths = default_injection_strengths()
    assert strengths == {"chieff.mean.linear_q": 0.54, "pairing.beta.linear_m1": 0.27}
    with pytest.raises(ValueError, match="unsupported atom"):
        evidence_check_model_specs(["mass.family.powerlaw"])


# ---------------------------------------------------------------------------
# Root equivalence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v2_catalog():
    return generate_baseline_synthetic_dataset(seed=11, config=tiny_survey())


def test_declarative_root_equals_the_phase3_baseline_on_a_v2_catalog(v2_catalog):
    check = root_equivalence_check(v2_catalog.posterior, v2_catalog.selection, seed=3)
    assert check["passed"], check
    assert check["priors_equal"] and check["context_equal"]
    assert check["support_mismatches"] == 0 and check["nan_or_posinf_values"] == 0
    assert check["max_abs_difference"] <= 1e-10
    assert check["n_points"] == 17
    # Prior draws put support edges inside the samples: both finite and -inf values occur.
    assert 0 < check["n_finite"] < check["n_values"]


def test_root_equivalence_detects_a_different_model(v2_catalog, monkeypatch):
    monkeypatch.setattr(
        models_module, "GwcatChiEffBBHModel", lambda: GwcatChiEffBBHModel(q_floor=0.2)
    )
    check = root_equivalence_check(v2_catalog.posterior, v2_catalog.selection, seed=3)
    assert not check["passed"]
    assert not check["context_equal"] and check["support_mismatches"] > 0


# ---------------------------------------------------------------------------
# Injected catalogs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atom", EVIDENCE_CHECK_ATOMS)
def test_injected_catalog_is_dag_consistent(atom):
    survey = tiny_survey()
    strength = default_injection_strengths()[atom]
    dataset = generate_injected_catalog(
        atom=atom, strength=strength, data_seed=5, survey_config=survey
    )
    assert dataset.truth_hyperparameters[ATOM_SLOPE_PARAMETERS[atom]] == strength
    for name, value in DEFAULT_BASELINE_HYPERPARAMETERS.items():
        assert dataset.truth_hyperparameters[name] == value
    assert dataset.event_observations is not None
    assert dataset.posterior.metadata["observation_model"] == "noisy_observation"
    assert dataset.selection.metadata["injection_draw"] == "population_proxy"
    bounds = _detector_prior_bounds(GwcatChiEffBBHModel(), survey, dataset.truth_hyperparameters)
    assert np.all(observed_detection_mask(dataset.event_observations, survey, bounds))
    # PE is centred on the observed data (noise 0.12), not on the truth.
    pe_means = dataset.posterior.samples["chi_eff"].reshape(6, 32).mean(axis=1)
    to_observed = np.sqrt(np.mean((pe_means - dataset.event_observations["chi_eff"]) ** 2))
    to_truth = np.sqrt(np.mean((pe_means - dataset.event_truths["chi_eff"]) ** 2))
    assert to_observed < 0.5 * to_truth
    with pytest.raises(ValueError):
        generate_injected_catalog(atom=atom, strength=5.0, data_seed=5, survey_config=survey)


def test_structured_draws_follow_the_compiled_child_densities():
    """The injected population is exactly the declarative child model.

    The structured sampler's conditional draws are uniform under the analytic
    CDF (probability integral transform), and the compiled child's conditional
    log density equals the analytic density at every draw.
    """
    survey = tiny_survey()
    model = GwcatChiEffBBHModel()
    specs = evidence_check_model_specs()
    hp = dict(DEFAULT_BASELINE_HYPERPARAMETERS)
    rng = np.random.default_rng(9)

    atom = "chieff.mean.linear_q"
    s = 0.54
    injection = StructuredScoutInjection(atom, s)
    draw = _draw_structured_population(rng, 20_000, hp, model, survey, injection)
    mu = hp["chi_mu"] + s * (draw["q"] - 0.7)
    sigma = hp["chi_sigma"]
    a, b = (-1.0 - mu) / sigma, (1.0 - mu) / sigma
    pit = truncnorm.cdf(draw["chi_eff"], a, b, loc=mu, scale=sigma)
    assert kstest(pit, "uniform").pvalue > 1e-3
    child_hp = dict(hp, chi_mu_q_slope=s)
    compiled = np.asarray(
        chieff_logpdf_from_spec(
            specs[atom].chieff, draw["chi_eff"], draw["m1_source"], draw["q"], draw["z"], child_hp
        )
    )
    expected = truncnorm.logpdf(draw["chi_eff"], a, b, loc=mu, scale=sigma)
    np.testing.assert_allclose(compiled, expected, rtol=0.0, atol=1e-9)

    atom = "pairing.beta.linear_m1"
    s = 0.27
    injection = StructuredScoutInjection(atom, s)
    draw = _draw_structured_population(rng, 20_000, hp, model, survey, injection)
    m1 = draw["m1_source"]
    q = draw["q"]
    beta = hp["beta_q"] + s * (m1 - 30.0)
    qmin = np.maximum(model.q_floor, hp["mmin"] / m1)
    tail = np.power(qmin, beta + 1.0)
    pit = (np.power(q, beta + 1.0) - tail) / (1.0 - tail)
    assert kstest(pit, "uniform").pvalue > 1e-3
    compiled = np.asarray(
        pairing_logpdf_from_spec(specs[atom].pairing, q, m1, dict(hp, beta_q_m1_slope=s))
    )
    expected = beta * np.log(q) - np.log((1.0 - tail) / (beta + 1.0))  # beta + 1 may be < 0
    np.testing.assert_allclose(compiled, expected, rtol=0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Savage-Dickey
# ---------------------------------------------------------------------------


class SlopeLikelihood:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, X):
        x = np.asarray(X, dtype=float)[:, 0]
        norm = math.log(self.sigma * math.sqrt(2 * math.pi))
        return -0.5 * ((x - self.mu) / self.sigma) ** 2 - norm


def test_savage_dickey_reproduces_an_analytic_nested_bayes_factor():
    """Child: slope ~ U(-0.6, 0.6) with a Gaussian likelihood; root: slope = 0.

    ``ln BF(child/root) = ln Z_child - ln L(0)`` exactly; the SDDR and the NS
    evidence ratio must both reproduce it.
    """
    mu, sigma = 0.1, 0.15
    prior = PriorSpec("uniform", low=-0.6, high=0.6)
    names, transform = prior_transform_for({"slope": prior})
    likelihood = SlopeLikelihood(mu, sigma)
    config = DynestyConfig(nlive=400, bound="single", sample="unif", dlogz=0.01, batch_size=16)
    runs = [
        run_dynesty(likelihood, transform, 1, seed=seed, config=config, names=names)
        for seed in (1, 2)
    ]
    mass = ndtr((0.6 - mu) / sigma) - ndtr((-0.6 - mu) / sigma)
    ln_z_child = math.log(mass / 1.2)
    ln_l0 = float(likelihood(np.zeros((1, 1)))[0])
    exact = ln_z_child - ln_l0

    sddr = savage_dickey_ln_bf(runs, "slope", prior, n_bootstrap=30, seed=4)
    assert sddr["ln_prior_density"] == pytest.approx(-math.log(1.2))
    assert sddr["ln_bf"] == pytest.approx(exact, abs=0.06)
    assert sddr["bootstrap_std"] is not None and 0.0 < sddr["bootstrap_std"] < 0.1
    assert sddr["posterior_mean"] == pytest.approx(mu, abs=0.02)
    assert sddr["posterior_std"] == pytest.approx(sigma, rel=0.1)
    ns = np.mean([run.log_evidence for run in runs]) - ln_l0
    assert ns == pytest.approx(exact, abs=4 * max(run.log_evidence_error for run in runs))

    with pytest.raises(ValueError, match="strictly inside"):
        savage_dickey_ln_bf(runs, "slope", prior, null_value=0.6, seed=1)
    with pytest.raises(ValueError, match="uniform"):
        savage_dickey_ln_bf(runs, "slope", PriorSpec("normal", loc=0.0, scale=1.0), seed=1)
    with pytest.raises(ValueError, match="not a parameter"):
        savage_dickey_ln_bf(runs, "other", prior, seed=1)


# ---------------------------------------------------------------------------
# Pass rule
# ---------------------------------------------------------------------------


def make_case(kind, catalog, atom, ln_bf, *, sigma=0.3, sddr=None):
    return {
        "kind": kind,
        "catalog": catalog,
        "atom": atom,
        "ln_bf": ln_bf,
        "ln_bf_sigma": sigma,
        "sddr": {"ln_bf": ln_bf if sddr is None else sddr},
    }


def good_cases():
    null = [
        make_case("null", f"catalog_{i:03d}", atom, -0.8, sddr=-0.9)
        for i in range(4)
        for atom in EVIDENCE_CHECK_ATOMS
    ]
    injected = [
        make_case("injected", f"injected/{atom}", atom, 3.0, sddr=6.0)
        for atom in EVIDENCE_CHECK_ATOMS
    ]
    fits = [{"passed": True}] * 20
    return null, injected, fits


def failing(verdict):
    return [item["name"] for item in verdict["checks"] if not item["passed"]]


def test_pass_rule_accepts_the_expected_outcome():
    null, injected, fits = good_cases()
    verdict = evaluate_evidence_check_rule(null, injected, fits)
    assert verdict["passed"], failing(verdict)
    assert verdict["rule"]["format_version"] == EvidenceCheckPassRule.FORMAT_VERSION


def test_pass_rule_rejects_each_violation():
    null, injected, fits = good_cases()
    null[3]["ln_bf"] = 3.0  # must be strictly below 3
    null[3]["sddr"]["ln_bf"] = 3.0
    assert failing(evaluate_evidence_check_rule(null, injected, fits)) == [
        "null.catalog_001.pairing.beta.linear_m1.ln_bf"
    ]

    null, injected, fits = good_cases()
    injected[0]["ln_bf"] = 0.0  # must be strictly above 0
    assert failing(evaluate_evidence_check_rule(null, injected, fits)) == [
        "injected.chieff.mean.linear_q.ln_bf"
    ]

    null, injected, fits = good_cases()
    null[0]["sddr"]["ln_bf"] = null[0]["ln_bf"] + 0.61  # tolerance max(0.3, 2 x 0.3) = 0.6
    assert failing(evaluate_evidence_check_rule(null, injected, fits)) == [
        "null.catalog_000.chieff.mean.linear_q.sddr_agreement"
    ]
    null[0]["sddr"]["ln_bf"] = null[0]["ln_bf"] + 0.59
    assert evaluate_evidence_check_rule(null, injected, fits)["passed"]
    null[0]["ln_bf_sigma"] = 0.05  # tolerance falls back to the 0.3 floor
    assert failing(evaluate_evidence_check_rule(null, injected, fits)) == [
        "null.catalog_000.chieff.mean.linear_q.sddr_agreement"
    ]

    for mutate in (
        lambda case: case.update(ln_bf_sigma=None),
        lambda case: case.update(sddr={"ln_bf": None}),
        lambda case: case.update(ln_bf=float("nan")),
    ):
        null, injected, fits = good_cases()
        mutate(null[5])
        assert not evaluate_evidence_check_rule(null, injected, fits)["passed"]

    null, injected, fits = good_cases()
    fits = [*fits[:-1], {"passed": False}]
    assert failing(evaluate_evidence_check_rule(null, injected, fits)) == [
        "all_fits_numerically_valid"
    ]
    assert not evaluate_evidence_check_rule(null, injected, [])["passed"]

    null, injected, fits = good_cases()
    assert failing(evaluate_evidence_check_rule(null[:-1], injected, fits)) == ["n_null_cases"]
    assert failing(evaluate_evidence_check_rule(null, injected[:1], fits)) == ["n_injected_cases"]


def test_ln_bf_sigma_divides_by_the_repeat_count_and_takes_the_largest_term():
    """MODEL_COMPARISON_MATH.md 9.1(6): sigma_NS^2 / R, not the single-run error."""
    from gwpop_search.inference.phase3_evidence import ln_bf_sigma_ns
    from gwpop_search.inference.phase3_ns import ns_evidence_sigma

    evidence = {
        "n_repeats": 2,
        "repeat_std": 0.05,
        "mean_reported_error": 0.108,
        "max_reported_error": 0.15,
        "predicted_error_sqrt_h_over_nlive": [0.10, 0.107],
    }
    single = ns_evidence_sigma(evidence)
    assert single["sigma_ns"] == pytest.approx(0.108)  # the mean logzerr wins
    assert single["sigma_of_mean"] == pytest.approx(0.108 / math.sqrt(2))
    assert single["terms"]["kappa_hat_sqrt_h_over_nlive"] == pytest.approx(0.107)
    assert ns_evidence_sigma(evidence, kappa_hat=2.0)["sigma_ns"] == pytest.approx(0.214)

    budget = ln_bf_sigma_ns(evidence, evidence)
    assert budget["sigma"] == pytest.approx(0.108)  # sqrt(2 x 0.108^2 / 2)
    # The single-run formula of the task text is sqrt(R) = 1.41 times larger.
    task_text = math.hypot(*(max(0.05, 0.15),) * 2)
    assert task_text / budget["sigma"] == pytest.approx(0.15 * math.sqrt(2) / 0.108, rel=1e-9)

    lonely = ns_evidence_sigma({**evidence, "n_repeats": 1, "repeat_std": None})
    assert lonely["sigma_ns"] == pytest.approx(0.108) == lonely["sigma_of_mean"]
    assert ln_bf_sigma_ns({**evidence, "n_repeats": 1}, evidence)["sigma"] == pytest.approx(
        math.sqrt(0.108**2 + 0.108**2 / 2)
    )
    with pytest.raises(ValueError):
        ns_evidence_sigma(evidence, kappa_hat=0.0)


def test_sddr_tolerance_includes_the_savage_dickey_bootstrap_error():
    """MODEL_COMPARISON_MATH.md 9.1(8) adds sigma_SDDR^2 under the square root."""
    null, injected, fits = good_cases()
    # sigma = 0.3, difference 0.61: outside 2 x 0.3 = 0.6 without sigma_SDDR.
    null[0]["sddr"] = {"ln_bf": null[0]["ln_bf"] + 0.61}
    name = "null.catalog_000.chieff.mean.linear_q.sddr_agreement"
    verdict = evaluate_evidence_check_rule(null, injected, fits)
    record = next(item for item in verdict["checks"] if item["name"] == name)
    assert record["passed"] is False
    assert record["combined_sigma"] == pytest.approx(0.3)
    assert record["note"] == "SDDR bootstrap std unavailable; tolerance omits sigma_SDDR"

    # 2 sqrt(0.3^2 + 0.2^2) = 0.721 > 0.61: the same difference now passes.
    null[0]["sddr"]["bootstrap_std"] = 0.2
    verdict = evaluate_evidence_check_rule(null, injected, fits)
    record = next(item for item in verdict["checks"] if item["name"] == name)
    assert record["passed"] is True
    assert record["combined_sigma"] == pytest.approx(math.hypot(0.3, 0.2))
    assert record["limit"] == pytest.approx(2.0 * math.hypot(0.3, 0.2))
    assert "note" not in record
    assert verdict["passed"]


def test_expected_case_counts_follow_the_plan_not_a_hard_wired_eight():
    rule = EvidenceCheckPassRule.for_plan(n_catalogs=2, n_atoms=len(EVIDENCE_CHECK_ATOMS))
    assert (rule.expected_null_cases, rule.expected_injected_cases) == (4, 2)
    assert EvidenceCheckPassRule.for_plan(n_catalogs=4, n_atoms=2) == EvidenceCheckPassRule()

    null = [
        make_case("null", f"catalog_{i:03d}", atom, -0.8, sddr=-0.9)
        for i in range(2)
        for atom in EVIDENCE_CHECK_ATOMS
    ]
    injected = [
        make_case("injected", f"injected/{atom}", atom, 3.0, sddr=6.0)
        for atom in EVIDENCE_CHECK_ATOMS
    ]
    fits = [{"passed": True}] * 12
    # The default rule fails a perfect two-catalog outcome on the count alone.
    assert failing(evaluate_evidence_check_rule(null, injected, fits)) == ["n_null_cases"]
    assert evaluate_evidence_check_rule(null, injected, fits, rule)["passed"]


def test_pass_rule_round_trips_and_validates():
    rule = EvidenceCheckPassRule()
    assert rule.fit_criteria == NSRecoveryAcceptanceCriteria.f3_level()
    payload = json.loads(json.dumps(rule.to_dict()))
    assert payload["fit_criteria"]["max_r_hat"] == 1.05
    assert EvidenceCheckPassRule.from_dict(payload) == rule
    with pytest.raises(ValueError, match="format"):
        EvidenceCheckPassRule.from_dict({**payload, "format_version": "x"})
    with pytest.raises(ValueError):
        EvidenceCheckPassRule(sddr_tolerance_floor=0.0)
    with pytest.raises(TypeError):
        EvidenceCheckPassRule(fit_criteria={"max_r_hat": 1.05})


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def run_tiny_check(root, **overrides):
    kwargs = dict(
        n_catalogs=1,
        repeats=2,
        survey_config=tiny_survey(),
        dynesty_config=default_phase3_dynesty_config(
            nlive=30, batch_size=8, maxiter=100, dlogz=0.5, num_posterior_samples=200, slices=None
        ),
        # No explicit rule: the expected case counts must follow the plan.
        importance_draws=32,
        rhat_draws_per_run=100,
        sddr_bootstrap=10,
    )
    kwargs.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_evidence_check(root, **kwargs)


def test_tiny_evidence_check_end_to_end_and_resume(tmp_path):
    root = tmp_path / "evidence"
    summary = run_tiny_check(root)
    assert summary["format_version"] == "gwpop-search-phase3-evidence-check-1.0"
    plan = json.loads((root / EVIDENCE_CHECK_PLAN_NAME).read_text())
    assert plan["format_version"] == "gwpop-search-phase3-evidence-check-plan-1.0"
    assert plan["slices_rule"] == "2*(3+ndim)" and plan["dynesty_config_base"]["slices"] is None
    assert plan["models"]["root"]["dynesty_config"]["slices"] == 26
    for atom in EVIDENCE_CHECK_ATOMS:
        assert plan["models"][atom]["dynesty_config"]["slices"] == 28
        assert plan["models"][atom]["ndim"] == len(BASELINE_SYNTHETIC_PRIORS) + 1
    assert plan["catalogs"][0]["data_seed"] == 1685370682
    assert [item["strength"] for item in plan["injected"]] == [0.54, 0.27]

    assert [item["catalog"] for item in summary["root_equivalence"]] == [
        "catalog_000",
        "injected/chieff.mean.linear_q",
        "injected/pairing.beta.linear_m1",
    ]
    assert all(item["passed"] for item in summary["root_equivalence"])
    assert len(summary["null_cases"]) == 2 and len(summary["injected_cases"]) == 2
    assert len(summary["fit_assessments"]) == 3 + 4

    for case in summary["null_cases"] + summary["injected_cases"]:
        where = "null/catalog_000" if case["kind"] == "null" else f"injected/{case['atom']}"
        prefix = root / where
        root_fit = json.loads((prefix / "root" / "fit_summary.json").read_text())
        child_fit = json.loads((prefix / case["atom"] / "fit_summary.json").read_text())
        assert root_fit["model_hash"] == case["root_model_hash"]
        root_ev, child_ev = root_fit["fit"]["evidence"], child_fit["fit"]["evidence"]
        assert case["ln_bf"] == pytest.approx(child_ev["mean"] - root_ev["mean"])
        # MODEL_COMPARISON_MATH.md 9.1(6): the error of the MEAN of R runs.
        expected_sigma = math.sqrt(
            sum(
                max(
                    evidence["repeat_std"],
                    evidence["mean_reported_error"],
                    max(evidence["predicted_error_sqrt_h_over_nlive"]),
                )
                ** 2
                / evidence["n_repeats"]
                for evidence in (child_ev, root_ev)
            )
        )
        assert case["ln_bf_sigma"] == pytest.approx(expected_sigma)
        assert case["ln_bf_sigma_budget"]["includes_sigma_mc"] is False
        assert case["ln_bf_sigma_budget"]["child"]["kappa_hat"] == 1.0
        # The literal task-text formula is recorded but not used: with R = 2 it
        # is about sqrt(2) times larger.
        assert case["ln_bf_sigma_task_text"] == pytest.approx(
            math.hypot(child_ev["conservative_error"], root_ev["conservative_error"])
        )
        assert case["ln_bf_sigma"] < case["ln_bf_sigma_task_text"]
        assert case["sddr"]["parameter"] == ATOM_SLOPE_PARAMETERS[case["atom"]]
        assert math.isfinite(case["sddr"]["ln_bf"])
        assert child_fit["dynesty_config"]["slices"] == 28
        # Root and child fits of a case see the same catalog.
        assert root_fit["fit"]["data_identity"] == child_fit["fit"]["data_identity"]
        assert case["data_identity"] == root_fit["fit"]["data_identity"]
    for case in summary["injected_cases"]:
        assert case["injected_slope"] == default_injection_strengths()[case["atom"]]
        assert {"q05", "median", "q95", "truth_in_90pct_interval"} <= set(case["slope_posterior"])
    names = [item["name"] for item in summary["pass_rule"]["checks"]]
    assert names[:3] == ["all_fits_numerically_valid", "n_null_cases", "n_injected_cases"]
    # One catalog, two atoms: the counts came from the plan, not from a
    # hard-wired eight, so the count checks pass.
    assert summary["rule"]["expected_null_cases"] == 2
    assert summary["rule"]["expected_injected_cases"] == 2
    assert all(
        item["passed"] for item in summary["pass_rule"]["checks"][1:3]
    ), summary["pass_rule"]["checks"][1:3]
    assert isinstance(summary["evidence_check_passed"], bool)
    # 100-iteration test runs cannot pass the F3 gates.
    assert not summary["evidence_check_passed"]

    before = ns_run_fingerprints(root)
    assert "evidence_check_plan.json" in before
    assert "null/catalog_000/root/repeat_000/result.npz" in before
    again = run_tiny_check(root)
    assert ns_run_fingerprints(root) == before
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(summary))

    with pytest.raises(ValueError, match="refusing to resume"):
        run_tiny_check(
            root,
            injection_strengths={"chieff.mean.linear_q": 0.5, "pairing.beta.linear_m1": 0.27},
        )
