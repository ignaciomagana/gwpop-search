"""D2 cut bracketing: P(sigma^2 <= c) recorded by the evaluator and recomputed post hoc.

Under a sharp variance cut at c the evidence at c' < c is Z(c') = Z(c) P_post(sigma^2 <= c').
"""
import json
import math
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from test_fidelity_evaluator import (  # noqa: E402
    _compiled,
    _fake_runs,
    _lenient,
    _small_config,
    dataset,  # noqa: F401  (module fixture)
)

from gwpop_search.analysis._common import AnalysisInputError  # noqa: E402
from gwpop_search.analysis.claims_v2 import collect_v2_evaluations  # noqa: E402
from gwpop_search.analysis.posthoc_cut import posthoc_mass_below, posthoc_report  # noqa: E402
from gwpop_search.grammar import baseline_model_spec  # noqa: E402
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.hbi.taper import (  # noqa: E402
    VarianceTaper,
    cut_key,
    posterior_mass_below,
    taper_region_summary,
)
from gwpop_search.inference import build_importance_diagnostics  # noqa: E402
from gwpop_search.inference.dynesty_backend import posterior_taper_mass  # noqa: E402
from gwpop_search.inference.fidelity import (  # noqa: E402
    DeterministicHBIEvaluator,
    NumericalCriteria,
    fidelity_run_config_from_dict,
    fidelity_run_config_to_dict,
    read_evaluation,
    summarize_dynesty_fit,
)
from gwpop_search.inference.v2_numerics import (  # noqa: E402
    V2_POSTERIOR_MASS_BELOW_CUTS,
    v2_campaign_numerics,
    v2_fidelity_run_config,
)
from gwpop_search.models import compile_model_spec  # noqa: E402
from gwpop_search.search.scheduler import Fidelity  # noqa: E402

SHARP = VarianceTaper(kind="sharp", threshold=1.0)


def test_posterior_mass_below_fractions_and_kish_binomial_errors():
    v = np.array([0.1, 0.5, 0.85, 0.92, 0.97, 1.0, np.nan])
    w = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 0.0])
    out = posterior_mass_below(v, w, [1.0, 0.9])
    assert list(out) == ["0.9", "1.0"] and cut_key(0.9) == "0.9"
    wn = w / w.sum()
    n_eff = 1.0 / np.sum(wn**2)
    p = 4.0 / 10.0
    assert out["0.9"]["fraction"] == pytest.approx(p)
    assert out["0.9"]["n_eff"] == pytest.approx(n_eff)
    assert out["0.9"]["error"] == pytest.approx(math.sqrt(p * (1 - p) / n_eff))
    # sigma^2 = 1 is kept by the sharp cut at 1; NaN counts as cut
    assert out["1.0"]["fraction"] == pytest.approx(1.0) and out["1.0"]["error"] == 0.0
    # a resampled subset: the two sampling stages add in 1/n_eff
    sub = posterior_mass_below(v, w, [0.9], reference_kish_ess=100.0)
    assert sub["0.9"]["n_eff"] == pytest.approx(1.0 / (1.0 / n_eff + 1.0 / 100.0))
    summary = taper_region_summary(v, w, SHARP, cuts=(0.9,))
    assert summary["posterior_mass_below"] == posterior_mass_below(v, w, (0.9,))
    assert "posterior_mass_below" not in taper_region_summary(v, w, SHARP)
    with pytest.raises(ValueError, match="positive"):
        posterior_mass_below(v, w, [0.0])


def test_sharp_cut_identity_on_weighted_prior_samples():
    """Z(c') = Z(c) P(sigma^2 <= c' | cut c), checked on prior-sampled toy evidences."""
    rng = np.random.default_rng(3)
    n = 200_000
    variance = rng.gamma(4.0, 0.25, size=n)
    logl = -0.5 * (rng.normal(size=n)) ** 2 - variance
    def log_z(cut):
        return logsumexp(np.where(variance <= cut, logl, -np.inf)) - math.log(n)
    w = np.exp(np.where(variance <= 1.0, logl, -np.inf) - log_z(1.0) - math.log(n))
    frac = posterior_mass_below(variance, w, [0.9])["0.9"]["fraction"]
    assert log_z(0.9) == pytest.approx(log_z(1.0) + math.log(frac), abs=1e-12)


def test_criteria_serialize_cuts_and_keep_old_hashes():
    plain = NumericalCriteria()
    assert "posterior_mass_below_cuts" not in plain.to_dict()
    with_cuts = NumericalCriteria(posterior_mass_below_cuts=[1.0, 0.9, 0.9])
    assert with_cuts.posterior_mass_below_cuts == (0.9, 1.0)
    payload = json.loads(json.dumps(with_cuts.to_dict()))
    assert payload["posterior_mass_below_cuts"] == [0.9, 1.0]
    assert NumericalCriteria.from_dict(payload) == with_cuts
    with pytest.raises(TypeError):
        NumericalCriteria(posterior_mass_below_cuts=0.9)
    with pytest.raises(ValueError):
        NumericalCriteria(posterior_mass_below_cuts=[-1.0])
    # v2: the pre-declared 0.9 plus the primary cut 1, at both rungs and in the cut-4 config
    assert V2_POSTERIOR_MASS_BELOW_CUTS == (0.9,)
    for threshold in (1.0, 4.0):
        config = v2_fidelity_run_config(threshold)
        assert config.f3_criteria.posterior_mass_below_cuts == (0.9, 1.0)
        assert config.f4_criteria.posterior_mass_below_cuts == (0.9, 1.0)
        assert fidelity_run_config_from_dict(fidelity_run_config_to_dict(config)) == config
    campaign = v2_campaign_numerics()
    assert campaign["likelihood"]["tighter_cuts_reported"] == [0.9, 1.0]
    assert "uses 2" not in json.dumps(campaign) and "cut-at-2" not in json.dumps(campaign)


def test_summarize_records_posterior_mass_below_at_the_cuts_and_threshold(dataset):  # noqa: F811
    spec = baseline_model_spec()
    model = compile_model_spec(spec)
    hbi = HBIConfig(selection_chunk_size=None, variance_taper=VarianceTaper(kind="sharp", threshold=50.0))
    results = _fake_runs(dataset, spec, log_evidences=[-10.0, -10.05], hbi=hbi)
    loglike, _ = _compiled(dataset, spec, hbi)
    diagnostics_fn = build_importance_diagnostics(
        dataset.posterior, dataset.selection, model, results[0].names, hbi_config=hbi
    )
    comps = [loglike.components(r.samples)["variance"] for r in results]
    cut = float(np.median(np.concatenate(comps)[np.isfinite(np.concatenate(comps))]))

    def summarize(criteria):
        return summarize_dynesty_fit(
            results, dataset.posterior, dataset.selection, model, hbi_config=hbi,
            criteria=criteria, n_draws=16, rhat_draws_per_run=40,
            diagnostics_fn=diagnostics_fn, taper_loglike=loglike,
        )

    assert "posterior_mass_below" not in summarize(_lenient())["taper"]["pooled"]
    block = summarize(_lenient(posterior_mass_below_cuts=(cut,)))["taper"]
    assert block["posterior_mass_below_cuts"] == sorted([cut, 50.0])
    expected = 0.0
    for run, variance, summary in zip(results, comps, block["runs"]):
        frac = float(np.sum(run.weights[variance <= cut]))
        assert summary["posterior_mass_below"][cut_key(cut)]["fraction"] == pytest.approx(frac)
        assert summary["posterior_mass_below"][cut_key(cut)]["n_eff"] == pytest.approx(summary["kish_ess"])
        expected += 0.5 * frac
    pooled = block["pooled"]["posterior_mass_below"]
    assert pooled[cut_key(cut)]["fraction"] == pytest.approx(expected)
    assert set(pooled) == {cut_key(cut), cut_key(50.0)}
    assert 0.0 < pooled[cut_key(cut)]["error"] < 0.5


def test_subsampled_taper_mass_is_consistent_and_flags_the_subsample(dataset):  # noqa: F811
    spec = baseline_model_spec()
    hbi = HBIConfig(selection_chunk_size=None, variance_taper=VarianceTaper(kind="sharp", threshold=50.0))
    results = _fake_runs(dataset, spec, n=300, hbi=hbi)
    loglike, _ = _compiled(dataset, spec, hbi)
    variance = np.concatenate([loglike.components(r.samples)["variance"] for r in results])
    cut = float(np.median(variance[np.isfinite(variance)]))
    full = posterior_taper_mass(results, loglike, cuts=(cut,))
    sub = posterior_taper_mass(results, loglike, cuts=(cut,), max_points_per_run=120, subsample_seed=1)
    assert "subsample" not in full["pooled"] and sub["pooled"]["subsample"]["max_points_per_run"] == 120
    assert sub["runs"][0]["subsample"]["n_draws"] == 120
    f, s = full["pooled"]["posterior_mass_below"][cut_key(cut)], sub["pooled"]["posterior_mass_below"][cut_key(cut)]
    # the subsample error includes both stages and covers the full-sample estimate
    assert s["n_eff"] < f["n_eff"]
    assert abs(s["fraction"] - f["fraction"]) < 4.0 * s["error"]
    # identical reproduction checks on the evaluated subset
    assert sub["reproduces_sampled_log_likelihood"] is True
    # a run with fewer points than the cap is evaluated whole
    same = posterior_taper_mass(results, loglike, cuts=(cut,), max_points_per_run=10_000)
    assert same["pooled"]["posterior_mass_below"] == full["pooled"]["posterior_mass_below"]


def test_posthoc_path_recomputes_an_evaluation_that_lacks_the_field(dataset, tmp_path):  # noqa: F811
    """A tiny real tapered F3 evaluation written without the cuts, then recomputed post hoc."""
    hbi = HBIConfig(selection_chunk_size=None, variance_taper=VarianceTaper(kind="sharp", threshold=50.0))
    config = _small_config(hbi=hbi)
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior, dataset.selection, config=config, dataset_identity="tiny"
    )
    spec = baseline_model_spec()
    run_dir = tmp_path / "F3" / spec.model_hash
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluator.evaluate(spec, Fidelity.F3_EVIDENCE, seed=5, run_dir=run_dir)
    before = (run_dir / "evaluation.json").read_bytes()
    stored = read_evaluation(run_dir / "evaluation.json")["diagnostics"]["taper"]["pooled"]
    assert "posterior_mass_below" not in stored
    assert collect_v2_evaluations([tmp_path])["mass_below"] == {}

    row = posthoc_mass_below(run_dir, dataset.posterior, dataset.selection, config,
                             dataset_identity="tiny", cuts=(20.0,))
    assert (run_dir / "evaluation.json").read_bytes() == before  # read-only
    assert row["model_hash"] == spec.model_hash and row["fidelity"] == "F3"
    assert row["consistency"]["band_mass_matches"] is True
    assert row["consistency"]["reproduces_sampled_log_likelihood"] is True
    entry = row["entry"]
    assert entry["threshold"] == 50.0 and entry["kind"] == "sharp"
    assert set(entry["cuts"]) == {"20.0", "50.0"}
    # direct check against the stored dynesty results
    from gwpop_search.analysis.posthoc_cut import evaluation_results

    _, results, _ = evaluation_results(run_dir)
    loglike, _ = _compiled(dataset, spec, hbi)
    expected = np.mean([np.sum(r.weights[loglike.components(r.samples)["variance"] <= 20.0]) for r in results])
    assert entry["cuts"]["20.0"]["fraction"] == pytest.approx(expected)
    assert entry["cuts"]["50.0"]["fraction"] == pytest.approx(1.0)
    report = posthoc_report([row], cuts=(20.0,))
    assert report["models"][spec.model_hash] == entry
    # a subsampled post-hoc estimate carries the subsample record
    sub = posthoc_mass_below(run_dir, dataset.posterior, dataset.selection, config,
                             dataset_identity="tiny", cuts=(20.0,), max_points_per_run=15)
    assert sub["consistency"]["subsampled"] is True and "band_mass_matches" not in sub["consistency"]
    # the evaluation's configuration and dataset are verified
    with pytest.raises(AnalysisInputError, match="dataset"):
        posthoc_mass_below(run_dir, dataset.posterior, dataset.selection, config,
                           dataset_identity="other", cuts=(20.0,))
    with pytest.raises(AnalysisInputError, match="fidelity configuration"):
        posthoc_mass_below(run_dir, dataset.posterior, dataset.selection,
                           _small_config(hbi=HBIConfig(selection_chunk_size=None,
                                                       variance_taper=VarianceTaper(kind="sharp", threshold=40.0))),
                           dataset_identity="tiny", cuts=(20.0,))

    # the same evaluation with the cuts configured records the same fractions itself
    recorded = _small_config(hbi=hbi, f3_criteria=_lenient(posterior_mass_below_cuts=(20.0,)))
    again = tmp_path / "with_cuts" / "F3" / spec.model_hash
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DeterministicHBIEvaluator(dataset.posterior, dataset.selection, config=recorded,
                                  dataset_identity="tiny").evaluate(
            spec, Fidelity.F3_EVIDENCE, seed=5, run_dir=again)
    collected = collect_v2_evaluations([tmp_path / "with_cuts"])["mass_below"][spec.model_hash]
    assert collected["cuts"]["20.0"]["fraction"] == pytest.approx(entry["cuts"]["20.0"]["fraction"])
