import json

import numpy as np

from gwpop_search.grammar import baseline_model_spec
from gwpop_search.inference.evidence import (
    EvidenceResult,
    NestedSamplingConfig,
)
from gwpop_search.inference.evidence_campaign import EvidenceCampaignConfig
from gwpop_search.inference.fidelity import (
    DeterministicHBIEvaluator,
    FidelityRunConfig,
    NumericalCriteria,
)
from gwpop_search.inference.numpyro import NUTSConfig, NUTSResult
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.search.scheduler import Fidelity


def _small_dataset():
    return generate_baseline_synthetic_dataset(
        seed=44,
        config=SyntheticSurveyConfig(
            n_events=4,
            posterior_samples_per_event=32,
            n_injections=1_000,
            population_batch_size=256,
            redshift_sampling_grid=1024,
        ),
    )


def _lenient_criteria(**kwargs):
    values = dict(
        max_r_hat=2.0,
        min_mcmc_n_eff=1.0,
        max_divergences=100,
        min_event_ess=1.0,
        min_selection_ess=1.0,
        max_event_weight_fraction=1.0,
        max_selection_weight_fraction=1.0,
        max_shape_log_likelihood_variance=1e9,
        max_evidence_error=10.0,
        max_evidence_repeat_std=10.0,
    )
    values.update(kwargs)
    return NumericalCriteria(**values)


def test_f0_runs_real_standardized_hbi_on_closed_synthetic_data(tmp_path):
    dataset = _small_dataset()
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior,
        dataset.selection,
        config=FidelityRunConfig(
            f0_pe_samples_per_event=16,
            f0_selected_per_campaign=128,
        ),
        dataset_identity="synthetic-test",
    )
    record = evaluator.evaluate(
        baseline_model_spec(),
        Fidelity.F0_SANITY,
        seed=123,
        run_dir=tmp_path / "f0",
    )

    assert record.status == "complete"
    assert record.diagnostics_pass
    assert record.screen_value == 0.0
    payload = json.loads((tmp_path / "f0" / "evaluation.json").read_text())
    assert payload["dataset_identity"] == "synthetic-test"
    assert payload["screen_value_semantics"] == "sanity_constant"
    assert np.isfinite(payload["diagnostics"]["importance"]["log_likelihood"])


def test_f1_uses_reduced_data_nuts_and_marks_score_as_allocation_only(
    monkeypatch,
    tmp_path,
):
    dataset = _small_dataset()
    spec = baseline_model_spec()
    centers = {
        name: (
            0.5 * (prior.parameters["low"] + prior.parameters["high"])
            if prior.family == "uniform"
            else (
                np.sqrt(prior.parameters["low"] * prior.parameters["high"])
                if prior.family == "log_uniform"
                else prior.parameters["loc"]
            )
        )
        for name, prior in spec.priors.items()
    }
    rng = np.random.default_rng(5)
    samples = {
        name: center + 1e-4 * rng.normal(size=(2, 20))
        for name, center in centers.items()
    }
    fake = NUTSResult(
        samples=samples,
        extra_fields={"diverging": np.zeros((2, 20), dtype=bool)},
        seed=77,
        config=NUTSConfig(
            num_warmup=10,
            num_samples=20,
            num_chains=2,
            progress_bar=False,
        ),
    )

    monkeypatch.setattr(
        "gwpop_search.inference.fidelity.run_resumable_chains",
        lambda *args, **kwargs: fake,
    )
    monkeypatch.setattr(
        "gwpop_search.inference.fidelity.chain_diagnostics",
        lambda samples: {
            "max_r_hat": 1.001,
            "min_n_eff": 30.0,
            "per_parameter": {},
        },
    )

    config = FidelityRunConfig(
        f1_pe_samples_per_event=16,
        f1_selected_per_campaign=128,
        f1_nuts=fake.config,
        f1_criteria=_lenient_criteria(),
    )
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior,
        dataset.selection,
        config=config,
    )
    record = evaluator.evaluate(
        spec,
        Fidelity.F1_SCREEN,
        seed=77,
        run_dir=tmp_path / "f1",
    )

    assert record.diagnostics_pass
    assert np.isfinite(record.screen_value)
    payload = json.loads((tmp_path / "f1" / "evaluation.json").read_text())
    assert payload["screen_value_semantics"] == "bic_like_allocation_only"
    assert (tmp_path / "f1" / "posterior.npz").exists()


def test_f3_uses_repeated_evidence_as_score(monkeypatch, tmp_path):
    dataset = _small_dataset()
    spec = baseline_model_spec()
    posterior_samples = {
        name: np.full(
            12,
            (
                0.5 * (prior.parameters["low"] + prior.parameters["high"])
                if prior.family == "uniform"
                else (
                    np.sqrt(prior.parameters["low"] * prior.parameters["high"])
                    if prior.family == "log_uniform"
                    else prior.parameters["loc"]
                )
            ),
        )
        for name, prior in spec.priors.items()
    }
    nested = NestedSamplingConfig(
        num_live_points=20,
        max_samples=200,
        dlogz=0.5,
        num_posterior_samples=12,
    )
    result = EvidenceResult(
        log_evidence=-12.3,
        log_evidence_error=0.1,
        posterior_samples=posterior_samples,
        diagnostics={"nested_ess": 50.0},
        seed=9,
        backend="numpyro-jaxns",
        config=nested,
    )
    summary = {
        "model_hash": spec.model_hash,
        "n_repeats": 2,
        "log_evidence_mean": -12.25,
        "repeat_std": 0.08,
        "mean_reported_error": 0.1,
        "max_reported_error": 0.1,
        "conservative_error": 0.1,
        "estimates": [-12.3, -12.2],
        "errors": [0.1, 0.1],
    }
    seen = {}

    def fake_repeats(*args, dataset_identity, **kwargs):
        seen["dataset_identity"] = dataset_identity
        return [result, result], summary

    monkeypatch.setattr(
        "gwpop_search.inference.fidelity.run_model_evidence_repeats",
        fake_repeats,
    )
    config = FidelityRunConfig(
        f3_evidence=EvidenceCampaignConfig(
            repeats=2,
            nested_sampling=nested,
        ),
        f3_criteria=_lenient_criteria(
            max_r_hat=None,
            min_mcmc_n_eff=None,
            max_divergences=None,
        ),
    )
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior,
        dataset.selection,
        config=config,
        dataset_identity="manifest-sha",
    )
    record = evaluator.evaluate(
        spec,
        Fidelity.F3_EVIDENCE,
        seed=9,
        run_dir=tmp_path / "f3",
    )

    assert record.diagnostics_pass
    assert record.screen_value == -12.25
    assert seen["dataset_identity"] == "manifest-sha"
    payload = json.loads((tmp_path / "f3" / "evaluation.json").read_text())
    assert payload["screen_value_semantics"] == "log_evidence_mean"
