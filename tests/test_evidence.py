"""Repeated dynesty evidence: summaries, one-run contract and typed failures."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("dynesty")

from gwpop_search.data.fixtures import (  # noqa: E402
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference import (  # noqa: E402
    DynestyConfig,
    NoFiniteSupportError,
    PriorSpec,
    run_dynesty_population,
    run_hbi_evidence,
    summarize_evidence_repeats,
)
from gwpop_search.inference.dynesty_backend import load_dynesty_result  # noqa: E402
from gwpop_search.inference.evidence import (  # noqa: E402
    SUPPORTED_DYNESTY_VERSIONS,
    dynesty_version,
    run_diagnostics,
    sampler_backend_identity,
)


def _fake(logz, error, information=None):
    return SimpleNamespace(log_evidence=logz, log_evidence_error=error, information=information)


def test_repeat_summary_keeps_between_run_scatter_explicit():
    summary = summarize_evidence_repeats(
        [_fake(-10.0, 0.1, 3.0), _fake(-9.8, 0.15, 3.2), _fake(-10.2, 0.12, 2.8)]
    )
    assert summary.log_evidence_mean == pytest.approx(-10.0)
    assert summary.repeat_std == pytest.approx(np.std([-10.0, -9.8, -10.2], ddof=1))
    assert summary.max_reported_error == 0.15
    assert summary.mean_reported_error == pytest.approx(np.mean([0.1, 0.15, 0.12]))
    assert summary.conservative_error == pytest.approx(max(summary.repeat_std, 0.15))
    assert summary.n_repeats == 3
    assert summary.max_pairwise_z == pytest.approx(0.4 / np.hypot(0.15, 0.12))
    payload = summary.to_dict()
    assert payload["information_mean"] == pytest.approx(3.0)
    assert payload["conservative_error"] == summary.conservative_error
    assert json.loads(json.dumps(payload)) == payload


def test_repeat_summary_of_one_run_and_invalid_inputs():
    single = summarize_evidence_repeats([_fake(-3.0, 0.2)])
    assert single.repeat_std == 0.0 and single.max_pairwise_z is None
    assert single.conservative_error == 0.2
    assert single.to_dict()["information_mean"] is None
    with pytest.raises(ValueError, match="at least one"):
        summarize_evidence_repeats([])
    with pytest.raises(ValueError, match="finite"):
        summarize_evidence_repeats([_fake(np.nan, 0.1)])
    with pytest.raises(ValueError, match=">= 0"):
        summarize_evidence_repeats([_fake(1.0, -0.1)])


def test_sampler_backend_identity_is_the_pinned_dynesty():
    assert dynesty_version() in SUPPORTED_DYNESTY_VERSIONS
    assert sampler_backend_identity() == {"name": "dynesty", "version": dynesty_version()}


TOY_PRIORS = {
    "a": PriorSpec("uniform", low=-1.0, high=2.0),
    "b": PriorSpec("uniform", low=-2.0, high=2.0),
    "mcut": PriorSpec("uniform", low=30.0, high=100.0),
}


def toy_density(samples, hp):
    value = (
        -0.02 * samples["m1_source"]
        + hp["a"] * samples["q"]
        + hp["b"] * samples["chi_eff"]
        - 0.1 * samples["z"]
    )
    return jnp.where(samples["m1_source"] <= hp["mcut"], value, -jnp.inf)


def unsupported_density(samples, hp):
    # No population support anywhere: every likelihood evaluation is -inf.
    return jnp.full_like(samples["q"], -jnp.inf) + 0.0 * hp["a"]


def test_identity_run_is_exactly_the_backend_population_run(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    cfg = DynestyConfig(nlive=30, batch_size=8, dlogz=0.5, num_posterior_samples=100)
    result = run_hbi_evidence(
        posterior,
        selection,
        toy_density,
        TOY_PRIORS,
        seed=7,
        config=cfg,
        hbi_config=HBIConfig(),
        run_dir=tmp_path / "run",
    )
    reference = run_dynesty_population(
        posterior,
        selection,
        toy_density,
        TOY_PRIORS,
        seed=7,
        config=cfg,
        hbi_config=HBIConfig(),
        run_dir=tmp_path / "reference",
    )
    assert result.log_evidence == reference.log_evidence
    np.testing.assert_array_equal(result.samples, reference.samples)
    assert (tmp_path / "run" / "manifest.json").read_text() == (
        tmp_path / "reference" / "manifest.json"
    ).read_text()
    loaded = load_dynesty_result(tmp_path / "run" / "result.npz")
    assert loaded.log_evidence == result.log_evidence

    row = run_diagnostics(result, repeat=1)
    assert row["repeat"] == 1 and row["seed"] == 7
    assert row["termination"] == "dlogz"
    assert row["kish_ess"] == pytest.approx(result.kish_ess)
    assert row["predicted_error"] == pytest.approx(np.sqrt(result.information / cfg.nlive))
    assert row["config"] == cfg.to_dict()
    assert json.loads(json.dumps(row)) == row
    with pytest.raises(TypeError, match="DynestyConfig"):
        run_hbi_evidence(
            posterior, selection, toy_density, TOY_PRIORS, seed=7, config=cfg.to_dict(),
            hbi_config=HBIConfig(), run_dir=tmp_path / "bad",
        )


def test_no_finite_support_is_a_typed_numerical_failure(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    cfg = DynestyConfig(nlive=4, batch_size=4, dlogz=0.5, num_posterior_samples=10)
    with pytest.raises(NoFiniteSupportError, match="no hyperparameter with finite"):
        run_hbi_evidence(
            posterior,
            selection,
            unsupported_density,
            {"a": PriorSpec("uniform", low=0.0, high=1.0)},
            seed=1,
            config=cfg,
            hbi_config=HBIConfig(),
            run_dir=tmp_path / "empty",
        )
