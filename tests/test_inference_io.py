from dataclasses import replace
import importlib.util
import json

import numpy as np
import pytest

from gwpop_search import __version__
from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.inference import (
    BASELINE_SYNTHETIC_PRIORS,
    NUTSConfig,
    NUTSResult,
    NumPyroUnavailableError,
    PriorSpec,
    build_run_manifest,
    load_result,
    run_resumable_chains,
    save_result,
    serialize_prior_map,
)
from gwpop_search.inference.numpyro import _chain_seed, _require_numpyro
from gwpop_search.models import GwcatChiEffBBHModel


def test_prior_spec_roundtrip_and_baseline_prior_map():
    spec = PriorSpec("log_uniform", low=0.1, high=2.0)
    assert PriorSpec.from_dict(spec.to_dict()) == spec
    serialized = serialize_prior_map(BASELINE_SYNTHETIC_PRIORS)
    assert set(serialized) == {
        "alpha",
        "mmin",
        "mmax",
        "peak_fraction",
        "peak_mu",
        "peak_sigma",
        "beta_q",
        "kappa",
        "chi_mu",
        "chi_sigma",
    }


def test_nuts_config_validation():
    with pytest.raises(ValueError):
        NUTSConfig(num_samples=0)
    with pytest.raises(ValueError):
        NUTSConfig(target_accept_prob=1.0)
    with pytest.raises(ValueError):
        NUTSConfig(chain_method="not-a-method")


def test_nuts_result_roundtrip(tmp_path):
    result = NUTSResult(
        samples={"x": np.arange(6).reshape(1, 6)},
        extra_fields={"diverging": np.zeros((1, 6), dtype=bool)},
        seed=4,
        config=NUTSConfig(
            num_warmup=2,
            num_samples=6,
            num_chains=1,
            progress_bar=False,
        ),
    )
    path = tmp_path / "chain.npz"
    save_result(path, result)
    loaded = load_result(path)

    np.testing.assert_array_equal(loaded.samples["x"], result.samples["x"])
    np.testing.assert_array_equal(
        loaded.extra_fields["diverging"],
        result.extra_fields["diverging"],
    )
    assert loaded.config == result.config
    assert loaded.seed == result.seed


def test_chain_seed_is_deterministic_and_distinct():
    seeds = [_chain_seed(11, i) for i in range(5)]
    assert seeds == [_chain_seed(11, i) for i in range(5)]
    assert len(set(seeds)) == 5


def test_run_manifest_contains_scientific_and_numerical_configuration():
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    model = GwcatChiEffBBHModel()
    config = NUTSConfig(
        num_warmup=10,
        num_samples=20,
        num_chains=2,
        progress_bar=False,
    )

    manifest = build_run_manifest(
        posterior,
        selection,
        model,
        BASELINE_SYNTHETIC_PRIORS,
        seed=17,
        config=config,
    )

    assert manifest["code"]["package_version"] == __version__
    assert "git_commit" in manifest["code"]
    assert manifest["pe_basis"] == posterior.basis.identity
    assert manifest["selection_basis"] == selection.basis.identity
    assert manifest["model"]["cosmology"]["H0"] == model.cosmology.H0
    assert manifest["nuts_config"]["num_chains"] == 2
    assert manifest["hbi_config"]["rate_treatment"] == "shape"
    assert manifest["priors"]["chi_sigma"]["family"] == "log_uniform"


def test_resumable_chains_load_completed_chain_checkpoints_without_numpyro(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    model = GwcatChiEffBBHModel()
    config = NUTSConfig(
        num_warmup=2,
        num_samples=3,
        num_chains=2,
        progress_bar=False,
    )
    seed = 23

    manifest = build_run_manifest(
        posterior,
        selection,
        model,
        BASELINE_SYNTHETIC_PRIORS,
        seed=seed,
        config=config,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2)
    )

    single = replace(config, num_chains=1, chain_method="sequential")
    for chain_index in range(config.num_chains):
        value = np.full((1, config.num_samples), float(chain_index))
        result = NUTSResult(
            samples={"alpha": value},
            extra_fields={
                "diverging": np.zeros_like(value, dtype=bool),
                "num_steps": np.ones_like(value, dtype=np.int32),
            },
            seed=_chain_seed(seed, chain_index),
            config=single,
        )
        save_result(tmp_path / f"chain_{chain_index:03d}.npz", result)

    combined = run_resumable_chains(
        tmp_path,
        posterior,
        selection,
        model,
        BASELINE_SYNTHETIC_PRIORS,
        seed=seed,
        config=config,
    )

    assert combined.samples["alpha"].shape == (2, 3)
    np.testing.assert_array_equal(combined.samples["alpha"][0], 0.0)
    np.testing.assert_array_equal(combined.samples["alpha"][1], 1.0)


def test_resume_manifest_mismatch_fails_before_sampling(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    model = GwcatChiEffBBHModel()
    config = NUTSConfig(
        num_warmup=2,
        num_samples=3,
        num_chains=1,
        progress_bar=False,
    )

    manifest = build_run_manifest(
        posterior,
        selection,
        model,
        BASELINE_SYNTHETIC_PRIORS,
        seed=1,
        config=config,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2)
    )

    with pytest.raises(ValueError, match="resume manifest"):
        run_resumable_chains(
            tmp_path,
            posterior,
            selection,
            model,
            BASELINE_SYNTHETIC_PRIORS,
            seed=2,
            config=config,
        )


def test_missing_numpyro_has_explicit_error():
    if importlib.util.find_spec("numpyro") is None:
        with pytest.raises(NumPyroUnavailableError):
            _require_numpyro()
