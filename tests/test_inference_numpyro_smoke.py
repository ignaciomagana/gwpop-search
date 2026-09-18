import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")

from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.inference import NUTSConfig, PriorSpec, run_nuts


def _tilted_chi_density(samples, hyperparameters):
    import jax.numpy as jnp

    eta = hyperparameters["eta"]
    chi = samples["chi_eff"]
    # Normalized exp(eta*chi) density on [-1, 1], with the eta -> 0 limit.
    small = jnp.abs(eta) < 1e-5
    norm = jnp.where(
        small,
        2.0 + eta**2 / 3.0,
        2.0 * jnp.sinh(eta) / eta,
    )
    # Other toy coordinates have parameter-independent factors and therefore
    # can be omitted from this wrapper smoke test.
    return eta * chi - jnp.log(norm)


def test_numpyro_wrapper_runs_short_chain_when_dependency_is_available():
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    result = run_nuts(
        posterior,
        selection,
        _tilted_chi_density,
        {"eta": PriorSpec("uniform", low=-2.0, high=2.0)},
        seed=7,
        config=NUTSConfig(
            num_warmup=20,
            num_samples=20,
            num_chains=1,
            target_accept_prob=0.8,
            progress_bar=False,
        ),
    )
    assert result.samples["eta"].shape == (1, 20)
    assert np.all(np.isfinite(result.samples["eta"]))
    assert result.extra_fields["diverging"].shape == (1, 20)
