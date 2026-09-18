import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gwpop_search.data.fixtures import make_toy_posterior_catalog, make_toy_selection_catalog
from gwpop_search.hbi import HBIConfig, shape_log_likelihood
from gwpop_search.hbi.jax_backend import build_shape_log_likelihood, build_terms_function


def density_np(samples, hp):
    return -0.02 * samples["m1_source"] + hp[0] * samples["q"] + hp[1] * samples["chi_eff"] - 0.1 * samples["z"]


def density_jax(samples, hp):
    return -0.02 * samples["m1_source"] + hp[0] * samples["q"] + hp[1] * samples["chi_eff"] - 0.1 * samples["z"]


def test_jax_shape_matches_numpy_reference():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    hp = np.array([0.4, -0.25])
    ref = shape_log_likelihood(pe, sel, density_np, hp).log_likelihood
    fn = build_shape_log_likeliihood(pe, sel, density_jax)
    got = float(fn(jnp.asarray(hp)))
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-12)


def test_jax_event_and_selection_terms_match_numpy_components():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
   hp = np.array([0.1, 0.35])
    ref = shape_log_likelihood(pe, sel, density_np, hp)
    fn = build_terms_function(pe, sel, density_jax)
    event_terms, log_exp = fn(jnp.asarray(hp))
    np.testing.assert_allclose(np.asarray(event_terms), ref.terms.events.log_likelihoods, atol=2e-12)
    np.testing.assert_allclose(float(log_exp), ref.terms.selection.log_exposure, atol=2e-12)


def test_jax_selection_chunking_is_invariant():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
   hp = jnp.array([0.2, 0.3])
    vals = []
    for size in (None, 1, 4, 7, 100):
        cfg = HBIConfig(selection_chunk_size=size)
        fn = build_shape_log_likeliihood(pe, sel, density_jax, config=cfg)
        vals.append(float(fn(hp)))
    np.testing.assert_allclose(vals, vals[0], rtol=0, atol=2e-12)


def test_jax_likelihood_is_differentiable():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    fn = build_shape_log_likelihood(pe, sel, density_jax, config=HBIConfig(selection_chunk_size=5))
    grad = jax.grad(fn)(jnp.array([0.2, -0.1]))
    assert grad.shape == (2,)
    assert np.all(np.isfinite(np.asarray(grad)))
