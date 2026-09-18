import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gwpop_search.hbi import HBIConfig, build_jax_shape_log_likelihood
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import (
    DEFAULT_BASELINE_HYPERPARAMETERS,
    GwcatChiEffBBHModel,
)


def test_actual_baseline_hbi_value_and_gradient_are_finite_at_truth():
    model = GwcatChiEffBBHModel(redshift_quadrature_order=32)
    dataset = generate_baseline_synthetic_dataset(
        seed=1618,
        model=model,
        config=SyntheticSurveyConfig(
            n_events=4,
            posterior_samples_per_event=8,
            n_injections=300,
            population_batch_size=128,
            redshift_sampling_grid=1024,
        ),
    )
    likelihood = build_jax_shape_log_likelihood(
        dataset.posterior,
        dataset.selection,
        model,
        config=HBIConfig(selection_chunk_size=64),
    )
    parameters = {
        name: jnp.asarray(value)
        for name, value in DEFAULT_BASELINE_HYPERPARAMETERS.items()
    }

    value, gradient = jax.value_and_grad(likelihood)(parameters)

    assert np.isfinite(float(value))
    for name, grad in gradient.items():
        assert np.isfinite(float(grad)), f"non-finite gradient for {name}"
