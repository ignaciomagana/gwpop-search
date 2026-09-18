import numpy as np
import pytest

pytest.importorskip("numpyro")

from gwpop_search.hbi import HBIConfig
from gwpop_search.inference import (
    BASELINE_SYNTHETIC_PRIORS,
    NUTSConfig,
    run_nuts,
)
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import GwcatChiEffBBHModel


def test_actual_baseline_runs_tiny_numpyro_chain():
    model = GwcatChiEffBBHModel(redshift_quadrature_order=32)
    dataset = generate_baseline_synthetic_dataset(
        seed=31415,
        model=model,
        config=SyntheticSurveyConfig(
            n_events=6,
            posterior_samples_per_event=16,
            n_injections=600,
            population_batch_size=256,
            redshift_sampling_grid=1024,
        ),
    )

    result = run_nuts(
        dataset.posterior,
        dataset.selection,
        model,
        BASELINE_SYNTHETIC_PRIORS,
        seed=2718,
        config=NUTSConfig(
            num_warmup=12,
            num_samples=12,
            num_chains=1,
            target_accept_prob=0.8,
            max_tree_depth=6,
            progress_bar=False,
        ),
        hbi_config=HBIConfig(selection_chunk_size=128),
    )

    assert set(result.samples) == set(BASELINE_SYNTHETIC_PRIORS)
    for values in result.samples.values():
        assert values.shape == (1, 12)
        assert np.all(np.isfinite(values))
    assert result.extra_fields["diverging"].shape == (1, 12)
