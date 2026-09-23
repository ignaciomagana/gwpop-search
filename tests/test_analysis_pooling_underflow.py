"""Regression test: pooled dynesty weights that underflow to exactly zero are dropped.

Observed on the GWTC-5 production results (2026-09-23): a handful of dead points per run have
a normalised weight equal to the smallest subnormal double (~4.94e-324). Dividing by the number of
runs rounds that to 0.0, and ``WeightedPosterior`` then refused the pooled sample with
"posterior weights must be finite and positive", which crashed ``analyze-sddr`` and
``analyze-prior-sensitivity``. Such points carry no mass and must be dropped like plateau points.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gwpop_search.analysis._common import pool_dynesty_results


def _run(samples, log_weights, names=("x", "y")):
    return SimpleNamespace(
        names=tuple(names),
        samples=np.asarray(samples, dtype=float),
        log_weights=np.asarray(log_weights, dtype=float),
        log_evidence=0.0,
        log_evidence_error=0.1,
    )


def test_subnormal_weights_that_round_to_zero_are_dropped():
    # After normalisation the second point of run ``a`` has weight exp(-745) ~ 4.94e-324 > 0,
    # which becomes exactly 0.0 once divided by the two runs.
    a = _run([[0.0, 0.0], [1.0, 1.0]], [0.0, -745.0])
    b = _run([[2.0, 2.0], [3.0, 3.0]], [0.0, 0.0])
    tiny = np.exp(-745.0)
    assert tiny > 0.0 and tiny / 2.0 == 0.0  # the mechanism being guarded against

    pooled = pool_dynesty_results([a, b])

    assert pooled.points.shape == (3, 2)
    assert np.all(pooled.weights > 0.0)
    assert pooled.weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pooled.weights, [0.5, 0.25, 0.25])
    np.testing.assert_array_equal(pooled.run_index, [0, 1, 1])
    assert pooled.n_runs == 2
    assert pooled.metadata["n_points_weighted"] == 3


def test_ordinary_pooling_is_unchanged():
    a = _run([[0.0, 0.0], [1.0, 1.0]], [np.log(1.0), np.log(3.0)])
    b = _run(np.arange(8.0).reshape(4, 2), np.full(4, 100.0))
    pooled = pool_dynesty_results([a, b])
    np.testing.assert_allclose(pooled.weights, [0.125, 0.375, 0.125, 0.125, 0.125, 0.125])
    assert pooled.points.shape == (6, 2)
