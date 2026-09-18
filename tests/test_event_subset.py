import numpy as np
import pytest

from gwpop_search.data import (
    drop_posterior_events,
    subset_posterior_events,
)
from gwpop_search.data.fixtures import make_toy_posterior_catalog


def test_subset_posterior_events_preserves_requested_order_and_samples():
    posterior = make_toy_posterior_catalog(seed=7)
    subset = subset_posterior_events(
        posterior,
        ("GWTOY_C", "GWTOY_A"),
        reason="stress",
    )

    assert subset.event_names == ("GWTOY_C", "GWTOY_A")
    assert subset.n_events == 2
    assert subset.sample_count(0) == posterior.sample_count(2)
    assert subset.sample_count(1) == posterior.sample_count(0)
    np.testing.assert_allclose(
        subset.get_event(0)["q"],
        posterior.get_event(2)["q"],
    )
    np.testing.assert_allclose(
        subset.get_event(1)["q"],
        posterior.get_event(0)["q"],
    )
    assert subset.metadata["event_subset"]["reason"] == "stress"
    assert subset.basis.identity == posterior.basis.identity


def test_drop_posterior_events_removes_only_named_events():
    posterior = make_toy_posterior_catalog()
    subset = drop_posterior_events(
        posterior,
        ("GWTOY_B",),
        reason="leave-one-out",
    )
    assert subset.event_names == ("GWTOY_A", "GWTOY_C")
    assert subset.metadata["event_subset"]["source_n_events"] == 3


def test_event_subset_rejects_unknown_duplicate_or_empty_requests():
    posterior = make_toy_posterior_catalog()
    with pytest.raises(KeyError, match="unknown"):
        subset_posterior_events(posterior, ("missing",))
    with pytest.raises(ValueError, match="unique"):
        subset_posterior_events(
            posterior,
            ("GWTOY_A", "GWTOY_A"),
        )
    with pytest.raises(ValueError, match="at least one"):
        subset_posterior_events(posterior, ())


def test_event_drop_cannot_remove_every_event():
    posterior = make_toy_posterior_catalog()
    with pytest.raises(ValueError, match="every event"):
        drop_posterior_events(
            posterior,
            posterior.event_names,
        )
