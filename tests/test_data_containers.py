import numpy as np
import pytest

from gwpop_search.data import (
    Campaign,
    DataContractError,
    MissingCoordinateError,
    PosteriorCatalog,
    ReferenceDensityError,
    SelectionCatalog,
    SelectionMode,
    validate_pair,
)
from gwpop_search.data.fixtures import (
    TOY_BASIS,
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)


def test_posterior_ragged_access_and_roundtrip(tmp_path):
    cat = make_toy_posterior_catalog()
    assert cat.n_events == 3
    assert [cat.sample_count(i) for i in range(3)] == [5, 8, 6]
    assert cat.get_event(1, ["q"])["q"].shape == (8,)

    path = tmp_path / "pe.h5"
    cat.to_hdf5(path)
    loaded = PosteriorCatalog.from_hdf5(path)
    assert loaded.event_names == cat.event_names
    np.testing.assert_array_equal(loaded.offsets, cat.offsets)
    np.testing.assert_allclose(loaded.log_ref_density, cat.log_ref_density)
    assert loaded.basis.identity == cat.basis.identity
    for field in cat.field_names:
        np.testing.assert_allclose(loaded.samples[field], cat.samples[field])


def test_selection_roundtrip_and_campaign_rows(tmp_path):
    sel = make_toy_selection_catalog()
    assert sel.mode is SelectionMode.RAW_DRAW
    assert sel.rows_for_campaign("O3").size == 9
    assert sel.rows_for_campaign("O4").size == 13

    path = tmp_path / "selection.h5"
    sel.to_hdf5(path)
    loaded = SelectionCatalog.from_hdf5(path)
    assert loaded.mode is SelectionMode.RAW_DRAW
    assert [c.n_draw for c in loaded.campaigns] == [1000, 1600]
    assert loaded.basis.identity == sel.basis.identity
    np.testing.assert_allclose(loaded.samples["m1_source"], sel.samples["m1_source"])


def test_missing_required_coordinate_names_events():
    cat = make_toy_posterior_catalog()
    q_idx = cat.field_names.index("q")
    cat.availability[1, q_idx] = False
    with pytest.raises(MissingCoordinateError, match="GWTOY_B"):
        cat.require(["q"])


def test_pair_validation_requires_coordinates_on_both_sides():
    pe = make_toy_posterior_catalog()
    sel = make_toy_selection_catalog()
    validate_pair(pe, sel, required_coordinates=("m1_source", "q"))
    with pytest.raises(MissingCoordinateError, match="not_here"):
        validate_pair(pe, sel, required_coordinates=("not_here",))


def test_reference_density_must_be_finite():
    pe = make_toy_posterior_catalog()
    bad = pe.log_ref_density.copy()
    bad[0] = -np.inf
    with pytest.raises(ReferenceDensityError):
        PosteriorCatalog(
            event_names=pe.event_names,
            offsets=pe.offsets,
            samples=pe.samples,
            log_ref_density=bad,
            basis=pe.basis,
        )


def test_raw_selection_requires_ndraw():
    sel = make_toy_selection_catalog()
    with pytest.raises(DataContractError, match="n_draw"):
        SelectionCatalog(
            samples=sel.samples,
            log_draw_density=sel.log_draw_density,
            campaign_id=sel.campaign_id,
            campaigns=(Campaign("O3"), Campaign("O4", n_draw=10)),
            basis=TOY_BASIS,
            mode=SelectionMode.RAW_DRAW,
        )


def test_estimator_ready_requires_semantics():
    sel = make_toy_selection_catalog()
    with pytest.raises(DataContractError, match="estimator_semantics"):
        SelectionCatalog(
            samples=sel.samples,
            log_draw_density=sel.log_draw_density,
            campaign_id=np.asarray(["combined"] * sel.n_selected),
            campaigns=(Campaign("combined", n_draw=100),),
            basis=TOY_BASIS,
            mode=SelectionMode.ESTIMATOR_READY,
        )
