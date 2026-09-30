import h5py
import numpy as np
import pytest

from gwpop_search.data import (
    BasisMismatchError,
    PosteriorCatalog,
    SelectionCatalog,
    SelectionMode,
    canonicalize_gwcat_v2_pair,
)
from gwpop_search.data.adapters.gwcat_v2 import load_pair, load_selection


def _write_pe(path, *, spin_basis="chieff"):
    nobs, nsamp = 2, 3
    n = nobs * nsamp
    m1 = np.linspace(30.0, 45.0, n)
    m2 = np.linspace(15.0, 30.0, n)
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-pe-2.0"
        f.attrs["spin_basis"] = spin_basis
        f.attrs["nobs"] = nobs
        f.attrs["nsamp"] = nsamp
        f.attrs["event_names"] = np.asarray(
            ["GWTEST_A", "GWTEST_B"], dtype=h5py.string_dtype("utf-8")
        )
        f.attrs["p_pe_state"] = "fixture PE denominator"
        f.create_dataset("m1det", data=m1)
        f.create_dataset("m2det", data=m2)
        f.create_dataset("dL", data=np.linspace(500.0, 1000.0, n))
        f.create_dataset("ra", data=np.linspace(0.1, 1.0, n))
        f.create_dataset("dec", data=np.linspace(-0.4, 0.4, n))
        f.create_dataset("m1src", data=m1 / 1.1)
        f.create_dataset("m2src", data=m2 / 1.1)
        f.create_dataset("redshift", data=np.full(n, 0.1))
        f.create_dataset("chieff", data=np.linspace(-0.2, 0.2, n))
        f.create_dataset("p_pe", data=np.linspace(0.5, 1.5, n))
        if spin_basis == "chieff_chip":
            f.create_dataset("chip", data=np.linspace(0.05, 0.3, n))
        if spin_basis == "component":
            f.create_dataset("a1", data=np.full(n, 0.4))
            f.create_dataset("a2", data=np.full(n, 0.3))
            f.create_dataset("cost1", data=np.full(n, 0.5))
            f.create_dataset("cost2", data=np.full(n, -0.2))


def _write_selection(path, *, spin_basis="chieff"):
    n = 5
    m1 = np.linspace(25.0, 55.0, n)
    m2 = np.linspace(15.0, 30.0, n)
    pdraw = np.asarray([0.20, 0.25, 0.40, 0.50, 0.80])
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-2.0"
        f.attrs["spin_basis"] = spin_basis
        f.attrs["n_detected"] = n
        f.attrs["ndraw"] = 5000
        f.attrs["T_obs_yr"] = 1.7
        f.attrs["n_campaigns"] = 2
        f.attrs["campaign_ndraws"] = np.asarray([2000, 3000], dtype=np.int64)
        f.attrs["pdraw_state"] = (
            "fixture estimator-ready density; campaign fractions and exposure already encoded"
        )
        f.create_dataset("m1det", data=m1)
        f.create_dataset("m2det", data=m2)
        f.create_dataset("q", data=m2 / m1)
        f.create_dataset("dL", data=np.linspace(400.0, 1400.0, n))
        f.create_dataset("ra", data=np.linspace(0.2, 1.1, n))
        f.create_dataset("dec", data=np.linspace(-0.3, 0.3, n))
        f.create_dataset("m1src", data=m1 / 1.1)
        f.create_dataset("m2src", data=m2 / 1.1)
        f.create_dataset("redshift", data=np.full(n, 0.1))
        f.create_dataset("chieff", data=np.linspace(-0.3, 0.3, n))
        f.create_dataset("pdraw", data=pdraw)
        if spin_basis == "chieff_chip":
            f.create_dataset("chip", data=np.linspace(0.05, 0.3, n))
        if spin_basis == "component":
            f.create_dataset("a1", data=np.full(n, 0.4))
            f.create_dataset("a2", data=np.full(n, 0.3))
            f.create_dataset("cost1", data=np.full(n, 0.5))
            f.create_dataset("cost2", data=np.full(n, -0.2))


def test_gwcat_chieff_pair_loads_without_reconstructing_priors(tmp_path):
    pe_path = tmp_path / "pe.h5"
    sel_path = tmp_path / "sel.h5"
    _write_pe(pe_path)
    _write_selection(sel_path)

    pe, sel = load_pair(pe_path, sel_path)
    assert pe.event_names == ("GWTEST_A", "GWTEST_B")
    assert [pe.sample_count(i) for i in range(2)] == [3, 3]
    np.testing.assert_allclose(
        pe.samples["q"], pe.samples["m2_source"] / pe.samples["m1_source"]
    )
    assert ("ra", "dec") == pe.basis.coordinates[3:5]
    assert sel.mode is SelectionMode.ESTIMATOR_READY
    assert sel.campaigns[0].n_draw == 5000
    assert sel.campaigns[0].metadata["campaign_ndraws"] == [2000, 3000]
    assert pe.basis.identity == sel.basis.identity


def test_gwcat_pdraw_is_preserved_as_estimator_ready_denominator(tmp_path):
    path = tmp_path / "sel.h5"
    _write_selection(path)
    sel = load_selection(path)
    pdraw = np.asarray([0.20, 0.25, 0.40, 0.50, 0.80])
    log_pop = np.log(np.asarray([0.3, 0.1, 0.5, 0.7, 0.2]))
    direct = np.sum(np.exp(log_pop) / pdraw)
    adapted = np.sum(np.exp(log_pop - sel.log_draw_density))
    np.testing.assert_allclose(adapted, direct, rtol=1e-14)
    assert not np.isclose(adapted, direct / sel.campaigns[0].n_draw)


def test_gwcat_chieff_chip_basis_maps_joint_spin_coordinates(tmp_path):
    pe_path = tmp_path / "pe.h5"
    sel_path = tmp_path / "sel.h5"
    _write_pe(pe_path, spin_basis="chieff_chip")
    _write_selection(sel_path, spin_basis="chieff_chip")
    pe, sel = load_pair(pe_path, sel_path)
    assert pe.basis.coordinates[-2:] == ("chi_eff", "chi_p")
    assert sel.basis.coordinates == pe.basis.coordinates


def test_gwcat_component_basis_excludes_derived_chieff_from_measure(tmp_path):
    pe_path = tmp_path / "pe.h5"
    sel_path = tmp_path / "sel.h5"
    _write_pe(pe_path, spin_basis="component")
    _write_selection(sel_path, spin_basis="component")
    pe, sel = load_pair(pe_path, sel_path)
    assert "chi_eff" in pe.samples
    assert "chi_eff" not in pe.basis.coordinates
    assert pe.basis.coordinates[-4:] == ("a1", "a2", "cos_tilt1", "cos_tilt2")


def test_gwcat_basis_mismatch_fails_before_inference(tmp_path):
    pe_path = tmp_path / "pe.h5"
    sel_path = tmp_path / "sel.h5"
    _write_pe(pe_path, spin_basis="chieff")
    _write_selection(sel_path, spin_basis="chieff_chip")
    with pytest.raises(BasisMismatchError):
        load_pair(pe_path, sel_path)



def test_gwcat_canonicalization_writes_audited_internal_pair(tmp_path):
    pe_export = tmp_path / "gwcat_pe.h5"
    selection_export = tmp_path / "gwcat_selection.h5"
    output = tmp_path / "canonical"
    _write_pe(pe_export, spin_basis="chieff")
    _write_selection(selection_export, spin_basis="chieff")

    report = canonicalize_gwcat_v2_pair(
        pe_export,
        selection_export,
        output,
        required_spin_basis="chieff",
    )

    assert report["format_version"] == "gwpop-search-gwcat-canonicalization-1.1"
    assert report["required_selection_spin_basis"] == "chieff"
    assert report["selection_reference_pairing"] is None
    assert report["required_spin_basis"] == "chieff"
    assert report["selection_mode"] == "estimator_ready"
    assert report["n_events"] == 2
    assert report["n_pe_samples"] == 6
    assert report["n_selected_injections"] == 5
    assert report["source_pe"]["sha256"]
    assert report["canonical_pe"]["sha256"]
    assert (
        report["denominator_contract"]["selection"]
        .startswith("gwcat exported estimator-ready pdraw")
    )

    pe = PosteriorCatalog.from_hdf5(output / "pe.h5")
    selection = SelectionCatalog.from_hdf5(output / "selection.h5")
    assert pe.basis.identity == selection.basis.identity
    assert selection.mode is SelectionMode.ESTIMATOR_READY
    np.testing.assert_allclose(
        np.exp(pe.log_ref_density),
        np.linspace(0.5, 1.5, 6),
    )
    np.testing.assert_allclose(
        np.exp(selection.log_draw_density),
        np.asarray([0.20, 0.25, 0.40, 0.50, 0.80]),
    )


def test_gwcat_canonicalization_is_idempotent_for_exact_same_inputs(tmp_path):
    pe_export = tmp_path / "gwcat_pe.h5"
    selection_export = tmp_path / "gwcat_selection.h5"
    output = tmp_path / "canonical"
    _write_pe(pe_export)
    _write_selection(selection_export)

    first = canonicalize_gwcat_v2_pair(
        pe_export,
        selection_export,
        output,
        required_spin_basis="chieff",
    )
    second = canonicalize_gwcat_v2_pair(
        pe_export,
        selection_export,
        output,
        required_spin_basis="chieff",
    )
    assert second == first


def test_gwcat_canonicalization_refuses_spin_basis_mismatch(tmp_path):
    pe_export = tmp_path / "gwcat_pe.h5"
    selection_export = tmp_path / "gwcat_selection.h5"
    _write_pe(pe_export, spin_basis="chieff")
    _write_selection(selection_export, spin_basis="chieff")

    with pytest.raises(ValueError, match="does not match explicit requirement"):
        canonicalize_gwcat_v2_pair(
            pe_export,
            selection_export,
            tmp_path / "canonical",
            required_spin_basis="chieff_chip",
        )


def test_gwcat_canonicalization_refuses_conflicting_existing_outputs(tmp_path):
    pe_export = tmp_path / "gwcat_pe.h5"
    selection_export = tmp_path / "gwcat_selection.h5"
    output = tmp_path / "canonical"
    _write_pe(pe_export)
    _write_selection(selection_export)
    canonicalize_gwcat_v2_pair(
        pe_export,
        selection_export,
        output,
        required_spin_basis="chieff",
    )

    with h5py.File(pe_export, "a") as f:
        f["m1det"][0] += 1.0

    with pytest.raises(ValueError, match="different PE export"):
        canonicalize_gwcat_v2_pair(
            pe_export,
            selection_export,
            output,
            required_spin_basis="chieff",
        )
