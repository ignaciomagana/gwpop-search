"""gwcat (chieff PE, chieff_reference selection) pair: exact reference pairing."""

import json

import h5py
import numpy as np
import pytest

from gwpop_search.data import (
    BasisMismatchError,
    DataContractError,
    SelectionCatalog,
    canonicalize_gwcat_v2_pair,
)
from gwpop_search.data.adapters.gwcat_v2 import (
    GWCAT_CHIEFF_REFERENCE_PDRAW_STATE,
    basis_for_spin,
    load_pair,
    load_selection,
)

A_REF = 0.99
SENTINEL = 1e300
# rows 1 and 4 lie outside the reference support (a1 > a_ref, a2 > a_ref)
A1 = np.asarray([0.20, 0.995, 0.50, 0.70, 0.10, 0.90])
A2 = np.asarray([0.30, 0.40, 0.10, 0.60, 0.999, 0.20])
PDRAW = np.asarray([0.20, SENTINEL, 0.40, 0.50, SENTINEL, 0.80])


def _write_pe(path, *, spin_basis="chieff", amax=(A_REF, A_REF), sources=None,
              record_amax=True, unrecognized=()):
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
        if record_amax and spin_basis == "chieff":
            f.attrs["chi_eff_amax_1_per_event"] = np.asarray(amax, dtype=float)
            f.attrs["chi_eff_amax_2_per_event"] = np.asarray(amax, dtype=float)
            f.attrs["chi_eff_amax_source_per_event"] = np.asarray(
                sources or ["analytic"] * nobs, dtype=h5py.string_dtype("utf-8")
            )
            f.attrs["chi_eff_amax_mode"] = "per_event"
            f.attrs["spin_prior_unrecognized_events"] = np.asarray(
                list(unrecognized), dtype=h5py.string_dtype("utf-8")
            )
        f.create_dataset("m1det", data=m1)
        f.create_dataset("m2det", data=m2)
        f.create_dataset("dL", data=np.linspace(500.0, 1000.0, n))
        f.create_dataset("ra", data=np.linspace(0.1, 1.0, n))
        f.create_dataset("dec", data=np.linspace(-0.4, 0.4, n))
        f.create_dataset("chieff", data=np.linspace(-0.2, 0.2, n))
        f.create_dataset("p_pe", data=np.linspace(0.5, 1.5, n))
        if spin_basis == "component":
            f.create_dataset("a1", data=np.full(n, 0.4))
            f.create_dataset("a2", data=np.full(n, 0.3))
            f.create_dataset("cost1", data=np.full(n, 0.5))
            f.create_dataset("cost2", data=np.full(n, -0.2))


def _write_reference_selection(path, *, pdraw=PDRAW, a1=A1, a2=A2,
                               declared=None, coverage_ok=True,
                               state=GWCAT_CHIEFF_REFERENCE_PDRAW_STATE,
                               chieff=None):
    n = pdraw.size
    m1 = np.linspace(25.0, 55.0, n)
    m2 = np.linspace(15.0, 30.0, n)
    declared = int(np.sum(pdraw == SENTINEL)) if declared is None else declared
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-2.0"
        f.attrs["spin_basis"] = "chieff_reference"
        f.attrs["n_detected"] = n
        f.attrs["ndraw"] = 5000
        f.attrs["T_obs_yr"] = 2.6
        f.attrs["n_campaigns"] = 2
        f.attrs["campaign_ndraws"] = np.asarray([2000, 3000], dtype=np.int64)
        f.attrs["pdraw_state"] = state
        f.attrs["spin_reference_amax"] = A_REF
        f.attrs["spin_reference_excluded_pdraw"] = SENTINEL
        f.attrs["spin_reference_excluded_rows"] = declared
        f.attrs["spin_reference_coverage_ok"] = coverage_ok
        f.create_dataset("m1det", data=m1)
        f.create_dataset("m2det", data=m2)
        f.create_dataset("q", data=m2 / m1)
        f.create_dataset("dL", data=np.linspace(400.0, 1400.0, n))
        f.create_dataset("ra", data=np.linspace(0.2, 1.1, n))
        f.create_dataset("dec", data=np.linspace(-0.3, 0.3, n))
        f.create_dataset(
            "chieff",
            data=np.linspace(-0.3, 0.3, n) if chieff is None else chieff,
        )
        f.create_dataset("a1", data=a1)
        f.create_dataset("a2", data=a2)
        f.create_dataset("cost1", data=np.full(n, 0.5))
        f.create_dataset("cost2", data=np.full(n, -0.2))
        f.create_dataset("pdraw", data=pdraw)


def test_reference_selection_loads_in_chieff_basis_without_zero_weight_rows(tmp_path):
    path = tmp_path / "sel.h5"
    _write_reference_selection(path)
    sel = load_selection(path)

    assert sel.basis.identity == basis_for_spin("chieff").identity
    assert sel.n_selected == 4
    np.testing.assert_allclose(np.exp(sel.log_draw_density), [0.20, 0.40, 0.50, 0.80])
    np.testing.assert_allclose(sel.samples["a1"], A1[[0, 2, 3, 5]])
    assert sel.campaigns[0].n_draw == 5000
    reference = sel.metadata["spin_reference"]
    assert reference["spin_reference_amax"] == A_REF
    assert reference["n_detected_in_export"] == 6
    assert reference["n_zero_weight_rows_removed"] == 2
    assert sel.metadata["spin_basis"] == "chieff_reference"
    assert sel.estimator_semantics == GWCAT_CHIEFF_REFERENCE_PDRAW_STATE


def test_reference_row_removal_leaves_estimator_ready_sum_unchanged(tmp_path):
    path = tmp_path / "sel.h5"
    _write_reference_selection(path)
    sel = load_selection(path)
    p_pop = np.asarray([0.3, 0.9, 0.5, 0.7, 0.4, 0.2])
    direct = np.sum(p_pop / PDRAW)  # sentinel rows contribute ~1e-300
    kept = p_pop[[0, 2, 3, 5]]
    adapted = np.sum(np.exp(np.log(kept) - sel.log_draw_density))
    np.testing.assert_allclose(adapted, direct, rtol=1e-14)


def test_reference_pair_loads_when_every_pe_ceiling_equals_reference(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    _write_pe(pe_path)
    _write_reference_selection(sel_path)
    pe, sel = load_pair(pe_path, sel_path)
    assert pe.basis.identity == sel.basis.identity
    assert pe.metadata["chi_eff_amax"]["amax_1_per_event"] == [A_REF, A_REF]


def test_reference_pair_rejects_pe_ceiling_different_from_reference(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    _write_pe(pe_path, amax=(A_REF, 0.998))
    _write_reference_selection(sel_path)
    with pytest.raises(DataContractError, match="spin_reference_amax"):
        load_pair(pe_path, sel_path)


def test_reference_pair_rejects_fallback_or_caller_pe_ceilings(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    _write_pe(pe_path, sources=["analytic", "fallback"])
    _write_reference_selection(sel_path)
    with pytest.raises(DataContractError, match="own sampling prior"):
        load_pair(pe_path, sel_path)


def test_reference_pair_rejects_unrecognized_pe_spin_prior(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    _write_pe(pe_path, unrecognized=["GWTEST_B"])
    _write_reference_selection(sel_path)
    with pytest.raises(DataContractError, match="recognized"):
        load_pair(pe_path, sel_path)


def test_reference_pair_requires_recorded_pe_ceilings(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    _write_pe(pe_path, record_amax=False)
    _write_reference_selection(sel_path)
    with pytest.raises(DataContractError, match="per-event chi_eff prior ceilings"):
        load_pair(pe_path, sel_path)


def test_reference_selection_pairs_only_with_chieff_pe(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    _write_pe(pe_path, spin_basis="component")
    _write_reference_selection(sel_path)
    with pytest.raises(BasisMismatchError):
        load_pair(pe_path, sel_path)


def test_reference_selection_sentinel_count_must_match_declaration(tmp_path):
    path = tmp_path / "sel.h5"
    _write_reference_selection(path, declared=1)
    with pytest.raises(DataContractError, match="sentinel rows"):
        load_selection(path)


def test_reference_selection_outside_support_row_must_carry_sentinel(tmp_path):
    path = tmp_path / "sel.h5"
    pdraw = PDRAW.copy()
    pdraw[4] = 0.33  # a2 > a_ref but a finite density
    _write_reference_selection(path, pdraw=pdraw)
    with pytest.raises(DataContractError, match="outside the declared reference"):
        load_selection(path)


def test_reference_selection_sentinel_inside_support_is_rejected(tmp_path):
    path = tmp_path / "sel.h5"
    pdraw = PDRAW.copy()
    pdraw[2] = SENTINEL  # a1, a2 and chi_eff inside the reference support
    _write_reference_selection(path, pdraw=pdraw)
    with pytest.raises(DataContractError, match="inside the declared reference"):
        load_selection(path)


def test_reference_selection_coverage_hole_is_rejected(tmp_path):
    path = tmp_path / "sel.h5"
    _write_reference_selection(path, coverage_ok=False)
    with pytest.raises(DataContractError, match="coverage hole"):
        load_selection(path)


def test_reference_selection_requires_reviewed_pdraw_state(tmp_path):
    path = tmp_path / "sel.h5"
    _write_reference_selection(path, state="hand-edited component pdraw")
    with pytest.raises(DataContractError, match="pdraw_state"):
        load_selection(path)


def test_reference_canonicalization_records_pairing_and_is_idempotent(tmp_path):
    pe_export, sel_export = tmp_path / "gwcat_pe.h5", tmp_path / "gwcat_sel.h5"
    output = tmp_path / "canonical"
    _write_pe(pe_export)
    _write_reference_selection(sel_export)

    report = canonicalize_gwcat_v2_pair(
        pe_export,
        sel_export,
        output,
        required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference",
    )
    assert report["required_selection_spin_basis"] == "chieff_reference"
    pairing = report["selection_reference_pairing"]
    assert pairing["spin_reference_amax"] == A_REF
    assert pairing["n_selection_zero_weight_rows_removed"] == 2
    assert report["n_selected_injections"] == 4
    assert "2 declared zero-weight rows" in report["denominator_contract"]["selection"]

    selection = SelectionCatalog.from_hdf5(output / "selection.h5")
    assert selection.n_selected == 4
    assert selection.metadata["spin_reference"]["spin_reference_amax"] == A_REF

    again = canonicalize_gwcat_v2_pair(
        pe_export,
        sel_export,
        output,
        required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference",
    )
    assert again == report
    with pytest.raises(ValueError, match="selection spin-basis requirement differs"):
        canonicalize_gwcat_v2_pair(
            pe_export, sel_export, output, required_spin_basis="chieff"
        )


def test_reference_selection_requires_explicit_canonicalization_basis(tmp_path):
    pe_export, sel_export = tmp_path / "gwcat_pe.h5", tmp_path / "gwcat_sel.h5"
    _write_pe(pe_export)
    _write_reference_selection(sel_export)
    with pytest.raises(ValueError, match="does not match explicit requirement"):
        canonicalize_gwcat_v2_pair(
            pe_export, sel_export, tmp_path / "canonical", required_spin_basis="chieff"
        )


def test_canonicalization_refuses_unpaired_selection_basis_request(tmp_path):
    pe_export, sel_export = tmp_path / "gwcat_pe.h5", tmp_path / "gwcat_sel.h5"
    _write_pe(pe_export, spin_basis="component")
    _write_reference_selection(sel_export)
    with pytest.raises(ValueError, match="reference basis paired with it"):
        canonicalize_gwcat_v2_pair(
            pe_export,
            sel_export,
            tmp_path / "canonical",
            required_spin_basis="component",
            required_selection_spin_basis="chieff_reference",
        )


def test_legacy_canonicalization_report_is_still_accepted_on_rerun(tmp_path):
    pe_export, sel_export = tmp_path / "gwcat_pe.h5", tmp_path / "gwcat_sel.h5"
    output = tmp_path / "canonical"
    _write_pe(pe_export)
    _write_reference_selection(sel_export)
    canonicalize_gwcat_v2_pair(
        pe_export,
        sel_export,
        output,
        required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference",
    )
    report_path = output / "canonicalization_report.json"
    legacy = json.loads(report_path.read_text())
    legacy["format_version"] = "gwpop-search-gwcat-canonicalization-1.0"
    del legacy["required_selection_spin_basis"]
    report_path.write_text(json.dumps(legacy))
    # a 1.0 report declared one basis for both halves, so it cannot vouch for
    # a reference selection: the rerun must refuse rather than reinterpret it
    with pytest.raises(ValueError, match="selection spin-basis requirement differs"):
        canonicalize_gwcat_v2_pair(
            pe_export,
            sel_export,
            output,
            required_spin_basis="chieff",
            required_selection_spin_basis="chieff_reference",
        )


def test_legacy_plain_pair_report_is_accepted_on_identical_rerun(tmp_path):
    pe_export, sel_export = tmp_path / "gwcat_pe.h5", tmp_path / "gwcat_sel.h5"
    output = tmp_path / "canonical"
    _write_pe(pe_export, record_amax=False)
    _write_reference_selection(sel_export)
    with h5py.File(sel_export, "a") as f:  # a plain substituting chieff export
        f.attrs["spin_basis"] = "chieff"
        f.attrs["pdraw_state"] = "fixture estimator-ready density"
        del f["pdraw"]
        f.create_dataset("pdraw", data=np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.8]))
    canonicalize_gwcat_v2_pair(
        pe_export, sel_export, output, required_spin_basis="chieff"
    )
    report_path = output / "canonicalization_report.json"
    legacy = json.loads(report_path.read_text())
    legacy["format_version"] = "gwpop-search-gwcat-canonicalization-1.0"
    del legacy["required_selection_spin_basis"]
    del legacy["selection_reference_pairing"]
    report_path.write_text(json.dumps(legacy))

    again = canonicalize_gwcat_v2_pair(
        pe_export, sel_export, output, required_spin_basis="chieff"
    )
    assert again == legacy
