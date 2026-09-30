"""v2 adapter (BUILD_PLAN Section 4.4): sky-marginal basis, cumulative-mixture
attrs, the GW-40c spin-prior allow-list (OD-7) and exact priors (GW-41)."""

import json

import numpy as np
import pytest
from gwcat_v2_fixtures import (
    COMPONENT_OF_RUN,
    EVENTS,
    LALINFERENCE_SIX,
    N_COMPONENT,
    OD7_LIST,
    RULES,
    RUNS,
    SNR_COL,
    SPIN_KINDS,
    TDEF_COMPONENT,
    TDEF_RUN,
    _str,
    write_v2_pe,
    write_v2_selection,
)
from gwcat_v2_fixtures import (
    loaded as _loaded,
)

from gwpop_search.data import (
    BasisMismatchError,
    DataContractError,
    MissingCoordinateError,
    PosteriorCatalog,
    SelectionCatalog,
    validate_pair,
)
from gwpop_search.data.adapters.gwcat_v2 import (
    basis_for_spin,
    load_pair,
    load_pe,
    load_selection,
    load_spin_prior_allow_list,
    validate_exact_priors,
    validate_reference_pairing,
    validate_spin_prior_sources,
)


@pytest.fixture
def v2_pair(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path)
    write_v2_selection(sel_path)
    return pe_path, sel_path


# --------------------------------------------------------------------- sky


def test_sky_marginal_basis_has_no_sky_and_its_own_identity():
    marginal = basis_for_spin("chieff", sky_marginal=True)
    resolved = basis_for_spin("chieff")
    assert marginal.coordinates == ("m1_detector", "q", "luminosity_distance", "chi_eff")
    assert "dOmega" not in marginal.density_measure
    assert "dOmega" in resolved.density_measure
    assert marginal.identity != resolved.identity
    assert marginal.name == "gwcat_v2_chieff_sky_marginal"
    component = basis_for_spin("component", sky_marginal=True)
    assert component.coordinates[3:] == ("a1", "a2", "cos_tilt1", "cos_tilt2")


def test_sky_marginalized_selection_loads_and_pe_follows_it(v2_pair):
    pe_path, sel_path = v2_pair
    sel = load_selection(sel_path)
    assert sel.metadata["sky_marginal"] is True
    assert sel.metadata["sky_mode_reason"] == "sky_marginalized"
    assert "ra" not in sel.samples and "dec" not in sel.samples
    assert sel.basis.identity == basis_for_spin("chieff", sky_marginal=True).identity
    pe, sel2 = load_pair(pe_path, sel_path, spin_prior_allow_list=OD7_LIST)
    assert pe.basis.identity == sel2.basis.identity
    assert "ra" not in pe.samples and "dec" not in pe.samples
    assert pe.metadata["sky_columns_dropped"] == ["ra", "dec"]


def test_sky_marginalized_selection_that_ships_a_sky_is_refused(tmp_path):
    path = tmp_path / "sel.h5"
    write_v2_selection(path, ds_ra=np.zeros(12), ds_dec=np.zeros(12))
    with pytest.raises(DataContractError, match="must omit the sky columns"):
        load_selection(path)


def test_sky_position_unavailable_everywhere_is_sky_marginal(tmp_path):
    path = tmp_path / "sel.h5"
    write_v2_selection(
        path,
        sky_marginalized=None,
        ds_ra=np.full(12, np.nan),
        ds_dec=np.full(12, np.nan),
    )
    sel = load_selection(path)
    assert sel.metadata["sky_mode_reason"] == "sky_position_available_false"
    assert "ra" not in sel.samples  # the NaN fill is never read


def test_partial_sky_availability_is_refused(tmp_path):
    path = tmp_path / "sel.h5"
    write_v2_selection(
        path,
        sky_marginalized=None,
        n_campaigns=2,
        campaign_kind_per_campaign=_str(["cumulative_mixture", "events"]),
        campaign_ndraws=np.asarray([600, 400]),
        sky_position_available=np.asarray([False, True]),
        ds_ra=np.zeros(12),
        ds_dec=np.zeros(12),
    )
    with pytest.raises(DataContractError, match="neither sky-resolved"):
        load_selection(path)


def test_sky_marginal_selection_never_pairs_with_sky_resolved_pe(v2_pair):
    pe_path, sel_path = v2_pair
    sel = load_selection(sel_path)
    pe = load_pe(pe_path)  # sky-resolved basis
    with pytest.raises(BasisMismatchError):
        validate_pair(pe, sel)


def test_sky_marginal_basis_survives_the_canonical_round_trip(tmp_path, v2_pair):
    pe_path, sel_path = v2_pair
    pe, sel = _loaded(pe_path, sel_path)
    pe.to_hdf5(tmp_path / "pe_c.h5")
    sel.to_hdf5(tmp_path / "sel_c.h5")
    pe2 = PosteriorCatalog.from_hdf5(tmp_path / "pe_c.h5")
    sel2 = SelectionCatalog.from_hdf5(tmp_path / "sel_c.h5")
    assert pe2.basis.identity == sel2.basis.identity == sel.basis.identity
    assert sel2.metadata["cumulative_mixture"]["run_labels"] == list(RUNS)


def test_sky_resolved_population_models_fail_closed_on_sky_marginal_data(v2_pair):
    """The current models request ra/dec; on a sky-marginal pair that is a loud
    MissingCoordinateError, never a silent use of absent sky columns."""
    pytest.importorskip("jax")
    from gwpop_search.models.baseline import GwcatChiEffBBHModel

    pe, sel = _loaded(*v2_pair)
    fields = GwcatChiEffBBHModel.required_fields
    with pytest.raises(MissingCoordinateError):
        pe.require(fields)
    with pytest.raises(MissingCoordinateError):
        sel.require(fields)


# ----------------------------------------------------------- mixture attrs


def test_cumulative_mixture_record_is_validated_and_carried(v2_pair):
    sel = load_selection(v2_pair[1])
    record = sel.metadata["cumulative_mixture"]
    assert record["run_labels"] == list(RUNS)
    assert record["component_of_run"] == list(COMPONENT_OF_RUN)
    assert record["N_per_component"] == list(N_COMPONENT)
    assert record["T_definition_per_component"] == list(TDEF_COMPONENT)
    assert record["detection_rule_per_run"] == list(RULES)
    assert record["bookkeeping_derived"] is True
    assert record["n_detected_mixture"] == 12
    assert sel.metadata["campaign_kind_per_campaign"] == ["cumulative_mixture"]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"N_per_component": None}, "are missing"),
        ({"run_labels": _str(["O1", "O2", "O3a", "O3b", "O4a", "O5"])}, "unknown or repeated"),
        ({"n_detected_per_run": np.full(5, 2)}, "has 5 entries"),
        ({"T_per_component_s": np.asarray([1.0, 2.0, 3.0])}, "has 3 entries"),
        (
            {"detection_rule_per_run": _str(("none(semianalytic_rows_excluded)",) + RULES[1:])},
            "exposure bias",
        ),
        (
            {"detection_rule_per_run": _str(("min(o3_cwb_far)<1",) + RULES[1:])},
            "exposure bias",
        ),
        (
            {"detection_rule_per_run": _str(RULES[:4] + ("min(o3_cwb_far,o4a_pycbc_far)<1",) + RULES[5:])},
            "of another run",
        ),
        (
            {"detection_rule_per_run": _str(RULES[:5] + ("min(o4b_gstlal_far)<2",))},
            "FAR threshold",
        ),
        (
            {"detection_rule_per_run": _str((f"{SNR_COL}>9",) + RULES[1:])},
            "significance_snr",
        ),
        (
            {"detection_rule_per_run": _str(RULES[:5] + ("min(o4b_mbta_far)<1",))},
            "significance_columns",
        ),
        ({"T_definition_per_run": _str(TDEF_RUN[:4] + ("endo3_analysis_time",) * 2)}, "is defined as"),
        ({"component_of_run": _str(("O1", "O2", "O3", "O4", "O4", "O4"))}, "defines it as"),
        ({"T_definition_per_component": _str(TDEF_COMPONENT[:3] + ("wall",))}, "unknown T_definition"),
        ({"N_per_component": np.asarray([100.0, 200.0, 300.0, 401.0])}, "!= ndraw"),
        ({"T_per_component_s": np.asarray([1e6, 2e6, 3e6, 4.1e6])}, "!= T_obs_yr"),
        ({"n_detected_per_run": np.asarray([2, 2, 2, 2, 2, 1])}, "do not add up"),
        ({"n_passing_detection_rule_per_run": np.asarray([2, 2, 2, 2, 1, 2])}, "not ordered"),
        ({"cumulative_mixture": False}, "does not record cumulative_mixture=True"),
        ({"campaign_kind_per_campaign": _str(["events"])}, "names no cumulative_mixture"),
        (
            {"mixture_bookkeeping_status": "derived_from_weights_and_anchors",
             "N_per_component": np.full(4, np.nan)},
            "not finite",
        ),
    ],
)
def test_inconsistent_cumulative_mixture_attrs_are_refused(tmp_path, overrides, match):
    path = tmp_path / "sel.h5"
    write_v2_selection(path, **overrides)
    with pytest.raises(DataContractError, match=match):
        load_selection(path)


# ------------------------------------------------- spin-prior allow-list


def test_six_lalinference_events_fail_without_the_allow_list(tmp_path):
    """G14-adapter: the six LALInference events (spin prior config-declared,
    mass prior assumed_default) are refused unless the OD-7 list names them."""
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    only_six = {n: "config_file_declared" for n in LALINFERENCE_SIX}
    kinds = ("own_analytic",) + ("config_file_declared",) * 6 + ("own_analytic",) * 2
    write_v2_pe(
        pe_path,
        chi_eff_amax_source_per_event=_str(kinds),
        prior_source_kind_spin_per_event=_str(kinds),
        spin_prior_non_own_analytic_events=_str(LALINFERENCE_SIX),
    )
    write_v2_selection(sel_path)
    pe, sel = _loaded(pe_path, sel_path)
    with pytest.raises(DataContractError, match=r"config_file_declared \[6\]") as err:
        validate_reference_pairing(pe, sel)
    for name in LALINFERENCE_SIX:
        assert name in str(err.value)
    with pytest.raises(DataContractError, match="not on the spin-prior allow-list"):
        load_pair(pe_path, sel_path)
    out = validate_reference_pairing(pe, sel, spin_prior_allow_list=only_six)
    sources = out["spin_prior_sources"]
    assert sources["spin_prior_source_kind_counts"] == {
        "config_file_declared": 6,
        "own_analytic": 3,
    }
    assert sources["mass_prior_assumed_default_events"] == list(LALINFERENCE_SIX)


def test_assumed_default_spin_priors_fail_even_against_a_config_declared_list(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    kinds = ("sibling_inherited",) + ("assumed_default",) * 6 + ("own_analytic",) * 2
    write_v2_pe(
        pe_path,
        chi_eff_amax_source_per_event=_str(kinds),
        prior_source_kind_spin_per_event=_str(kinds),
        spin_prior_assumed_events=_str(LALINFERENCE_SIX),
    )
    write_v2_selection(sel_path)
    pe, sel = _loaded(pe_path, sel_path)
    with pytest.raises(DataContractError, match=r"assumed_default \[6\]"):
        validate_reference_pairing(pe, sel)
    with pytest.raises(DataContractError, match="DIFFERENT kind"):
        validate_reference_pairing(pe, sel, spin_prior_allow_list=OD7_LIST)
    names_only = list(OD7_LIST)  # a names-only list accepts any kind
    out = validate_reference_pairing(pe, sel, spin_prior_allow_list=names_only)
    assert out["spin_prior_sources"]["spin_prior_source_kind_counts"]["assumed_default"] == 6


def test_od7_allow_list_admits_every_non_own_event_and_must_be_exact(v2_pair):
    pe, sel = _loaded(*v2_pair)
    with pytest.raises(DataContractError, match=r"sibling_inherited \[1\]"):
        validate_reference_pairing(pe, sel)
    out = validate_reference_pairing(
        pe, sel, spin_prior_allow_list=OD7_LIST, require_exact_allow_list=True
    )
    assert out["spin_prior_sources"]["n_non_own_analytic_allow_listed"] == 7
    stale = dict(OD7_LIST) | {"GW230601_224134": "sibling_inherited"}
    assert validate_reference_pairing(pe, sel, spin_prior_allow_list=stale)[
        "spin_prior_sources"
    ]["allow_list_unused_entries"] == ["GW230601_224134"]
    with pytest.raises(DataContractError, match="exactly"):
        validate_reference_pairing(
            pe, sel, spin_prior_allow_list=stale, require_exact_allow_list=True
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"chi_eff_amax_resolution_per_event": _str(["analytic"] * 8 + ["fallback"])}, "resolution"),
        ({"chi_eff_amax_source_per_event": _str(("own_analytic",) * 9)}, "disagrees with"),
        ({"spin_prior_non_own_analytic_events": _str(["GW150914_095045"])}, "non_own_analytic_events"),
        ({"spin_prior_assumed_events": _str(["GW150914_095045"])}, "spin_prior_assumed_events"),
        (
            {"chi_eff_amax_source_per_event": _str(("mystery",) + SPIN_KINDS[1:]),
             "prior_source_kind_spin_per_event": _str(("mystery",) + SPIN_KINDS[1:])},
            "unknown spin prior source",
        ),
        ({"prior_source_kind_spin_per_event": None}, "predates gwcat GW-40c"),
    ],
)
def test_inconsistent_spin_prior_provenance_is_refused(tmp_path, overrides, match):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path, **overrides)
    write_v2_selection(sel_path)
    pe, sel = _loaded(pe_path, sel_path)
    with pytest.raises(DataContractError, match=match):
        validate_reference_pairing(pe, sel, spin_prior_allow_list=OD7_LIST)


def test_legacy_pe_keeps_the_analytic_rule_and_refuses_an_allow_list(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(
        pe_path,
        chi_eff_amax_resolution_per_event=None,
        prior_source_kind_spin_per_event=None,
        chi_eff_amax_source_per_event=_str(["analytic"] * len(EVENTS)),
    )
    write_v2_selection(sel_path)
    pe, sel = _loaded(pe_path, sel_path)
    assert "spin_prior_sources" not in validate_reference_pairing(pe, sel)
    with pytest.raises(DataContractError, match="predates gwcat GW-40c"):
        validate_reference_pairing(pe, sel, spin_prior_allow_list=OD7_LIST)
    with pytest.raises(DataContractError, match="predates gwcat GW-40c"):
        validate_spin_prior_sources(pe)


def test_allow_list_loader_matches_gwcat_forms(tmp_path):
    assert load_spin_prior_allow_list(None) is None
    assert load_spin_prior_allow_list(["A", "B"]) == {"A": None, "B": None}
    assert load_spin_prior_allow_list({"A": {"kind": "x"}, "B": "*"}) == {"A": "x", "B": None}
    js = tmp_path / "list.json"
    js.write_text(json.dumps(OD7_LIST))
    assert load_spin_prior_allow_list(js) == OD7_LIST
    txt = tmp_path / "list.txt"
    txt.write_text("# comment\nA sibling_inherited\nB  # any kind\n\n")
    assert load_spin_prior_allow_list(str(txt)) == {"A": "sibling_inherited", "B": None}


# ----------------------------------------------------------- exact priors


def test_exact_priors_pass_on_the_v2_pair(v2_pair):
    pe, sel = _loaded(*v2_pair)
    out = validate_exact_priors(pe, sel)
    assert out == {
        "pe_chi_eff_prior_impl": "exact",
        "selection_chi_eff_prior_impl": "exact",
        "pe_dL_prior_impl_counts": {"exact": 9},
    }


@pytest.mark.parametrize(
    "pe_overrides, sel_overrides, match",
    [
        ({"chi_eff_prior_impl": "grid"}, {}, "PE chi_eff_prior_impl='grid'"),
        ({"chi_eff_prior_impl": None}, {}, "PE chi_eff_prior_impl=None"),
        ({"dL_prior_impl_per_event": _str(["exact"] * 8 + ["bilby"])}, {}, "non-exact distance"),
        ({"dL_prior_impl_per_event": None}, {}, "no dL_prior_impl_per_event"),
        ({}, {"chi_eff_prior_impl": "grid"}, "selection chi_eff_prior_impl"),
        ({}, {"chi_eff_prior_impl": None}, "selection chi_eff_prior_impl"),
    ],
)
def test_non_exact_priors_are_refused(tmp_path, pe_overrides, sel_overrides, match):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path, **pe_overrides)
    write_v2_selection(sel_path, **sel_overrides)
    pe, sel = _loaded(pe_path, sel_path)
    with pytest.raises(DataContractError, match=match):
        validate_exact_priors(pe, sel)


def test_analytic_dl_prior_counts_as_exact(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path, dL_prior_impl_per_event=_str(["analytic"] * 9))
    write_v2_selection(sel_path)
    pe, sel = _loaded(pe_path, sel_path)
    assert validate_exact_priors(pe, sel)["pe_dL_prior_impl_counts"] == {"analytic": 9}


