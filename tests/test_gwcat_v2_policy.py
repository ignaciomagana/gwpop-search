"""v2 data policy (BUILD_PLAN Sections 4.4, 10-12): G17 redshift support, the
policy object, and the 2.0 canonicalization report (OD-9 sentinel drop)."""

import json

import numpy as np
import pytest
from gwcat_v2_fixtures import (
    EVENTS,
    MASS_KINDS,
    MEASURED_Z_FRACTIONS,
    OD7_LIST,
    SEL_PDRAW,
    SENTINEL,
    Z_ALLOW,
    Z_MAX,
    _str,
    write_v2_pe,
    write_v2_selection,
)
from gwcat_v2_fixtures import (
    loaded as _loaded,
)

from gwpop_search.data import (
    DataContractError,
    GwcatV2DataPolicy,
    PosteriorCatalog,
    SelectionCatalog,
    canonicalize_gwcat_v2_pair,
    evaluate_gwcat_v2_policy,
)
from gwpop_search.data.v2_policy import (
    validate_redshift_support,
    z_of_luminosity_distance,
)


def policy(**overrides):
    payload = {
        "policy_id": "test-v2",
        "z_max": Z_MAX,
        "spin_prior_allow_list": dict(OD7_LIST),
        "z_allow_list": dict(Z_ALLOW),
        "z_fraction_above_zmax": dict(MEASURED_Z_FRACTIONS),
    }
    payload.update(overrides)
    return GwcatV2DataPolicy.from_dict(payload)


@pytest.fixture
def v2_pair(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path)
    write_v2_selection(sel_path)
    return pe_path, sel_path


def test_underived_bookkeeping_loads_but_the_v2_policy_refuses_it(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path)
    write_v2_selection(
        sel_path,
        mixture_bookkeeping_status="unavailable_no_anchor",
        N_per_component=np.full(4, np.nan),
        T_per_component_s=np.full(4, np.nan),
    )
    pe, sel = _loaded(pe_path, sel_path)
    assert sel.metadata["cumulative_mixture"]["bookkeeping_derived"] is False
    with pytest.raises(DataContractError, match="derived per-component exposure"):
        evaluate_gwcat_v2_policy(pe, sel, policy())


# --------------------------------------------------------- redshift (G17)


def _z_check(pe, sel, **kwargs):
    args = {"z_max": Z_MAX, "z_allow_list": Z_ALLOW}
    args.update(kwargs)
    return validate_redshift_support(pe, sel, **args)


def test_g17_allow_list_admits_the_approved_event(v2_pair):
    pe, sel = _loaded(*v2_pair)
    out = _z_check(pe, sel, z_fraction_above_zmax=MEASURED_Z_FRACTIONS)
    assert out["fraction_basis"] == "measured"
    assert out["events_above_threshold"] == [("GW230704_212616", 0.0192)]
    # without measured fractions the export's upper bound still passes here
    bound = _z_check(pe, sel)
    assert bound["fraction_basis"] == "export_upper_bound"
    assert [n for n, _ in bound["events_above_threshold"]] == ["GW230704_212616"]


def test_g17_event_over_threshold_off_the_list_is_refused(v2_pair):
    pe, sel = _loaded(*v2_pair)
    with pytest.raises(DataContractError, match="not on the z allow-list") as err:
        _z_check(pe, sel, z_allow_list={}, z_fraction_above_zmax=MEASURED_Z_FRACTIONS)
    assert "GW230704_212616" in str(err.value)
    with pytest.raises(DataContractError, match="upper bound"):
        _z_check(pe, sel, z_allow_list={})


def test_g17_measured_fractions_refine_the_export_bound(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    cut = np.zeros(len(EVENTS), dtype=np.int64)
    cut[EVENTS.index("GW230704_212616")] = 400
    cut[EVENTS.index("GW230601_224134")] = 1  # bound 1/5 = 20 %, true 0.3 %
    write_v2_pe(pe_path, n_samples_cut_by_z_max=cut)
    write_v2_selection(sel_path)
    pe, sel = _loaded(pe_path, sel_path)
    with pytest.raises(DataContractError, match="GW230601_224134"):
        _z_check(pe, sel)
    fractions = dict(MEASURED_Z_FRACTIONS) | {"GW230601_224134": 0.003}
    out = _z_check(pe, sel, z_fraction_above_zmax=fractions)
    assert out["n_events_with_samples_cut_by_z_max"] == 2


@pytest.mark.parametrize(
    "pe_overrides, sel_overrides, kwargs, match",
    [
        ({"z_max": 2.5}, {}, {}, "PE export z_max=2.5"),
        ({}, {"z_max": 2.5}, {}, "selection export z_max=2.5"),
        ({"z_max": None}, {}, {}, "PE export records no finite z_max"),
        ({"ds_redshift": np.linspace(0.05, 1.95, 36)}, {}, {}, "PE export has samples at z="),
        ({}, {"ds_redshift": np.linspace(0.1, 2.5, 12)}, {}, "selection export has samples"),
        ({"n_samples_cut_by_z_max": None}, {}, {}, "n_samples_cut_by_z_max"),
        ({}, {}, {"z_allow_list": Z_ALLOW | {"GW999999_000000": "x"}}, "not in the catalog"),
        ({}, {}, {"z_allow_list": Z_ALLOW | {"GW230601_224134": "x"}}, "had no samples"),
        ({}, {}, {"z_fraction_above_zmax": {"GW230704_212616": 0.0192}}, "cover exactly"),
    ],
)
def test_g17_contract_violations_are_refused(tmp_path, pe_overrides, sel_overrides, kwargs, match):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path, **pe_overrides)
    write_v2_selection(sel_path, **sel_overrides)
    pe, sel = _loaded(pe_path, sel_path)
    with pytest.raises(DataContractError, match=match):
        _z_check(pe, sel, **kwargs)


def test_population_cosmology_redshift_diagnostic():
    from scipy.integrate import quad

    H0, Om0 = 67.74, 0.3089
    z_true = np.asarray([0.02, 0.2, 1.0, 1.8, 1.9, 2.0])
    d_l = np.asarray(
        [
            (1.0 + z)
            * (299792.458 / H0)
            * quad(lambda x: 1.0 / np.sqrt(Om0 * (1.0 + x) ** 3 + 1.0 - Om0), 0.0, z,
                   epsabs=0.0, epsrel=1e-13)[0]
            for z in z_true
        ]
    )
    np.testing.assert_allclose(
        z_of_luminosity_distance(d_l, H0=H0, Om0=Om0), z_true, rtol=1e-8
    )


def test_population_cosmology_diagnostic_counts_samples_above_zmax(tmp_path):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    d_l = np.linspace(300.0, 6000.0, 36)
    d_l[-1] = 15500.0  # z ~ 1.99 at the population cosmology
    write_v2_pe(pe_path, ds_dL=d_l)
    write_v2_selection(sel_path)
    pe, sel = _loaded(pe_path, sel_path)
    out = _z_check(
        pe,
        sel,
        z_fraction_above_zmax=MEASURED_Z_FRACTIONS,
        population_cosmology={"H0": 67.74, "Om0": 0.3089},
    )
    assert out["n_exported_pe_samples_above_zmax_at_population_cosmology"] == 1
    assert out["per_event_exported_samples_above_zmax_at_population_cosmology"] == {
        "GW230704_212616": 1
    }


# ----------------------------------------------------------------- policy


def test_policy_round_trips_and_hash_is_stable(tmp_path):
    first = policy()
    path = tmp_path / "policy.json"
    first.save(path)
    again = GwcatV2DataPolicy.from_json(path)
    assert again == first
    assert again.policy_hash == first.policy_hash
    assert policy(z_max=2.0).policy_hash != first.policy_hash
    with pytest.raises(DataContractError, match="unknown v2 data policy key"):
        GwcatV2DataPolicy.from_dict({**first.to_dict(), "surprise": 1})
    with pytest.raises(DataContractError, match="format"):
        GwcatV2DataPolicy.from_dict({**first.to_dict(), "format_version": "x"})
    with pytest.raises(DataContractError, match="outside"):
        policy(z_fraction_above_zmax={"A": 1.5})


def test_policy_passes_on_the_v2_pair_and_records_every_check(v2_pair):
    pe, sel = _loaded(*v2_pair)
    checks = evaluate_gwcat_v2_policy(pe, sel, policy())
    assert set(checks) == {
        "reference_pairing",
        "exact_priors",
        "sky",
        "cumulative_mixture",
        "redshift_support",
    }
    assert checks["sky"]["sky_marginal"] is True
    assert checks["reference_pairing"]["n_selection_zero_weight_rows_removed"] == 2


@pytest.mark.parametrize(
    "pe_overrides, sel_overrides, policy_overrides, match",
    [
        ({}, {}, {"spin_prior_allow_list": {}}, "not on the spin-prior allow-list"),
        ({"chi_eff_prior_impl": "grid"}, {}, {}, "chi_eff_prior_impl"),
        ({}, {"sky_marginalized": False, "sky_position_available": np.asarray([True]),
              "ds_ra": np.zeros(12), "ds_dec": np.zeros(12)}, {}, "sky-marginal pair"),
        ({}, {"cumulative_mixture": None, "campaign_kind_per_campaign": _str(["events"])}, {},
         "cumulative-mixture selection"),
        ({}, {}, {"z_max": 2.5}, "z_max"),
    ],
)
def test_policy_refuses_each_violation(tmp_path, pe_overrides, sel_overrides, policy_overrides, match):
    pe_path, sel_path = tmp_path / "pe.h5", tmp_path / "sel.h5"
    write_v2_pe(pe_path, **pe_overrides)
    write_v2_selection(sel_path, **sel_overrides)
    pe, sel = _loaded(pe_path, sel_path)
    with pytest.raises(DataContractError, match=match):
        evaluate_gwcat_v2_policy(pe, sel, policy(**policy_overrides))


# --------------------------------------------------------- canonicalization


def test_v2_canonicalization_drops_sentinels_and_records_the_policy(tmp_path, v2_pair):
    pe_path, sel_path = v2_pair
    out = tmp_path / "canonical"
    pol = policy()
    report = canonicalize_gwcat_v2_pair(
        pe_path,
        sel_path,
        out,
        required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference",
        policy=pol,
    )
    assert report["format_version"] == "gwpop-search-gwcat-canonicalization-2.0"
    assert report["v2_policy_hash"] == pol.policy_hash
    assert report["v2_policy"] == json.loads(json.dumps(pol.to_dict()))
    assert report["sky_marginal"] is True
    assert report["spin_prior_allow_list"] == dict(sorted(OD7_LIST.items()))
    assert report["n_selected_injections"] == 10
    assert report["selection_reference_pairing"]["n_selection_zero_weight_rows_removed"] == 2
    assert report["coordinate_basis"]["name"] == "gwcat_v2_chieff_sky_marginal"
    sel = SelectionCatalog.from_hdf5(out / "selection.h5")
    np.testing.assert_allclose(
        np.exp(sel.log_draw_density), SEL_PDRAW[SEL_PDRAW != SENTINEL]
    )
    pe = PosteriorCatalog.from_hdf5(out / "pe.h5")
    assert pe.basis.identity == sel.basis.identity == report["coordinate_basis_identity"]
    assert pe.metadata["prior_provenance"]["prior_source_kind_mass_per_event"] == list(MASS_KINDS)

    again = canonicalize_gwcat_v2_pair(
        pe_path, sel_path, out, required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference", policy=pol,
    )
    assert again == report
    with pytest.raises(ValueError, match="different v2 data policy"):
        canonicalize_gwcat_v2_pair(
            pe_path, sel_path, out, required_spin_basis="chieff",
            required_selection_spin_basis="chieff_reference", policy=policy(z_fraction_threshold=0.02),
        )
    with pytest.raises(ValueError, match="different v2 data policy"):
        canonicalize_gwcat_v2_pair(
            pe_path, sel_path, out, required_spin_basis="chieff",
            required_selection_spin_basis="chieff_reference",
            spin_prior_allow_list=OD7_LIST,
        )


def test_v2_canonicalization_without_allow_list_refuses_and_writes_nothing(tmp_path, v2_pair):
    pe_path, sel_path = v2_pair
    out = tmp_path / "canonical"
    with pytest.raises(DataContractError, match="spin-prior allow-list"):
        canonicalize_gwcat_v2_pair(
            pe_path, sel_path, out, required_spin_basis="chieff",
            required_selection_spin_basis="chieff_reference",
        )
    assert not any(out.iterdir())


def test_v2_canonicalization_with_a_plain_allow_list_is_recorded(tmp_path, v2_pair):
    pe_path, sel_path = v2_pair
    out = tmp_path / "canonical"
    report = canonicalize_gwcat_v2_pair(
        pe_path, sel_path, out, required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference",
        spin_prior_allow_list=OD7_LIST,
    )
    assert report["v2_policy_hash"] is None
    assert report["spin_prior_allow_list"] == dict(sorted(OD7_LIST.items()))
    with pytest.raises(ValueError, match="different spin-prior allow-list"):
        canonicalize_gwcat_v2_pair(
            pe_path, sel_path, out, required_spin_basis="chieff",
            required_selection_spin_basis="chieff_reference",
            spin_prior_allow_list=OD7_LIST | {"GW230601_224134": None},
        )
    with pytest.raises(ValueError, match="through the v2 policy"):
        canonicalize_gwcat_v2_pair(
            pe_path, sel_path, tmp_path / "other", required_spin_basis="chieff",
            required_selection_spin_basis="chieff_reference",
            policy=policy(), spin_prior_allow_list=OD7_LIST,
        )


def test_legacy_1_1_report_cannot_vouch_for_a_v2_policy(tmp_path, v2_pair):
    pe_path, sel_path = v2_pair
    out = tmp_path / "canonical"
    canonicalize_gwcat_v2_pair(
        pe_path, sel_path, out, required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference", policy=policy(),
    )
    report_path = out / "canonicalization_report.json"
    legacy = json.loads(report_path.read_text())
    legacy["format_version"] = "gwpop-search-gwcat-canonicalization-1.1"
    for key in ("v2_policy", "v2_policy_hash", "v2_policy_checks", "spin_prior_allow_list", "sky_marginal"):
        legacy.pop(key)
    report_path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="predates the v2 data policy"):
        canonicalize_gwcat_v2_pair(
            pe_path, sel_path, out, required_spin_basis="chieff",
            required_selection_spin_basis="chieff_reference", policy=policy(),
        )
    again = canonicalize_gwcat_v2_pair(
        pe_path, sel_path, out, required_spin_basis="chieff",
        required_selection_spin_basis="chieff_reference",
    )
    assert again == legacy


def test_cli_canonicalize_accepts_a_v2_policy(tmp_path, v2_pair, capsys):
    from gwpop_search.cli import build_parser

    pe_path, sel_path = v2_pair
    policy_path = tmp_path / "policy.json"
    policy().save(policy_path)
    args = build_parser().parse_args(
        [
            "canonicalize-gwcat-v2",
            "--pe-export", str(pe_path),
            "--selection-export", str(sel_path),
            "--spin-basis", "chieff",
            "--selection-spin-basis", "chieff_reference",
            "--output-dir", str(tmp_path / "canonical"),
            "--v2-policy", str(policy_path),
        ]
    )
    args.func(args)
    report = json.loads(capsys.readouterr().out)
    assert report["v2_policy_hash"] == policy().policy_hash
