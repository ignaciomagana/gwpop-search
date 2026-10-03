"""Synthetic gwcat v2r2-shaped exports shared by the v2 adapter tests.

They mirror the attrs of the approved v2r2 gwcat exports (gwcat 8263ae9): a
sky-marginal ``chieff_reference`` cumulative O1-O4b mixture and a ``chieff``
PE export with PR-B per-prior provenance, including the six LALInference
events (spin ``config_file_declared``, mass ``assumed_default``).
"""

import json

import h5py
import numpy as np

from gwpop_search.data.adapters.gwcat_v2 import (
    GWCAT_CHIEFF_REFERENCE_PDRAW_STATE,
    load_pe,
    load_selection,
)

A_REF = 0.99
SENTINEL = 1e300
Z_MAX = 1.9
JULIAN_YEAR_S = 365.25 * 86400.0

LALINFERENCE_SIX = (
    "GW170608_020116",
    "GW190707_093326",
    "GW190720_000836",
    "GW190725_174728",
    "GW190728_064510",
    "GW190924_021846",
)
EVENTS = ("GW150914_095045",) + LALINFERENCE_SIX + ("GW230601_224134", "GW230704_212616")
SPIN_KINDS = ("sibling_inherited",) + ("config_file_declared",) * 6 + ("own_analytic",) * 2
MASS_KINDS = ("sibling_inherited",) + ("assumed_default",) * 6 + ("own_analytic",) * 2
OD7_LIST = {n: k for n, k in zip(EVENTS, SPIN_KINDS) if k != "own_analytic"}
Z_ALLOW = {"GW230704_212616": "1.92% of PE samples above z = 1.9"}
MEASURED_Z_FRACTIONS = {n: 0.0 for n in EVENTS} | {"GW230704_212616": 0.0192}
NSAMP = 4

RUNS = ("O1", "O2", "O3a", "O3b", "O4a", "O4b")
COMPONENT_OF_RUN = ("O1", "O2", "O3", "O3", "O4", "O4")
COMPONENTS = ("O1", "O2", "O3", "O4")
TDEF_COMPONENT = (
    "coincident_livetime_semianalytic",
    "coincident_livetime_semianalytic",
    "endo3_analysis_time",
    "monthly_wall_clock",
)
TDEF_RUN = (TDEF_COMPONENT[0], TDEF_COMPONENT[1]) + (TDEF_COMPONENT[2],) * 2 + (
    TDEF_COMPONENT[3],
) * 2
SNR_COL = "semianalytic_observed_phase_maximized_snr_net"
RULES = (
    f"{SNR_COL}>10",
    f"{SNR_COL}>10",
    "min(o3_cwb_far,o3_gstlal_far)<1",
    "min(o3_cwb_far,o3_gstlal_far)<1",
    "min(o4a_gstlal_far,o4a_pycbc_far)<1",
    "min(o4b_gstlal_far)<1",
)
N_COMPONENT = (100.0, 200.0, 300.0, 400.0)
T_COMPONENT = (1.0e6, 2.0e6, 3.0e6, 4.0e6)
NDRAW = 1000
TOBS_YR = sum(T_COMPONENT) / JULIAN_YEAR_S

# 12 detected rows, 2 per run; rows 3 and 8 lie outside the reference support
SEL_A1 = np.asarray([0.2, 0.3, 0.4, 0.995, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
SEL_A2 = np.asarray([0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.999, 0.2, 0.3, 0.4])
SEL_PDRAW = np.asarray(
    [0.2, 0.3, 0.4, SENTINEL, 0.6, 0.7, 0.8, 0.9, SENTINEL, 1.1, 1.2, 1.3]
)


def _str(values):
    return np.asarray(list(values), dtype=h5py.string_dtype("utf-8"))


def write_v2_selection(path, **overrides):
    n = SEL_PDRAW.size
    m1 = np.linspace(25.0, 60.0, n)
    m2 = np.linspace(12.0, 30.0, n)
    attrs = {
        "format_version": "gwcat-selection-2.0",
        "spin_basis": "chieff_reference",
        "n_detected": n,
        "ndraw": NDRAW,
        "T_obs_yr": TOBS_YR,
        "n_campaigns": 1,
        "campaign_ndraws": np.asarray([NDRAW], dtype=np.int64),
        "campaign_kind_per_campaign": _str(["cumulative_mixture"]),
        "pdraw_state": GWCAT_CHIEFF_REFERENCE_PDRAW_STATE,
        "spin_reference_amax": A_REF,
        "spin_reference_excluded_pdraw": SENTINEL,
        "spin_reference_excluded_rows": 2,
        "spin_reference_coverage_ok": True,
        "spin_reference_coverage_per_run": np.ones(6, dtype=bool),
        "spin_reference_coverage_bound_per_run": np.full(6, 0.998),
        "chi_eff_prior_impl": "exact",
        "z_max": Z_MAX,
        "sky_marginalized": True,
        "sky_position_available": np.asarray([False]),
        "cosmology_override_used": False,
        "cumulative_mixture": True,
        "run_labels": _str(RUNS),
        "n_rows_per_run": np.full(6, 10, dtype=np.int64),
        "n_detected_per_run": np.full(6, 2, dtype=np.int64),
        "n_detected_mixture": n,
        "n_passing_detection_rule_per_run": np.asarray([2, 2, 2, 2, 3, 2]),
        "detection_rule_per_run": _str(RULES),
        "T_definition_per_run": _str(TDEF_RUN),
        "component_of_run": _str(COMPONENT_OF_RUN),
        "mixture_components": _str(COMPONENTS),
        "mixture_component_runs": json.dumps(
            {"O1": ["O1"], "O2": ["O2"], "O3": ["O3a", "O3b"], "O4": ["O4a", "O4b"]}
        ),
        "semianalytic_components": _str(["O1", "O2"]),
        "N_per_component": np.asarray(N_COMPONENT),
        "T_per_component_s": np.asarray(T_COMPONENT),
        "T_definition_per_component": _str(TDEF_COMPONENT),
        "N_integrality_residual_per_semianalytic_component": np.asarray([1e-7, -1e-7]),
        "mixture_bookkeeping_status": "derived_from_weights_and_anchors",
        "z_draw_max_per_run": np.asarray([1.3, 1.66, 1.896, 1.888, 2.997, 2.999]),
        "significance_columns": _str(
            [
                "o3_cwb_far",
                "o3_gstlal_far",
                "o4a_gstlal_far",
                "o4a_pycbc_far",
                "o4b_gstlal_far",
            ]
        ),
        "significance_snr_column": SNR_COL,
        "significance_snr_threshold": 10.0,
        "significance_far_threshold": 1.0,
    }
    datasets = {
        "m1det": m1,
        "m2det": m2,
        "q": m2 / m1,
        "dL": np.linspace(400.0, 9000.0, n),
        "m1src": m1 / 1.5,
        "m2src": m2 / 1.5,
        "redshift": np.linspace(0.1, 1.85, n),
        "chieff": np.linspace(-0.3, 0.3, n),
        "a1": SEL_A1,
        "a2": SEL_A2,
        "cost1": np.full(n, 0.5),
        "cost2": np.full(n, -0.2),
        "pdraw": SEL_PDRAW,
    }
    for key, value in overrides.items():
        if key.startswith("ds_"):
            name = key[3:]
            if value is None:
                datasets.pop(name, None)
            else:
                datasets[name] = value
        elif value is None:
            attrs.pop(key, None)
        else:
            attrs[key] = value
    with h5py.File(path, "w") as f:
        for key, value in attrs.items():
            f.attrs[key] = value
        for name, value in datasets.items():
            f.create_dataset(name, data=value)


def write_v2_pe(path, **overrides):
    nobs = len(EVENTS)
    n = nobs * NSAMP
    m1 = np.linspace(30.0, 60.0, n)
    m2 = np.linspace(15.0, 30.0, n)
    cut = np.zeros(nobs, dtype=np.int64)
    cut[EVENTS.index("GW230704_212616")] = 400
    attrs = {
        "format_version": "gwcat-pe-2.0",
        "spin_basis": "chieff",
        "nobs": nobs,
        "nsamp": NSAMP,
        "event_names": _str(EVENTS),
        "chi_eff_amax_1_per_event": np.full(nobs, A_REF),
        "chi_eff_amax_2_per_event": np.full(nobs, A_REF),
        "chi_eff_amax_source_per_event": _str(SPIN_KINDS),
        "chi_eff_amax_resolution_per_event": _str(["analytic"] * nobs),
        "chi_eff_amax_mode": "per_event",
        "spin_prior_unrecognized_events": _str([]),
        "prior_source_kind_spin_per_event": _str(SPIN_KINDS),
        "prior_source_kind_mass_per_event": _str(MASS_KINDS),
        "prior_source_kind_dL_per_event": _str(["release_reweighted"] * 7 + ["own_analytic"] * 2),
        "spin_prior_non_own_analytic_events": _str(list(OD7_LIST)),
        "spin_prior_assumed_events": _str([]),
        "mass_prior_unverified_events": _str(LALINFERENCE_SIX),
        "chi_eff_prior_impl": "exact",
        "dL_prior_impl_per_event": _str(["exact"] * nobs),
        "z_max": Z_MAX,
        "n_samples_cut_by_z_max": cut,
        "n_unique_samples_per_event": np.full(nobs, NSAMP, dtype=np.int64),
        "n_dropped_spin_above_ceiling_per_event": np.zeros(nobs, dtype=np.int64),
        "writer_commit": "8263ae9ca9137ec1b9e896f909c30bbc0bbe7cf6",
    }
    datasets = {
        "m1det": m1,
        "m2det": m2,
        "q": m2 / m1,
        "dL": np.linspace(300.0, 6000.0, n),
        "ra": np.linspace(0.1, 6.0, n),
        "dec": np.linspace(-1.0, 1.0, n),
        "m1src": m1 / 1.4,
        "m2src": m2 / 1.4,
        "redshift": np.linspace(0.05, 1.8, n),
        "chieff": np.linspace(-0.2, 0.3, n),
        "p_pe": np.linspace(0.5, 1.5, n),
    }
    for key, value in overrides.items():
        if key.startswith("ds_"):
            name = key[3:]
            if value is None:
                datasets.pop(name, None)
            else:
                datasets[name] = value
        elif value is None:
            attrs.pop(key, None)
        else:
            attrs[key] = value
    with h5py.File(path, "w") as f:
        for key, value in attrs.items():
            f.attrs[key] = value
        for name, value in datasets.items():
            f.create_dataset(name, data=value)


def loaded(pe_path, sel_path):
    sel = load_selection(sel_path)
    pe = load_pe(pe_path, sky_marginal=bool(sel.metadata["sky_marginal"]))
    return pe, sel
