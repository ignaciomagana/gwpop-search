"""The v2 claim table: criteria D1-D6 of the GWTC-5 atom search v2 (plan 2026-09-30).

This evaluates the pre-declared v2 criteria literally from the numbers the
analysis tools report; it computes nothing new from the data. It is the v2
successor of ``runs/gwtc5-bbh-search-v1/prereg_claims.py`` (whose v1 lessons
-- the ``|bias|`` is subtracted, the SDDR band has a ``max(0.5, .)`` floor, the
model-prior clause is the child/parent ratio, a partial suite is never
complete, and DISFAVOURED is implemented -- are kept).

SUPPORTED requires all six:

* **D1 numerics.** Every nested-sampling and evidence check passes for both
  endpoints (the frozen per-model gate file, ``numerics.production_gates``;
  the models were scored at one fidelity rung, see
  :func:`collect_v2_evaluations` for the F3/F4 second-seed rule). The fraction
  of posterior mass inside the variance-taper region must be *reported* for
  both endpoints (``--taper-mass``; D1 is ``incomplete`` without it); the
  tool's G-MC1..5 defaults are reported beside D1, not binding (as in v1).
* **D2 strength.** ``ln BF - 2 sigma_total - |bias| >= 3``; the number of
  atoms tried (the trials count: every distinct evaluated non-root model, so
  a depth-2 model reached by two parent edges counts once) is disclosed next
  to every claim.
* **D3 prior sensitivity.** Halving and doubling each added parameter's prior
  (exact reweighting for the narrowed prior, the Occam relation for the widened
  one, or -- where neither is valid -- an explicit rerun, ``--d3-reruns``)
  keeps ``ln BF >= 1`` with the same sign; every model-prior variant keeps the
  child's probability relative to its parent ``p_c / (p_c + p_p) >= 0.75``;
  and the taper-at-2 rerun (``--taper2``) keeps the sign.
* **D4 SDDR.** For nested edges the Savage-Dickey estimate agrees with the
  evidence ratio within ``max(0.5, 2 sigma)``,
  ``sigma = hypot(sigma_total, sigma_SDDR)``. Family changes are
  ``not_applicable``; a one-sided bound satisfied by the evidence ratio is
  ``bound_consistent``. A disagreement blocks SUPPORTED and is flagged.
* **D5 alternative roots.** The sign of ``ln BF`` holds on both alternative
  roots, A1 = BP2P + beta per mass component and A2 = BP2P + kappa(m1)
  (``--alt-root A1=<suite or scenario summary>``). An edge whose mutation is
  inapplicable on an alternative root (it is already part of that root) is
  ``not_applicable`` there. PSIS-LOO is reported, not binding. Event-drop
  scenarios are deferred (not part of D5 in v2).
* **D6 posterior predictive check** of the claimed model (the child for a
  positive edge): no pre-declared statistic at ``p < 0.01``
  (:mod:`gwpop_search.analysis.ppc`).

DISFAVOURED: ``ln BF + 2 sigma_total + |bias| <= -3`` with D1 passing and the
sign stable under D3 (width variants, and the taper-2 rerun where one exists).
INCONCLUSIVE: everything else.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from gwpop_search.grammar.paths import edge_path_key, mutation_paths
from gwpop_search.hbi.taper import DEFAULT_TAPER_KIND

from ._common import AnalysisInputError, json_ready
from .ppc import PPC_ALPHA, ppc_criterion

CLAIMS_V2_FORMAT = "gwpop-search-v2-claims-1.0"

# --- v2 pre-declared constants (plan 2026-09-30; never edit after the freeze) ---
STRENGTH_THRESHOLD = 3.0          # D2 / DISFAVOURED
STRENGTH_SIGMAS = 2.0             # D2 / DISFAVOURED
WIDTH_MIN_LOG_BF = 1.0            # D3
WIDTH_FACTOR = 2.0                # D3: halving and doubling
MODEL_PRIOR_MIN_RATIO = 0.75      # D3, child relative to its parent
SDDR_FLOOR = 0.5                  # D4: max(0.5, 2 sigma)
SDDR_SIGMAS = 2.0                 # D4
PPC_LEVEL = PPC_ALPHA             # D6
ALT_ROOTS = ("A1", "A2")          # D5
#: D2 (DRAFT, operator approves at freeze): the Monte-Carlo error of a tapered
#: edge is the first-order error of the untapered estimator
#: (``first_order_untapered_mc``); the fluctuation of ln T(sigma^2_hat) is
#: neglected. Under the v2 sharp cut (the LVK form) ln T is 0 or -inf, and the
#: neglected term is the posterior mass that flips in or out of the cut under
#: another Monte-Carlo realisation of sigma^2_hat: the taper region is the band
#: sigma^2 > (1 - 0.05) x threshold (VarianceTaper.sharp_region_band, the
#: assumed 5% relative error of sigma^2_hat). (Smooth alternative: d ln T /
#: d ln sigma^2 reaches -p/2 = -15 at the threshold, ~0.75 nat per 5% error.)
#: Above this posterior taper-region mass (either endpoint) the approximation
#: is not trusted: D2 is ``incomplete`` (and DISFAVOURED is not available).
#: OPEN: the LVK Default posterior has 67% of its mass at sigma^2 > 0.95
#: (staging/v2/TAPER_FORM.md), so an R0 that piles against the cut the same
#: way makes D2 incomplete everywhere at 0.10 (pilot item a-4 decides).
TAPER_MASS_D2_LIMIT = 0.10
ALT_ROOT_DESCRIPTIONS = {
    "A1": "BP2P + beta per mass component (LVK 'Extended' pairing)",
    "A2": "BP2P + kappa(m1)",
}
SUPPORTED, DISFAVOURED, INCONCLUSIVE = "SUPPORTED", "DISFAVOURED", "INCONCLUSIVE"
_D4_OK = ("agree", "not_applicable", "bound_consistent")


def _sign(x: float) -> float:
    return math.copysign(1.0, float(x))


# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------


def d1_numerics(claim: Mapping, *, taper_mass: Mapping[str, float] | None = None) -> dict:
    tool = (claim.get("criteria") or {}).get("numerics") or {}
    gates = tool.get("production_gates")
    gmc_failed = []
    for name, ok in (tool.get("gmc_edge") or {}).items():
        if ok is False:
            gmc_failed.append(name)
    for model_hash, per_model in (tool.get("gmc_models") or {}).items():
        for name, ok in per_model.items():
            if ok is False:
                gmc_failed.append(f"{name}({model_hash[:8]})")
    if not isinstance(gates, list) or len(gates) != 2:
        status = "missing"
    elif any(g is False for g in gates):
        status = "fail"
    elif any(g is None for g in gates):
        status = "missing"
    elif tool.get("rung_homogeneous") is False:
        status = "fail"
    else:
        status = "pass"
    parent, child = claim["parent_hash"], claim["child_hash"]
    masses = None
    if taper_mass is not None:
        masses = {"parent": taper_mass.get(parent), "child": taper_mass.get(child)}
    reported = bool(masses and None not in masses.values())
    reason = None
    if status == "pass" and not reported:
        # D1: "the fraction of posterior mass inside the taper region is reported"
        status, reason = "incomplete", "taper-region posterior mass not reported for both endpoints"
    return {
        "status": status,
        "reason": reason,
        "production_gates": gates,
        "rung_homogeneous": tool.get("rung_homogeneous"),
        "gmc_failed_reported_not_binding": gmc_failed,
        "taper_mass_fraction": masses,
        "taper_mass_reported": reported,
    }


# ---------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------


def d2_strength(edge: Mapping, *, n_atoms_tried: int, taper_mass: Mapping | None = None) -> dict:
    """D2 strength. ``taper_mass`` = ``{"parent": m_p, "child": m_c}`` (D1's block)."""
    lnbf = edge.get("log_bayes_factor")
    if lnbf is None:
        return {"status": "missing", "reason": "no ln BF", "n_atoms_tried": n_atoms_tried}
    sigma_total = edge.get("sigma_total")
    mc = edge.get("mc")
    bias = None if not mc else mc.get("bias")
    if sigma_total is None or not edge.get("mc_error_included") or bias is None:
        return {"status": "incomplete", "log_bayes_factor": float(lnbf), "n_atoms_tried": n_atoms_tried,
                "reason": "no Monte-Carlo error/bias for this edge"}
    sigma_total = float(sigma_total)
    bias = abs(float(bias))
    lower = float(lnbf) - STRENGTH_SIGMAS * sigma_total - bias
    upper = float(lnbf) + STRENGTH_SIGMAS * sigma_total + bias
    status = "pass" if lower >= STRENGTH_THRESHOLD else "fail"
    disfavoured = bool(upper <= -STRENGTH_THRESHOLD)
    reason = None
    masses = [m for m in (taper_mass or {}).values() if m is not None]
    over = bool(masses) and max(masses) > TAPER_MASS_D2_LIMIT
    if over:
        # sigma_total neglects the Monte-Carlo fluctuation of ln T(sigma^2_hat)
        status, disfavoured = "incomplete", False
        reason = (f"posterior taper-region mass {max(masses):.3g} > {TAPER_MASS_D2_LIMIT:g}: the first-order "
                  "(untapered) Monte-Carlo error may understate sigma_total")
    return {
        "status": status,
        "reason": reason,
        "log_bayes_factor": float(lnbf),
        "sigma_total": sigma_total,
        "abs_bias": bias,
        "lower": lower,
        "upper": upper,
        "disfavoured": disfavoured,
        "taper_mass_limit": TAPER_MASS_D2_LIMIT,
        "taper_mass_over_limit": over,
        "n_atoms_tried": int(n_atoms_tried),
    }


# ---------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------


def _edge_matches(row: Mapping, claim: Mapping) -> bool:
    if row.get("parent_hash") and row.get("child_hash"):
        return row["parent_hash"] == claim["parent_hash"] and row["child_hash"] == claim["child_hash"]
    return row.get("mutation_id") == claim["mutation_id"]


def apply_d3_reruns(width: list[dict], claim: Mapping, reruns: Sequence[Mapping]) -> list[dict]:
    """Fill width variants with no value (Occam not valid, or ESS too small) from reruns.

    A rerun row is ``{parent_hash, child_hash | mutation_id, parameter, variant
    ("narrowed"|"widened"), log_bayes_factor, valid, rerun_job}``; ``valid``
    means the rerun passed its frozen numerical gates. Only null entries are
    touched (the rerun path of D3 is for variants the reweighting/Occam routes
    cannot evaluate).
    """
    used = []
    for variant in width:
        if variant.get("log_bayes_factor") is not None:
            continue
        for row in reruns:
            if not _edge_matches(row, claim):
                continue
            if row.get("parameter") != variant.get("parameter"):
                continue
            if row.get("variant", variant.get("variant")) != variant.get("variant"):
                continue
            used.append({"parameter": row["parameter"], "variant": variant.get("variant"),
                         "rerun_job": row.get("rerun_job"), "valid": bool(row.get("valid")),
                         "log_bayes_factor": row.get("log_bayes_factor")})
            if row.get("valid") and row.get("log_bayes_factor") is not None:
                variant["log_bayes_factor"] = float(row["log_bayes_factor"])
                variant["flag"] = "rerun"
            else:
                variant["flag"] = "rerun_invalid"
            break
    return used


def _taper2_row(claim: Mapping, taper2: Sequence[Mapping]) -> Mapping | None:
    for row in taper2:
        if _edge_matches(row, claim):
            return row
    return None


def d3_prior(
    claim: Mapping,
    variants: Sequence[Mapping],
    lnbf: float,
    *,
    reruns: Sequence[Mapping] = (),
    taper2: Sequence[Mapping] = (),
    factor: float | None = None,
) -> dict:
    """Width variants, model-prior variants and the taper-2 sign (see the module docstring).

    For a negative edge the "keeps ln BF >= 1" clause is replaced by sign
    stability (the v2 DISFAVOURED rule; v1 Amendment 1 item 10 settled in the
    spec), the model-prior clause is reported but inapplicable, and a missing
    taper-2 rerun does not block (taper-2 reruns are budgeted for claimed
    edges only).
    """
    criteria = (claim.get("criteria") or {}).get("prior_robustness") or {}
    width = [dict(v) for v in (criteria.get("width_variants") or [])]
    negative = lnbf < 0.0
    out: dict[str, object] = {}
    reruns_used = apply_d3_reruns(width, claim, reruns) if width else []
    if reruns_used:
        out["reruns_used"] = reruns_used
    out["width_variants"] = [
        {"parameter": v.get("parameter"), "variant": v.get("variant"),
         "log_bayes_factor": v.get("log_bayes_factor"), "flag": v.get("flag")}
        for v in width
    ]
    if factor is not None and abs(float(factor) - WIDTH_FACTOR) > 1e-12:
        width_status = "incomplete"
        out["width_note"] = f"prior-sensitivity factor {factor} != {WIDTH_FACTOR} (halving/doubling)"
    elif not width:
        width_status = "missing"
    else:
        sides = {(v.get("parameter"), v.get("variant")) for v in width}
        params = {p for p, _ in sides}
        lacking = sorted(p for p in params if not {(p, "narrowed"), (p, "widened")} <= sides)
        values = [v.get("log_bayes_factor") for v in width]
        if lacking:
            width_status = "incomplete"
            out["width_note"] = f"parameters without both a halved and a doubled variant: {lacking}"
        elif any(v is None for v in values):
            width_status = "incomplete"
        elif any(_sign(v) != _sign(lnbf) for v in values):
            width_status = "fail"
        elif negative:
            width_status = "sign_stable"
        else:
            width_status = "pass" if all(v >= WIDTH_MIN_LOG_BF for v in values) else "fail"
    out["width_status"] = width_status

    ratios = []
    parent, child = claim["parent_hash"], claim["child_hash"]
    for variant in variants:
        probs = variant.get("posterior_model_probabilities") or {}
        p_c, p_p = probs.get(child), probs.get(parent)
        ratio = None
        if p_c is not None and p_p is not None and (p_c + p_p) > 0.0:
            ratio = float(p_c) / (float(p_c) + float(p_p))
        ratios.append({"penalty_per_axis": variant.get("penalty_per_axis"), "child_relative_probability": ratio})
    out["model_prior_variants"] = ratios
    values = [r["child_relative_probability"] for r in ratios]
    if not values or any(v is None for v in values):
        prior_status = "missing"
    elif negative:
        prior_status = "inapplicable_negative_edge"
    else:
        prior_status = "pass" if all(v >= MODEL_PRIOR_MIN_RATIO for v in values) else "fail"
    out["model_prior_status"] = prior_status

    row = _taper2_row(claim, taper2)
    if row is None:
        taper_status = "not_required_negative_edge" if negative else "missing"
        out["taper2"] = None
    elif not row.get("valid", True) or row.get("log_bayes_factor") is None:
        taper_status = "incomplete"
        out["taper2"] = dict(row)
    else:
        taper_lnbf = float(row["log_bayes_factor"])
        taper_status = "pass" if _sign(taper_lnbf) == _sign(lnbf) else "fail"
        out["taper2"] = {"log_bayes_factor": taper_lnbf, "rerun_job": row.get("rerun_job")}
    out["taper2_status"] = taper_status

    statuses = (width_status, prior_status, taper_status)
    if "fail" in statuses:
        status = "fail"
    elif negative:
        # DISFAVOURED needs sign stability: width sign-stable and taper-2 not reversed
        if width_status == "sign_stable" and taper_status in ("pass", "not_required_negative_edge"):
            status = "sign_stable"
        else:
            status = "incomplete" if "incomplete" in statuses else "missing"
    elif all(s == "pass" for s in statuses):
        status = "pass"
    else:
        status = "incomplete" if "incomplete" in statuses else "missing"
    out["status"] = status
    return out


# ---------------------------------------------------------------------------
# D4
# ---------------------------------------------------------------------------


def index_sddr(payload) -> dict:
    """Index ``analyze-sddr`` output (``{"edges": [row, ...]}``) by edge and mutation id."""
    if payload is None:
        return {}
    rows = payload["edges"] if isinstance(payload, Mapping) and "edges" in payload else payload
    if isinstance(rows, Mapping):
        return dict(rows)
    index: dict[str, Mapping] = {}
    mutation_count: dict[str, int] = {}
    for row in rows:
        if row.get("mutation_id"):
            mutation_count[row["mutation_id"]] = mutation_count.get(row["mutation_id"], 0) + 1
    for row in rows:
        parent, child = row.get("parent_hash"), row.get("child_hash")
        if parent and child:
            index[f"{parent}->{child}"] = row
        mid = row.get("mutation_id")
        if mid and mutation_count.get(mid) == 1:
            index[mid] = row
    return index


def d4_sddr(claim: Mapping, edge: Mapping, sddr: Mapping) -> dict:
    check = None
    for key in (f"{edge.get('graph_parent_hash')}->{edge.get('graph_child_hash')}",
                f"{edge['parent_hash']}->{edge['child_hash']}", edge.get("mutation_id")):
        if key and key in sddr:
            check = sddr[key]
            break
    if check is None:
        tool = (claim.get("criteria") or {}).get("sddr") or {}
        if tool.get("status") == "not_applicable":
            return {"status": "not_applicable", "reason": "evidence-only edge (no exact nesting)"}
        return {"status": "missing", "reason": "no SDDR check for this edge"}
    ns = check.get("log_bf_child_over_parent_ns")
    sd = check.get("log_bf_child_over_parent_sddr")
    tool_status = check.get("check_status") or check.get("status")
    if ns is None or sd is None:
        if tool_status == "not_applicable" and check.get("classification") not in ("exact", "approximate"):
            return {"status": "not_applicable", "tool_status": tool_status,
                    "reason": check.get("reason") or "evidence-only edge (no exact nesting)"}
        if tool_status == "bound_consistent":
            return {"status": "bound_consistent", "tool_status": tool_status,
                    "reason": "SDDR is a one-sided bound at the nested value; the evidence ratio satisfies it"}
        if tool_status == "bound_violated":
            return {"status": "flag_investigate", "tool_status": tool_status,
                    "reason": "the evidence ratio violates the one-sided SDDR bound"}
        return {"status": "missing", "tool_status": tool_status, "reason": "SDDR not estimable for this edge"}
    difference = float(ns) - float(sd)
    sigma = math.hypot(float(edge.get("sigma_total") or 0.0), float(check.get("sigma_sddr") or 0.0))
    band = max(SDDR_FLOOR, SDDR_SIGMAS * sigma)
    return {
        "status": "agree" if abs(difference) <= band else "flag_investigate",
        "difference_ns_minus_sddr": difference,
        "band": band,
        "sigma_used": sigma,
        "tool_status": tool_status,
    }


# ---------------------------------------------------------------------------
# D5
# ---------------------------------------------------------------------------


def alt_root_edge_values(summary: Mapping) -> dict[str, object]:
    """``{edge_key: ln BF}`` and the inapplicable keys of one alternative-root summary.

    Accepts a nearby-baseline scenario summary (``root_edge_log_bayes_factors``,
    written by :func:`gwpop_search.validation.baselines.run_nearby_baseline_suite`
    for restricted D5 scenarios), a suite summary with exactly one scenario, or
    a plain ``{"edges": {key: lnbf}, "inapplicable": [...]}`` mapping. Keys
    are root-independent edge keys (:mod:`gwpop_search.grammar.paths`):
    ``"<mutation id>"`` for an edge out of the root, ``"<a>|<b>"`` for the
    edge applying ``b`` to the root + ``a`` node.
    """
    if "scenarios" in summary:
        scenarios = list(summary["scenarios"])
        if len(scenarios) != 1:
            raise AnalysisInputError(
                f"an alternative-root suite summary must hold exactly one scenario; got {len(scenarios)}"
            )
        summary = scenarios[0]
    if "edges" in summary and isinstance(summary["edges"], Mapping):
        values = {str(k): (None if v is None else float(v)) for k, v in summary["edges"].items()}
        return {"values": values, "inapplicable": set(summary.get("inapplicable") or ()),
                "root_hash": summary.get("graph_root_hash")}
    if "root_edge_log_bayes_factors" not in summary:
        raise AnalysisInputError(
            "alternative-root summary carries no root_edge_log_bayes_factors; run the D5 scenario "
            "with a restricted mutation set (validation.baselines.v2_alt_root_scenario)"
        )
    values = {str(k): (None if v is None else float(v)) for k, v in summary["root_edge_log_bayes_factors"].items()}
    restriction = (summary.get("scenario") or {}).get("restriction") or {}
    return {"values": values, "inapplicable": set(summary.get("inapplicable_paths") or restriction.get("inapplicable") or ()),
            "root_hash": summary.get("graph_root_hash")}


def d5_alt_roots(
    claim: Mapping,
    lnbf: float,
    edge_key: str | None,
    alt_roots: Mapping[str, Mapping[str, object]],
    *,
    loo: Mapping | None = None,
) -> dict:
    per_root = {}
    missing = []
    reversed_ = []
    for name in ALT_ROOTS:
        data = alt_roots.get(name)
        if data is None:
            per_root[name] = {"status": "missing"}
            missing.append(name)
            continue
        if edge_key is None:
            per_root[name] = {"status": "missing", "reason": "no mutation path for this edge"}
            missing.append(name)
            continue
        if edge_key in data["inapplicable"]:
            per_root[name] = {"status": "not_applicable", "edge_key": edge_key,
                              "reason": "the mutation is inapplicable on this root (already part of it, "
                                        "or defined only relative to the constant form the root replaces; "
                                        "grammar.v2.v2_d5_atom_semantics)"}
            continue
        value = data["values"].get(edge_key)
        if value is None:
            per_root[name] = {"status": "missing", "edge_key": edge_key}
            missing.append(name)
            continue
        same = _sign(value) == _sign(lnbf)
        per_root[name] = {"status": "pass" if same else "fail", "edge_key": edge_key,
                          "log_bayes_factor": value}
        if not same:
            reversed_.append(name)
    if reversed_:
        status = "fail"
    elif missing:
        status = "incomplete"
    elif all(per_root[n]["status"] == "not_applicable" for n in ALT_ROOTS):
        status = "incomplete"
    else:
        status = "pass"
    loo_detail = None
    if loo is not None:
        for e in loo.get("edges", []):
            if (e.get("parent_hash"), e.get("child_hash")) == (claim["parent_hash"], claim["child_hash"]):
                loo_detail = {
                    "any_sign_reversal": e.get("any_sign_reversal"),
                    "n_flagged_events": e.get("n_flagged_events"),
                    "max_abs_delta_log_bayes_factor": e.get("max_abs_delta_log_bayes_factor"),
                }
    return {
        "status": status,
        "alt_roots": per_root,
        "sign_reversals": reversed_,
        "missing": missing,
        "psis_loo_reported_not_binding": loo_detail,
        "event_drop": "deferred to a follow-up (v2 plan)",
    }


# ---------------------------------------------------------------------------
# D6
# ---------------------------------------------------------------------------


def d6_ppc(claim: Mapping, lnbf: float, ppc: Mapping[str, Mapping]) -> dict:
    """PPC of the claimed model: the child for a positive edge, the parent for a negative one."""
    claimed = claim["child_hash"] if lnbf > 0 else claim["parent_hash"]
    result = ppc_criterion(ppc.get(claimed), alpha=PPC_LEVEL)
    return {"claimed_model": claimed, **result}


# ---------------------------------------------------------------------------
# Paths and the table
# ---------------------------------------------------------------------------


def _edge_key(edge: Mapping, root_hash: str | None, paths: Mapping[str, tuple] | None) -> str | None:
    parent = edge.get("graph_parent_hash") or edge["parent_hash"]
    if paths is not None and parent in paths:
        return edge_path_key(paths[parent], edge["mutation_id"])
    if root_hash is not None and parent == root_hash:
        return str(edge["mutation_id"])
    return None


#: ``DynestyConfig`` fields that do not define the trajectory rung (as
#: :data:`gwpop_search.analysis._common._NON_RUNG_FIELDS`)
_NON_TRAJECTORY_FIELDS = ("sample", "slices", "walks", "batch_size", "seed")
_RUNG_ORDER = {"F3": 0, "F4": 1}


def _trajectory(payload: Mapping) -> dict | None:
    nested = ((payload.get("diagnostics") or {}).get("nested_sampling") or {})
    resolved = nested.get("resolved_dynesty_config")
    if not isinstance(resolved, Mapping):
        return None
    return {k: v for k, v in resolved.items() if k not in _NON_TRAJECTORY_FIELDS}


def collect_v2_evaluations(paths) -> dict[str, dict]:
    """Per-model D1 inputs from the production evaluator's ``evaluation.json`` files.

    ``paths`` are ``evaluation.json`` files or directories searched recursively.
    Returns ``{"gates": {model_hash: passed}, "taper_mass": {model_hash:
    pooled posterior mass fraction inside the taper region}, "fidelity":
    {model_hash: rung}, "files": {model_hash: path}, "superseded": {model_hash:
    path}}``.

    **F3/F4 rule (v2 second seeds).** Every model has one F3 evaluation (one
    dynesty run); decision-relevant edges are re-evaluated at F4 (the
    second-seed rung). A model with both uses its **F4** evaluation for D1
    (its gates include the cross-run checks over the F4 runs) and its taper
    mass; the F3 evaluation is recorded as ``superseded``. This is not a
    mixture of procedures: v2 F3 and F4 runs share one dynesty trajectory
    configuration (nlive, dlogz, bound, caps; checked here when the
    evaluations record it) and differ only in the number of seeds, so the
    evidence analysis pools all of a model's runs as repeats of one rung
    (:func:`gwpop_search.analysis._common.discover_dynesty_results`; the F4
    root seed differs from the F3 one, so the F4 runs are fresh seeds).
    Refused: two evaluations of one model at the same rung, a rung other than
    F3/F4, or F3/F4 evaluations with different trajectory configurations.
    The taper mass is the ``diagnostics.taper.pooled`` block the tapered
    evaluator writes (``summarize_dynesty_fit``); it is absent (not reported)
    for an untapered likelihood.
    """
    per_model: dict[str, dict[str, tuple]] = {}
    for path, payload in _evidence_evaluations(paths):
        model_hash = str(payload["model_hash"])
        rung = str(payload.get("fidelity"))
        if rung not in _RUNG_ORDER:
            raise AnalysisInputError(f"{path}: unsupported evaluation rung {rung!r} (v2 uses F3 and F4)")
        slot = per_model.setdefault(model_hash, {})
        if rung in slot:
            raise AnalysisInputError(
                f"model {model_hash} has more than one {rung} evaluation ({slot[rung][0]}, {path})"
            )
        slot[rung] = (path, payload)
    out: dict[str, dict] = {"gates": {}, "taper_mass": {}, "fidelity": {}, "files": {}, "superseded": {}}
    trajectories: dict[str, dict] = {}
    for model_hash, slot in per_model.items():
        for rung, (path, payload) in slot.items():
            trajectory = _trajectory(payload)
            if trajectory is None:
                continue
            key = json_dumps_sorted(trajectory)
            trajectories.setdefault(key, {"config": trajectory, "files": []})["files"].append(str(path))
        rung = max(slot, key=_RUNG_ORDER.__getitem__)
        path, payload = slot[rung]
        if len(slot) > 1:
            out["superseded"][model_hash] = str(slot["F3"][0])
        diagnostics = payload.get("diagnostics") or {}
        out["files"][model_hash] = str(path)
        out["fidelity"][model_hash] = rung
        out["gates"][model_hash] = bool(diagnostics.get("passed"))
        pooled = ((diagnostics.get("taper") or {}).get("pooled") or {})
        if "posterior_mass_in_taper_region" in pooled:
            out["taper_mass"][model_hash] = float(pooled["posterior_mass_in_taper_region"])
    if len(trajectories) > 1:
        listing = "; ".join(f"{v['config']} ({len(v['files'])} files)" for v in trajectories.values())
        raise AnalysisInputError(
            "the evaluations use more than one dynesty trajectory configuration; F3 and F4 may be "
            f"combined only when they differ in the number of seeds alone: {listing}"
        )
    return out


def json_dumps_sorted(payload) -> str:
    import json

    return json.dumps(payload, sort_keys=True, default=str)


def _evidence_evaluations(paths):
    """``(path, payload)`` of every evidence-rung ``evaluation.json`` under ``paths``."""
    import json
    from pathlib import Path

    files = []
    for item in paths:
        item = Path(item)
        files.extend(sorted(item.rglob("evaluation.json")) if item.is_dir() else [item])
    for path in files:
        payload = json.loads(path.read_text())
        if payload.get("fidelity") == "F0":
            continue
        yield path, payload


#: D3: the taper-at-2 sensitivity rerun (plan 2026-09-30, Numerics)
TAPER2_THRESHOLD = 2.0
#: ... under the same taper form as the primary runs (the LVK sharp cut).
TAPER2_KIND = "sharp"


def taper2_rows_from_evaluations(
    graph, paths, *, threshold: float = TAPER2_THRESHOLD, kind: str | None = TAPER2_KIND
) -> list[dict]:
    """D3 taper-2 rows (``v2-claim-table --taper2`` format) from rerun evaluations.

    ``paths`` hold the ``evaluation.json`` files of the reruns under the
    taper-at-``threshold`` fidelity configuration
    (``v2_fidelity_run_config(2.0)``). Every evaluation must carry the taper
    diagnostic of that threshold and of taper form ``kind`` (a rerun under
    another taper is refused; ``kind=None`` skips the form check). A recorded
    taper without a ``kind`` is read with the :class:`VarianceTaper` default.
    One row per graph edge whose two endpoints were rerun:
    ``log_bayes_factor = ln Z_child - ln Z_parent`` and ``valid`` when both
    evaluations pass their numerical checks.
    """
    runs: dict[str, dict] = {}
    for path, payload in _evidence_evaluations(paths):
        diagnostics = payload.get("diagnostics") or {}
        taper = (((diagnostics.get("taper") or {}).get("pooled") or {}).get("taper") or {})
        if taper.get("threshold") is None or float(taper["threshold"]) != float(threshold):
            raise AnalysisInputError(
                f"{path}: not a taper-at-{threshold:g} evaluation (taper {taper or None})"
            )
        if kind is not None and str(taper.get("kind", DEFAULT_TAPER_KIND)) != kind:
            raise AnalysisInputError(
                f"{path}: taper-at-{threshold:g} evaluation under a {taper.get('kind')!r} taper, "
                f"not the primary {kind!r} form"
            )
        model_hash = str(payload["model_hash"])
        if model_hash in runs:
            raise AnalysisInputError(f"model {model_hash} has more than one taper-2 evaluation")
        evidence = diagnostics.get("evidence") or {}
        runs[model_hash] = {
            "passed": bool(diagnostics.get("passed")),
            "log_evidence": evidence.get("log_evidence_mean"),
            "error": evidence.get("conservative_error"),
            "path": str(path),
        }
    rows = []
    for edge in graph.edges:
        p, c = runs.get(edge.parent_hash), runs.get(edge.child_hash)
        if p is None or c is None:
            continue
        lnbf = None
        if p["log_evidence"] is not None and c["log_evidence"] is not None:
            lnbf = float(c["log_evidence"]) - float(p["log_evidence"])
        rows.append({
            "parent_hash": edge.parent_hash,
            "child_hash": edge.child_hash,
            "mutation_id": edge.mutation_id,
            "log_bayes_factor": lnbf,
            "valid": bool(p["passed"] and c["passed"] and lnbf is not None),
            "taper_threshold": float(threshold),
            "rerun_job": {"parent": p["path"], "child": c["path"]},
        })
    return rows


def atom_labels_from_graph_payload(payload: Mapping) -> dict[str, str]:
    """``{mutation_id: atom label}`` from a v2 graph JSON's ``metadata.atoms``."""
    atoms = ((payload.get("metadata") or {}).get("atoms") or {}) if isinstance(payload, Mapping) else {}
    return {str(v["mutation_id"]): str(k) for k, v in atoms.items() if isinstance(v, Mapping) and "mutation_id" in v}


def _evaluated_edges(report: Mapping) -> list[Mapping]:
    return [e for e in report.get("edges", []) if not e.get("skipped") and e.get("log_bayes_factor") is not None]


def count_atoms_tried(report: Mapping) -> int:
    """Trials count: distinct non-root models tested by an evaluated edge (every depth).

    A depth-2 model has an edge from each of its two depth-1 parents; it is
    one hypothesis tried, so it counts once (distinct child hashes).
    """
    return len({str(e["child_hash"]) for e in _evaluated_edges(report)})


def count_edges_evaluated(report: Mapping) -> int:
    """Evaluated edges (reported beside the trials count)."""
    return len(_evaluated_edges(report))


def label_for(d1: Mapping, d2: Mapping, d3: Mapping, d4: Mapping, d5: Mapping, d6: Mapping) -> str:
    supported = (
        d1["status"] == "pass"
        and d2["status"] == "pass"
        and d3["status"] == "pass"
        and d4["status"] in _D4_OK
        and d5["status"] == "pass"
        and d6["status"] == "pass"
    )
    if supported:
        return SUPPORTED
    disfavoured = (
        d1["status"] == "pass"
        and d2.get("disfavoured") is True
        and d3["status"] == "sign_stable"
    )
    return DISFAVOURED if disfavoured else INCONCLUSIVE


def build_claim_table(
    report: Mapping,
    *,
    sddr=None,
    prior_sensitivity: Mapping | None = None,
    d3_reruns: Sequence[Mapping] = (),
    taper2: Sequence[Mapping] = (),
    alt_roots: Mapping[str, Mapping] | None = None,
    ppc: Mapping[str, Mapping] | None = None,
    loo: Mapping | None = None,
    taper_mass: Mapping[str, float] | None = None,
    graph=None,
    n_atoms_tried: int | None = None,
    atom_labels: Mapping[str, str] | None = None,
) -> dict:
    """The v2 claim table from an ``analyze-model-comparison`` report and the D3-D6 inputs.

    ``alt_roots`` maps ``"A1"``/``"A2"`` to :func:`alt_root_edge_values`
    outputs (or raw summaries); ``ppc`` maps model hashes to PPC payloads;
    ``atom_labels`` optionally maps mutation ids to the plan's atom names
    (``C2`` ...). ``graph`` (the searched model graph) is needed to key
    depth-2 edges on the alternative roots.
    """
    edges = {(e["parent_hash"], e["child_hash"]): e for e in report.get("edges", []) if not e.get("skipped")}
    variants = report.get("model_prior_variants") or []
    sddr_index = index_sddr(sddr)
    tried = count_atoms_tried(report) if n_atoms_tried is None else int(n_atoms_tried)
    alt = {}
    for name, value in (alt_roots or {}).items():
        if name not in ALT_ROOTS:
            raise AnalysisInputError(f"unknown alternative root {name!r}; expected one of {ALT_ROOTS}")
        alt[name] = value if "values" in value and "inapplicable" in value else alt_root_edge_values(value)
    paths = None if graph is None else mutation_paths(graph)
    root_hash = report.get("graph_root_hash")
    factors = {}
    if prior_sensitivity is not None:
        factors = {(row["parent_hash"], row["child_hash"]): row.get("factor")
                   for row in prior_sensitivity.get("edges", [])}
    rows = []
    for claim in report.get("claims", []):
        edge = edges.get((claim["parent_hash"], claim["child_hash"]))
        if edge is None or edge.get("log_bayes_factor") is None:
            continue
        lnbf = float(edge["log_bayes_factor"])
        d1 = d1_numerics(claim, taper_mass=taper_mass)
        d2 = d2_strength(edge, n_atoms_tried=tried, taper_mass=d1["taper_mass_fraction"])
        d3 = d3_prior(claim, variants, lnbf, reruns=d3_reruns, taper2=taper2,
                      factor=factors.get((claim["parent_hash"], claim["child_hash"])))
        d4 = d4_sddr(claim, edge, sddr_index)
        key = _edge_key(edge, root_hash, paths)
        d5 = d5_alt_roots(claim, lnbf, key, alt, loo=loo)
        d6 = d6_ppc(claim, lnbf, ppc or {})
        label = label_for(d1, d2, d3, d4, d5, d6)
        flags = []
        if d4["status"] == "flag_investigate":
            flags.append("sddr_disagreement_requires_human_review")
        if not d1["taper_mass_reported"]:
            flags.append("taper_mass_not_reported")
        if lnbf < 0 and d3.get("taper2_status") == "not_required_negative_edge":
            flags.append("negative_edge_without_taper2_rerun")
        if any(r["status"] == "not_applicable" for r in d5["alt_roots"].values()):
            flags.append("d5_alt_root_not_applicable_for_this_atom")
        if d6.get("borderline_statistics"):
            flags.append("ppc_borderline_statistic")
        rows.append({
            "atom": claim.get("atom"),
            "atom_label": None if atom_labels is None else atom_labels.get(claim["mutation_id"]),
            "mutation_id": claim["mutation_id"],
            "parent_hash": claim["parent_hash"],
            "child_hash": claim["child_hash"],
            "edge_key": key,
            "log_bayes_factor": lnbf,
            "n_atoms_tried": tried,
            "label": label,
            "D1_numerics": d1,
            "D2_strength": d2,
            "D3_prior_sensitivity": d3,
            "D4_sddr": d4,
            "D5_alternative_roots": d5,
            "D6_ppc": d6,
            "flags": flags,
        })
    rows.sort(key=lambda r: -r["log_bayes_factor"])
    return json_ready({
        "format_version": CLAIMS_V2_FORMAT,
        "specification": "gwpop-search v2 plan 2026-09-30, claim criteria D1-D6",
        "graph_root_hash": root_hash,
        "n_atoms_tried": tried,
        "n_atoms_tried_definition": "distinct non-root models tested by an evaluated edge",
        "n_edges_evaluated": count_edges_evaluated(report),
        "constants": {
            "strength_threshold": STRENGTH_THRESHOLD, "strength_sigmas": STRENGTH_SIGMAS,
            "width_min_log_bf": WIDTH_MIN_LOG_BF, "width_factor": WIDTH_FACTOR,
            "model_prior_min_ratio": MODEL_PRIOR_MIN_RATIO, "sddr_floor": SDDR_FLOOR,
            "sddr_sigmas": SDDR_SIGMAS, "ppc_alpha": PPC_LEVEL, "alt_roots": ALT_ROOT_DESCRIPTIONS,
            "taper_mass_d2_limit": TAPER_MASS_D2_LIMIT,
        },
        "interpretations_pending_operator_confirmation": [
            "D1 is decided by the frozen per-model gate file; the tool's G-MC1..5 defaults are "
            "reported, not binding (v1 resolution carried over; the v2 taper replaces the variance guard).",
            "D3 for a negative edge: sign stability of the width variants and of the taper-2 rerun "
            "where one exists; the model-prior clause is reported but inapplicable; a missing taper-2 "
            "rerun does not block DISFAVOURED (taper-2 reruns are budgeted for claimed edges).",
            "D2 for a tapered edge (DRAFT): sigma_total is the first-order Monte-Carlo error of the "
            "untapered estimator; above a posterior taper-region mass of 0.10 at either endpoint D2 is "
            "incomplete (the neglected ln T(sigma^2_hat) fluctuation can reach ~0.75 nat at the threshold).",
            "D5: an atom that is already part of an alternative root (P2 on A1, Z2 on A2), or that is "
            "defined only relative to the constant form the root replaces (P1 on A1, Z1 on A2), is "
            "not_applicable on that root; the other root must then pass. On A1 the mass atoms add/remove "
            "the component's pairing slope with the component (grammar.v2.v2_d5_atom_semantics).",
            "D6 checks the claimed model: the child of a positive edge (the parent of a negative one; "
            "D6 is not required for DISFAVOURED).",
            "D6 p-values: one-sided P(T_pred >= T_obs) for the four KS distances, two-sided for the "
            "two Spearman correlations; alpha = 0.01 per statistic without multiplicity correction "
            "(family-wise false-fail rate of a correct model <= 5.9% over the six statistics); at least "
            "1000 posterior draws (n_draws * alpha / 2 >= 5).",
        ],
        "edges": rows,
    })


def render_claims_markdown(table: Mapping) -> str:
    lines = [
        "# v2 claim table (criteria D1-D6)",
        "",
        f"Atoms tried (trials count, distinct evaluated non-root models): **{table['n_atoms_tried']}** "
        f"({table.get('n_edges_evaluated', '?')} edges evaluated).",
        "",
        "| atom | ln BF | D1 | D2 (lower) | D3 | D4 | D5 | D6 | label |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for r in table["edges"]:
        d2 = r["D2_strength"]
        d2_cell = f"{d2['status']} ({d2['lower']:+.2f})" if d2.get("lower") is not None else d2["status"]
        name = r.get("atom_label") or r.get("atom") or r["mutation_id"]
        lines.append(
            f"| `{name}` | {r['log_bayes_factor']:+.2f} | {r['D1_numerics']['status']} | {d2_cell} | "
            f"{r['D3_prior_sensitivity']['status']} | {r['D4_sddr']['status']} | "
            f"{r['D5_alternative_roots']['status']} | {r['D6_ppc']['status']} | "
            f"**{r['label']}** ({r['n_atoms_tried']} atoms tried) |"
        )
    lines += ["", "Interpretations pending operator confirmation:", ""]
    lines += [f"- {text}" for text in table.get("interpretations_pending_operator_confirmation", [])]
    return "\n".join(lines) + "\n"

