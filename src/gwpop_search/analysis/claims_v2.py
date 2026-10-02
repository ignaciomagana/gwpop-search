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
* **D2 strength.** ``ln BF - 2 sigma_total - |bias| >= 3`` at the primary
  variance cut ``c = 1`` **and** at the tighter cut ``c' = 0.9``
  (:data:`D2_TIGHTER_CUTS`; operator decision 2026-10-01). Under the sharp
  LVK cut (likelihood ``x 1[sigma^2 <= c]``) the evidence at ``c' < c`` is
  exactly ``Z(c') = Z(c) P_post(sigma^2 <= c' | cut c)``, so per edge
  ``ln BF(c') = ln BF(c) + ln P_child(sigma^2 <= c') - ln P_parent(sigma^2 <= c')``;
  at ``c'`` the same ``sigma_total`` and ``|bias|`` are used, with the
  binomial errors of the two fractions (Kish ESS of the importance-weighted
  dead + live points, ``sqrt(p (1 - p) / n_eff) / p`` in ``ln P``) added in
  quadrature. The fractions come from the evaluations
  (``diagnostics.taper.pooled.posterior_mass_below``) or from the post-hoc
  recomputation (:mod:`gwpop_search.analysis.posthoc_cut`); without them D2
  is ``incomplete``. With the cut-4 D3 rerun this brackets the cut. The
  near-cut band mass (``sigma^2 > 0.95``) is reported, not gating. The
  number of atoms tried (the trials count: every distinct evaluated
  non-root model, so a depth-2 model reached by two parent edges counts
  once) is disclosed next to every claim.
* **D3 prior sensitivity.** Halving and doubling each added parameter's prior
  (exact reweighting for the narrowed prior, the Occam relation for the widened
  one, or -- where neither is valid -- an explicit rerun, ``--d3-reruns``)
  keeps ``ln BF >= 1`` with the same sign; every model-prior variant keeps the
  child's probability relative to its parent ``p_c / (p_c + p_p) >= 0.75``;
  and the cut-at-4 rerun (``--taper2``) keeps the sign.
* **D4 SDDR.** For nested edges the Savage-Dickey estimate agrees with the
  evidence ratio within ``max(0.5, 2 sigma)``,
  ``sigma = hypot(sigma_total, sigma_SDDR)``. Family changes are
  ``not_applicable``; a one-sided bound satisfied by the evidence ratio is
  ``bound_consistent``. A disagreement blocks SUPPORTED and is flagged.
* **D5 alternative roots.** The sign of ``ln BF`` holds on both alternative
  roots, A1 = BP2P + beta per mass component and A2 = BP2P + kappa(m1)
  (local mass function convention R(m1, z) = R(m1, 0) (1+z)^kappa(m1): BP2P is
  the z = 0 mass spectrum; operator decision 2026-10-01)
  (``--alt-root A1=<suite or scenario summary>``). An edge whose mutation is
  inapplicable on an alternative root (it is already part of that root) is
  ``not_applicable`` there. PSIS-LOO is reported, not binding. Event-drop
  scenarios are deferred (not part of D5 in v2).
* **D6 posterior predictive check** of the claimed model (the child for a
  positive edge): no pre-declared statistic at ``p < 0.01``
  (:mod:`gwpop_search.analysis.ppc`).

**Pairwise attribution (chi_eff block; operator decision 2026-10-02).** On
the pilot (b) closure mock (truth: width(q), C2) both C2 and C4 (width(z))
passed D2 at depth 1: the selection couples q and z, so a chi_eff atom can
pass D2 by absorbing another atom's signal. A chi_eff-block atom (C1-C6,
S1-S4) can therefore be SUPPORTED only if, for every OTHER chi_eff atom that
passes D2 at depth 1 (the *family*), the depth-2 pair (this atom + the
other) was evaluated and the edge adding this atom to the other's depth-1
model still passes D2 -- ``ln BF - 2 sigma_total - |bias| >= 3`` at the cuts
1 and 0.9, the same :func:`d2_strength` evaluation as every edge. Otherwise
the label is INCONCLUSIVE with the reason ``not attributable (family:
...)``; a pair that was not evaluated (over the depth-2 cap, or not
composable: any two of S1-S4 are alternative chi_eff families) counts as not
attributable. A chi_eff atom of the searched graph whose depth-1 D2 is
undetermined (root edge not evaluated, or D2 incomplete) may itself pass D2,
so it also blocks attribution (status ``incomplete``) until it is resolved.
The table reports the
family and the pairwise matrix (:func:`chieff_attribution`); the depth-2
enumeration makes these pairs mandatory
(:func:`gwpop_search.grammar.v2.plan_depth2`). The rule applies to the
depth-1 chi_eff edges (out of the root); depth-2 edges are themselves the
attribution comparisons and are labelled by D1-D6 alone.

DISFAVOURED: ``ln BF + 2 sigma_total + |bias| <= -3`` at both cuts ``c = 1``
and ``c' = 0.9`` (the symmetric D2 condition) with D1 passing and the sign
stable under D3 (width variants, and the taper-2 rerun where one exists).
INCONCLUSIVE: everything else.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from gwpop_search.grammar.paths import edge_path_key, mutation_paths
from gwpop_search.grammar.v2 import CHIEFF_ATOMS, V2_ATOM_ORDER, V2_MUTATION_ATOM
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
#: D2 cut bracketing (operator decision 2026-10-01; replaces the DRAFT near-cut
#: mass limit 0.10): D2 / DISFAVOURED must hold at the primary sharp cut and at
#: every tighter cut, ln BF(c') = ln BF(c) + ln P_child(sigma^2 <= c') -
#: ln P_parent(sigma^2 <= c') (exact for a sharp cut; Z(c') = Z(c) P_post).
D2_PRIMARY_CUT = 1.0
D2_TIGHTER_CUTS = (0.9,)
ALT_ROOT_DESCRIPTIONS = {
    "A1": "BP2P + beta per mass component (LVK 'Extended' pairing)",
    "A2": "BP2P + kappa(m1), R(m1, z) = R(m1, 0) (1+z)^kappa(m1) (BP2P = z = 0 mass spectrum)",
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


def mass_below_entry(taper_summary: Mapping, *, source=None) -> dict | None:
    """Per-model D2 input from a taper summary block (``diagnostics.taper.pooled``).

    Returns ``{"threshold", "kind", "kish_ess", "near_cut_band_mass", "cuts":
    {cut_key: {"cut", "fraction", "error", "n_eff"}}, "source"}``, or ``None``
    when the block carries no ``posterior_mass_below``.
    """
    below = taper_summary.get("posterior_mass_below")
    if not below:
        return None
    taper = taper_summary.get("taper") or {}
    return {
        "threshold": None if taper.get("threshold") is None else float(taper["threshold"]),
        "kind": str(taper.get("kind", DEFAULT_TAPER_KIND)),
        "kish_ess": taper_summary.get("kish_ess"),
        "near_cut_band_mass": taper_summary.get("posterior_mass_in_taper_region"),
        "cuts": {str(k): dict(v) for k, v in below.items()},
        "source": None if source is None else str(source),
    }


def _fraction_at(entry: Mapping | None, cut: float):
    if not entry:
        return None
    from gwpop_search.hbi.taper import cut_key

    row = (entry.get("cuts") or {}).get(cut_key(cut))
    if row is None:
        return None
    return float(row["fraction"]), float(row.get("error") or 0.0)


def d2_at_cut(
    lnbf: float,
    sigma_total: float,
    bias: float,
    cut: float,
    *,
    parent: Mapping | None,
    child: Mapping | None,
    primary_cut: float = D2_PRIMARY_CUT,
) -> dict:
    """The D2 inequality at a tighter sharp cut ``cut < primary_cut``.

    ``parent``/``child`` are :func:`mass_below_entry` blocks. ``ln BF(c') =
    ln BF + ln p_c - ln p_p``; ``sigma(c') = hypot(sigma_total, e_c / p_c,
    e_p / p_p)`` (delta method on ``ln p``, ``e`` the Kish-ESS binomial
    error); ``lower/upper = ln BF(c') -/+ (2 sigma(c') + |bias|)``. Status
    ``pass``/``fail`` (D2 lower bound), ``disfavoured`` (upper bound), or
    ``missing``/``incomplete`` (no fractions; a smooth taper or a cut that is
    not tighter than the evaluations' threshold, for which the identity does
    not hold; a zero fraction, which the samples cannot resolve).
    """
    out: dict[str, object] = {"cut": float(cut), "primary_cut": float(primary_cut)}
    if not float(cut) < float(primary_cut):
        return {**out, "status": "incomplete", "reason": "cut is not tighter than the primary cut"}
    fp, fc = _fraction_at(parent, cut), _fraction_at(child, cut)
    if fp is None or fc is None:
        return {**out, "status": "missing",
                "reason": f"posterior fraction below sigma^2 = {cut:g} not reported for both endpoints"}
    for name, entry in (("parent", parent), ("child", child)):
        if entry.get("kind", DEFAULT_TAPER_KIND) != "sharp":
            return {**out, "status": "incomplete",
                    "reason": f"{name} evaluation is not under a sharp cut (Z(c') = Z(c) P holds only there)"}
        threshold = entry.get("threshold")
        if threshold is None or float(threshold) != float(primary_cut):
            return {**out, "status": "incomplete",
                    "reason": f"{name} evaluation cut {threshold} is not the primary cut {primary_cut:g}"}
    (p_p, e_p), (p_c, e_c) = fp, fc
    out.update({"fraction_parent": p_p, "fraction_parent_error": e_p,
                "fraction_child": p_c, "fraction_child_error": e_c})
    if not (p_p > 0.0 and p_c > 0.0):
        return {**out, "status": "incomplete",
                "reason": "no posterior mass below the cut at one endpoint (ln BF(c') not resolved)"}
    delta = math.log(p_c) - math.log(p_p)
    sigma_fraction = math.hypot(e_c / p_c, e_p / p_p)
    sigma_cut = math.hypot(float(sigma_total), sigma_fraction)
    lnbf_cut = float(lnbf) + delta
    lower = lnbf_cut - STRENGTH_SIGMAS * sigma_cut - bias
    upper = lnbf_cut + STRENGTH_SIGMAS * sigma_cut + bias
    return {
        **out,
        "status": "pass" if lower >= STRENGTH_THRESHOLD else "fail",
        "log_bayes_factor": lnbf_cut,
        "delta_log_bayes_factor": delta,
        "sigma_fraction": sigma_fraction,
        "sigma_total": sigma_cut,
        "lower": lower,
        "upper": upper,
        "disfavoured": bool(upper <= -STRENGTH_THRESHOLD),
    }


def d2_strength(
    edge: Mapping,
    *,
    n_atoms_tried: int,
    mass_below: Mapping | None = None,
    taper_mass: Mapping | None = None,
    tighter_cuts: Sequence[float] = D2_TIGHTER_CUTS,
) -> dict:
    """D2 strength at the primary cut and at each tighter cut (module docstring).

    ``mass_below`` = ``{"parent": entry, "child": entry}``
    (:func:`mass_below_entry`); ``taper_mass`` = D1's near-cut band mass
    block, reported here, never gating. D2 passes only if the inequality
    holds at the primary cut and at every tighter cut; a primary pass with a
    tighter cut not evaluable is ``incomplete``. DISFAVOURED (``disfavoured``)
    needs the symmetric upper bound at every cut.
    """
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
    primary = "pass" if lower >= STRENGTH_THRESHOLD else "fail"
    mass_below = mass_below or {}
    cuts = [
        d2_at_cut(float(lnbf), sigma_total, bias, cut,
                  parent=mass_below.get("parent"), child=mass_below.get("child"))
        for cut in tighter_cuts
    ]
    evaluable = [c for c in cuts if c["status"] in ("pass", "fail")]
    reason = None
    if primary == "fail":
        status = "fail"
    elif any(c["status"] == "fail" for c in evaluable):
        status = "fail"
        reason = "the D2 inequality fails at the tighter cut " + ", ".join(
            f"{c['cut']:g} (lower {c['lower']:+.3g})" for c in evaluable if c["status"] == "fail")
    elif len(evaluable) < len(cuts):
        status = "incomplete"
        reason = "; ".join(f"cut {c['cut']:g}: {c['reason']}" for c in cuts if c not in evaluable)
    else:
        status = "pass"
    disfavoured = bool(upper <= -STRENGTH_THRESHOLD) and len(evaluable) == len(cuts) and all(
        c["disfavoured"] for c in evaluable)
    if upper <= -STRENGTH_THRESHOLD and len(evaluable) < len(cuts):
        disfavoured_status = "incomplete"
    else:
        disfavoured_status = "pass" if disfavoured else "fail"
    return {
        "status": status,
        "reason": reason,
        "log_bayes_factor": float(lnbf),
        "sigma_total": sigma_total,
        "abs_bias": bias,
        "lower": lower,
        "upper": upper,
        "primary_cut": D2_PRIMARY_CUT,
        "primary_status": primary,
        "tighter_cuts": cuts,
        "min_lower": min([lower, *(c["lower"] for c in evaluable)]),
        "max_upper": max([upper, *(c["upper"] for c in evaluable)]),
        "disfavoured": disfavoured,
        "disfavoured_status": disfavoured_status,
        "near_cut_band_mass_reported_not_gating": None if taper_mass is None else dict(taper_mass),
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
    pooled posterior mass fraction inside the taper region (the near-cut
    band)}, "mass_below": {model_hash: D2 cut-bracketing entry
    (:func:`mass_below_entry`; only evaluations that recorded
    ``posterior_mass_below``)}, "fidelity": {model_hash: rung}, "files":
    {model_hash: path}, "superseded": {model_hash: path}}``.

    **F3/F4 rule (v2 second seeds).** Every model has one F3 evaluation (one
    dynesty run); decision-relevant edges are re-evaluated at F4 (the
    second-seed rung). A model with both uses its **F4** evaluation for D1
    (its gates include the cross-run checks over the F4 runs) and its taper
    mass and posterior fractions; the F3 evaluation is recorded as ``superseded``. This is not a
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
    out: dict[str, dict] = {"gates": {}, "taper_mass": {}, "mass_below": {}, "fidelity": {}, "files": {},
                            "superseded": {}}
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
        entry = mass_below_entry(pooled, source=path)
        if entry is not None:
            out["mass_below"][model_hash] = entry
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


#: D3: the variance-cut sensitivity rerun at 4 (the LVK relaxed-cut release; operator decision
#: 2026-10-01; the flag keeps its historical name ``--taper2``).
TAPER2_THRESHOLD = 4.0
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


# ---------------------------------------------------------------------------
# Pairwise attribution of chi_eff atoms
# ---------------------------------------------------------------------------

_ATTRIBUTION_OK = ("pass", "not_applicable")


def _graph_parent(edge: Mapping) -> str:
    return str(edge.get("graph_parent_hash") or edge["parent_hash"])


def _graph_child(edge: Mapping) -> str:
    return str(edge.get("graph_child_hash") or edge["child_hash"])


def _composable(a: str, b: str) -> bool | None:
    """Whether two v2 atoms compose into one depth-2 model on R0 (``None``: not v2 atoms)."""
    from gwpop_search.grammar.v2 import NotComposable, compose_atoms, v2_root_model_spec

    if a not in V2_ATOM_ORDER or b not in V2_ATOM_ORDER:
        return None
    key = tuple(sorted((a, b)))
    if key not in _COMPOSABLE_CACHE:
        try:
            compose_atoms(v2_root_model_spec(), a, b)
            _COMPOSABLE_CACHE[key] = True
        except NotComposable:
            _COMPOSABLE_CACHE[key] = False
    return _COMPOSABLE_CACHE[key]


_COMPOSABLE_CACHE: dict[tuple[str, str], bool] = {}


def chieff_attribution(
    report: Mapping,
    d2_of_edge: Mapping[tuple[str, str], Mapping],
    *,
    atom_of: Mapping[str, str] | None = None,
) -> dict:
    """The pairwise attribution rule of the chi_eff block (module docstring).

    ``d2_of_edge`` maps ``(parent_hash, child_hash)`` of every evaluated edge
    to its :func:`d2_strength` result; ``atom_of`` maps mutation ids to atom
    ids (default: the v2 table). Returns ``{"family": [D2-passing chi_eff
    atoms at depth 1], "undetermined": [...], "matrix": {A: {B: entry}},
    "atoms": {A: {"status", "reason", "pairs"}}}`` where ``matrix[A][B]`` is
    the edge that adds ``A`` to ``B``'s depth-1 model (status
    ``pass``/``fail``/``incomplete`` from D2, or ``missing`` when the pair was
    not evaluated, with ``not_composable`` when the two atoms cannot form one
    model).

    A family member's status is ``pass`` only if every entry against the
    other family members passes (vacuously when it is the only member) *and*
    no other chi_eff atom of the searched graph has an undetermined depth-1
    D2 (its root edge not evaluated, or D2 ``incomplete``/``missing``): such
    an atom may pass D2, so the rule cannot be verified and the status is
    ``incomplete``. A failing or unevaluated pair gives ``fail``. Atoms that
    do not pass D2 at depth 1 are ``not_applicable`` (they cannot be
    SUPPORTED, so attribution is moot); atoms outside the chi_eff block are
    not listed.
    """
    labels = dict(V2_MUTATION_ATOM)
    labels.update(atom_of or {})
    root = report.get("graph_root_hash")
    edges = _evaluated_edges(report)
    order = {a: i for i, a in enumerate(V2_ATOM_ORDER)}
    depth1: dict[str, Mapping] = {}
    for e in edges:
        atom = labels.get(e["mutation_id"])
        if root is not None and _graph_parent(e) == root and atom in CHIEFF_ATOMS:
            depth1[atom] = e
    # chi_eff atoms of the searched graph (evaluated or not; alias edges excluded)
    searched = {labels.get(e["mutation_id"]) for e in report.get("edges", [])
                if not e.get("skipped") and root is not None and _graph_parent(e) == root}
    searched = {a for a in searched if a in CHIEFF_ATOMS}

    def d2_status(e: Mapping) -> Mapping:
        return d2_of_edge.get((e["parent_hash"], e["child_hash"])) or {"status": "missing"}

    family = sorted((a for a, e in depth1.items() if d2_status(e)["status"] == "pass"),
                    key=lambda a: order.get(a, len(order)))
    depth1_status = {a: (d2_status(depth1[a])["status"] if a in depth1 else "not_evaluated")
                     for a in searched | set(depth1)}
    undetermined = sorted((a for a, st in depth1_status.items() if st not in ("pass", "fail")),
                          key=lambda a: order.get(a, len(order)))
    base_of_child = {_graph_child(e): a for a, e in depth1.items()}
    pair_edges: dict[tuple[str, str], Mapping] = {}
    for e in edges:
        base = base_of_child.get(_graph_parent(e))
        added = labels.get(e["mutation_id"])
        if base is not None and added in CHIEFF_ATOMS and added != base:
            pair_edges[(added, base)] = e

    def entry(added: str, base: str) -> dict:
        e = pair_edges.get((added, base))
        if e is None:
            composable = _composable(added, base)
            reason = ("not composable (the two atoms cannot form one model)" if composable is False
                      else "depth-2 pair not evaluated")
            return {"status": "missing", "reason": reason, "not_composable": composable is False}
        d2 = d2_status(e)
        return {
            "status": d2["status"],
            "parent_hash": e["parent_hash"],
            "child_hash": e["child_hash"],
            "log_bayes_factor": d2.get("log_bayes_factor"),
            "lower": d2.get("lower"),
            "min_lower": d2.get("min_lower"),
            "reason": d2.get("reason"),
        }

    matrix = {a: {b: entry(a, b) for b in family if b != a} for a in family}
    family_text = ", ".join(family) if family else "none"
    atoms = {}
    for a in sorted(depth1, key=lambda x: order.get(x, len(order))):
        if a not in family:
            atoms[a] = {"status": "not_applicable",
                        "reason": f"D2 at depth 1 is {depth1_status[a]!r}, not 'pass' (attribution moot)",
                        "family": list(family), "pairs": {}, "blocking": [], "undetermined": []}
            continue
        others = [b for b in family if b != a]
        pairs = matrix[a]
        blocking = [b for b in others if pairs[b]["status"] != "pass"]
        unknown = [b for b in undetermined if b != a]
        if blocking:
            status = "fail"
            reason = (f"not attributable (family: {family_text}; adding {a} fails or is unevaluated "
                      f"against {', '.join(blocking)})")
        elif unknown:
            status = "incomplete"
            reason = (f"not attributable (family: {family_text}; depth-1 D2 undetermined for "
                      f"{', '.join(unknown)}, which may also pass D2)")
        elif not others:
            status, reason = "pass", "no other D2-passing chi_eff atom"
        else:
            status, reason = "pass", f"attributable against every other family member ({', '.join(others)})"
        atoms[a] = {"status": status, "reason": reason, "family": list(family), "pairs": pairs,
                    "blocking": blocking, "undetermined": unknown}
    return {
        "rule": "a chi_eff atom can be SUPPORTED only if adding it to every other chi_eff atom that "
                "passes D2 at depth 1 still passes D2 (ln BF - 2 sigma_total - |bias| >= 3 at cuts 1 "
                "and 0.9); a chi_eff atom of the graph with an undetermined depth-1 D2 (not evaluated, "
                "incomplete) blocks attribution; matrix[A][B] = the edge adding A to B's depth-1 model",
        "family": family,
        "undetermined": undetermined,
        "matrix": matrix,
        "atoms": atoms,
    }


def label_for(d1: Mapping, d2: Mapping, d3: Mapping, d4: Mapping, d5: Mapping, d6: Mapping,
              attribution: Mapping | None = None) -> str:
    supported = (
        d1["status"] == "pass"
        and d2["status"] == "pass"
        and d3["status"] == "pass"
        and d4["status"] in _D4_OK
        and d5["status"] == "pass"
        and d6["status"] == "pass"
        and (attribution is None or attribution["status"] in _ATTRIBUTION_OK)
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
    mass_below: Mapping[str, Mapping] | None = None,
    graph=None,
    n_atoms_tried: int | None = None,
    atom_labels: Mapping[str, str] | None = None,
) -> dict:
    """The v2 claim table from an ``analyze-model-comparison`` report and the D3-D6 inputs.

    ``taper_mass`` maps model hashes to the near-cut band mass (D1: reported);
    ``mass_below`` maps model hashes to the D2 cut-bracketing entries
    (:func:`mass_below_entry`: ``collect_v2_evaluations(...)["mass_below"]``
    or the post-hoc :func:`gwpop_search.analysis.posthoc_cut.posthoc_mass_below`).

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
    d2_of_edge = {}
    for e in _evaluated_edges(report):
        below = {"parent": (mass_below or {}).get(e["parent_hash"]),
                 "child": (mass_below or {}).get(e["child_hash"])}
        d2_of_edge[(e["parent_hash"], e["child_hash"])] = d2_strength(e, n_atoms_tried=tried, mass_below=below)
    attribution_table = chieff_attribution(report, d2_of_edge, atom_of=atom_labels)
    labels_of = dict(V2_MUTATION_ATOM)
    labels_of.update(atom_labels or {})
    rows = []
    for claim in report.get("claims", []):
        edge = edges.get((claim["parent_hash"], claim["child_hash"]))
        if edge is None or edge.get("log_bayes_factor") is None:
            continue
        lnbf = float(edge["log_bayes_factor"])
        d1 = d1_numerics(claim, taper_mass=taper_mass)
        below = {"parent": (mass_below or {}).get(claim["parent_hash"]),
                 "child": (mass_below or {}).get(claim["child_hash"])}
        d2 = d2_strength(edge, n_atoms_tried=tried, mass_below=below, taper_mass=d1["taper_mass_fraction"])
        atom_id = labels_of.get(claim["mutation_id"])
        if (atom_id in CHIEFF_ATOMS and root_hash is not None and _graph_parent(edge) == root_hash
                and atom_id in attribution_table["atoms"]):
            attribution = dict(attribution_table["atoms"][atom_id])
        elif atom_id in CHIEFF_ATOMS and root_hash is not None and _graph_parent(edge) == root_hash:
            attribution = {"status": "missing", "reason": "depth-1 edge not evaluated"}
        else:
            attribution = {"status": "not_applicable",
                           "reason": "not a depth-1 chi_eff-block edge"}
        d3 = d3_prior(claim, variants, lnbf, reruns=d3_reruns, taper2=taper2,
                      factor=factors.get((claim["parent_hash"], claim["child_hash"])))
        d4 = d4_sddr(claim, edge, sddr_index)
        key = _edge_key(edge, root_hash, paths)
        d5 = d5_alt_roots(claim, lnbf, key, alt, loo=loo)
        d6 = d6_ppc(claim, lnbf, ppc or {})
        label = label_for(d1, d2, d3, d4, d5, d6, attribution)
        criteria_met = label_for(d1, d2, d3, d4, d5, d6) == SUPPORTED
        label_reason = None
        if criteria_met and label != SUPPORTED:
            label_reason = attribution["reason"]
        flags = []
        if attribution["status"] not in _ATTRIBUTION_OK:
            flags.append("chieff_not_attributable")
        if d4["status"] == "flag_investigate":
            flags.append("sddr_disagreement_requires_human_review")
        if not d1["taper_mass_reported"]:
            flags.append("taper_mass_not_reported")
        if any(c["status"] == "missing" for c in d2.get("tighter_cuts", ())):
            flags.append("posterior_mass_below_cut_not_reported")
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
            "label_reason": label_reason,
            "D1_numerics": d1,
            "D2_strength": d2,
            "D3_prior_sensitivity": d3,
            "D4_sddr": d4,
            "D5_alternative_roots": d5,
            "D6_ppc": d6,
            "attribution": attribution,
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
            "d2_primary_cut": D2_PRIMARY_CUT, "d2_tighter_cuts": list(D2_TIGHTER_CUTS),
        },
        "chieff_attribution": {k: v for k, v in attribution_table.items() if k != "atoms"},
        "interpretations_pending_operator_confirmation": [
            "D1 is decided by the frozen per-model gate file; the tool's G-MC1..5 defaults are "
            "reported, not binding (v1 resolution carried over; the v2 taper replaces the variance guard).",
            "D3 for a negative edge: sign stability of the width variants and of the taper-2 rerun "
            "where one exists; the model-prior clause is reported but inapplicable; a missing taper-2 "
            "rerun does not block DISFAVOURED (taper-2 reruns are budgeted for claimed edges).",
            "D2 cut bracketing (operator decision 2026-10-01): D2 (and DISFAVOURED, symmetrically) must "
            "hold at the primary sharp cut sigma^2 <= 1 and at sigma^2 <= 0.9, where ln BF(0.9) = ln BF(1) "
            "+ ln P_child(sigma^2 <= 0.9) - ln P_parent(sigma^2 <= 0.9) (exact for a sharp cut), with the "
            "same sigma_total and |bias| plus the Kish-ESS binomial errors of the two fractions in "
            "quadrature; with the cut-4 D3 rerun this brackets the cut. The near-cut band mass "
            "(sigma^2 > 0.95) is reported, not gating (the DRAFT 0.10 limit is withdrawn).",
            "D5: an atom that is already part of an alternative root (P2 on A1, Z2 on A2), or that is "
            "defined only relative to the constant form the root replaces (P1 on A1, Z1 on A2), is "
            "not_applicable on that root; the other root must then pass. On A1 the mass atoms add/remove "
            "the component's pairing slope with the component (grammar.v2.v2_d5_atom_semantics).",
            "D6 checks the claimed model: the child of a positive edge (the parent of a negative one; "
            "D6 is not required for DISFAVOURED).",
            "D6 p-values: one-sided P(T_pred >= T_obs) for the four KS distances, two-sided for the "
            "two Spearman correlations and the twelve width statistics (IQR of chi_eff in terciles of "
            "q, z, m1; Spearman(x, |chi_eff - median|) for x = q, z, m1; operator decision "
            "2026-10-02); alpha = 0.01 per statistic without multiplicity correction (a D6 false fail "
            "only removes SUPPORTED); the family-wise false-fail rate of a correct model is disclosed "
            "(independence bound 16.5% over 18 statistics; empirical estimate from the predicted "
            "replicates in the PPC output); at least 1000 posterior draws (n_draws * alpha / 2 >= 5).",
            "Pairwise attribution (operator decision 2026-10-02): a depth-1 chi_eff atom is SUPPORTED "
            "only if adding it to every other D2-passing depth-1 chi_eff atom passes D2 at cuts 1 and "
            "0.9; an unevaluated or non-composable pair (any two of S1-S4, which are alternative chi_eff "
            "families), or another chi_eff atom of the graph whose depth-1 D2 is undetermined (not "
            "evaluated or incomplete), leaves it INCONCLUSIVE ('not attributable'); a direct S_i vs S_j depth-1 "
            "comparison is not substituted without an operator decision. The family is D2-passing (not D1 + D2): an atom with failing "
            "numerics still competes for the signal.",
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
        "| atom | ln BF | D1 | D2 (lower; at tighter cuts) | D3 | D4 | D5 | D6 | attribution | label |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for r in table["edges"]:
        d2 = r["D2_strength"]
        d2_cell = f"{d2['status']} ({d2['lower']:+.2f})" if d2.get("lower") is not None else d2["status"]
        for cut in d2.get("tighter_cuts", ()):
            d2_cell += (f"; {cut['cut']:g}: {cut['lower']:+.2f}" if cut.get("lower") is not None
                        else f"; {cut['cut']:g}: {cut['status']}")
        name = r.get("atom_label") or r.get("atom") or r["mutation_id"]
        lines.append(
            f"| `{name}` | {r['log_bayes_factor']:+.2f} | {r['D1_numerics']['status']} | {d2_cell} | "
            f"{r['D3_prior_sensitivity']['status']} | {r['D4_sddr']['status']} | "
            f"{r['D5_alternative_roots']['status']} | {r['D6_ppc']['status']} | "
            f"{(r.get('attribution') or {}).get('status', 'n/a')} | "
            f"**{r['label']}** ({r['n_atoms_tried']} atoms tried) |"
        )
    att = table.get("chieff_attribution") or {}
    family = list(att.get("family") or [])
    lines += ["", f"chi_eff attribution family (D2-passing chi_eff atoms at depth 1): "
                  f"**{', '.join(family) if family else 'none'}**"]
    undetermined = list(att.get("undetermined") or [])
    if undetermined:
        lines += ["", "chi_eff atoms with undetermined depth-1 D2 (not evaluated or incomplete; they block "
                      f"attribution until resolved): **{', '.join(undetermined)}**"]
    if len(family) > 1:
        matrix = att.get("matrix") or {}
        lines += ["", "Pairwise matrix: row = atom added, column = the other atom's depth-1 model; "
                      "cell = D2 status (minimum lower bound over cuts 1 and 0.9).", "",
                  "| added \\ base | " + " | ".join(family) + " |",
                  "| --- | " + " | ".join("---" for _ in family) + " |"]
        for a in family:
            cells = []
            for b in family:
                if a == b:
                    cells.append("-")
                    continue
                e = (matrix.get(a) or {}).get(b) or {}
                if e.get("min_lower") is not None:
                    cells.append(f"{e['status']} ({e['min_lower']:+.2f})")
                elif e.get("not_composable"):
                    cells.append("not composable")
                else:
                    cells.append(str(e.get("status", "missing")))
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
    reasons = [(r.get("atom_label") or r.get("atom") or r["mutation_id"], r["label_reason"])
               for r in table["edges"] if r.get("label_reason")]
    if reasons:
        lines += ["", "Labels changed by the attribution rule:", ""]
        lines += [f"- `{name}`: {why}" for name, why in reasons]
    unattributed = [(r.get("atom_label") or r["mutation_id"], (r.get("attribution") or {}).get("reason"))
                    for r in table["edges"]
                    if (r.get("attribution") or {}).get("status") not in (None, "pass", "not_applicable")]
    if unattributed:
        lines += ["", "Attribution status (blocks SUPPORTED):", ""]
        lines += [f"- `{name}`: {why}" for name, why in unattributed]
    lines += ["", "Interpretations pending operator confirmation:", ""]
    lines += [f"- {text}" for text in table.get("interpretations_pending_operator_confirmation", [])]
    return "\n".join(lines) + "\n"

