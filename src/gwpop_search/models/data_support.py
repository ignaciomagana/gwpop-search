"""Model-support versus dataset checks for v2 models (gate G12, model + data).

A v2 model spec carries its own support block (``zmax``, ``q_floor``,
``mmin``, ``mmax``, the sky convention and the cosmology; see
:data:`gwpop_search.grammar.v2.V2_SUPPORT`); the gwcat v2 adapter records the
dataset's side (``z_max`` and ``sky_marginal`` in the canonical metadata, the
sky-marginal density basis, the :class:`~gwpop_search.data.v2_policy.GwcatV2DataPolicy`).
:func:`v2_data_support_report` confronts the two:

* **sky** -- ``support.sky = "marginalized"`` needs the sky-marginal basis
  (no ``ra``/``dec`` coordinates, ``sky_marginal`` metadata not False) on both
  catalogs; ``"isotropic"`` needs ``ra``/``dec`` in the basis;
* **zmax** -- the model ``zmax`` equals the selection's declared ``z_max``
  (required), the PE's when declared, and the data policy's when given;
* **per-event support** -- every event keeps at least one PE sample inside the
  fixed population support (``mmin <= m1_source <= mmax``,
  ``q >= max(q_floor, mmin / m1_source)``, ``z <= zmax``) at the model's
  cosmology, otherwise its likelihood is zero for every hyperparameter;
* **injection draw support** -- the selection estimate ``xi`` is unbiased
  only if the injection draw distribution covers the population support
  wherever detection is possible:

  - *declared* draw bounds (``selection.metadata["draw_support"]`` or the
    ``draw_support`` argument; keys ``m1_source_min``, ``m1_source_max``,
    ``q_min``, ``z_max``) must contain the population support (the z bound
    within a 1% finite-draw tolerance); not evaluated when none is declared
    (the v2r2 canonical selection records none);
  - *found-injection edges* (always): at each support edge (``m1`` in
    ``[mmin, mmax]``, ``q >= q_floor``, ``z <= zmax``) the found injections
    either reach the edge (within 1%, or within the 2% test window of the found
    range) or thin out smoothly before it. A
    found-injection density that stops inside the support at a cliff (the
    density in the last 2% of the found range at least 10% of the mean)
    means the draws, not the detector, end there, and fails (not evaluated
    below 1000 found injections). Limitation: on a cumulative mixture a
    per-run draw bound (e.g. O1/O2 draws to z = 1.30/1.66) is hidden by the
    other runs; it is checked only through declared bounds;
* **reported only** -- the PE mass above ``zmax``, the found injections above
  ``zmax`` and below ``q_floor``, and the least-supported event.

:func:`require_v2_data_support` raises :class:`V2DataSupportError` on any
failed check. The production evaluator calls it for every v2 model before any
sampling (``DeterministicHBIEvaluator``).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

V2_DATA_SUPPORT_FORMAT = "gwpop-search-v2-data-support-1.0"
_SKY = ("ra", "dec")


class V2DataSupportError(ValueError):
    """The dataset does not match the support a v2 model declares."""


def _compiled(model):
    from .declarative import DeclarativeGwcatChiEffModel, compile_model_spec

    if isinstance(model, DeclarativeGwcatChiEffModel):
        return model
    return compile_model_spec(model)


def _metadata(catalog) -> Mapping[str, object]:
    return getattr(catalog, "metadata", None) or {}


def _source_frame(samples: Mapping[str, np.ndarray], cosmology):
    d_l = np.asarray(samples["luminosity_distance"], dtype=np.float64)
    z = np.asarray(cosmology.z_of_dL(d_l), dtype=np.float64)
    m1 = np.asarray(samples["m1_detector"], dtype=np.float64) / (1.0 + z)
    return m1, np.asarray(samples["q"], dtype=np.float64), z


def _inside(m1, q, z, *, mmin, mmax, q_floor, zmax):
    safe_m1 = np.where(m1 > 0.0, m1, np.inf)
    return (
        (m1 >= mmin) & (m1 <= mmax)
        & (q >= np.maximum(q_floor, mmin / safe_m1)) & (q <= 1.0)
        & (z > 0.0) & (z <= zmax)
    )


#: found-injection edge test: an edge is reached within this relative tolerance
EDGE_REACH_TOLERANCE = 0.01
#: width of the top window as a fraction of the found range, and the density
#: ratio (window density / mean density) at or above which the stop is a cliff
EDGE_WINDOW_FRACTION = 0.02
EDGE_CLIFF_DENSITY_RATIO = 0.1
#: below this many found injections the window test has no power: not evaluated
EDGE_MIN_FOUND = 1000
_DRAW_KEYS = ("m1_source_min", "m1_source_max", "q_min", "z_max")


def _edge_test(values, edge: float, side: str) -> dict[str, object]:
    """Does the found-injection extent reach ``edge`` or thin out smoothly before it?"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"edge": edge, "side": side, "n_found": 0, "reached": False, "cliff": True,
                "passed": False, "note": "no found injections"}
    lo, hi = float(values.min()), float(values.max())
    if values.size < EDGE_MIN_FOUND:
        return {"edge": float(edge), "side": side, "n_found": int(values.size),
                "found_extent": hi if side == "upper" else lo, "reached": None, "cliff": None,
                "passed": True, "note": f"not evaluated: fewer than {EDGE_MIN_FOUND} found injections"}
    width = hi - lo
    # "reached": the gap to the edge is within 1% of the edge or within the
    # test window (finite sampling cannot resolve a smaller gap)
    slack = max(EDGE_REACH_TOLERANCE * abs(edge), EDGE_WINDOW_FRACTION * width)
    if side == "upper":
        extent = hi
        reached = hi >= edge - slack
        in_window = values >= hi - EDGE_WINDOW_FRACTION * width
    else:
        extent = lo
        reached = lo <= edge + slack
        in_window = values <= lo + EDGE_WINDOW_FRACTION * width
    ratio = float(np.mean(in_window) / EDGE_WINDOW_FRACTION) if width > 0 else float("inf")
    cliff = (not reached) and ratio >= EDGE_CLIFF_DENSITY_RATIO
    return {"edge": float(edge), "side": side, "n_found": int(values.size), "found_extent": extent,
            "reached": bool(reached), "window_density_ratio": ratio, "cliff": bool(cliff),
            "passed": bool(not cliff)}


def v2_data_support_report(model, posterior, selection, *, policy=None,
                           draw_support: Mapping[str, float] | None = None) -> dict[str, object]:
    """Check a v2 model's declared support against a canonical PE/selection pair.

    ``model`` is a v2 :class:`~gwpop_search.grammar.ModelSpec` or its compiled
    model; ``policy`` an optional :class:`GwcatV2DataPolicy` (or anything with
    a ``z_max`` attribute / key); ``draw_support`` optional declared injection
    draw bounds (else ``selection.metadata["draw_support"]``). Returns a
    JSON-ready report with ``pass``.
    """
    compiled = _compiled(model)
    if not compiled.is_v2:
        raise ValueError("the v2 data-support check applies to v2 models only")
    support = dict(compiled.spec.support)
    zmax = float(support["zmax"])
    q_floor = float(support["q_floor"])
    mmin, mmax = float(support["mmin"]), float(support["mmax"])
    q_edge = max(q_floor, mmin / mmax)
    sky = str(support["sky"])
    checks: list[dict[str, object]] = []

    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})

    # -- sky convention ---------------------------------------------------
    for label, catalog in (("pe", posterior), ("selection", selection)):
        coords = tuple(catalog.basis.coordinates)
        has_sky = all(x in coords for x in _SKY)
        declared = _metadata(catalog).get("sky_marginal")
        if sky == "marginalized":
            ok = not any(x in coords for x in _SKY) and declared is not False
        else:
            ok = has_sky and declared is not True
        check(f"sky.{label}", ok, model_sky=sky, basis=catalog.basis.name,
              basis_has_sky=has_sky, declared_sky_marginal=declared)

    # -- zmax ---------------------------------------------------------------
    declared_sel = _metadata(selection).get("z_max")
    check("zmax.selection_declared", declared_sel is not None and float(declared_sel) == zmax,
          model_zmax=zmax, dataset_z_max=None if declared_sel is None else float(declared_sel))
    declared_pe = _metadata(posterior).get("z_max")
    if declared_pe is not None:
        check("zmax.pe_declared", float(declared_pe) == zmax, model_zmax=zmax,
              dataset_z_max=float(declared_pe))
    if policy is not None:
        value = policy.get("z_max") if isinstance(policy, Mapping) else getattr(policy, "z_max")
        check("zmax.policy", float(value) == zmax, model_zmax=zmax, policy_z_max=float(value))

    # -- per-event support ---------------------------------------------------
    cosmo = compiled.cosmology
    m1, q, z = _source_frame(posterior.samples, cosmo)
    inside = _inside(m1, q, z, mmin=mmin, mmax=mmax, q_floor=q_floor, zmax=zmax)
    fractions, above = [], []
    for i in range(posterior.n_events):
        sl = posterior.event_slice(i)
        fractions.append(float(np.mean(inside[sl])))
        above.append(float(np.mean(z[sl] > zmax)))
    fractions = np.asarray(fractions)
    unsupported = [posterior.event_names[i] for i in np.flatnonzero(fractions <= 0.0)]
    worst = int(np.argmin(fractions))
    check("support.every_event_has_supported_samples", not unsupported,
          unsupported_events=unsupported[:20], n_unsupported=len(unsupported))

    s_m1, s_q, s_z = _source_frame(selection.samples, cosmo)
    s_inside = _inside(s_m1, s_q, s_z, mmin=mmin, mmax=mmax, q_floor=q_floor, zmax=zmax)

    # -- injection draw support ---------------------------------------------
    declared = draw_support if draw_support is not None else _metadata(selection).get("draw_support")
    draw_status = "not_declared"
    if declared is not None:
        declared = {k: float(v) for k, v in dict(declared).items() if k in _DRAW_KEYS and v is not None}
        draw_status = "declared"
        needs = {
            "m1_source_min": ("<=", mmin), "m1_source_max": (">=", mmax),
            "q_min": ("<=", q_edge),
            "z_max": (">=", zmax * (1.0 - EDGE_REACH_TOLERANCE)),
        }
        for key, value in declared.items():
            op, bound = needs[key]
            ok = value <= bound if op == "<=" else value >= bound
            check(f"draw_support.declared.{key}", ok, declared=value, population_bound=bound)
    edges = {
        "m1_upper": _edge_test(s_m1, mmax, "upper"),
        "m1_lower": _edge_test(s_m1, mmin, "lower"),
        # the population's lowest reachable q is max(q_floor, mmin / mmax)
        # (m2 >= mmin), not q_floor itself
        "q_lower": _edge_test(s_q, q_edge, "lower"),
        "z_upper": _edge_test(s_z, zmax, "upper"),
    }
    for name, row in edges.items():
        check(f"draw_support.found_edge.{name}", row["passed"],
              **{k: v for k, v in row.items() if k != "passed"})
    report = {
        "format_version": V2_DATA_SUPPORT_FORMAT,
        "model_hash": compiled.spec.model_hash,
        "support": support,
        "basis_identity": posterior.basis.identity,
        "checks": checks,
        "pass": all(c["passed"] for c in checks),
        "reported": {
            "pe_min_event_fraction_in_support": float(fractions[worst]),
            "pe_least_supported_event": posterior.event_names[worst],
            "pe_max_event_fraction_above_zmax": float(np.max(above)) if above else 0.0,
            "pe_fraction_above_zmax": float(np.mean(z > zmax)),
            "selection_rows": int(selection.n_selected),
            "selection_rows_above_zmax": int(np.sum(s_z > zmax)),
            "selection_rows_below_q_floor": int(np.sum(s_q < q_floor)),
            "population_q_lower_edge": float(q_edge),
            "selection_fraction_in_support": float(np.mean(s_inside)),
            "draw_support_declaration": draw_status,
        },
    }
    return report


def require_v2_data_support(model, posterior, selection, *, policy=None, draw_support=None,
                            context: str = "") -> dict[str, object]:
    """:func:`v2_data_support_report`, raising :class:`V2DataSupportError` on a failure."""
    report = v2_data_support_report(model, posterior, selection, policy=policy, draw_support=draw_support)
    if not report["pass"]:
        failed = [c for c in report["checks"] if not c["passed"]]
        where = f" ({context})" if context else ""
        raise V2DataSupportError(
            f"dataset does not match the v2 model support{where}: "
            + "; ".join(f"{c['name']}: {({k: v for k, v in c.items() if k not in ('name', 'passed')})}" for c in failed)
        )
    return report
