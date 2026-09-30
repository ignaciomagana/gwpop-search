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


def v2_data_support_report(model, posterior, selection, *, policy=None) -> dict[str, object]:
    """Check a v2 model's declared support against a canonical PE/selection pair.

    ``model`` is a v2 :class:`~gwpop_search.grammar.ModelSpec` or its compiled
    model; ``policy`` an optional :class:`GwcatV2DataPolicy` (or anything with
    a ``z_max`` attribute / key). Returns a JSON-ready report with ``pass``.
    """
    compiled = _compiled(model)
    if not compiled.is_v2:
        raise ValueError("the v2 data-support check applies to v2 models only")
    support = dict(compiled.spec.support)
    zmax = float(support["zmax"])
    q_floor = float(support["q_floor"])
    mmin, mmax = float(support["mmin"]), float(support["mmax"])
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
            "selection_fraction_in_support": float(np.mean(s_inside)),
        },
    }
    return report


def require_v2_data_support(model, posterior, selection, *, policy=None, context: str = "") -> dict[str, object]:
    """:func:`v2_data_support_report`, raising :class:`V2DataSupportError` on a failure."""
    report = v2_data_support_report(model, posterior, selection, policy=policy)
    if not report["pass"]:
        failed = [c for c in report["checks"] if not c["passed"]]
        where = f" ({context})" if context else ""
        raise V2DataSupportError(
            f"dataset does not match the v2 model support{where}: "
            + "; ".join(f"{c['name']}: {({k: v for k, v in c.items() if k not in ('name', 'passed')})}" for c in failed)
        )
    return report
