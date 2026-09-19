"""Prior sensitivity of evidences, Bayes factors and model probabilities.

Hyperprior width (MODEL_COMPARISON_MATH.md Sec. 3):

* **Narrowed prior** ``pi'`` with ``supp pi' ⊆ supp pi`` (exact)::

      Z' = Z * E_post[pi'(Lambda) / pi(Lambda)]   =>   ln Z' - ln Z = ln sum_k W_k r_k

  estimated from the weighted posterior points ``W_k`` of every repeat (pooled
  as the equal mixture of runs and also per repeat, for the repeat scatter),
  with the reweighting ESS ``(sum W r)^2 / sum (W r)^2``. The narrowed prior
  keeps only the posterior draws inside its support, so the ESS collapses when
  the posterior sits away from the new support: below
  ``min_reweighting_ess`` no number is reported (flag
  ``low_reweighting_ess``), because an importance estimate from a handful of
  effective draws is not a prior-robustness result. Without that refusal a
  single surviving draw yields a confident-looking multi-nat shift.
* **Widened prior** (support beyond the sampled region): only the analytic
  Occam relation is used, and only when the posterior is contained in the old
  prior (negligible posterior mass near both bounds). Widening a uniform (or
  log-uniform, in log space) prior by a factor ``kappa`` about its centre
  lowers the prior density by ``1/kappa`` wherever the likelihood lives, so
  ``ln Z' - ln Z = -ln kappa``. Otherwise the result is flagged
  ``posterior_not_contained`` and no number is given (a rerun is needed).
* ``d ln BF / d ln W`` per parameter from the halved and doubled widths.
* **Model-prior variants**: posterior model probabilities and structural
  masses for the complexity penalty ``lambda in {0, ln 2, ln 4}`` (formula (5)).

For an edge, the parameters that exist only in the child (added) are varied
in the child's evidence and those that exist only in the parent (removed) in
the parent's, so ``ln BF_{child/parent}`` changes by ``+Delta ln Z_child`` or
``-Delta ln Z_parent``.

**Parameters whose null sits at a prior bound** (the mixture fractions
``peak_fraction``, ``peak1_fraction``, ``peak2_fraction``, ``chi_fraction``,
whose edges test ``f = 0``) are rescaled about the *bound that carries the
null*, not about the prior's centre: halving ``U(0, 0.5)`` about its centre
gives ``U(0.125, 0.375)``, which deletes the tested hypothesis from the prior,
so the variant is not a width perturbation of the hypothesis at all. Anchored
at the bound it gives ``U(0, 0.25)``, which is. The anchor is read off the
edge's nesting embeddings (:func:`gwpop_search.analysis.sddr.classify_edge`)
and reported per parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from gwpop_search.grammar import ModelSpec
from gwpop_search.inference.priors import PriorSpec

from ._common import AnalysisInputError, WeightedPosterior, as_float, json_ready
from .sddr import prior_log_density_array
from .structure import (
    complexity_prior,
    iter_lambda_variants,
    normalized_log_prior,
    posterior_probabilities,
    structural_masses,
)

PRIOR_SENSITIVITY_FORMAT = "gwpop-search-prior-sensitivity-1.0"
DEFAULT_LAMBDAS = (0.0, math.log(2.0), math.log(4.0))


def _as_prior_spec(prior) -> PriorSpec:
    if isinstance(prior, PriorSpec):
        return prior
    params = dict(prior.parameters)
    if prior.family in {"uniform", "log_uniform"}:
        return PriorSpec(prior.family, low=params["low"], high=params["high"])
    return PriorSpec("normal", loc=params["loc"], scale=params["scale"])


def rescaled_prior(prior, factor: float, *, anchor: str | None = None) -> PriorSpec:
    """Width multiplied by ``factor``.

    ``anchor=None`` rescales about the centre (log-centre for a log-uniform
    prior). ``anchor="low"``/``"high"`` holds that bound fixed and moves the
    other, which is what a parameter whose tested null sits at a bound needs:
    rescaling about the centre would move the null out of the prior and the
    variant would no longer perturb the hypothesis under test. A normal prior
    has no bounds, so it is always rescaled about its location.
    """
    prior = _as_prior_spec(prior)
    factor = as_float("factor", factor, positive=True)
    if anchor not in (None, "low", "high"):
        raise ValueError("anchor must be None, 'low' or 'high'")
    if prior.family == "uniform":
        width = prior.high - prior.low
        if anchor == "low":
            return PriorSpec("uniform", low=prior.low, high=prior.low + factor * width)
        if anchor == "high":
            return PriorSpec("uniform", low=prior.high - factor * width, high=prior.high)
        centre, half = 0.5 * (prior.low + prior.high), 0.5 * width
        return PriorSpec("uniform", low=centre - factor * half, high=centre + factor * half)
    if prior.family == "log_uniform":
        lo, hi = math.log(prior.low), math.log(prior.high)
        if anchor == "low":
            return PriorSpec("log_uniform", low=prior.low, high=math.exp(lo + factor * (hi - lo)))
        if anchor == "high":
            return PriorSpec("log_uniform", low=math.exp(hi - factor * (hi - lo)), high=prior.high)
        lc, lh = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return PriorSpec("log_uniform", low=math.exp(lc - factor * lh), high=math.exp(lc + factor * lh))
    return PriorSpec("normal", loc=prior.loc, scale=factor * prior.scale)


def _contained_support(new: PriorSpec, old: PriorSpec) -> bool:
    if old.family == "normal":
        return True  # every support is inside the real line; the ratio stays finite
    if new.family == "normal":
        return False
    return old.low <= new.low and new.high <= old.high


#: Reweighting ESS below which a narrowed-prior estimate is not reported.
#: The self-normalized importance estimate has relative standard error of order
#: ``1/sqrt(ESS)``, so 50 effective draws is already a ~14% error on ``E[pi'/pi]``
#: and the shift enters ``ln BF`` directly. Below it the estimator is declining,
#: not agreeing: with ``peak_fraction ~ U(0, 0.5)`` halved about its centre and a
#: posterior piled up near 0, one surviving draw of 4096 (ESS 1.0) returns
#: ``Delta ln Z = -7.6`` while neighbouring seeds return nothing at all.
MIN_REWEIGHTING_ESS = 50.0


def prior_reweighting(
    sample: WeightedPosterior,
    parameter: str,
    old_prior,
    new_prior,
    *,
    min_ess: float = MIN_REWEIGHTING_ESS,
) -> dict[str, object]:
    """Exact ``ln Z' - ln Z`` for a prior change with ``supp pi' ⊆ supp pi``.

    ``delta_log_evidence`` is ``None`` with ``flag="low_reweighting_ess"`` when
    the reweighting ESS falls below ``min_ess`` (the raw value is kept as
    ``delta_log_evidence_unreliable`` so the refusal can be inspected).
    """
    old, new = _as_prior_spec(old_prior), _as_prior_spec(new_prior)
    min_ess = as_float("min_ess", min_ess, nonnegative=True)
    if not _contained_support(new, old):
        raise AnalysisInputError(
            f"{parameter}: the new prior's support is not inside the sampled prior's; "
            "reweighting cannot extend the prior (rerun, or use the Occam relation)"
        )
    values = sample.column(parameter)
    log_r = prior_log_density_array(new, values) - prior_log_density_array(old, values)
    log_w = np.log(sample.weights)
    terms = log_w + log_r
    total = float(logsumexp(terms))
    if not math.isfinite(total):
        return {"parameter": parameter, "old_prior": old.to_dict(), "new_prior": new.to_dict(),
                "delta_log_evidence": None, "reweighting_ess": 0.0, "min_reweighting_ess": min_ess,
                "flag": "no_posterior_mass",
                "reason": "no posterior mass inside the new prior's support"}
    wn = np.exp(terms - total)
    ess = float(1.0 / np.sum(wn * wn))
    r = np.exp(log_r)
    mean_r = math.exp(total)
    se = float(math.sqrt(float(np.sum(sample.weights**2 * (r - mean_r) ** 2))) / mean_r)
    per_run = []
    for k in range(sample.n_runs):
        mask = sample.run_index == k
        if mask.any():
            wk = log_w[mask] - logsumexp(log_w[mask])
            per_run.append(float(logsumexp(wk + log_r[mask])))
    out = {
        "parameter": parameter,
        "old_prior": old.to_dict(),
        "new_prior": new.to_dict(),
        "delta_log_evidence": total,
        "importance_se": se,
        "per_run_delta_log_evidence": per_run,
        "repeat_std": float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None,
        "reweighting_ess": ess,
        "min_reweighting_ess": min_ess,
        "base_kish_ess": sample.kish_ess,
        "flag": None,
    }
    if ess < min_ess:
        out["delta_log_evidence_unreliable"] = total
        out["delta_log_evidence"] = None
        out["flag"] = "low_reweighting_ess"
        out["reason"] = (
            f"reweighting ESS {ess:.3g} < {min_ess:g}: too few posterior draws inside the "
            "narrowed prior's support to estimate E[pi'/pi]"
        )
    return out


def posterior_edge_mass(
    sample: WeightedPosterior, parameter: str, prior, *, edge_fraction: float = 0.05
) -> dict[str, float]:
    """Posterior mass within ``edge_fraction`` of the prior range of each bound
    (log space for log-uniform priors)."""
    prior = _as_prior_spec(prior)
    if prior.family == "normal":
        return {"lower": 0.0, "upper": 0.0}
    values = sample.column(parameter)
    if prior.family == "log_uniform":
        values = np.log(values)
        lo, hi = math.log(prior.low), math.log(prior.high)
    else:
        lo, hi = prior.low, prior.high
    width = edge_fraction * (hi - lo)
    return {
        "lower": float(np.sum(sample.weights[values <= lo + width])),
        "upper": float(np.sum(sample.weights[values >= hi - width])),
    }


def occam_widening(
    sample: WeightedPosterior,
    parameter: str,
    prior,
    *,
    factor: float = 2.0,
    edge_fraction: float = 0.05,
    max_edge_mass: float = 1e-3,
) -> dict[str, object]:
    """``ln Z' - ln Z = -ln(factor)`` for a widened prior, if the posterior is contained."""
    factor = as_float("factor", factor, positive=True)
    if factor < 1.0:
        raise ValueError("the Occam relation is for widening (factor >= 1); narrow by reweighting")
    prior = _as_prior_spec(prior)
    edges = posterior_edge_mass(sample, parameter, prior, edge_fraction=edge_fraction)
    contained = prior.family != "normal" and max(edges.values()) <= max_edge_mass
    return {
        "parameter": parameter,
        "factor": factor,
        "new_prior": rescaled_prior(prior, factor).to_dict(),
        "posterior_edge_mass": edges,
        "edge_fraction": edge_fraction,
        "max_edge_mass": max_edge_mass,
        "contained": bool(contained),
        "delta_log_evidence": -math.log(factor) if contained else None,
        "flag": None if contained else "posterior_not_contained",
    }


def null_anchors(parent: ModelSpec, child: ModelSpec) -> dict[str, str]:
    """``{parameter: 'low' | 'high'}`` for edge nulls that sit at a prior bound.

    Read off the edge's nesting embeddings. A parameter listed here must be
    rescaled about that bound (see the module docstring): moving it would delete
    the tested null from the prior.
    """
    from .sddr import classify_edge

    try:
        nesting = classify_edge(parent, child, "")
    except (KeyError, AttributeError, ValueError):
        # a pair the grammar cannot classify as an edge has no registered null,
        # so there is no bound to anchor at; the centred rescaling stands
        return {}
    return {
        emb.tested_parameter: emb.location
        for emb in nesting.embeddings
        if emb.location in {"lower", "upper"}
    }


def edge_prior_sensitivity(
    parent: ModelSpec,
    child: ModelSpec,
    parent_sample: WeightedPosterior,
    child_sample: WeightedPosterior,
    *,
    log_bayes_factor: float | None = None,
    factor: float = 2.0,
    edge_fraction: float = 0.05,
    max_edge_mass: float = 1e-3,
    parameters: Iterable[str] | None = None,
    min_reweighting_ess: float = MIN_REWEIGHTING_ESS,
) -> dict[str, object]:
    """Width sensitivity of ``ln BF_{child/parent}`` for the edge-specific parameters."""
    factor = as_float("factor", factor, positive=True)
    if factor <= 1.0:
        raise ValueError("factor must exceed 1 (the width is divided and multiplied by it)")
    added = sorted(set(child.priors) - set(parent.priors))
    removed = sorted(set(parent.priors) - set(child.priors))
    selected = None if parameters is None else set(parameters)
    anchors = {"lower": "low", "upper": "high"}
    bound_nulls = null_anchors(parent, child)
    rows = []
    for owner, names, spec, sample, sign in (
        ("child", added, child, child_sample, 1.0),
        ("parent", removed, parent, parent_sample, -1.0),
    ):
        for name in names:
            if selected is not None and name not in selected:
                continue
            prior = spec.priors[name]
            anchor = anchors.get(bound_nulls.get(name, ""))
            narrow = prior_reweighting(
                sample, name, prior, rescaled_prior(prior, 1.0 / factor, anchor=anchor),
                min_ess=min_reweighting_ess,
            )
            wide = occam_widening(
                sample, name, prior, factor=factor, edge_fraction=edge_fraction,
                max_edge_mass=max_edge_mass,
            )
            d_narrow = narrow.get("delta_log_evidence")
            d_wide = wide.get("delta_log_evidence")
            row = {
                "parameter": name,
                "model": owner,
                "null_at_prior_bound": bound_nulls.get(name),
                "narrow_anchor": anchor,
                "narrowed": narrow,
                "widened": wide,
                "delta_log_bayes_factor_narrowed": None if d_narrow is None else sign * d_narrow,
                "delta_log_bayes_factor_widened": None if d_wide is None else sign * d_wide,
            }
            if d_narrow is not None and d_wide is not None:
                row["dlogbf_dlogwidth"] = sign * (d_wide - d_narrow) / (2.0 * math.log(factor))
            elif d_narrow is not None:
                row["dlogbf_dlogwidth_narrow_side"] = sign * (-d_narrow) / math.log(factor)
            if log_bayes_factor is not None:
                row["log_bayes_factor_narrowed"] = (
                    None if d_narrow is None else log_bayes_factor + sign * d_narrow
                )
                row["log_bayes_factor_widened"] = (
                    None if d_wide is None else log_bayes_factor + sign * d_wide
                )
            rows.append(row)
    variants = []
    if log_bayes_factor is not None:
        for row in rows:
            for key, side in (("log_bayes_factor_narrowed", "narrowed"),
                              ("log_bayes_factor_widened", "widened")):
                variants.append({
                    "parameter": row["parameter"],
                    "variant": side,
                    "log_bayes_factor": row.get(key),
                    "anchor": row["narrow_anchor"] if side == "narrowed" else None,
                    # why a value is absent: low reweighting ESS / no posterior
                    # mass (narrowed) or a posterior not contained (widened)
                    "flag": row[side].get("flag"),
                    "reweighting_ess": row["narrowed"].get("reweighting_ess") if side == "narrowed" else None,
                })
    return json_ready(
        {
            "parent_hash": parent.model_hash,
            "child_hash": child.model_hash,
            "factor": factor,
            "log_bayes_factor": log_bayes_factor,
            "parameters": rows,
            "variants": variants,
        }
    )


@dataclass(frozen=True)
class ModelPriorVariant:
    penalty: float
    posterior: Mapping[str, float]
    prior: Mapping[str, float]
    atoms: Mapping[str, Mapping[str, object]]
    axes: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return json_ready(
            {
                "penalty_per_axis": self.penalty,
                "posterior_model_probabilities": dict(self.posterior),
                "prior_model_probabilities": dict(self.prior),
                "atoms": {k: dict(v) for k, v in self.atoms.items()},
                "axes": {k: dict(v) for k, v in self.axes.items()},
            }
        )


def model_prior_variants(
    root: ModelSpec,
    specs: Mapping[str, ModelSpec],
    log_evidence: Mapping[str, float],
    *,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> list[ModelPriorVariant]:
    """Posterior probabilities and structural masses per complexity penalty.

    ``specs``/``log_evidence`` must describe the deduplicated model set (one
    entry per effective model; see :func:`structure.find_alias_groups`).
    """
    if set(specs) != set(log_evidence):
        raise AnalysisInputError("specs and log evidences must cover the same models")
    out = []
    ordered = [specs[k] for k in sorted(specs)]
    for penalty in iter_lambda_variants(lambdas):
        log_prior = normalized_log_prior(ordered, root, complexity_prior(penalty))
        post = posterior_probabilities(log_evidence, log_prior)
        prior = {k: math.exp(v) for k, v in log_prior.items()}
        out.append(
            ModelPriorVariant(
                penalty=float(penalty),
                posterior=post,
                prior=prior,
                atoms=structural_masses(root, specs, post, prior, level="atom"),
                axes=structural_masses(root, specs, post, prior, level="axis"),
            )
        )
    return out


def prior_sensitivity_report(
    edges: Sequence[Mapping[str, object]],
    variants: Sequence[ModelPriorVariant] = (),
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "format_version": PRIOR_SENSITIVITY_FORMAT,
        "edges": list(edges),
        "model_prior_variants": [variant.to_dict() for variant in variants],
    }
    if extra:
        payload.update(dict(extra))
    return json_ready(payload)
