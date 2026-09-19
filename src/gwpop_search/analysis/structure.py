"""Structural atoms, effective model identity and model-space probabilities.

* ``atoms_of(root, model)``: the structural atoms of ``model`` relative to the
  root, ``"<block>.family=<family>"`` or ``"<block>.options.<option>=<value>"``;
  ``axes_of`` drops the values (``structural_diff_axes``).
* Effective model identity (MODEL_COMPARISON_MATH.md Sec. 1.4, finding F1):
  two graph nodes are the same statistical model when they have the same
  structure and the same priors on the hyperparameters the compiled density
  actually uses. A prior on a *dead* parameter (kept by an option replacement)
  integrates to one and does not change the evidence, but it changes the
  model hash. Used parameters are found exactly from the traced computation
  graph (``jax.make_jaxpr`` plus backward reachability from the output), not
  from a hand-written table.
* Posterior model probabilities ``p(M | D) ∝ p(M) Z_M`` and structural masses
  ``P(S | D) = sum_{M ∋ S} p(M | D)`` with prior masses ``P(S)`` and the
  model-averaged structural Bayes factor
  ``BF_S = [P(S|D) / (1 - P(S|D))] / [P(S) / (1 - P(S))]`` (formula (5)).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from gwpop_search.grammar import ModelSpec, structural_diff_axes

from ._common import AnalysisInputError


def atoms_of(root: ModelSpec, model: ModelSpec) -> frozenset[str]:
    atoms = []
    for axis in structural_diff_axes(root, model):
        block = axis.split(".")[0]
        spec = getattr(model, block)
        if axis.endswith(".family"):
            atoms.append(f"{block}.family={spec.family}")
        else:
            option = axis.split(".options.")[1]
            atoms.append(f"{axis}={json.dumps(spec.options.get(option))}")
    return frozenset(atoms)


def axes_of(root: ModelSpec, model: ModelSpec) -> frozenset[str]:
    return frozenset(structural_diff_axes(root, model))


def edge_atom(root: ModelSpec, parent: ModelSpec, child: ModelSpec) -> str:
    """The atom an edge isolates (``atoms(child) - atoms(parent)``, else the removed one).

    For an option replacement the new option value is the atom; for a change
    that removes an atom present in the parent (relative to the root) the
    removed atom is reported with a ``not:`` prefix.
    """
    added = atoms_of(root, child) - atoms_of(root, parent)
    if len(added) == 1:
        return next(iter(added))
    removed = atoms_of(root, parent) - atoms_of(root, child)
    if not added and len(removed) == 1:
        return "not:" + next(iter(removed))
    raise AnalysisInputError(
        f"edge {parent.short_hash}->{child.short_hash} does not isolate one atom "
        f"(added {sorted(added)}, removed {sorted(removed)})"
    )


# ---------------------------------------------------------------------------
# Effective model identity
# ---------------------------------------------------------------------------


def used_hyperparameters(population_model, names: Sequence[str], samples: Mapping[str, object]) -> tuple[str, ...]:
    """Hyperparameters that reach the output of ``population_model(samples, hp)``.

    Traced with ``jax.make_jaxpr`` (no values are evaluated) and resolved by
    backward reachability over the equations; any equation with a needed
    output marks all of its inputs needed (conservative for nested jaxprs).
    """
    import jax
    import jax.numpy as jnp
    from jax.extend.core import Var

    names = tuple(str(name) for name in names)

    def fn(values):
        return population_model(samples, dict(zip(names, values)))

    closed = jax.make_jaxpr(fn)([jnp.asarray(0.5, dtype=jnp.float64) for _ in names])
    jaxpr = closed.jaxpr
    needed = {v for v in jaxpr.outvars if isinstance(v, Var)}
    for eqn in reversed(jaxpr.eqns):
        if any(isinstance(v, Var) and v in needed for v in eqn.outvars):
            needed.update(v for v in eqn.invars if isinstance(v, Var))
    return tuple(name for name, var in zip(names, jaxpr.invars) if var in needed)


def _dummy_samples(population_model, n: int = 4) -> dict[str, np.ndarray]:
    fields = getattr(population_model, "required_fields", None)
    if not fields:
        raise AnalysisInputError("population model declares no required_fields to trace with")
    return {str(name): np.linspace(0.2, 0.8, n) for name in fields}


@dataclass(frozen=True)
class EffectiveModel:
    model_hash: str
    identity: str
    used_parameters: tuple[str, ...]
    dead_parameters: tuple[str, ...]


def effective_model(spec: ModelSpec) -> EffectiveModel:
    """Effective identity: structure plus the priors of used hyperparameters."""
    from gwpop_search.models import compile_model_spec

    model = compile_model_spec(spec)
    names = tuple(sorted(spec.priors))
    used = used_hyperparameters(model, names, _dummy_samples(model))
    dead = tuple(name for name in names if name not in used)
    payload = {
        "structure": spec.structure_dict(),
        "priors": {name: spec.priors[name].to_dict() for name in used},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return EffectiveModel(
        model_hash=spec.model_hash,
        identity=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        used_parameters=used,
        dead_parameters=dead,
    )


@dataclass(frozen=True)
class AliasGroups:
    """Graph nodes grouped by effective identity (first node in graph order is canonical)."""

    canonical: Mapping[str, str]  # model hash -> canonical model hash
    groups: Mapping[str, tuple[str, ...]]  # canonical -> all members
    effective: Mapping[str, EffectiveModel]

    @property
    def aliases(self) -> dict[str, tuple[str, ...]]:
        return {k: v for k, v in self.groups.items() if len(v) > 1}

    def to_dict(self) -> dict[str, object]:
        return {
            "n_nodes": len(self.canonical),
            "n_effective_models": len(self.groups),
            "alias_groups": {k: list(v) for k, v in self.aliases.items()},
            "dead_parameters": {
                h: list(e.dead_parameters) for h, e in self.effective.items() if e.dead_parameters
            },
        }


def find_alias_groups(specs: Sequence[ModelSpec]) -> AliasGroups:
    effective = {spec.model_hash: effective_model(spec) for spec in specs}
    canonical: dict[str, str] = {}
    first: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for spec in specs:
        ident = effective[spec.model_hash].identity
        head = first.setdefault(ident, spec.model_hash)
        canonical[spec.model_hash] = head
        groups.setdefault(head, []).append(spec.model_hash)
    return AliasGroups(
        canonical=canonical,
        groups={k: tuple(v) for k, v in groups.items()},
        effective=effective,
    )


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


def normalized_log_prior(
    specs: Sequence[ModelSpec], root: ModelSpec, model_prior
) -> dict[str, float]:
    values = {}
    for spec in specs:
        value = float(model_prior.log_prior(spec, root=root))
        if not math.isfinite(value):
            raise AnalysisInputError(f"model prior is not finite for {spec.model_hash}")
        values[spec.model_hash] = value
    total = logsumexp(list(values.values()))
    return {k: v - total for k, v in values.items()}


def posterior_probabilities(
    log_evidence: Mapping[str, float], log_prior: Mapping[str, float]
) -> dict[str, float]:
    keys = sorted(log_evidence)
    if set(log_prior) != set(keys):
        raise AnalysisInputError("log priors must cover exactly the models with evidence")
    lw = np.asarray([log_evidence[k] + log_prior[k] for k in keys])
    return dict(zip(keys, np.exp(lw - logsumexp(lw)).tolist()))


def structural_masses(
    root: ModelSpec,
    specs: Mapping[str, ModelSpec],
    posterior: Mapping[str, float],
    prior: Mapping[str, float],
    *,
    level: str = "atom",
) -> dict[str, dict[str, float | None]]:
    """``P(S|D)``, ``P(S)``, ``BF_S`` and ``ln BF_S`` for every atom (or axis)."""
    if level not in {"atom", "axis"}:
        raise ValueError("level must be 'atom' or 'axis'")
    membership: dict[str, list[str]] = {}
    for key, spec in specs.items():
        labels = atoms_of(root, spec) if level == "atom" else axes_of(root, spec)
        for label in labels:
            membership.setdefault(label, []).append(key)
    out = {}
    for label, members in sorted(membership.items()):
        post = float(sum(posterior[m] for m in members))
        pri = float(sum(prior[m] for m in members))
        out[label] = {
            "prior_mass": pri,
            "posterior_mass": post,
            **structural_bayes_factor(post, pri),
            "members": sorted(members),
        }
    return out


def structural_bayes_factor(posterior_mass: float, prior_mass: float) -> dict[str, float | None]:
    """``BF_S`` (``None`` when a mass is 0 or 1, where the odds are undefined)."""
    if not (0.0 < prior_mass < 1.0) or not (0.0 < posterior_mass < 1.0):
        return {"bayes_factor": None, "log_bayes_factor": None}
    log_bf = (
        math.log(posterior_mass) - math.log1p(-posterior_mass)
        - math.log(prior_mass) + math.log1p(-prior_mass)
    )
    return {"bayes_factor": math.exp(log_bf), "log_bayes_factor": log_bf}


def membership_by_atom(root: ModelSpec, specs: Mapping[str, ModelSpec]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, spec in specs.items():
        for label in atoms_of(root, spec):
            out.setdefault(label, []).append(key)
    return out


def complexity_prior(penalty: float):
    from gwpop_search.search import ComplexityModelPrior

    return ComplexityModelPrior(penalty_per_axis=float(penalty))


def iter_lambda_variants(lambdas: Iterable[float]) -> list[float]:
    values = [float(x) for x in lambdas]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("model-prior penalties must be finite and non-negative")
    return values
