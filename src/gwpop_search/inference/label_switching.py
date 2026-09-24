"""Evidence-preserving canonical labels for exactly exchangeable mixture components.

``chieff.family.gaussian_mixture`` has two truncated-Gaussian chi_eff
components with identical priors (``chi_mu_k ~ U(-0.5, 0.5)``,
``chi_sigma_k ~ LU(0.02, 0.5)``) and a mixture weight ``chi_fraction ~ U(0, 1)``
of component 2. Prior and likelihood are exactly invariant under the label
swap

    S: (mu_1, sigma_1, mu_2, sigma_2, f) -> (mu_2, sigma_2, mu_1, sigma_1, 1 - f),

so the posterior has two mirror modes, coordinate-wise medians are
meaningless and a run that loses one mode underestimates ``ln Z`` by up to
``ln 2``.

The canonicalisation used here is a *bijective order-statistic prior
transform* (design note ``MODEL_COMPARISON_MATH.md`` Sec. 7.5). With ``F`` the
common prior CDF of the ordered pair and ``(u_1, u_2)`` their unit-cube
coordinates::

    v_1 = 1 - sqrt(1 - u_1)          (evaluated as u_1 / (1 + sqrt(1 - u_1)))
    v_2 = v_1 + u_2 (1 - v_1)
    mu_(1) = F^-1(v_1) <= mu_(2) = F^-1(v_2)

``v_1`` is distributed as the minimum of two iid U(0, 1) variables (density
``2 (1 - v)``) and, given ``v_1``, ``v_2`` is uniform on ``[v_1, 1]``, so the
map sends the unit square one-to-one onto the half-space ``mu_(1) <= mu_(2)``
with density ``2 pi(mu_1) pi(mu_2)`` (Jacobian ``1/2``); every other
coordinate keeps its own prior. Because ``S`` maps the other half-space onto
this one while preserving prior and likelihood,

    Z = int L pi = 2 int_{mu_1 <= mu_2} L pi = int_{[0,1]^d} L(T(u)) du,

i.e. the evidence is *identical*; only the parameterization dynesty explores
changes (one mode instead of two). Posterior samples are then on canonical
labels: component 1 has the smaller mean and ``chi_fraction`` is the weight
of the larger-mean component. Merely sorting two iid draws would not help:
the likelihood would stay mirror-symmetric in the unit cube.

Exchangeability is verified against the declared priors before the
transform is used (identical priors for the swapped pairs, a mixture-weight
prior symmetric under ``f -> 1 - f``); anything else is refused because the
canonicalisation would then change the evidence. ``mass.family.pl_two_peak``
has different peak priors and is therefore never canonicalised.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .priors import PriorSpec

PARAMETERIZATION_FORMAT_VERSION = "gwpop-search-parameterization-1.0"
IDENTITY_PARAMETERIZATION = "identity"
ORDERED_PAIRS_PARAMETERIZATION = "ordered_exchangeable_pairs"


@dataclass(frozen=True)
class ExchangeableComponents:
    """One exactly exchangeable pair of population components.

    ``order_by`` are the parameters used to order the pair (the first becomes
    the smaller one); ``swapped`` are companion parameter pairs exchanged by
    the label swap; ``reflected`` are parameters mapped ``x -> 1 - x`` by the
    swap (mixture weights). ``source`` names the grammar atom.
    """

    order_by: tuple[str, str]
    swapped: tuple[tuple[str, str], ...] = ()
    reflected: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        order_by = tuple(str(name) for name in self.order_by)
        if len(order_by) != 2 or order_by[0] == order_by[1]:
            raise ValueError("order_by must name two distinct parameters")
        swapped = tuple(tuple(str(name) for name in pair) for pair in self.swapped)
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in swapped):
            raise ValueError("every swapped entry must name two distinct parameters")
        reflected = tuple(str(name) for name in self.reflected)
        names = [*order_by, *(name for pair in swapped for name in pair), *reflected]
        if len(set(names)) != len(names):
            raise ValueError("a parameter can appear only once in an exchangeable group")
        object.__setattr__(self, "order_by", order_by)
        object.__setattr__(self, "swapped", swapped)
        object.__setattr__(self, "reflected", reflected)

    @property
    def parameters(self) -> tuple[str, ...]:
        return (
            *self.order_by,
            *(name for pair in self.swapped for name in pair),
            *self.reflected,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "order_by": list(self.order_by),
            "swapped": [list(pair) for pair in self.swapped],
            "reflected": list(self.reflected),
            "source": self.source,
        }

    def exchangeability_violations(self, priors: Mapping[str, PriorSpec]) -> tuple[str, ...]:
        """Reasons the declared priors are not invariant under the label swap."""
        problems: list[str] = []
        for name in self.parameters:
            if name not in priors:
                problems.append(f"no prior for {name!r}")
        if problems:
            return tuple(problems)
        for a, b in (self.order_by, *self.swapped):
            if priors[a] != priors[b]:
                problems.append(
                    f"priors of {a!r} and {b!r} differ ({priors[a].to_dict()} vs "
                    f"{priors[b].to_dict()})"
                )
        for name in self.reflected:
            spec = priors[name]
            symmetric = (
                spec.family == "uniform"
                and math.isclose(float(spec.low) + float(spec.high), 1.0, abs_tol=1e-15)
            )
            if not symmetric:
                problems.append(
                    f"prior of {name!r} ({spec.to_dict()}) is not symmetric under x -> 1 - x"
                )
        return tuple(problems)


def exchangeable_components_for_spec(spec) -> tuple[ExchangeableComponents, ...]:
    """Exactly exchangeable component groups of a declarative model spec.

    Only ``chieff.family.gaussian_mixture`` with a constant mixture fraction
    qualifies in the current grammar (``chi_eff_mixture_logpdf`` weights
    component 1 by ``1 - chi_fraction`` and component 2 by ``chi_fraction``,
    both with the same truncated-normal density). The follow-up
    ``fraction_dependence="logistic_q"`` mixture is *not* exchangeable (its
    weight depends on q and its component priors differ by design), so it is
    sampled in the identity parameterization with labels fixed by the priors.
    """
    chieff = getattr(spec, "chieff", None)
    if chieff is None or chieff.family != "gaussian_mixture":
        return ()
    if chieff.options.get("fraction_dependence", "constant") != "constant":
        return ()
    return (
        ExchangeableComponents(
            order_by=("chi_mu_1", "chi_mu_2"),
            swapped=(("chi_sigma_1", "chi_sigma_2"),),
            reflected=("chi_fraction",),
            source="chieff.family.gaussian_mixture",
        ),
    )


class OrderedPairPriorTransform:
    """Picklable prior transform with order-statistic coordinates for exchangeable pairs.

    Wraps a base per-column transform (``names``, ``specs``, ``__call__``;
    :class:`~gwpop_search.inference.dynesty_backend.PriorTransform`). For every
    ``(lower, upper)`` pair the unit-cube coordinates are replaced by
    ``v_1 = u_1 / (1 + sqrt(1 - u_1))`` and ``v_2 = v_1 + u_2 (1 - v_1)``
    before the base inverse CDFs are applied, so ``theta[lower] <=
    theta[upper]``. Both members of a pair must have identical priors.
    Accepts ``u`` of shape ``[ndim]`` or ``[m, ndim]``.
    """

    def __init__(self, base, pairs: Sequence[tuple[str, str]]):
        names = tuple(base.names)
        specs = tuple(base.specs)
        index = {name: k for k, name in enumerate(names)}
        resolved = []
        used: set[str] = set()
        for pair in pairs:
            lower, upper = (str(name) for name in pair)
            for name in (lower, upper):
                if name not in index:
                    raise ValueError(f"ordered parameter {name!r} is not a model parameter")
                if name in used:
                    raise ValueError(f"parameter {name!r} appears in more than one pair")
                used.add(name)
            if specs[index[lower]] != specs[index[upper]]:
                raise ValueError(
                    f"ordered pair ({lower!r}, {upper!r}) has different priors; the "
                    "order-statistic transform preserves the evidence only for "
                    "identically distributed (exchangeable) components"
                )
            resolved.append((lower, upper))
        if not resolved:
            raise ValueError("at least one ordered pair is required")
        self.base = base
        self.names = names
        self.specs = specs
        self.ndim = len(names)
        self.pairs = tuple(resolved)
        self._index = np.asarray(
            [(index[lower], index[upper]) for lower, upper in resolved], dtype=np.int64
        )

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim not in (1, 2) or u.shape[-1] != self.ndim:
            raise ValueError(
                f"unit-cube input must have shape [{self.ndim}] or [m, {self.ndim}]; "
                f"got {u.shape}"
            )
        v = np.array(u, dtype=float, copy=True)
        for i, j in self._index:
            first = u[..., i]
            # 1 - sqrt(1 - u) without cancellation for small u.
            lower = first / (1.0 + np.sqrt(1.0 - first))
            v[..., i] = lower
            v[..., j] = lower + u[..., j] * (1.0 - lower)
        return self.base(v)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": ORDERED_PAIRS_PARAMETERIZATION,
            "pairs": [list(pair) for pair in self.pairs],
            "base": self.base.to_dict(),
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, OrderedPairPriorTransform):
            return NotImplemented
        return self.base == other.base and self.pairs == other.pairs

    def __hash__(self) -> int:
        return hash((self.base, self.pairs))

    def __repr__(self) -> str:
        return f"OrderedPairPriorTransform(pairs={self.pairs!r})"


@dataclass(frozen=True)
class ModelParameterization:
    """How the unit cube is mapped onto a model's (unchanged) hyperprior.

    ``identity``: independent inverse CDFs per parameter. ``ordered_exchangeable_pairs``:
    the bijective order-statistic transform on each exchangeable group's
    ``order_by`` pair (evidence identical, posterior on canonical labels).
    """

    kind: str = IDENTITY_PARAMETERIZATION
    groups: tuple[ExchangeableComponents, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in (IDENTITY_PARAMETERIZATION, ORDERED_PAIRS_PARAMETERIZATION):
            raise ValueError(f"unknown parameterization {self.kind!r}")
        groups = tuple(self.groups)
        if self.kind == IDENTITY_PARAMETERIZATION and groups:
            raise ValueError("the identity parameterization has no exchangeable groups")
        if self.kind == ORDERED_PAIRS_PARAMETERIZATION and not groups:
            raise ValueError("ordered_exchangeable_pairs needs at least one group")
        object.__setattr__(self, "groups", groups)

    @property
    def is_identity(self) -> bool:
        return self.kind == IDENTITY_PARAMETERIZATION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format_version": PARAMETERIZATION_FORMAT_VERSION,
            "kind": self.kind,
        }
        if not self.is_identity:
            payload["groups"] = [group.to_dict() for group in self.groups]
            payload["canonical_labels"] = (
                "within each group the first order_by parameter is the smaller; swapped "
                "companions follow their component; reflected mixture weights refer to "
                "the second (larger) component"
            )
            payload["evidence"] = (
                "identical to the identity parameterization: bijective order-statistic "
                "transform onto the half-space with density 2*pi, exact label-swap symmetry"
            )
        return payload

    def validate(self, priors: Mapping[str, PriorSpec]) -> None:
        """Refuse groups that are not exchangeable under ``priors``."""
        for group in self.groups:
            problems = group.exchangeability_violations(priors)
            if problems:
                raise ValueError(
                    f"{group.source or group.order_by} components are not exchangeable "
                    "under the declared priors; label canonicalisation would change the "
                    "evidence: " + "; ".join(problems)
                )

    def prior_transform(self, priors: Mapping[str, PriorSpec]):
        """``(names, transform)`` for ``priors`` in this parameterization."""
        from .dynesty_backend import prior_transform_for

        names, base = prior_transform_for(priors)
        if self.is_identity:
            return names, base
        self.validate(priors)
        return names, OrderedPairPriorTransform(
            base, [group.order_by for group in self.groups]
        )


IDENTITY = ModelParameterization()


def parameterization_for_spec(spec, *, canonicalize: bool = True) -> ModelParameterization:
    """The sampling parameterization of a declarative model.

    With ``canonicalize`` every exactly exchangeable component group of the
    spec (see :func:`exchangeable_components_for_spec`) is sampled with the
    order-statistic transform; otherwise, or when there is none, the identity.
    """
    if not canonicalize:
        return IDENTITY
    groups = exchangeable_components_for_spec(spec)
    if not groups:
        return IDENTITY
    return ModelParameterization(kind=ORDERED_PAIRS_PARAMETERIZATION, groups=groups)


def canonicalize_samples(
    samples: np.ndarray,
    names: Sequence[str],
    groups: Sequence[ExchangeableComponents],
) -> np.ndarray:
    """Relabel samples onto canonical labels by applying the swap where needed.

    Post-hoc helper for samples drawn in the identity parameterization (it is
    the map used to compare with the ordered-transform posterior): rows with
    ``order_by[0] > order_by[1]`` get the label swap (companions exchanged,
    reflected parameters mapped ``x -> 1 - x``).
    """
    out = np.array(samples, dtype=float, copy=True)
    index = {str(name): k for k, name in enumerate(names)}
    for group in groups:
        a, b = (index[name] for name in group.order_by)
        swap = out[:, a] > out[:, b]
        for first, second in (group.order_by, *group.swapped):
            i, j = index[first], index[second]
            left = out[swap, i].copy()
            out[swap, i] = out[swap, j]
            out[swap, j] = left
        for name in group.reflected:
            k = index[name]
            out[swap, k] = 1.0 - out[swap, k]
    return out
