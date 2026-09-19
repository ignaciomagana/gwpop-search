"""Shared helpers of the model-comparison analysis package.

Posterior samples
-----------------
Every analysis here consumes a *weighted* posterior sample
(:class:`WeightedPosterior`): points ``[K, ndim]`` with normalized weights.
dynesty repeats are pooled by :func:`pool_dynesty_results` as the equal-weight
mixture of separately normalized runs (orchestrator decision D1): the dead and
final live points of run ``r`` keep their nested-sampling weights
``exp(logwt - logsumexp(logwt_r)) / R``. ``jitter_run``/``resample_run``/
``merge_runs`` are never used (they are biased on runs with a ``-inf``
plateau). Points with zero weight (the plateau) are dropped from the
posterior sample; analyses that need them (the PSIS-LOO support-complement
mass) read the dynesty results directly.

Likelihood identity
-------------------
Estimates that combine two models (Bayes factors, Monte-Carlo covariances) are
only meaningful when both models were evaluated on the same data with the same
estimator (condition C1 of MODEL_COMPARISON_MATH.md). Every dynesty result of
an HBI likelihood stores :func:`build_likelihood_identity`; the helpers below
compare those identities (ignoring ``selection_chunk_size``, which only changes
the floating-point summation order) and refuse mismatches.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
import numbers
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp


class AnalysisInputError(ValueError):
    """Inputs of an analysis are inconsistent (different data, names, models)."""


# ---------------------------------------------------------------------------
# Validation and JSON helpers
# ---------------------------------------------------------------------------


def as_int(name: str, value, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def as_float(
    name: str,
    value,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    allow_inf: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    value = float(value)
    if math.isnan(value) or (not allow_inf and not math.isfinite(value)):
        raise ValueError(f"{name} must be finite; got {value}")
    if positive and not value > 0.0:
        raise ValueError(f"{name} must be positive; got {value}")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value}")
    return value


def json_ready(value):
    """Convert NumPy scalars/arrays and tuples into plain JSON types.

    Non-finite floats are written as JSON ``NaN``/``Infinity`` by the callers
    (``allow_nan=True``) and read back unchanged by :mod:`json`.
    """
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically write ``payload`` as sorted, indented JSON."""
    atomic_write_text(
        path,
        json.dumps(json_ready(payload), sort_keys=True, indent=2, allow_nan=True) + "\n",
    )


def read_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"{path} must contain one JSON object")
    return payload


def require_format(payload: Mapping[str, object], expected: str | Sequence[str], *, what: str):
    allowed = (expected,) if isinstance(expected, str) else tuple(expected)
    found = payload.get("format_version")
    if found not in allowed:
        raise AnalysisInputError(
            f"{what}: unsupported format_version {found!r}; expected one of {list(allowed)}"
        )
    return found


# ---------------------------------------------------------------------------
# Likelihood identity
# ---------------------------------------------------------------------------


def without_chunk_size(identity: Mapping[str, object]) -> dict[str, object]:
    """Likelihood identity minus ``hbi_config.selection_chunk_size``."""
    result = copy.deepcopy(dict(identity))
    hbi = result.get("hbi_config")
    if isinstance(hbi, dict):
        hbi.pop("selection_chunk_size", None)
    return result


def mapping_differences(existing: Mapping, requested: Mapping, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two nested mappings."""
    keys = sorted(set(existing) | set(requested))
    diffs: list[str] = []
    for key in keys:
        a, b = existing.get(key, "<missing>"), requested.get(key, "<missing>")
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            diffs.extend(mapping_differences(a, b, prefix=f"{prefix}{key}."))
        elif a != b:
            diffs.append(f"{prefix}{key}")
    return diffs


def normalized_identity(identity: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(json_ready(without_chunk_size(identity)), sort_keys=True))


def require_same_data(
    identity_a: Mapping[str, object] | None,
    identity_b: Mapping[str, object] | None,
    *,
    what: str = "models",
) -> None:
    """Condition C1: both likelihoods read the same data with the same estimator.

    Compares the ``data`` block (event names, per-event sample counts, bases,
    selection mode and campaigns, SHA-256 digests of every array) and the HBI
    configuration (without the chunk size). The model and parameter names are
    allowed to differ. Missing identities are refused: without them the
    common-data assumption cannot be verified.
    """
    if identity_a is None or identity_b is None:
        raise AnalysisInputError(
            f"{what}: a result carries no likelihood identity; the common-data condition "
            "(same events, PE samples, injections and HBI configuration) cannot be verified"
        )
    a = normalized_identity(identity_a)
    b = normalized_identity(identity_b)
    diffs = mapping_differences(
        {"data": a.get("data"), "hbi_config": a.get("hbi_config")},
        {"data": b.get("data"), "hbi_config": b.get("hbi_config")},
    )
    if diffs:
        raise AnalysisInputError(
            f"{what} were not evaluated on the same data/estimator; differing keys: {diffs}"
        )


def identity_for(posterior, selection, population_model, names: Sequence[str], hbi_config):
    from gwpop_search.inference.dynesty_backend import build_likelihood_identity

    return build_likelihood_identity(posterior, selection, population_model, names, hbi_config)


def require_identity_matches(
    stored: Mapping[str, object] | None,
    posterior,
    selection,
    population_model,
    names: Sequence[str],
    hbi_config,
    *,
    what: str,
) -> None:
    """The requested evaluation reproduces the likelihood the samples came from."""
    if stored is None:
        raise AnalysisInputError(
            f"{what}: the posterior sample carries no likelihood identity; refusing to "
            "evaluate it against data that cannot be verified to be the sampled data"
        )
    requested = identity_for(posterior, selection, population_model, names, hbi_config)
    diffs = mapping_differences(normalized_identity(stored), normalized_identity(requested))
    if diffs:
        raise AnalysisInputError(
            f"{what}: the requested data/model differ from the likelihood that was sampled; "
            f"differing keys: {diffs}"
        )


def hbi_config_from_identity(identity: Mapping[str, object] | None):
    """The run's HBIConfig with ``selection_chunk_size=None`` (decision D5)."""
    from gwpop_search.hbi import HBIConfig

    if identity is None:
        raise AnalysisInputError("no likelihood identity: pass hbi_config explicitly")
    payload = dict(identity["hbi_config"])
    return HBIConfig(
        rate_treatment=payload["rate_treatment"],
        raw_selection_use_observing_time=bool(payload["raw_selection_use_observing_time"]),
        selection_chunk_size=None,
    )


def model_hash_of(identity: Mapping[str, object] | None) -> str | None:
    if identity is None:
        return None
    model = identity.get("model") or {}
    value = model.get("model_hash") if isinstance(model, Mapping) else None
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Weighted posterior samples
# ---------------------------------------------------------------------------


def normalized_log_weights(log_weights) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if log_weights.ndim != 1 or log_weights.size == 0:
        raise ValueError("log weights must be a non-empty 1-D array")
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError("log weights must not contain NaN or +inf")
    total = logsumexp(log_weights)
    if not np.isfinite(total):
        raise ValueError("log weights are all -inf")
    return log_weights - total


def kish_ess(weights) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if not total > 0:
        raise ValueError("weights must have positive total")
    w = weights / total
    return float(1.0 / np.sum(w * w))


@dataclass(frozen=True, eq=False)
class WeightedPosterior:
    """A weighted posterior sample.

    ``points[k]`` has normalized weight ``weights[k] > 0`` (the weights sum to
    one). ``run_index[k]`` is the repeat a point came from (0 for a single
    sample). ``likelihood_identity`` is the identity of the likelihood that
    was sampled (``None`` for samples of unknown provenance, which the
    HBI-evaluating analyses refuse). ``log_evidence``/``log_evidence_error``
    are the per-run dynesty values when the sample came from runs.
    """

    names: tuple[str, ...]
    points: np.ndarray
    weights: np.ndarray
    run_index: np.ndarray
    n_runs: int
    source: str
    likelihood_identity: Mapping[str, object] | None = None
    log_evidences: tuple[float, ...] = ()
    log_evidence_errors: tuple[float, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        if not names or len(set(names)) != len(names):
            raise ValueError(f"names must be unique and non-empty; got {names}")
        points = np.asarray(self.points, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        run_index = np.asarray(self.run_index, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != len(names) or points.shape[0] == 0:
            raise ValueError(f"points must have shape [K>0, {len(names)}]; got {points.shape}")
        if weights.shape != (points.shape[0],) or run_index.shape != weights.shape:
            raise ValueError("weights and run_index need one entry per point")
        if not np.all(np.isfinite(points)):
            raise ValueError("posterior points must be finite")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("posterior weights must be finite and positive")
        if abs(float(weights.sum()) - 1.0) > 1e-9:
            raise ValueError(f"posterior weights must sum to one; got {weights.sum()!r}")
        n_runs = as_int("n_runs", self.n_runs, minimum=1)
        if np.any(run_index < 0) or np.any(run_index >= n_runs):
            raise ValueError("run_index entries must lie in [0, n_runs)")
        set_ = object.__setattr__
        set_(self, "names", names)
        set_(self, "points", points)
        set_(self, "weights", weights)
        set_(self, "run_index", run_index)
        set_(self, "n_runs", n_runs)
        set_(self, "log_evidences", tuple(float(x) for x in self.log_evidences))
        set_(self, "log_evidence_errors", tuple(float(x) for x in self.log_evidence_errors))

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def kish_ess(self) -> float:
        return kish_ess(self.weights)

    def column(self, name: str) -> np.ndarray:
        try:
            k = self.names.index(str(name))
        except ValueError as exc:
            raise KeyError(f"{name!r} is not a parameter of this posterior {self.names}") from exc
        return self.points[:, k]

    def run(self, r: int) -> "WeightedPosterior":
        """The points of repeat ``r`` with their weights renormalized."""
        mask = self.run_index == int(r)
        if not mask.any():
            raise ValueError(f"no points from run {r}")
        w = self.weights[mask]
        return WeightedPosterior(
            names=self.names,
            points=self.points[mask],
            weights=w / w.sum(),
            run_index=np.zeros(int(mask.sum()), dtype=np.int64),
            n_runs=1,
            source=f"{self.source}[run {r}]",
            likelihood_identity=self.likelihood_identity,
            log_evidences=self.log_evidences[r : r + 1] if self.log_evidences else (),
            log_evidence_errors=(
                self.log_evidence_errors[r : r + 1] if self.log_evidence_errors else ()
            ),
        )

    def equal_weight_draws(self, n: int, seed: int) -> np.ndarray:
        """``n`` systematic-resampling draws (randomly permuted)."""
        from gwpop_search.inference.dynesty_backend import equal_weight_resample

        rng = np.random.default_rng(as_int("seed", seed, minimum=0))
        return equal_weight_resample(self.points, self.weights, as_int("n", n, minimum=1), rng)

    def weighted_quantile(self, name: str, q) -> np.ndarray:
        return weighted_quantile(self.column(name), self.weights, q)

    def weighted_median(self) -> np.ndarray:
        return np.asarray(
            [weighted_quantile(self.points[:, k], self.weights, 0.5) for k in range(len(self.names))]
        )


def weighted_quantile(values, weights, q) -> np.ndarray | float:
    """Inverted-CDF quantile of a weighted sample (``numpy`` ``inverted_cdf`` rule)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    qs = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if np.any((qs < 0) | (qs > 1)):
        raise ValueError("quantiles must lie in [0, 1]")
    idx = np.searchsorted(cdf, qs, side="left")
    idx = np.clip(idx, 0, values.size - 1)
    out = values[order][idx]
    return float(out[0]) if np.ndim(q) == 0 else out


def _result_log_weights(result) -> np.ndarray:
    log_weights = np.asarray(result.log_weights, dtype=np.float64)
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise AnalysisInputError("dynesty log weights contain NaN or +inf")
    return log_weights - logsumexp(log_weights)


def common_likelihood_identity(results: Sequence) -> Mapping[str, object] | None:
    """The shared likelihood identity of repeat runs (``ValueError`` if they differ)."""
    identities = [getattr(item, "likelihood_identity", None) for item in results]
    if all(identity is None for identity in identities):
        return None
    if any(identity is None for identity in identities):
        raise AnalysisInputError("some repeats carry a likelihood identity and others do not")
    reference = normalized_identity(identities[0])
    for k, identity in enumerate(identities[1:], start=1):
        diffs = mapping_differences(reference, normalized_identity(identity))
        if diffs:
            raise AnalysisInputError(
                f"repeat {k} sampled a different likelihood than repeat 0; differing keys: {diffs}"
            )
    return copy.deepcopy(dict(identities[0]))


def pool_dynesty_results(
    results: Sequence,
    *,
    n_draws: int | None = None,
    seed: int = 0,
) -> WeightedPosterior:
    """Equal-weight mixture of separately normalized dynesty runs.

    Every run gets total weight ``1/R``; inside a run the dead and final live
    points keep their nested-sampling weights. Zero-weight (plateau) points are
    dropped. With ``n_draws`` the pooled sample is replaced by ``n_draws``
    systematic-resampling draws, aggregated into unique points with
    multiplicity weights (an unbiased, lower-cost representation). All runs
    must share parameter names and likelihood identity.
    """
    results = tuple(results)
    if not results:
        raise ValueError("at least one dynesty result is required")
    names = tuple(results[0].names)
    for k, item in enumerate(results):
        if tuple(item.names) != names:
            raise AnalysisInputError(f"repeat {k} has parameters {item.names}; expected {names}")
    identity = common_likelihood_identity(results)
    n_runs = len(results)
    points, weights, runs = [], [], []
    for r, item in enumerate(results):
        log_w = _result_log_weights(item)
        keep = np.isfinite(log_w)
        w = np.exp(log_w[keep])
        keep_idx = np.flatnonzero(keep)[w > 0.0]
        w = w[w > 0.0]
        points.append(np.asarray(item.samples, dtype=np.float64)[keep_idx])
        weights.append(w / w.sum() / n_runs)
        runs.append(np.full(keep_idx.size, r, dtype=np.int64))
    points = np.concatenate(points)
    weights = np.concatenate(weights)
    runs = np.concatenate(runs)
    weights = weights / weights.sum()
    source = "dynesty_pooled"
    metadata: dict[str, object] = {"n_points_weighted": int(points.shape[0])}
    if n_draws is not None:
        from gwpop_search.inference.dynesty_backend import equal_weight_resample

        n_draws = as_int("n_draws", n_draws, minimum=1)
        rng = np.random.default_rng(as_int("seed", seed, minimum=0))
        index = equal_weight_resample(np.arange(points.shape[0]), weights, n_draws, rng)
        unique, counts = np.unique(index, return_counts=True)
        points, runs = points[unique], runs[unique]
        weights = counts.astype(np.float64) / float(n_draws)
        source = "dynesty_pooled_resampled"
        metadata.update({"n_draws": int(n_draws), "seed": int(seed)})
    return WeightedPosterior(
        names=names,
        points=points,
        weights=weights,
        run_index=runs,
        n_runs=n_runs,
        source=source,
        likelihood_identity=identity,
        log_evidences=tuple(float(item.log_evidence) for item in results),
        log_evidence_errors=tuple(float(item.log_evidence_error) for item in results),
        metadata=metadata,
    )


def posterior_from_equal_weight_draws(
    names: Sequence[str],
    draws,
    *,
    likelihood_identity: Mapping[str, object] | None = None,
    source: str = "equal_weight",
) -> WeightedPosterior:
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] == 0:
        raise ValueError("draws must have shape [S>0, ndim]")
    n = draws.shape[0]
    return WeightedPosterior(
        names=tuple(names),
        points=draws,
        weights=np.full(n, 1.0 / n),
        run_index=np.zeros(n, dtype=np.int64),
        n_runs=1,
        source=source,
        likelihood_identity=likelihood_identity,
    )


# ---------------------------------------------------------------------------
# Discovering saved dynesty results
# ---------------------------------------------------------------------------


def iter_result_paths(paths: Iterable[str | Path]) -> list[Path]:
    """``result.npz`` files given directly or found recursively below directories."""
    found: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("result.npz") if p.is_file()))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"no dynesty result or directory at {path}")
    unique = sorted({p.resolve() for p in found})
    if not unique:
        raise FileNotFoundError("no result.npz files were found")
    return unique


def discover_dynesty_results(
    paths: Iterable[str | Path],
    *,
    model_hashes: Iterable[str] | None = None,
) -> dict[str, list]:
    """Load saved dynesty results and group them by model hash.

    The model hash is read from each result's likelihood identity. All
    repeats of one model must share one trajectory configuration apart from
    the seed (``DynestyConfig.identity_dict()``); mixing rungs (e.g. F3 and F4
    runs of the same model) is refused so the caller must choose explicitly.
    """
    from gwpop_search.inference.dynesty_backend import load_dynesty_result

    wanted = None if model_hashes is None else {str(h) for h in model_hashes}
    grouped: dict[str, list] = {}
    configs: dict[str, dict] = {}
    for path in iter_result_paths(paths):
        result = load_dynesty_result(path)
        model_hash = model_hash_of(result.likelihood_identity)
        if model_hash is None:
            raise AnalysisInputError(f"{path} records no model hash in its likelihood identity")
        if wanted is not None and model_hash not in wanted:
            continue
        config = result.config.identity_dict()
        # heterogeneous-sampler repeats are allowed; the rung is defined by the rest
        for name in ("sample", "slices", "walks", "batch_size"):
            config.pop(name, None)
        if model_hash in configs and configs[model_hash] != config:
            raise AnalysisInputError(
                f"results for model {model_hash[:16]} use different dynesty configurations "
                f"({configs[model_hash]} vs {config}); pass the runs of one rung only"
            )
        configs.setdefault(model_hash, config)
        grouped.setdefault(model_hash, []).append(result)
    if wanted is not None:
        missing = sorted(wanted - set(grouped))
        if missing:
            raise FileNotFoundError(f"no dynesty results found for model hash(es) {missing}")
    for model_hash, items in grouped.items():
        seeds = [item.seed for item in items]
        if len(set(seeds)) != len(seeds):
            raise AnalysisInputError(
                f"model {model_hash[:16]} has duplicate repeats with the same seed {seeds}"
            )
        items.sort(key=lambda item: item.seed)
    return grouped
