"""Held-out detected-event predictive validation for population models."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from gwpop_search.hbi import HBIConfig, evaluate_events, evaluate_selection


def deterministic_event_folds(
    event_names: tuple[str, ...] | list[str],
    *,
    n_folds: int,
    seed: int = 20260917,
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    result = {}
    for name in event_names:
        digest = hashlib.sha256(
            f"{int(seed)}:{name}".encode("utf-8")
        ).digest()
        result[str(name)] = int.from_bytes(digest[:8], "big") % int(n_folds)
    return result


def _hyperparameter_draw(
    samples: Mapping[str, np.ndarray],
    index: int,
) -> dict[str, float]:
    return {
        name: float(np.asarray(values).reshape(-1)[index])
        for name, values in samples.items()
    }


def validate_hyperposterior_samples(
    samples: Mapping[str, np.ndarray],
) -> int:
    if not samples:
        raise ValueError("hyperposterior samples cannot be empty")
    sizes = {
        np.asarray(values).size
        for values in samples.values()
    }
    if len(sizes) != 1:
        raise ValueError("hyperposterior parameter arrays must have equal size")
    n = next(iter(sizes))
    if n <= 0:
        raise ValueError("hyperposterior samples cannot be empty")
    if any(not np.all(np.isfinite(values)) for values in samples.values()):
        raise ValueError("hyperposterior samples must be finite")
    return int(n)


def _validated_log_weights(log_weights, n_draws: int) -> np.ndarray:
    if log_weights is None:
        return np.full(n_draws, -np.log(n_draws))
    log_weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if log_weights.size != n_draws:
        raise ValueError("log_weights must have one entry per hyperposterior draw")
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError("log_weights must not contain NaN or +inf")
    if not np.any(np.isfinite(log_weights)):
        raise ValueError("log_weights are all -inf")
    return log_weights - logsumexp(log_weights)


def heldout_detected_log_predictive(
    posterior,
    selection,
    population_model,
    hyperposterior_samples: Mapping[str, np.ndarray],
    *,
    heldout_events: tuple[str, ...] | list[str],
    config: HBIConfig | None = None,
    log_weights=None,
    batch_size: int = 64,
) -> dict[str, float]:
    """Posterior-predictive log density of held-out detected events.

    For a detected event the shape-model density is ``lambda_i = ell_i(Lambda)
    / A(Lambda)``; the held-out score is ``log E_post[ell_i / A]`` over a
    hyperposterior learned without the held-out events::

        log p_i = logsumexp_s(log w_s + log ell_i(Lambda_s) - log A(Lambda_s)) - logsumexp_s(log w_s)

    ``log_weights`` (optional) are the posterior weights of the draws
    (weighted nested-sampling points); equal weights otherwise. The terms are
    evaluated in fixed-size jitted batches with the backend terms function
    (:class:`gwpop_search.analysis.terms.BatchedCatalogTerms`) on the held-out
    subset catalog; ``_reference_heldout_log_predictive`` is the NumPy
    reference. NaN/``+inf`` raise; a draw with positive weight whose selection
    exposure has no population support raises (it cannot belong to a
    posterior of the shape likelihood).
    """
    from gwpop_search.analysis.terms import BatchedCatalogTerms
    from gwpop_search.data import subset_posterior_events

    config = HBIConfig(selection_chunk_size=None) if config is None else config
    n_draws = validate_hyperposterior_samples(hyperposterior_samples)
    heldout = tuple(str(name) for name in heldout_events)
    if not heldout:
        raise ValueError("at least one held-out event is required")
    missing = [name for name in heldout if name not in set(posterior.event_names)]
    if missing:
        raise ValueError(f"unknown held-out event(s): {missing}")
    log_w = _validated_log_weights(log_weights, n_draws)
    keep = np.isfinite(log_w)
    names = tuple(sorted(hyperposterior_samples))
    X = np.column_stack(
        [np.asarray(hyperposterior_samples[name], dtype=np.float64).reshape(-1) for name in names]
    )[keep]
    log_w = log_w[keep]
    subset = subset_posterior_events(posterior, heldout, reason="heldout-predictive")
    terms = BatchedCatalogTerms(
        subset, selection, population_model, names, hbi_config=config, batch_size=batch_size
    )
    event_terms, log_exposure = terms(X)
    if np.any(~np.isfinite(log_exposure)):
        rows = np.flatnonzero(~np.isfinite(log_exposure))[:5]
        raise ValueError(
            "hyperposterior draws with zero selection exposure (no population support on the "
            f"injections) at rows {rows.tolist()}"
        )
    values = log_w[:, None] + event_terms - log_exposure[:, None]
    scores = logsumexp(values, axis=0)
    return {name: float(scores[k]) for k, name in enumerate(heldout)}


def _reference_heldout_log_predictive(
    posterior,
    selection,
    population_model,
    hyperposterior_samples: Mapping[str, np.ndarray],
    *,
    heldout_events: tuple[str, ...] | list[str],
    config: HBIConfig | None = None,
    log_weights=None,
) -> dict[str, float]:
    """NumPy reference of :func:`heldout_detected_log_predictive` (one draw at a time)."""
    config = HBIConfig() if config is None else config
    n_draws = validate_hyperposterior_samples(hyperposterior_samples)
    indices = {
        name: index
        for index, name in enumerate(posterior.event_names)
    }
    missing = [name for name in heldout_events if name not in indices]
    if missing:
        raise ValueError(f"unknown held-out event(s): {missing}")
    log_w = _validated_log_weights(log_weights, n_draws)

    log_predictive_draws = {
        name: np.empty(n_draws, dtype=float)
        for name in heldout_events
    }

    for draw_index in range(n_draws):
        hp = _hyperparameter_draw(hyperposterior_samples, draw_index)
        event_terms = evaluate_events(
            posterior,
            population_model,
            hp,
        )
        selection_term = evaluate_selection(
            selection,
            population_model,
            hp,
            raw_use_observing_time=config.raw_selection_use_observing_time,
            chunk_size=config.selection_chunk_size,
        )
        for name in heldout_events:
            event_index = indices[name]
            log_predictive_draws[name][draw_index] = (
                float(event_terms.log_likelihoods[event_index])
                - float(selection_term.log_exposure)
            )

    return {
        name: float(logsumexp(log_w + values))
        for name, values in log_predictive_draws.items()
    }


def compare_holdout_models(
    scores: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Sum per-event predictive scores for each model without ranking labels."""
    totals = {}
    for model_hash, event_scores in scores.items():
        values = np.asarray(list(event_scores.values()), dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"model {model_hash!r} has invalid held-out predictive scores"
            )
        totals[model_hash] = float(values.sum())
    return totals
