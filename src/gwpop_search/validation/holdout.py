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


def heldout_detected_log_predictive(
    posterior,
    selection,
    population_model,
    hyperposterior_samples: Mapping[str, np.ndarray],
    *,
    heldout_events: tuple[str, ...] | list[str],
    config: HBIConfig | None = None,
) -> dict[str, float]:
    """Compute posterior-predictive log density for held-out detected events.

    For each hyperposterior draw Lambda, the shape-model density for a detected
    event is proportional to ell_i(Lambda) / A(Lambda). We average that density
    over a hyperposterior learned without the held-out event(s).
    """
    config = HBIConfig() if config is None else config
    n_draws = validate_hyperposterior_samples(hyperposterior_samples)
    indices = {
        name: index
        for index, name in enumerate(posterior.event_names)
    }
    missing = [name for name in heldout_events if name not in indices]
    if missing:
        raise ValueError(f"unknown held-out event(s): {missing}")

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
        name: float(logsumexp(values) - np.log(n_draws))
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
