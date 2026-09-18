"""Deterministic Monte-Carlo reductions for non-production screening fidelities."""

from __future__ import annotations

import hashlib

import numpy as np

from .posterior import PosteriorCatalog
from .selection import SelectionCatalog


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def thin_posterior_catalog(
    posterior: PosteriorCatalog,
    *,
    max_samples_per_event: int,
    seed: int,
) -> PosteriorCatalog:
    """Randomly retain at most N posterior samples per event.

    Event likelihoods are posterior-sample means, so an iid/random subset needs
    no additional weight factor. This utility is for screening fidelities only.
    """
    if max_samples_per_event <= 0:
        raise ValueError("max_samples_per_event must be positive")

    selected_rows: list[np.ndarray] = []
    counts: list[int] = []
    for index, name in enumerate(posterior.event_names):
        sl = posterior.event_slice(index)
        rows = np.arange(sl.start, sl.stop, dtype=np.int64)
        keep = min(int(max_samples_per_event), rows.size)
        if keep < rows.size:
            rng = np.random.default_rng(_derived_seed(seed, f"pe:{name}"))
            rows = np.sort(rng.choice(rows, size=keep, replace=False))
        selected_rows.append(rows)
        counts.append(rows.size)

    rows = np.concatenate(selected_rows)
    offsets = np.concatenate(
        ([0], np.cumsum(np.asarray(counts, dtype=np.int64)))
    )
    return PosteriorCatalog(
        event_names=posterior.event_names,
        offsets=offsets,
        samples={
            name: np.asarray(values)[rows]
            for name, values in posterior.samples.items()
        },
        log_ref_density=np.asarray(posterior.log_ref_density)[rows],
        basis=posterior.basis,
        availability=np.asarray(posterior.availability, dtype=bool).copy(),
        metadata={
            **posterior.metadata,
            "screening_thin": {
                "seed": int(seed),
                "max_samples_per_event": int(max_samples_per_event),
                "source_n_samples_total": int(posterior.n_samples_total),
            },
        },
    )


def thin_selection_catalog(
    selection: SelectionCatalog,
    *,
    max_selected_per_campaign: int,
    seed: int,
) -> SelectionCatalog:
    """Subsample retained selection rows with inclusion-probability correction.

    If m of n retained rows from a campaign are sampled uniformly without
    replacement, each retained row has inclusion probability f=m/n. The
    Horvitz-Thompson correction multiplies retained importance contributions by
    1/f, implemented here by replacing p_draw with f*p_draw:

        log p_draw,reduced = log p_draw + log(f).

    Original campaign n_draw / observing-time metadata are therefore preserved.
    This is a screening approximation and must not be used for F3/F4 results.
    """
    if max_selected_per_campaign <= 0:
        raise ValueError("max_selected_per_campaign must be positive")

    selected_rows: list[np.ndarray] = []
    adjusted_log_draw: list[np.ndarray] = []
    for campaign in selection.campaigns:
        rows = selection.rows_for_campaign(campaign.campaign_id)
        if rows.size == 0:
            continue
        keep = min(int(max_selected_per_campaign), rows.size)
        if keep < rows.size:
            rng = np.random.default_rng(
                _derived_seed(seed, f"selection:{campaign.campaign_id}")
            )
            chosen = np.sort(rng.choice(rows, size=keep, replace=False))
        else:
            chosen = rows
        fraction = float(keep) / float(rows.size)
        selected_rows.append(chosen)
        adjusted_log_draw.append(
            np.asarray(selection.log_draw_density)[chosen] + np.log(fraction)
        )

    if not selected_rows:
        raise ValueError("selection reduction retained no rows")

    rows = np.concatenate(selected_rows)
    log_draw = np.concatenate(adjusted_log_draw)
    return SelectionCatalog(
        samples={
            name: np.asarray(values)[rows]
            for name, values in selection.samples.items()
        },
        log_draw_density=log_draw,
        campaign_id=np.asarray(selection.campaign_id)[rows],
        campaigns=selection.campaigns,
        basis=selection.basis,
        mode=selection.mode,
        estimator_semantics=selection.estimator_semantics,
        metadata={
            **selection.metadata,
            "screening_thin": {
                "seed": int(seed),
                "max_selected_per_campaign": int(max_selected_per_campaign),
                "source_n_selected": int(selection.n_selected),
                "inclusion_probability_corrected": True,
            },
        },
    )


def thin_catalog_pair(
    posterior: PosteriorCatalog,
    selection: SelectionCatalog,
    *,
    max_samples_per_event: int,
    max_selected_per_campaign: int,
    seed: int,
) -> tuple[PosteriorCatalog, SelectionCatalog]:
    return (
        thin_posterior_catalog(
            posterior,
            max_samples_per_event=max_samples_per_event,
            seed=seed,
        ),
        thin_selection_catalog(
            selection,
            max_selected_per_campaign=max_selected_per_campaign,
            seed=seed,
        ),
    )
