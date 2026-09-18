"""Exact event subsetting for ragged posterior catalogs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .posterior import PosteriorCatalog


def subset_posterior_events(
    posterior: PosteriorCatalog,
    event_names: Iterable[str],
    *,
    reason: str | None = None,
) -> PosteriorCatalog:
    """Return a catalog containing the requested events in requested order."""
    requested = tuple(str(name) for name in event_names)
    if not requested:
        raise ValueError("posterior subset must contain at least one event")
    if len(set(requested)) != len(requested):
        raise ValueError("posterior subset event names must be unique")

    lookup = {name: index for index, name in enumerate(posterior.event_names)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise KeyError(f"posterior subset references unknown events {missing}")

    indices = [lookup[name] for name in requested]
    rows = []
    counts = []
    for index in indices:
        sl = posterior.event_slice(index)
        event_rows = np.arange(sl.start, sl.stop, dtype=np.int64)
        rows.append(event_rows)
        counts.append(event_rows.size)

    row_index = np.concatenate(rows)
    offsets = np.concatenate(
        ([0], np.cumsum(np.asarray(counts, dtype=np.int64)))
    )
    return PosteriorCatalog(
        event_names=requested,
        offsets=offsets,
        samples={
            name: np.asarray(values)[row_index]
            for name, values in posterior.samples.items()
        },
        log_ref_density=np.asarray(posterior.log_ref_density)[row_index],
        basis=posterior.basis,
        availability=np.asarray(posterior.availability, dtype=bool)[indices],
        metadata={
            **posterior.metadata,
            "event_subset": {
                "source_n_events": int(posterior.n_events),
                "selected_n_events": len(requested),
                "selected_events": list(requested),
                "reason": reason,
            },
        },
    )


def drop_posterior_events(
    posterior: PosteriorCatalog,
    event_names: Iterable[str],
    *,
    reason: str | None = None,
) -> PosteriorCatalog:
    """Return a catalog with explicitly named events removed."""
    drop = tuple(str(name) for name in event_names)
    if not drop:
        raise ValueError("drop_posterior_events requires at least one event")
    unknown = sorted(set(drop) - set(posterior.event_names))
    if unknown:
        raise KeyError(f"cannot drop unknown events {unknown}")
    keep = tuple(name for name in posterior.event_names if name not in set(drop))
    if not keep:
        raise ValueError("event-drop scenario cannot remove every event")
    return subset_posterior_events(
        posterior,
        keep,
        reason=reason or f"drop:{','.join(drop)}",
    )
