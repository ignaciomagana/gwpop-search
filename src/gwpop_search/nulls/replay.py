"""Null replay and search-level look-elsewhere calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class SearchReplayResult:
    null_index: int
    seed: int
    max_log_bayes_factor: float
    max_log_posterior_odds: float
    n_models_evaluated: int
    best_model_hash: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("max_log_bayes_factor", "max_log_posterior_odds"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.n_models_evaluated <= 0:
            raise ValueError("n_models_evaluated must be positive")
        if not self.best_model_hash:
            raise ValueError("best_model_hash cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def null_replay_seed(root_seed: int, null_index: int) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:null:{int(null_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def empirical_tail_probability(
    null_statistics,
    observed_statistic: float,
) -> dict[str, float | int]:
    """Finite-sample corrected empirical upper-tail probability."""
    null_statistics = np.asarray(null_statistics, dtype=float)
    if null_statistics.ndim != 1 or null_statistics.size == 0:
        raise ValueError("null_statistics must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(null_statistics)):
        raise ValueError("null_statistics must be finite")
    if not math.isfinite(observed_statistic):
        raise ValueError("observed_statistic must be finite")

    exceedances = int(np.sum(null_statistics >= float(observed_statistic)))
    n = int(null_statistics.size)
    p = (exceedances + 1.0) / (n + 1.0)
    return {
        "n_null": n,
        "n_exceed": exceedances,
        "observed_statistic": float(observed_statistic),
        "tail_probability": float(p),
        "minimum_resolvable_tail_probability": float(1.0 / (n + 1.0)),
    }


def calibrate_search_replays(
    results: Iterable[SearchReplayResult],
    *,
    observed_max_log_bayes_factor: float | None = None,
    observed_max_log_posterior_odds: float | None = None,
) -> dict[str, object]:
    results = tuple(results)
    if not results:
        raise ValueError("at least one null replay is required")

    payload: dict[str, object] = {
        "format_version": "gwpop-search-null-calibration-1.0",
        "n_null_replays": len(results),
        "max_log_bayes_factor": {
            "median": float(np.median([item.max_log_bayes_factor for item in results])),
            "q90": float(np.quantile(
                [item.max_log_bayes_factor for item in results], 0.90
            )),
            "q99": float(np.quantile(
                [item.max_log_bayes_factor for item in results], 0.99
            )),
        },
        "max_log_posterior_odds": {
            "median": float(np.median(
                [item.max_log_posterior_odds for item in results]
            )),
            "q90": float(np.quantile(
                [item.max_log_posterior_odds for item in results], 0.90
            )),
            "q99": float(np.quantile(
                [item.max_log_posterior_odds for item in results], 0.99
            )),
        },
    }
    if observed_max_log_bayes_factor is not None:
        payload["observed_log_bf_calibration"] = empirical_tail_probability(
            [item.max_log_bayes_factor for item in results],
            observed_max_log_bayes_factor,
        )
    if observed_max_log_posterior_odds is not None:
        payload["observed_log_posterior_odds_calibration"] = empirical_tail_probability(
            [item.max_log_posterior_odds for item in results],
            observed_max_log_posterior_odds,
        )
    return payload


def run_null_replay_campaign(
    root: str | Path,
    *,
    n_nulls: int,
    root_seed: int,
    replay: Callable[[int, int], SearchReplayResult],
) -> dict[str, object]:
    """Run/resume complete-search null replays through a caller-supplied pipeline.

    The replay callback must execute the same frozen search procedure used for
    the observed catalog. This orchestrator deliberately does not substitute a
    reduced search for the production procedure.
    """
    if n_nulls <= 0:
        raise ValueError("n_nulls must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    results: list[SearchReplayResult] = []
    for index in range(n_nulls):
        path = root / f"null_{index:05d}.json"
        seed = null_replay_seed(root_seed, index)
        if path.exists():
            payload = json.loads(path.read_text())
            result = SearchReplayResult(**payload)
            if result.seed != seed or result.null_index != index:
                raise ValueError(
                    f"null replay checkpoint mismatch at index {index}"
                )
        else:
            result = replay(index, seed)
            if result.seed != seed or result.null_index != index:
                raise ValueError(
                    "null replay callback returned inconsistent index/seed"
                )
            path.write_text(
                json.dumps(result.to_dict(), sort_keys=True, indent=2)
            )
        results.append(result)

    summary = calibrate_search_replays(results)
    (root / "null_calibration.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
