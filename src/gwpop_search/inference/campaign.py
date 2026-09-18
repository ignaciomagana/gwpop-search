"""Phase-3 recovery campaign orchestration and acceptance logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .numpyro import NUTSConfig
from .recovery import run_synthetic_baseline_recovery
from .synthetic import SyntheticSurveyConfig


@dataclass(frozen=True)
class RecoveryAcceptanceCriteria:
    """Numerical acceptance thresholds for Phase-3 synthetic recovery.

    These defaults are deliberately strict enough to catch obviously unreliable
    chains/importance reweighting, but they are versioned run configuration rather
    than immutable scientific constants.
    """

    max_r_hat: float = 1.01
    min_mcmc_n_eff: float = 200.0
    max_divergences: int = 0
    min_event_ess: float = 20.0
    min_selection_ess: float = 200.0
    max_event_weight_fraction: float = 0.25
    max_selection_weight_fraction: float = 0.10
    max_shape_log_likelihood_variance: float = 1.0
    min_runs: int = 4

    def __post_init__(self) -> None:
        if self.max_r_hat <= 1.0:
            raise ValueError("max_r_hat must be greater than 1")
        for name in ("min_mcmc_n_eff", "min_event_ess", "min_selection_ess"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_event_weight_fraction", "max_selection_weight_fraction"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.max_shape_log_likelihood_variance <= 0:
            raise ValueError("max_shape_log_likelihood_variance must be positive")
        if self.max_divergences < 0:
            raise ValueError("max_divergences cannot be negative")
        if self.min_runs <= 0:
            raise ValueError("min_runs must be positive")


def _check(name: str, value, limit, *, comparison: str) -> dict[str, object]:
    if value is None or not np.isfinite(float(value)):
        passed = False
    elif comparison == "le":
        passed = float(value) <= float(limit)
    elif comparison == "ge":
        passed = float(value) >= float(limit)
    else:  # pragma: no cover - internal misuse
        raise ValueError(comparison)
    return {
        "name": name,
        "value": None if value is None else float(value),
        "comparison": comparison,
        "limit": float(limit),
        "passed": bool(passed),
    }


def assess_recovery_summary(
    summary: Mapping[str, object],
    criteria: RecoveryAcceptanceCriteria | None = None,
) -> dict[str, object]:
    """Evaluate one completed recovery summary against numerical criteria."""
    criteria = RecoveryAcceptanceCriteria() if criteria is None else criteria
    diagnostics = dict(summary["diagnostics"])
    importance = dict(summary["importance_diagnostics"])
    median = dict(importance["posterior_median"])

    checks = [
        _check(
            "max_r_hat",
            diagnostics.get("max_r_hat"),
            criteria.max_r_hat,
            comparison="le",
        ),
        _check(
            "min_mcmc_n_eff",
            diagnostics.get("min_n_eff"),
            criteria.min_mcmc_n_eff,
            comparison="ge",
        ),
        _check(
            "n_divergent",
            diagnostics.get("n_divergent"),
            criteria.max_divergences,
            comparison="le",
        ),
        _check(
            "min_event_ess",
            median.get("min_event_ess"),
            criteria.min_event_ess,
            comparison="ge",
        ),
        _check(
            "selection_ess",
            median.get("selection_ess"),
            criteria.min_selection_ess,
            comparison="ge",
        ),
        _check(
            "max_event_weight_fraction",
            median.get("max_event_weight_fraction"),
            criteria.max_event_weight_fraction,
            comparison="le",
        ),
        _check(
            "selection_max_weight_fraction",
            median.get("selection_max_weight_fraction"),
            criteria.max_selection_weight_fraction,
            comparison="le",
        ),
        _check(
            "shape_log_likelihood_variance",
            median.get("shape_log_likelihood_variance"),
            criteria.max_shape_log_likelihood_variance,
            comparison="le",
        ),
    ]
    return {
        "data_seed": int(summary["data_seed"]),
        "sampler_seed": int(summary["sampler_seed"]),
        "passed": bool(all(item["passed"] for item in checks)),
        "checks": checks,
    }


def aggregate_recovery_summaries(
    summaries: Iterable[Mapping[str, object]],
    criteria: RecoveryAcceptanceCriteria | None = None,
) -> dict[str, object]:
    """Aggregate independent recovery runs and report numerical/coverage summaries."""
    criteria = RecoveryAcceptanceCriteria() if criteria is None else criteria
    summaries = [dict(item) for item in summaries]
    assessments = [assess_recovery_summary(item, criteria) for item in summaries]

    parameter_names: set[str] = set()
    for item in summaries:
        parameter_names.update(dict(item["posterior"]).keys())

    coverage: dict[str, dict[str, object]] = {}
    for name in sorted(parameter_names):
        indicators: list[bool] = []
        standardized_offsets: list[float] = []
        for item in summaries:
            posterior = dict(item["posterior"])
            if name not in posterior:
                continue
            stats = dict(posterior[name])
            if "truth_in_90pct_interval" in stats:
                indicators.append(bool(stats["truth_in_90pct_interval"]))
            if (
                "truth" in stats
                and stats.get("std") is not None
                and float(stats["std"]) > 0.0
            ):
                standardized_offsets.append(
                    (float(stats["median"]) - float(stats["truth"])) / float(stats["std"])
                )
        coverage[name] = {
            "n_runs": len(indicators),
            "central_90pct_coverage_fraction": (
                float(np.mean(indicators)) if indicators else None
            ),
            "median_standardized_offset": (
                float(np.median(standardized_offsets))
                if standardized_offsets
                else None
            ),
        }

    numerical_passes = sum(bool(item["passed"]) for item in assessments)
    enough_runs = len(summaries) >= criteria.min_runs
    all_numerical = numerical_passes == len(summaries) and bool(summaries)

    # Coverage is reported, not thresholded, unless a later calibration phase
    # declares a sufficient number of catalogs for a formal coverage test.
    return {
        "format_version": "gwpop-search-phase3-campaign-1.0",
        "criteria": asdict(criteria),
        "n_runs": len(summaries),
        "n_numerical_pass": numerical_passes,
        "n_numerical_fail": len(summaries) - numerical_passes,
        "enough_runs": enough_runs,
        "all_numerical_pass": all_numerical,
        "phase3_numerical_gate_passed": bool(enough_runs and all_numerical),
        "coverage": coverage,
        "run_assessments": assessments,
    }


def load_recovery_summaries(root: str | Path) -> list[dict[str, object]]:
    """Load all completed recovery summaries below a campaign root."""
    root = Path(root)
    summaries = []
    for path in sorted(root.glob("run_*/recovery_summary.json")):
        summaries.append(json.loads(path.read_text()))
    return summaries


def assess_recovery_campaign(
    root: str | Path,
    criteria: RecoveryAcceptanceCriteria | None = None,
) -> dict[str, object]:
    """Assess already-completed recovery runs and write campaign_summary.json."""
    root = Path(root)
    result = aggregate_recovery_summaries(
        load_recovery_summaries(root),
        criteria=criteria,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign_summary.json").write_text(
        json.dumps(result, sort_keys=True, indent=2)
    )
    return result


def _derived_seed(root_seed: int, label: str, index: int) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{label}:{int(index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def recovery_seed_pairs(
    n_runs: int,
    *,
    root_seed: int = 20260917,
) -> tuple[tuple[int, int], ...]:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    return tuple(
        (
            _derived_seed(root_seed, "data", index),
            _derived_seed(root_seed, "sampler", index),
        )
        for index in range(int(n_runs))
    )


def build_campaign_plan(
    *,
    n_runs: int,
    root_seed: int,
    survey_config: SyntheticSurveyConfig,
    nuts_config: NUTSConfig,
    selection_chunk_size: int | None,
    criteria: RecoveryAcceptanceCriteria,
) -> dict[str, object]:
    pairs = recovery_seed_pairs(n_runs, root_seed=root_seed)
    return {
        "format_version": "gwpop-search-phase3-campaign-plan-1.0",
        "root_seed": int(root_seed),
        "n_runs": int(n_runs),
        "seed_pairs": [
            {"data_seed": int(data_seed), "sampler_seed": int(sampler_seed)}
            for data_seed, sampler_seed in pairs
        ],
        "survey_config": asdict(survey_config),
        "nuts_config": asdict(nuts_config),
        "selection_chunk_size": selection_chunk_size,
        "criteria": asdict(criteria),
    }


def run_recovery_campaign(
    root: str | Path,
    *,
    n_runs: int = 4,
    root_seed: int = 20260917,
    survey_config: SyntheticSurveyConfig | None = None,
    nuts_config: NUTSConfig | None = None,
    selection_chunk_size: int | None = 4096,
    criteria: RecoveryAcceptanceCriteria | None = None,
) -> dict[str, object]:
    """Run/resume a deterministic matrix of independent Phase-3 recoveries."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    survey_config = (
        SyntheticSurveyConfig() if survey_config is None else survey_config
    )
    nuts_config = NUTSConfig() if nuts_config is None else nuts_config
    criteria = (
        RecoveryAcceptanceCriteria() if criteria is None else criteria
    )

    plan = build_campaign_plan(
        n_runs=n_runs,
        root_seed=root_seed,
        survey_config=survey_config,
        nuts_config=nuts_config,
        selection_chunk_size=selection_chunk_size,
        criteria=criteria,
    )
    plan_path = root / "campaign_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing != plan:
            raise ValueError(
                "existing campaign plan does not match the requested configuration"
            )
    else:
        plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2))

    for index, pair in enumerate(plan["seed_pairs"]):
        run_dir = root / f"run_{index:03d}"
        run_synthetic_baseline_recovery(
            run_dir,
            data_seed=int(pair["data_seed"]),
            sampler_seed=int(pair["sampler_seed"]),
            survey_config=survey_config,
            nuts_config=nuts_config,
            selection_chunk_size=selection_chunk_size,
        )

    return assess_recovery_campaign(root, criteria=criteria)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprints(run_dir: str | Path) -> dict[str, str]:
    """Fingerprint completed chain artifacts for restart-integrity checks."""
    run_dir = Path(run_dir)
    chains = run_dir / "chains"
    result: dict[str, str] = {}
    for path in sorted(chains.glob("chain_*.npz")):
        result[path.name] = file_sha256(path)
        metadata = path.with_suffix(path.suffix + ".json")
        if metadata.exists():
            result[metadata.name] = file_sha256(metadata)
    return result
