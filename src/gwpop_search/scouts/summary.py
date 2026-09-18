"""Interpret conditionally normalized HSGP scouts as parametric hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .conditional import ConditionalHSGPResidualModel
from .proposals import (
    ResidualDependenceSummary,
    StructureProposal,
    proposal_from_summary,
)


@dataclass(frozen=True)
class ConditionalMomentSummaryConfig:
    covariate_grid_size: int = 15
    target_grid_size: int = 301
    max_posterior_draws: int = 256
    fixed_m1_source: float = 35.0
    fixed_q: float = 0.7
    fixed_z: float = 0.3
    proposal_minimum_abs_z: float = 2.0

    def __post_init__(self) -> None:
        if self.covariate_grid_size < 3:
            raise ValueError("covariate_grid_size must be at least three")
        if self.target_grid_size < 51:
            raise ValueError("target_grid_size must be at least 51")
        if self.max_posterior_draws <= 0:
            raise ValueError("max_posterior_draws must be positive")
        if self.fixed_m1_source <= 0.0:
            raise ValueError("fixed_m1_source must be positive")
        if not 0.0 < self.fixed_q <= 1.0:
            raise ValueError("fixed_q must lie in (0, 1]")
        if self.fixed_z < 0.0:
            raise ValueError("fixed_z cannot be negative")
        if self.proposal_minimum_abs_z <= 0.0:
            raise ValueError("proposal_minimum_abs_z must be positive")


def _flatten_coefficients(
    samples: Mapping[str, np.ndarray],
    model: ConditionalHSGPResidualModel,
) -> tuple[dict[str, np.ndarray], int]:
    flattened = {}
    sizes = set()
    for name in model.coefficient_names:
        if name not in samples:
            raise KeyError(f"missing HSGP posterior coefficient {name!r}")
        values = np.asarray(samples[name], dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"HSGP posterior coefficient {name!r} is non-finite")
        flattened[name] = values
        sizes.add(values.size)
    if len(sizes) != 1:
        raise ValueError("HSGP posterior coefficient arrays have unequal sizes")
    n = next(iter(sizes))
    if n <= 0:
        raise ValueError("HSGP posterior is empty")
    return flattened, int(n)


def _selected_draw_indices(n_draws: int, maximum: int) -> np.ndarray:
    keep = min(int(n_draws), int(maximum))
    if keep == n_draws:
        return np.arange(n_draws, dtype=int)
    return np.unique(
        np.linspace(0, n_draws - 1, keep, dtype=int)
    )


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - np.mean(x)
    denominator = float(np.sum(dx**2))
    if denominator <= 0.0:
        raise ValueError("covariate grid has zero variance")
    return float(np.sum(dx * (y - np.mean(y))) / denominator)


def _posterior_slope_summary(
    slopes: np.ndarray,
    *,
    target: str,
    covariate: str,
    method: str,
) -> ResidualDependenceSummary:
    slopes = np.asarray(slopes, dtype=float)
    if slopes.ndim != 1 or slopes.size < 2:
        raise ValueError("at least two posterior slope draws are required")
    if not np.all(np.isfinite(slopes)):
        raise ValueError("posterior slopes must be finite")

    median = float(np.median(slopes))
    scale = float(np.std(slopes, ddof=1))
    if scale <= 0.0:
        scale = np.sqrt(np.finfo(float).eps) * max(abs(median), 1.0)
    return ResidualDependenceSummary(
        target=target,
        covariate=covariate,
        slope=median,
        slope_error=scale,
        z_score=median / scale,
        n_effective=float(slopes.size),
        method=method,
    )


def _grids(
    model: ConditionalHSGPResidualModel,
    config: ConditionalMomentSummaryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    cov_axis = model.config.covariate_axis
    cov_lower = float(cov_axis.lower)
    cov_upper = float(cov_axis.upper)

    # For q|m1 the secondary-mass bound makes the conditional support collapse
    # to the singleton q=1 at m1=mmin. That boundary has zero measure and must
    # not be used for numerical moment summaries. Stay infinitesimally inside
    # the physically valid support while retaining the declared HSGP domain.
    if (
        model.config.target == "q"
        and model.config.covariate == "m1_source"
    ):
        mmin = float(model.base_hyperparameters["mmin"])
        if cov_upper <= mmin:
            raise ValueError(
                "q|m1 scout covariate domain has no nonzero conditional support"
            )
        cov_lower = max(
            cov_lower,
            np.nextafter(mmin, np.inf),
        )

    covariate = np.linspace(
        cov_lower,
        cov_upper,
        config.covariate_grid_size,
    )
    target_axis = model.config.target_axis
    target = np.linspace(
        target_axis.lower,
        target_axis.upper,
        config.target_grid_size,
    )
    return target, covariate


def _conditional_moments_for_draw(
    model: ConditionalHSGPResidualModel,
    coefficients: Mapping[str, float],
    *,
    target_grid: np.ndarray,
    covariate_grid: np.ndarray,
    config: ConditionalMomentSummaryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    target_mesh = np.broadcast_to(
        target_grid[None, :],
        (covariate_grid.size, target_grid.size),
    )
    cov_mesh = np.broadcast_to(
        covariate_grid[:, None],
        target_mesh.shape,
    )

    m1 = np.full_like(target_mesh, config.fixed_m1_source)
    q = np.full_like(target_mesh, config.fixed_q)
    z = np.full_like(target_mesh, config.fixed_z)

    if model.config.covariate == "m1_source":
        m1 = cov_mesh
    elif model.config.covariate == "q":
        q = cov_mesh
    elif model.config.covariate == "z":
        z = cov_mesh

    if model.config.target == "q":
        q = target_mesh
    else:
        chi = target_mesh

    logp = np.asarray(
        model.conditional_logpdf_source(
            target_mesh,
            m1_source=m1,
            q=q,
            z=z,
            hyperparameters=coefficients,
        ),
        dtype=float,
    )
    pdf = np.exp(logp)
    norm = np.trapezoid(pdf, target_grid, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError("conditional scout summary encountered zero/non-finite norm")
    pdf = pdf / norm[:, None]

    mean = np.trapezoid(
        pdf * target_mesh,
        target_grid,
        axis=1,
    )
    variance = np.trapezoid(
        pdf * (target_mesh - mean[:, None]) ** 2,
        target_grid,
        axis=1,
    )
    width = np.sqrt(np.maximum(variance, np.finfo(float).tiny))
    return mean, width


def summarize_conditional_hsgp(
    model: ConditionalHSGPResidualModel,
    posterior_samples: Mapping[str, np.ndarray],
    *,
    config: ConditionalMomentSummaryConfig | None = None,
) -> dict[str, object]:
    """Summarize flexible conditional structure and propose legal descendants."""
    config = (
        ConditionalMomentSummaryConfig()
        if config is None
        else config
    )
    flattened, n_draws = _flatten_coefficients(posterior_samples, model)
    indices = _selected_draw_indices(n_draws, config.max_posterior_draws)
    target_grid, covariate_grid = _grids(model, config)

    mean_slopes = np.empty(indices.size, dtype=float)
    log_width_slopes = np.empty(indices.size, dtype=float)
    mean_curves = []
    width_curves = []

    for output_index, draw_index in enumerate(indices):
        coefficients = {
            name: float(values[draw_index])
            for name, values in flattened.items()
        }
        mean, width = _conditional_moments_for_draw(
            model,
            coefficients,
            target_grid=target_grid,
            covariate_grid=covariate_grid,
            config=config,
        )
        mean_slopes[output_index] = _linear_slope(covariate_grid, mean)
        log_width_slopes[output_index] = _linear_slope(
            covariate_grid,
            np.log(width),
        )
        mean_curves.append(mean)
        width_curves.append(width)

    covariate_label = (
        "m1"
        if model.config.covariate == "m1_source"
        else model.config.covariate
    )
    if model.config.target == "q":
        mean_target = "pairing"
        width_target = None
    else:
        mean_target = "chieff_mean"
        width_target = "chieff_width"

    summaries = []
    proposals: list[StructureProposal] = []

    mean_summary = _posterior_slope_summary(
        mean_slopes,
        target=mean_target,
        covariate=covariate_label,
        method="hsgp-conditional-mean-v1",
    )
    summaries.append(mean_summary)
    proposal = proposal_from_summary(
        mean_summary,
        minimum_abs_z=config.proposal_minimum_abs_z,
    )
    if proposal is not None:
        proposals.append(proposal)

    if width_target is not None:
        width_summary = _posterior_slope_summary(
            log_width_slopes,
            target=width_target,
            covariate=covariate_label,
            method="hsgp-conditional-logwidth-v1",
        )
        summaries.append(width_summary)
        proposal = proposal_from_summary(
            width_summary,
            minimum_abs_z=config.proposal_minimum_abs_z,
        )
        if proposal is not None:
            proposals.append(proposal)

    mean_curves = np.asarray(mean_curves)
    width_curves = np.asarray(width_curves)
    return {
        "format_version": "gwpop-search-hsgp-structure-summary-1.0",
        "model": model.to_config(),
        "summary_config": asdict(config),
        "n_posterior_draws_total": n_draws,
        "n_posterior_draws_summarized": int(indices.size),
        "target_grid": target_grid.tolist(),
        "covariate_grid": covariate_grid.tolist(),
        "conditional_mean": {
            "q05": np.quantile(mean_curves, 0.05, axis=0).tolist(),
            "median": np.quantile(mean_curves, 0.50, axis=0).tolist(),
            "q95": np.quantile(mean_curves, 0.95, axis=0).tolist(),
        },
        "conditional_width": {
            "q05": np.quantile(width_curves, 0.05, axis=0).tolist(),
            "median": np.quantile(width_curves, 0.50, axis=0).tolist(),
            "q95": np.quantile(width_curves, 0.95, axis=0).tolist(),
        },
        "dependence_summaries": [asdict(item) for item in summaries],
        "proposals": [asdict(item) for item in proposals],
    }
