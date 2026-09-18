"""Backend-independent HBI validation and importance-sampling diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .types import ImportanceDiagnostics
from ..data import SelectionMode

class HBIError(RuntimeError):
    """Base error raised by the standardized HBI layer."""

class PopulationDensityError(HBIError):
    """A population density returned NaN/+inf or an incompatible shape."""

class SelectionSupportError(HBIError):
    """The selection Monte Carlo has zero population support."""


def validate_log_population(values, *, expected_shape=None, what="population log density") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if expected_shape is not None and arr.shape != tuple(expected_shape):
        raise PopulationDensityError(
            f"{what} returned shape {arr.shape}; expected {tuple(expected_shape)}"
        )
    bad = np.isnan(arr) | np.isposinf(arr)
    if bad.any():
        idx = np.argwhere(bad)[:8].tolist()
        raise PopulationDensityError(
            f"{what} contains {int(bad.sum())} NaN/+inf value(s); first indices={idx}. "
            "-inf is allowed for genuine zero population support."
        )
    return arr


def importance_diagnostics(log_weights, *, n_draw: int | None = None) -> ImportanceDiagnostics:
    """Stable diagnostics for retained importance weights.

    Missing/undetected draws are exactly zero weight and therefore need not be
    stored. ``n_draw`` is nevertheless required to obtain the delta-method
    Monte Carlo variance of the mean estimator.
    """
    logw = validate_log_population(log_weights, what="log importance weights")
    if logw.ndim != 1:
        raise PopulationDensityError(f"log importance weights must be 1-D; got {logw.shape}")
    n_retained = int(logw.size)
    total = n_retained if n_draw is None else int(n_draw)
    if total <= 0 or total < n_retained:
        raise HBIError(
            f"n_draw must be >= retained rows and positive; got n_draw={total}, "
            f"n_retained={n_retained}"
        )
    finite = np.isfinite(logw)
    n_zero = int((~finite).sum())
    if not finite.any():
        return ImportanceDiagnostics(
            n_retained=n_retained,
            n_draw=total,
            n_zero_weight=n_zero,
            ess=0.0,
            ess_fraction_of_draws=0.0,
            max_weight_fraction=0.0,
            variance_log_estimate=float("inf"),
        )
    lse = float(logsumexp(logw[finite]))
    normalized = np.exp(logw[finite] - lse)
    inv_ess = float(np.sum(normalized**2))
    ess = 1.0 / inv_ess
    variance_log = max(inv_ess - 1.0 / total, 0.0)
    return ImportanceDiagnostics(
        n_retained=n_retained,
        n_draw=total,
        n_zero_weight=n_zero,
        ess=float(ess),
        ess_fraction_of_draws=float(ess / total),
        max_weight_fraction=float(np.max(normalized)),
        variance_log_estimate=float(variance_log),
    )



def density_required_fields(log_density, basis) -> tuple[str, ...]:
    """Fields a population-density callable needs from each data product.

    Plain callables default to the independent density coordinates. Model
    objects may declare ``required_fields`` to request advisory coordinates
    needed for an explicit change of variables/Jacobian.
    """
    declared = getattr(log_density, "required_fields", None)
    if declared is None:
        return tuple(basis.coordinates)
    fields = tuple(str(x) for x in declared)
    if not fields:
        raise HBIError("population log-density required_fields cannot be empty")
    missing_basis = [x for x in basis.coordinates if x not in fields]
    if missing_basis:
        raise HBIError(
            "population log-density required_fields must include every density-basis "
            f"coordinate; missing {missing_basis}"
        )
    return fields

def selection_log_factors(selection, *, use_observing_time: bool = True) -> np.ndarray:
    """Return per-retained-row additive factors for the exposure estimator.

    raw_draw:
        log(T_k / N_draw,k), or -log(N_draw,k) when time weighting is explicitly
        disabled.
    estimator_ready:
        zero: the adapter-supplied pdraw already defines the full estimator.
    """
    n = selection.n_selected
    if selection.mode is SelectionMode.ESTIMATOR_READY:
        return np.zeros(n, dtype=float)
    factors = np.empty(n, dtype=float)
    for campaign in selection.campaigns:
        rows = selection.rows_for_campaign(campaign.campaign_id)
        if campaign.n_draw is None:
            raise HBIError(f"raw campaign {campaign.campaign_id!r} has no n_draw")
        factor = -np.log(float(campaign.n_draw))
        if use_observing_time:
            if campaign.observing_time_yr is None:
                raise HBIError(
                    f"raw campaign {campaign.campaign_id!r} has no observing_time_yr; "
                    "supply it or explicitly disable observing-time weighting"
                )
            factor += np.log(float(campaign.observing_time_yr))
        factors[rows] = factor
    return factors
