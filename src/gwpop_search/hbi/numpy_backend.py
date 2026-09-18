"""NumPy/SciPy reference implementation of the standardized HBI likelihood."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.special import logsumexp

from ..data import SelectionMode, validate_pair
from .common import (
    HBIError,
    SelectionSupportError,
    density_required_fields,
    importance_diagnostics,
    selection_log_factors,
    validate_log_population,
)
from .types import (
    CampaignSelectionResult,
    CatalogLikelihoodResult,
    CatalogTerms,
    EventLikelihoodResult,
    HBIConfig,
    LikelihoodVarianceDiagnostics,
    PopulationLogDensity,
    RateTreatment,
    SelectionResult,
)


def _call_density(log_density: PopulationLogDensity, samples, hyperparameters, *, what):
    result = log_density(samples, hyperparameters)
    return validate_log_population(
        result, expected_shape=next(iter(samples.values())).shape, what=what
    )


def evaluate_events(
    posterior,
    log_density: PopulationLogDensity,
    hyperparameters: Any = None,
) -> EventLikelihoodResult:
    """Evaluate all event importance integrals and their diagnostics."""
    fields = density_required_fields(log_density, posterior.basis)
    posterior.require(fields)
    log_terms = np.empty(posterior.n_events, dtype=float)
    diagnostics = []
    for i, event_name in enumerate(posterior.event_names):
        samples = posterior.get_event(i, fields)
        log_pop = _call_density(
            log_density,
            samples,
            hyperparameters,
            what=f"population log density for {event_name}"
        )
        logw = log_pop - posterior.event_log_ref_density(i)
        n = posterior.sample_count(i)
        log_terms[i] = float(logsumexp(logw) - np.log(n))
        diagnostics.append(importance_diagnostics(logw, n_draw=n))
    return EventLikelihoodResult(
        event_names=posterior.event_names,
        log_likelihoods=log_terms,
        diagnostics=tuple(diagnostics),
    )


def _evaluate_selection_log_population(
    selection,
    log_density: PopulationLogDensity,
    hyperparameters,
    *,
    chunk_size: int | None,
) -> np.ndarray:
    fields = density_required_fields(log_density, selection.basis)
    selection.require(fields)
    n = selection.n_selected
    if chunk_size is None:
        return _call_density(
            log_density,
            {name: selection.samples[name] for name in fields},
            hyperparameters,
            what="population log density for selection samples",
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = {name: selection.samples[name][start:stop] for name in fields}
        out[start:stop] = _call_density(
            log_density,
            chunk,
            hyperparameters,
            what=f"population log density for selection rows {start}:{stop}",
        )
    return out


def evaluate_selection(
    selection,
    log_density: PopulationLogDensity,
    hyperparameters: Any = None,
    *,
    raw_use_observing_time: bool = True,
    chunk_size: int | None = None,
) -> SelectionResult:
    """Evaluate the detectable exposure for either selection input contract."""
    log_pop = _evaluate_selection_log_population(
        selection, log_density, hyperparameters, chunk_size=chunk_size
    )
    base_logw = log_pop - selection.log_draw_density
    factors = selection_log_factors(
        selection, use_observing_time=raw_use_observing_time
    )
    effective_logw = base_logw + factors
    log_exposure = float(logsumexp(effective_logw))
    if not np.isfinite(log_exposure):
        raise SelectionSupportError(
            "selection exposure is zero: the population has no support on retained "
            "selection samples"
        )

    campaign_results = []
    exposure_logs = []
    campaign_variances = []
    for campaign in selection.campaigns:
        rows = selection.rows_for_campaign(campaign.campaign_id)
        if rows.size == 0:
            continue
        logw_k = base_logw[rows]
        diag = importance_diagnostics(logw_k, n_draw=campaign.n_draw)
        log_exp_k = float(logsumexp(effective_logw[rows]))
        if selection.mode is SelectionMode.RAW_DRAW:
            log_eff = float(logsumexp(logw_k) - np.log(campaign.n_draw))
        else:
            log_eff = None
        campaign_results.append(
            CampaignSelectionResult(
                campaign_id=campaign.campaign_id,
                log_exposure=log_exp_k,
                log_efficiency=log_eff,
                diagnostics=diag,
            )
        )
        exposure_logs.append(log_exp_k)
        campaign_variances.append(diag.variance_log_estimate)

    total_draw = (
        sum(int(c.n_draw) for c in selection.campaigns)
        if all(c.n_draw is not None for c in selection.campaigns)
        else effective_logw.size
    )
    combined_diag = importance_diagnostics(effective_logw, n_draw=total_draw)

    if campaign_results and all(np.isfinite(campaign_variances)):
        fractions = np.exp(np.asarray(exposure_logs) - log_exposure)
        variance_log_exposure = float(
            np.sum(fractions**2 * np.asarray(campaign_variances, dtype=float))
        )
    else:
        variance_log_exposure = float("nan")

    return SelectionResult(
        log_exposure=log_exposure,
        diagnostics=combined_diag,
        campaigns=tuple(campaign_results),
        variance_log_exposure=variance_log_exposure,
    )


def evaluate_catalog_terms(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    hyperparameters: Any = None,
    *,
    config: HBIConfig | None = None,
) -> CatalogTerms:
    config = HBIConfig() if config is None else config
    fields = density_required_fields(log_density, posterior.basis)
    validate_pair(posterior, selection, fields)
    events = evaluate_events(posterior, log_density, hyperparameters)
    sel = evaluate_selection(
        selection,
        log_density,
        hyperparameters,
        raw_use_observing_time=config.raw_selection_use_observing_time,
        chunk_size=config.selection_chunk_size,
    )
    event_variance = float(sum(d.variance_log_estimate for d in events.diagnostics))
    selection_variance = float(sel.variance_log_exposure)
    shape_variance = float(
        event_variance + posterior.n_events**2 * selection_variance
    )
    variance = LikelihoodVarianceDiagnostics(
        event_variance=event_variance,
        selection_variance=selection_variance,
        shape_log_likelihood_variance=shape_variance,
    )
    return CatalogTerms(events=events, selection=sel, variance=variance)


def shape_log_likelihood(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    hyperparameters: Any = None,
    *,
    config: HBIConfig | None = None,
) -> CatalogLikeliihoodResult:
    """Rate-marginalized shape likelihood, omitting model-independent constants."""
    cfg = HBIConfig() if config is None else config
    if cfg.rate_treatment is not RateTreatment.SHAPE:
        raise HBIError(
            f"shape_log_likelihood received rate_treatment={cfg.rate_treatment.value!r}"
        )
    terms = evaluate_catalog_terms(
        posterior, selection, log_density, hyperparameters, config=cfg
    )
    value = (
        terms.events.log_likelihood
        - posterior.n_events * terms.selection.log_exposure
     )
    return CatalogLikelihoodResult(
        log_likelihood=float(value),
        terms=terms,
        rate_treatment=RateTreatment.SHAPE,
    )


def poisson_log_likelihood(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    hyperparameters: Any,
    *,
    rate: float,
    config: HBIConfig | None = None,
) -> CatalogLikelihoodResult:
    """Poisson point-process likelihood with explicit positive rate parameter."""
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise HBIError(f"rate must be finite and positive; got {rate}")
    cfg = HBIConfig(rate_treatment=RateTreatment.POISSON) if config is None else config
    if cfg.rate_treatment is not RateTreatment.POISSON:
        raise HBIError(
            f"poisson_log_likelihood received rate_treatment={cfg.rate_treatment.value!r}"
        )
    terms = evaluate_catalog_terms(
        posterior, selection, log_density, hyperparameters, config=cfg
     )
    expected_count = rate * np.exp(terms.selection.log_exposure)
    value = (
        terms.events.log_likelihood
        + posterior.n_events * np.log(rate)
        - expected_count
    )
    return CatalogLikelihoodResult(
        log_likelihood=float(value),
        terms=terms,
        rate_treatment=RateTreatment.POISSON,
        rate=rate,
    )


def catalog_log_likelihood(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    hyperparameters: Any = None,
    *,
    config: HBIConfig | None = None,
    rate: float | None = None,
) -> CatalogLikelihoodResult:
    """Dispatch using the explicitly declared rate treatment."""
    cfg = HBIConfig() if config is None else config
    if cfg.rate_treatment is RateTreatment.SHAPE:
        if rate is not None:
            raise HBIError("shape likelihood does not take an explicit rate")
        return shape_log_likelihood(
            posterior, selection, log_density, hyperparameters, config=cfg
        )
    if rate is None:
        raise HBIError("Poisson rate treatment requires an explicit rate")
    return poisson_log_likelihood(
        posterior, selection, log_density, hyperparameters, rate=rate, config=cfg
    )
