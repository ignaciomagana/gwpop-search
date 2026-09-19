"""Tie the null-calibration PE precision to the frozen observed catalog.

D4 requires the observed statistic and every null statistic to come from the
identical F3 procedure. That is only true if the null catalogs are measured
like the real ones: the Monte-Carlo regime of the HBI likelihood
(``Var[log ell_i] ~ 1/ESS_i - 1/n_i``) is set by the PE sample count, and the
null distribution of the maximum edge ln BF is set by how sharply the events
are measured. A fixed synthetic noise model is neither.

This module measures the per-event PE precision of the frozen catalog and
builds the per-event measurement-noise scales the null survey uses:

``match_observed`` (default)
    Every null event inherits one observed event's four marginal posterior
    widths (``sigma[ln m1_det], sigma[q], sigma[ln d_L], sigma[chi_eff]``) and
    the observed per-event sample count. The assignment is the rank match on
    the detection-statistic proxy ``ln rho = (5/6) ln Mc_det - ln d_L``, so the
    null catalog reproduces both the observed *multiset* of measurement
    precisions and their correlation with event loudness.

``declared_fixed``
    The configured scalar scales and sample count are used unchanged, and the
    measured mismatch against the frozen catalog is recorded as a declared
    approximation (never silently).

Remaining approximation in both modes: only the four marginal widths are
matched; the real posteriors' shapes and correlations are not. The Gaussian
scales ``s`` are the measurement-noise scales, so the null posterior widths
equal ``s`` only up to the uniform-in-(m1, d_L) Jacobian shift and truncation
at the PE prior box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from gwpop_search.inference.synthetic import SyntheticSurveyConfig

OBSERVED_PE_SCALES_FORMAT_VERSION = "gwpop-search-observed-pe-scales-1.0"

PE_SCALE_POLICY_MATCH_OBSERVED = "match_observed"
PE_SCALE_POLICY_DECLARED_FIXED = "declared_fixed"
PE_SCALE_POLICIES = (
    PE_SCALE_POLICY_MATCH_OBSERVED,
    PE_SCALE_POLICY_DECLARED_FIXED,
)

# Observed-field order used by the synthetic noisy_observation machinery.
PE_SCALE_FIELDS = (
    "log_m1_detector",
    "q",
    "log_luminosity_distance",
    "chi_eff",
)
# Survey configuration field carrying each scale.
PE_SCALE_CONFIG_FIELDS = {
    "log_m1_detector": "pe_m1_fractional_sigma",
    "q": "pe_q_sigma",
    "log_luminosity_distance": "pe_d_l_fractional_sigma",
    "chi_eff": "pe_chi_eff_sigma",
}


def detection_statistic_proxy(
    m1_detector: np.ndarray,
    q: np.ndarray,
    luminosity_distance: np.ndarray,
) -> np.ndarray:
    """``ln rho = (5/6) ln Mc_detector - ln d_L`` (the survey's reach scaling).

    Only the ordering of this proxy is used (rank matching), so its additive
    normalization is irrelevant.
    """
    m1 = np.asarray(m1_detector, dtype=float)
    q = np.asarray(q, dtype=float)
    d_l = np.asarray(luminosity_distance, dtype=float)
    if np.any(m1 <= 0.0) or np.any(q <= 0.0) or np.any(d_l <= 0.0):
        raise ValueError(
            "the detection-statistic proxy needs positive m1_detector, q and "
            "luminosity_distance"
        )
    log_chirp_mass = np.log(m1) + 0.6 * np.log(q) - 0.2 * np.log1p(q)
    return (5.0 / 6.0) * log_chirp_mass - np.log(d_l)


@dataclass(frozen=True)
class ObservedPEScales:
    """Per-event PE precision measured on a frozen posterior catalog."""

    n_events: int
    samples_per_event: np.ndarray
    sigmas: Mapping[str, np.ndarray]
    loudness: np.ndarray

    def __post_init__(self) -> None:
        for name in PE_SCALE_FIELDS:
            values = np.asarray(self.sigmas[name], dtype=float)
            if values.shape != (self.n_events,):
                raise ValueError(f"sigma {name!r} must have one entry per event")
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"sigma {name!r} must be finite and positive")

    def uniform_samples_per_event(self) -> int:
        """The common per-event sample count; refuses a ragged catalog."""
        counts = np.unique(np.asarray(self.samples_per_event, dtype=np.int64))
        if counts.size != 1:
            raise ValueError(
                "matching the null PE to the observed catalog needs one common "
                "per-event sample count; the frozen catalog is ragged "
                f"(min={int(counts.min())}, max={int(counts.max())})"
            )
        return int(counts[0])

    def summary(self) -> dict[str, object]:
        """JSON-ready distribution of the measured precision."""
        counts = np.asarray(self.samples_per_event, dtype=np.int64)
        return {
            "format_version": OBSERVED_PE_SCALES_FORMAT_VERSION,
            "n_events": int(self.n_events),
            "samples_per_event": {
                "min": int(counts.min()),
                "median": float(np.median(counts)),
                "max": int(counts.max()),
            },
            "posterior_width": {
                name: {
                    "median": float(np.median(self.sigmas[name])),
                    "q10": float(np.quantile(self.sigmas[name], 0.10)),
                    "q90": float(np.quantile(self.sigmas[name], 0.90)),
                    "min": float(np.min(self.sigmas[name])),
                    "max": float(np.max(self.sigmas[name])),
                }
                for name in PE_SCALE_FIELDS
            },
        }


def measure_observed_pe_scales(posterior) -> ObservedPEScales:
    """Measure each event's marginal posterior widths and sample count.

    The widths are the sample standard deviations (``ddof=1``) of
    ``ln m1_detector``, ``q``, ``ln d_L`` and ``chi_eff`` over the event's own
    posterior samples: the four scales the ``noisy_observation`` PE model
    reproduces.
    """
    required = ("m1_detector", "q", "luminosity_distance", "chi_eff")
    missing = [name for name in required if name not in posterior.samples]
    if missing:
        raise ValueError(
            f"measuring the observed PE precision needs sample column(s) {missing}"
        )
    n_events = int(posterior.n_events)
    counts = np.empty(n_events, dtype=np.int64)
    loudness = np.empty(n_events, dtype=float)
    sigmas = {name: np.empty(n_events, dtype=float) for name in PE_SCALE_FIELDS}
    columns = {
        name: np.asarray(posterior.samples[name], dtype=float) for name in required
    }
    for index in range(n_events):
        sl = posterior.event_slice(index)
        m1 = columns["m1_detector"][sl]
        q = columns["q"][sl]
        d_l = columns["luminosity_distance"][sl]
        chi = columns["chi_eff"][sl]
        if m1.size < 2:
            raise ValueError(
                f"event {posterior.event_names[index]!r} has fewer than two "
                "posterior samples; its PE width is undefined"
            )
        if np.any(m1 <= 0.0) or np.any(d_l <= 0.0):
            raise ValueError(
                f"event {posterior.event_names[index]!r} has non-positive "
                "m1_detector or luminosity_distance"
            )
        counts[index] = m1.size
        sigmas["log_m1_detector"][index] = np.std(np.log(m1), ddof=1)
        sigmas["q"][index] = np.std(q, ddof=1)
        sigmas["log_luminosity_distance"][index] = np.std(np.log(d_l), ddof=1)
        sigmas["chi_eff"][index] = np.std(chi, ddof=1)
        loudness[index] = np.median(detection_statistic_proxy(m1, q, d_l))
    for name, values in sigmas.items():
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            bad = int(np.argmin(np.where(np.isfinite(values) & (values > 0.0), 1, 0)))
            raise ValueError(
                f"observed posterior width {name!r} is not finite and positive for "
                f"event {posterior.event_names[bad]!r}; the null PE cannot be "
                "matched to it"
            )
    return ObservedPEScales(
        n_events=n_events,
        samples_per_event=counts,
        sigmas=sigmas,
        loudness=loudness,
    )


def rank_matched_event_sigmas(
    scales: ObservedPEScales,
    truths: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Assign observed width vectors to null events by loudness rank.

    ``truths`` are the null event truths (``m1_detector``, ``q``,
    ``luminosity_distance``). The i-th loudest null event inherits the four
    widths of the i-th loudest observed event, so the null catalog carries the
    observed multiset of precisions with the observed precision-loudness
    correlation. Returns the per-event scales and a provenance block.
    """
    null_loudness = detection_statistic_proxy(
        truths["m1_detector"],
        truths["q"],
        truths["luminosity_distance"],
    )
    n_events = int(null_loudness.size)
    if n_events != scales.n_events:
        raise ValueError(
            "null and observed event counts must match to rank-match the PE "
            f"precision; null={n_events}, observed={scales.n_events}"
        )
    observed_order = np.argsort(scales.loudness, kind="stable")
    null_order = np.argsort(null_loudness, kind="stable")
    assignment = np.empty(n_events, dtype=np.int64)
    assignment[null_order] = observed_order
    matched = {
        name: np.asarray(scales.sigmas[name], dtype=float)[assignment]
        for name in PE_SCALE_FIELDS
    }
    provenance = {
        "policy": PE_SCALE_POLICY_MATCH_OBSERVED,
        "assignment": "rank match on ln rho = (5/6) ln Mc_detector - ln d_L",
        "observed_scales": scales.summary(),
        "applied_scales": {
            name: {
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in matched.items()
        },
    }
    return matched, provenance


def pe_scale_comparison(
    survey_config: SyntheticSurveyConfig,
    scales: ObservedPEScales,
) -> dict[str, object]:
    """Declared fixed scales and sample count against the observed catalog."""
    counts = np.asarray(scales.samples_per_event, dtype=np.int64)
    declared_samples = int(survey_config.posterior_samples_per_event)
    observed_median_samples = float(np.median(counts))
    comparison = {
        "posterior_samples_per_event": {
            "declared": declared_samples,
            "observed_median": observed_median_samples,
            "observed_min": int(counts.min()),
            "observed_max": int(counts.max()),
            "ratio_observed_over_declared": (
                observed_median_samples / declared_samples
            ),
        },
        "posterior_width": {},
    }
    for name in PE_SCALE_FIELDS:
        declared = float(getattr(survey_config, PE_SCALE_CONFIG_FIELDS[name]))
        observed_median = float(np.median(scales.sigmas[name]))
        comparison["posterior_width"][name] = {
            "declared": declared,
            "observed_median": observed_median,
            "observed_q10": float(np.quantile(scales.sigmas[name], 0.10)),
            "observed_q90": float(np.quantile(scales.sigmas[name], 0.90)),
            "ratio_observed_over_declared": observed_median / declared,
        }
    ratios = [
        float(item["ratio_observed_over_declared"])
        for item in comparison["posterior_width"].values()
    ]
    comparison["max_width_ratio"] = max(max(ratios), max(1.0 / r for r in ratios))
    return comparison


def require_pe_scale_policy(policy: str) -> str:
    if policy not in PE_SCALE_POLICIES:
        raise ValueError(
            f"unsupported null PE scale policy {policy!r}; supported: {PE_SCALE_POLICIES}"
        )
    return policy
