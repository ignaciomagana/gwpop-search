"""Monte-Carlo error and bias of ln Z and ln BF from the MC-estimated likelihood.

The sampled likelihood is the importance-sampling estimator (nothing about it
changes here)::

    log Lhat = sum_i log ellhat_i - N log Ahat,
    ellhat_i = (1/n_i) sum_j w_ij,       w_ij = p_pop(theta_ij | Lambda) / pi_ij
    Ahat     = sum_m u_m,                u_m  = (T_k/N_k) p_pop(theta_m | Lambda) / p_draw(theta_m)

With self-normalized weights ``omega_ij = w_ij / sum_j' w_ij'`` and
``omega_m = u_m / sum_m' u_m'``, campaign exposure fractions
``f_k = sum_{m in k} omega_m`` and ``N_k`` the number of draws of campaign
``k`` (``n_draw``; the retained-row count when ``n_draw`` is unknown, as in the
repository's importance diagnostics), the first-order Monte-Carlo covariance of
the log-likelihood estimator between two hyperparameter vectors is
(MODEL_COMPARISON_MATH.md Sec. 6.2)::

    C(Lambda, Lambda') = sum_i ( sum_j omega_ij omega'_ij - 1/n_i )
                         + N^2 ( sum_m omega_m omega'_m - sum_k f_k f'_k / N_k )

and at ``Lambda = Lambda'`` it is the repository's variance diagnostic
``V = sum_i sigma_i^2 + N^2 sigma_A^2``. Averaging over posteriors gives, with
``omega_bar = E_P[omega]``::

    C_PP(a)          = E_{Pa x Pa}[C]           (Var_MC of ln Zhat_a)
    Cov_MC(a, b)     = E_{Pa x Pb}[C]           (models share PE samples and injections)
    Var_MC(ln BF_ab) = C_PP(a) + C_PP(b) - 2 Cov_MC(a, b)
                     = sum_i ||omega_bar^a_i - omega_bar^b_i||^2
                       + N^2 ( ||omega_bar^a - omega_bar^b||^2 - sum_k (f_bar^a_k - f_bar^b_k)^2 / N_k )
    bias(ln Zhat_a)  = N(N+1)/2 E_Pa[sigma_A^2] - C_PP(a)/2
    bias(ln BF_ab)   = bias(ln Zhat_a) - bias(ln Zhat_b)

``C_PP`` is a double integral over *independent* ``Lambda, Lambda'``. From a
weighted sample (weights ``W_s``) it is estimated by the weighted U-statistic
that omits the ``s = s'`` diagonal (unbiased for the double integral; the
plug-in value ``sum_j omega_bar_j^2`` includes the diagonal and is biased high by
about ``E_P[1/ESS] / ESS_post``)::

    E_{PxP}[sum_j omega_j omega'_j] ~ ( sum_j omega_bar_j^2 - sum_s W_s^2 sum_j omega_j(Lambda_s)^2 ) / (1 - sum_s W_s^2)

Both the U-statistic (primary) and the plug-in values are reported. The
cross term needs no correction (independent samples of two models). These are
predictions to *report and gate on*; ln Zhat is never corrected by them
(Essick & Farr 2022).

The empirical estimator :func:`bootstrap_edge_mc_error` resamples, for every
replicate, the PE samples within each event and the injection draws of every
campaign (multinomially over all ``N_k`` draws, undetected ones included) with
the *same* replicate for both models (common random numbers), recomputes
``log L*`` at the posterior points and estimates
``ln Z* - ln Zhat = log sum_s W_s exp(log L*_s - log Lhat_s)`` by importance
reweighting. The replicate spread estimates the MC standard error and the mean
shift estimates the bias (bootstrap bias estimate). The reweighting ESS is
reported and the estimate is refused (``InsufficientReweightingESSError``)
when it falls below ``min_ess``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from ._common import (
    AnalysisInputError,
    WeightedPosterior,
    as_float,
    as_int,
    hbi_config_from_identity,
    json_ready,
    read_json,
    require_format,
    require_identity_matches,
    require_same_data,
    weighted_quantile,
    write_json,
)
from .terms import CatalogWeightEvaluator, PaddedCatalog, pad_catalog

MC_WEIGHTS_FORMAT = "gwpop-search-mc-weights-1.0"
EDGE_MC_FORMAT = "gwpop-search-edge-mc-error-1.0"
DEFAULT_VARIANCE_QUANTILES = (0.5, 0.9, 0.95, 0.99)


class InsufficientReweightingESSError(RuntimeError):
    """The bootstrap reweighting ESS fell below the requested minimum."""


# ---------------------------------------------------------------------------
# Per-model posterior-averaged weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelMCWeights:
    """Posterior-averaged normalized weights of one model and derived moments.

    ``omega_event_bar`` is ``[N, n_max]`` (zero on padding), ``omega_selection_bar``
    ``[M]`` in selection-row order, ``fraction_bar`` ``[K]``. ``diag_*`` are
    the ``sum_s W_s^2 (...)`` terms of the U-statistic and ``sum_w2`` is
    ``sum_s W_s^2``. ``expected_*`` are posterior means of the pointwise
    variances. ``selection_neff_ok_fraction`` is the posterior mass where
    Farr's ``N_eff = 1/sigma_A^2`` exceeds ``4N`` (gate G-MC2).
    """

    label: str
    names: tuple[str, ...]
    likelihood_identity: Mapping[str, object] | None
    event_names: tuple[str, ...]
    pe_counts: np.ndarray
    campaign_ids: tuple[str, ...]
    campaign_n_draw: np.ndarray
    omega_event_bar: np.ndarray
    omega_selection_bar: np.ndarray
    fraction_bar: np.ndarray
    diag_event: np.ndarray
    diag_selection: float
    diag_fraction: float
    sum_w2: float
    expected_event_sigma2: np.ndarray
    expected_selection_sigma2: float
    expected_variance: float
    variance_quantiles: Mapping[str, float]
    selection_neff_ok_fraction: float
    min_selection_ess: float
    min_event_ess: float
    n_points: int
    kish_ess: float
    source: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.event_names)

    def c_pp_components(self, *, u_statistic: bool = True) -> dict[str, float]:
        """Event and selection parts of ``C_PP`` (see the module docstring)."""
        ev_sq = np.sum(self.omega_event_bar**2, axis=1)  # [N]
        sel_sq = float(np.sum(self.omega_selection_bar**2))
        f_sq = float(np.sum(self.fraction_bar**2 / self.campaign_n_draw))
        if u_statistic:
            if not self.sum_w2 < 1.0:
                raise ValueError("the U-statistic needs at least two posterior points")
            scale = 1.0 / (1.0 - self.sum_w2)
            ev_sq = (ev_sq - self.diag_event) * scale
            sel_sq = (sel_sq - self.diag_selection) * scale
            f_sq = (f_sq - self.diag_fraction) * scale
        events = float(np.sum(ev_sq - 1.0 / self.pe_counts))
        selection = float(self.n_events**2 * (sel_sq - f_sq))
        return {"events": events, "selection": selection, "total": events + selection}

    def c_pp(self, *, u_statistic: bool = True) -> float:
        """``Var_MC[ln Zhat]`` predicted as the posterior-averaged covariance."""
        return self.c_pp_components(u_statistic=u_statistic)["total"]

    def predicted_log_evidence_bias(self, *, u_statistic: bool = True) -> float:
        n = self.n_events
        return 0.5 * n * (n + 1) * self.expected_selection_sigma2 - 0.5 * self.c_pp(
            u_statistic=u_statistic
        )

    def summary(self) -> dict[str, object]:
        return json_ready(
            {
                "label": self.label,
                "names": list(self.names),
                "n_events": self.n_events,
                "n_points": self.n_points,
                "kish_ess": self.kish_ess,
                "source": self.source,
                "c_pp": self.c_pp(),
                "c_pp_plug_in": self.c_pp(u_statistic=False),
                "c_pp_components": self.c_pp_components(),
                "expected_variance": self.expected_variance,
                "variance_quantiles": dict(self.variance_quantiles),
                "expected_selection_sigma2": self.expected_selection_sigma2,
                "expected_event_sigma2_total": float(np.sum(self.expected_event_sigma2)),
                "predicted_log_evidence_bias": self.predicted_log_evidence_bias(),
                "selection_neff_ok_fraction": self.selection_neff_ok_fraction,
                "min_selection_ess": self.min_selection_ess,
                "min_event_ess": self.min_event_ess,
            }
        )


def _check_compatible(a: ModelMCWeights, b: ModelMCWeights) -> None:
    require_same_data(a.likelihood_identity, b.likelihood_identity, what=f"{a.label} and {b.label}")
    if (
        a.event_names != b.event_names
        or a.omega_event_bar.shape != b.omega_event_bar.shape
        or a.omega_selection_bar.shape != b.omega_selection_bar.shape
        or a.campaign_ids != b.campaign_ids
        or not np.array_equal(a.pe_counts, b.pe_counts)
        or not np.array_equal(a.campaign_n_draw, b.campaign_n_draw)
    ):
        raise AnalysisInputError(
            f"{a.label} and {b.label} were averaged over different catalog layouts"
        )


def mc_covariance(a: ModelMCWeights, b: ModelMCWeights) -> float:
    """``Cov_MC[ln Zhat_a, ln Zhat_b]`` for two models sharing PE samples and injections."""
    if a is b:
        return a.c_pp()
    _check_compatible(a, b)
    events = float(
        np.sum(np.sum(a.omega_event_bar * b.omega_event_bar, axis=1) - 1.0 / a.pe_counts)
    )
    selection = float(np.sum(a.omega_selection_bar * b.omega_selection_bar)) - float(
        np.sum(a.fraction_bar * b.fraction_bar / a.campaign_n_draw)
    )
    return events + a.n_events**2 * selection


def mc_covariance_matrix(models: Sequence[ModelMCWeights]) -> np.ndarray:
    """``Sigma_MC[a, b] = Cov_MC[ln Zhat_a, ln Zhat_b]`` (diagonal ``C_PP``)."""
    models = tuple(models)
    k = len(models)
    out = np.empty((k, k))
    for i in range(k):
        out[i, i] = models[i].c_pp()
        for j in range(i + 1, k):
            out[i, j] = out[j, i] = mc_covariance(models[i], models[j])
    return out


@dataclass(frozen=True)
class EdgeMCError:
    """Predicted MC error budget of ``ln BF_ab = ln Zhat_a - ln Zhat_b``."""

    label_a: str
    label_b: str
    variance: float
    variance_plug_in: float
    covariance: float
    c_pp_a: float
    c_pp_b: float
    bias: float
    bias_a: float
    bias_b: float

    @property
    def sigma(self) -> float:
        """``sqrt(variance)``; a negative variance estimate (finite-sample noise
        of the U-statistic) is reported as ``variance`` and gives sigma 0 with
        ``variance_negative`` set in :meth:`to_dict`."""
        return math.sqrt(max(self.variance, 0.0))

    @property
    def naive_independent_sigma(self) -> float:
        """``sqrt(C_PP(a) + C_PP(b))``: what adding per-model errors in quadrature gives."""
        return math.sqrt(max(self.c_pp_a + self.c_pp_b, 0.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "variance": self.variance,
            "variance_negative": bool(self.variance < 0.0),
            "sigma": self.sigma,
            "variance_plug_in": self.variance_plug_in,
            "covariance": self.covariance,
            "c_pp_a": self.c_pp_a,
            "c_pp_b": self.c_pp_b,
            "naive_independent_sigma": self.naive_independent_sigma,
            "bias": self.bias,
            "bias_a": self.bias_a,
            "bias_b": self.bias_b,
        }


def edge_mc_error(a: ModelMCWeights, b: ModelMCWeights) -> EdgeMCError:
    """Predicted ``Var_MC``/bias of ``ln BF_ab`` (common random numbers; Sec. 6.2)."""
    _check_compatible(a, b)
    cov = mc_covariance(a, b)
    ca, cb = a.c_pp(), b.c_pp()
    ca_p, cb_p = a.c_pp(u_statistic=False), b.c_pp(u_statistic=False)
    bias_a, bias_b = a.predicted_log_evidence_bias(), b.predicted_log_evidence_bias()
    return EdgeMCError(
        label_a=a.label,
        label_b=b.label,
        variance=ca + cb - 2.0 * cov,
        variance_plug_in=ca_p + cb_p - 2.0 * cov,
        covariance=cov,
        c_pp_a=ca,
        c_pp_b=cb,
        bias=bias_a - bias_b,
        bias_a=bias_a,
        bias_b=bias_b,
    )


def compute_model_mc_weights(
    sample: WeightedPosterior,
    posterior,
    selection,
    population_model,
    *,
    label: str | None = None,
    hbi_config=None,
    batch_size: int = 16,
    backend: str = "jax",
    verify_identity: bool = True,
    variance_quantiles: Sequence[float] = DEFAULT_VARIANCE_QUANTILES,
    catalog: PaddedCatalog | None = None,
) -> ModelMCWeights:
    """Posterior-averaged normalized weights of one model (one pass over ``sample``).

    ``hbi_config`` defaults to the sampled run's configuration. With
    ``verify_identity`` (default) the sample's likelihood identity must match
    ``posterior``/``selection``/``population_model``.
    """
    identity = sample.likelihood_identity
    if hbi_config is None:
        hbi_config = hbi_config_from_identity(identity)
    if verify_identity:
        require_identity_matches(
            identity, posterior, selection, population_model, sample.names, hbi_config,
            what=f"MC weights of {label or 'model'}",
        )
    if catalog is None:
        catalog = pad_catalog(posterior, selection, population_model, hbi_config=hbi_config)
    elif bool(catalog.raw_selection_use_observing_time) != bool(hbi_config.raw_selection_use_observing_time):
        # the identity check above cannot catch this: the mismatch is in the
        # supplied catalog, whose sel_log_factor carries (or omits) the
        # per-campaign log(T_k / N_k) that the selection weights are built from
        raise AnalysisInputError(
            f"MC weights of {label or 'model'}: the supplied catalog was padded with "
            f"raw_selection_use_observing_time={catalog.raw_selection_use_observing_time} but the "
            f"sampled likelihood used {hbi_config.raw_selection_use_observing_time}; the "
            "self-normalized selection weights, C_PP and sigma_A^2 would come from a "
            "different estimator than the one that was sampled"
        )
    evaluator = CatalogWeightEvaluator(
        catalog, population_model, sample.names, batch_size=batch_size, backend=backend
    )
    W = sample.weights
    out = evaluator.moments(sample.points, W)
    n = catalog.n_events
    variance = out["variance"]
    quantiles = {
        f"q{q:g}": float(weighted_quantile(variance, W, q)) for q in variance_quantiles
    }
    neff_ok = out["sigma_a2"] < 1.0 / (4.0 * n)
    return ModelMCWeights(
        label=str(label) if label is not None else "model",
        names=tuple(sample.names),
        likelihood_identity=identity,
        event_names=catalog.event_names,
        pe_counts=catalog.pe_counts.astype(np.float64),
        campaign_ids=catalog.campaign_ids,
        campaign_n_draw=np.asarray(catalog.campaign_n_draw, dtype=np.float64),
        omega_event_bar=np.where(catalog.pe_mask, out["omega_e_sum"], 0.0),
        omega_selection_bar=np.asarray(out["omega_s_sum"])[: catalog.n_selected],
        fraction_bar=np.asarray(out["frac_sum"], dtype=np.float64),
        diag_event=np.asarray(out["diag_e"], dtype=np.float64),
        diag_selection=float(out["diag_s"]),
        diag_fraction=float(out["diag_f"]),
        sum_w2=float(np.sum(W * W)),
        expected_event_sigma2=np.asarray(out["sigma_e2_sum"], dtype=np.float64),
        expected_selection_sigma2=float(out["sigma_a2_sum"]),
        expected_variance=float(out["variance_sum"]),
        variance_quantiles=quantiles,
        selection_neff_ok_fraction=float(np.sum(W[neff_ok])),
        min_selection_ess=float(np.min(out["selection_ess"])),
        min_event_ess=float(np.min(out["min_event_ess"])),
        n_points=sample.n_points,
        kish_ess=sample.kish_ess,
        source=sample.source,
        metadata=dict(sample.metadata),
    )


def save_model_mc_weights(path: str | Path, weights: ModelMCWeights) -> None:
    """``path`` (npz arrays) plus ``path + '.json'`` (scalars and identity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": MC_WEIGHTS_FORMAT,
        "label": weights.label,
        "names": list(weights.names),
        "likelihood_identity": weights.likelihood_identity,
        "event_names": list(weights.event_names),
        "campaign_ids": list(weights.campaign_ids),
        "diag_selection": weights.diag_selection,
        "diag_fraction": weights.diag_fraction,
        "sum_w2": weights.sum_w2,
        "expected_selection_sigma2": weights.expected_selection_sigma2,
        "expected_variance": weights.expected_variance,
        "variance_quantiles": dict(weights.variance_quantiles),
        "selection_neff_ok_fraction": weights.selection_neff_ok_fraction,
        "min_selection_ess": weights.min_selection_ess,
        "min_event_ess": weights.min_event_ess,
        "n_points": weights.n_points,
        "kish_ess": weights.kish_ess,
        "source": weights.source,
        "metadata": dict(weights.metadata),
        "summary": weights.summary(),
    }
    write_json(path.with_suffix(path.suffix + ".json"), metadata)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            pe_counts=weights.pe_counts,
            campaign_n_draw=weights.campaign_n_draw,
            omega_event_bar=weights.omega_event_bar,
            omega_selection_bar=weights.omega_selection_bar,
            fraction_bar=weights.fraction_bar,
            diag_event=weights.diag_event,
            expected_event_sigma2=weights.expected_event_sigma2,
        )
    os.replace(tmp, path)


def load_model_mc_weights(path: str | Path) -> ModelMCWeights:
    path = Path(path)
    metadata = read_json(path.with_suffix(path.suffix + ".json"))
    require_format(metadata, MC_WEIGHTS_FORMAT, what=str(path))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return ModelMCWeights(
        label=str(metadata["label"]),
        names=tuple(metadata["names"]),
        likelihood_identity=metadata.get("likelihood_identity"),
        event_names=tuple(metadata["event_names"]),
        pe_counts=arrays["pe_counts"],
        campaign_ids=tuple(metadata["campaign_ids"]),
        campaign_n_draw=arrays["campaign_n_draw"],
        omega_event_bar=arrays["omega_event_bar"],
        omega_selection_bar=arrays["omega_selection_bar"],
        fraction_bar=arrays["fraction_bar"],
        diag_event=arrays["diag_event"],
        diag_selection=float(metadata["diag_selection"]),
        diag_fraction=float(metadata["diag_fraction"]),
        sum_w2=float(metadata["sum_w2"]),
        expected_event_sigma2=arrays["expected_event_sigma2"],
        expected_selection_sigma2=float(metadata["expected_selection_sigma2"]),
        expected_variance=float(metadata["expected_variance"]),
        variance_quantiles=dict(metadata["variance_quantiles"]),
        selection_neff_ok_fraction=float(metadata["selection_neff_ok_fraction"]),
        min_selection_ess=float(metadata["min_selection_ess"]),
        min_event_ess=float(metadata["min_event_ess"]),
        n_points=int(metadata["n_points"]),
        kish_ess=float(metadata["kish_ess"]),
        source=str(metadata["source"]),
        metadata=dict(metadata.get("metadata") or {}),
    )


# ---------------------------------------------------------------------------
# Bootstrap reweighting (empirical estimator)
# ---------------------------------------------------------------------------


def bootstrap_counts(
    catalog: PaddedCatalog, n_replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resampling multiplicities ``(counts_e [R, N, n_max], counts_s [R, M_pad])``.

    Replicate ``r`` uses ``numpy.random.default_rng([seed, r])`` (independent
    of how replicates are chunked). Each event's ``n_i`` PE samples are
    resampled with replacement; each campaign's ``N_k`` injection draws are
    resampled with replacement (multinomial over the retained rows plus one
    bucket for the undetected draws, which carry zero weight).
    """
    n_replicates = as_int("n_replicates", n_replicates, minimum=1)
    seed = as_int("seed", seed, minimum=0)
    counts_e = np.zeros((n_replicates, catalog.n_events, catalog.n_max), dtype=np.int32)
    counts_s = np.zeros((n_replicates, catalog.m_pad), dtype=np.int32)
    pe_counts = catalog.pe_counts
    for r in range(n_replicates):
        rng = np.random.default_rng([seed, r])
        for i in range(catalog.n_events):
            n = int(pe_counts[i])
            counts_e[r, i, :n] = rng.multinomial(n, np.full(n, 1.0 / n))
        for k, rows in enumerate(catalog.selection_rows_per_campaign):
            n_draw = int(round(catalog.campaign_n_draw[k]))
            n_rows = rows.size
            p = np.full(n_rows + 1, 1.0 / n_draw)
            p[-1] = max(0.0, 1.0 - n_rows / n_draw)
            p /= p.sum()
            counts_s[r, rows] = rng.multinomial(n_draw, p)[:n_rows]
    return counts_e, counts_s


@dataclass(frozen=True)
class BootstrapModelShift:
    """Per-replicate ``ln Z* - ln Zhat`` of one model."""

    label: str
    log_evidence_shifts: np.ndarray
    reweighting_ess: np.ndarray
    base_kish_ess: float

    def to_dict(self) -> dict[str, object]:
        shifts = self.log_evidence_shifts
        return {
            "label": self.label,
            "sd": float(np.std(shifts, ddof=1)) if shifts.size > 1 else None,
            "mean_shift": float(np.mean(shifts)),
            "se_mean_shift": (
                float(np.std(shifts, ddof=1) / math.sqrt(shifts.size)) if shifts.size > 1 else None
            ),
            "min_reweighting_ess": float(np.min(self.reweighting_ess)),
            "median_reweighting_ess": float(np.median(self.reweighting_ess)),
            "base_kish_ess": self.base_kish_ess,
        }


@dataclass(frozen=True)
class BootstrapEdgeMCError:
    """Empirical (bootstrap-reweighting) MC error of ``ln BF_ab``."""

    n_replicates: int
    seed: int
    model_a: BootstrapModelShift
    model_b: BootstrapModelShift
    min_ess: float

    @property
    def log_bayes_factor_shifts(self) -> np.ndarray:
        return self.model_a.log_evidence_shifts - self.model_b.log_evidence_shifts

    @property
    def sigma(self) -> float:
        return float(np.std(self.log_bayes_factor_shifts, ddof=1))

    @property
    def mean_shift(self) -> float:
        return float(np.mean(self.log_bayes_factor_shifts))

    def to_dict(self) -> dict[str, object]:
        shifts = self.log_bayes_factor_shifts
        return {
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "min_ess": self.min_ess,
            "sigma": self.sigma,
            "mean_shift": self.mean_shift,
            "se_mean_shift": float(np.std(shifts, ddof=1) / math.sqrt(shifts.size)),
            "model_a": self.model_a.to_dict(),
            "model_b": self.model_b.to_dict(),
        }


def _reweight(W, base, replicate_values):
    """``ln Z* - ln Zhat`` and reweighting ESS per replicate."""
    log_w = np.log(W)[None, :]
    with np.errstate(invalid="ignore"):
        delta = replicate_values - base[None, :]
    delta = np.where(np.isneginf(replicate_values), -np.inf, delta)
    terms = log_w + delta
    shift = logsumexp(terms, axis=1)
    finite = np.isfinite(shift)
    ess = np.zeros(shift.shape)
    if finite.any():
        norm = np.exp(terms[finite] - shift[finite, None])
        ess[finite] = 1.0 / np.sum(norm * norm, axis=1)
    return shift, ess


def bootstrap_model_shifts(
    sample: WeightedPosterior,
    evaluator: CatalogWeightEvaluator,
    counts_e: np.ndarray,
    counts_s: np.ndarray,
    *,
    label: str,
    replicate_chunk: int = 16,
) -> BootstrapModelShift:
    """Per-replicate evidence shifts of one model by importance reweighting."""
    replicate_chunk = as_int("replicate_chunk", replicate_chunk, minimum=1)
    X, W = sample.points, sample.weights
    cat = evaluator.catalog
    ones_e = cat.pe_mask[None].astype(counts_e.dtype)
    ones_s = cat.sel_mask[None].astype(counts_s.dtype)
    both = evaluator.bootstrap(
        X,
        np.concatenate([ones_e, counts_e], axis=0),
        np.concatenate([ones_s, counts_s], axis=0),
        replicate_chunk=replicate_chunk,
    )
    base, values = both[0], both[1:]
    if not np.all(np.isfinite(base)):
        raise ValueError(f"{label}: posterior points with zero likelihood under the original sample")
    shifts, ess = _reweight(W, base, values)
    return BootstrapModelShift(
        label=label,
        log_evidence_shifts=shifts,
        reweighting_ess=ess,
        base_kish_ess=sample.kish_ess,
    )


def bootstrap_edge_mc_error(
    sample_a: WeightedPosterior,
    sample_b: WeightedPosterior,
    posterior,
    selection,
    model_a,
    model_b,
    *,
    n_replicates: int = 200,
    seed: int = 0,
    min_ess: float = 100.0,
    hbi_config=None,
    labels: tuple[str, str] = ("a", "b"),
    batch_size: int = 16,
    replicate_chunk: int = 16,
    backend: str = "jax",
    verify_identity: bool = True,
    catalogs: tuple[PaddedCatalog, PaddedCatalog] | None = None,
) -> BootstrapEdgeMCError:
    """Empirical MC error of ``ln BF_ab`` by bootstrap reweighting (see module doc).

    The same resampling replicate is applied to both models (common random
    numbers). Raises :class:`InsufficientReweightingESSError` when the
    smallest reweighting ESS of either model is below ``min_ess``.
    """
    min_ess = as_float("min_ess", min_ess, positive=True)
    if hbi_config is None:
        hbi_config = hbi_config_from_identity(sample_a.likelihood_identity)
    if verify_identity:
        require_same_data(sample_a.likelihood_identity, sample_b.likelihood_identity, what="models")
        for sample, model, label in ((sample_a, model_a, labels[0]), (sample_b, model_b, labels[1])):
            require_identity_matches(
                sample.likelihood_identity, posterior, selection, model, sample.names, hbi_config,
                what=f"bootstrap of {label}",
            )
    if catalogs is None:
        catalogs = (
            pad_catalog(posterior, selection, model_a, hbi_config=hbi_config),
            pad_catalog(posterior, selection, model_b, hbi_config=hbi_config),
        )
    cat_a, cat_b = catalogs
    if (cat_a.n_max, cat_a.m_pad) != (cat_b.n_max, cat_b.m_pad):
        raise AnalysisInputError("the two models' catalog layouts differ")
    for cat, label in ((cat_a, labels[0]), (cat_b, labels[1])):
        if bool(cat.raw_selection_use_observing_time) != bool(hbi_config.raw_selection_use_observing_time):
            raise AnalysisInputError(
                f"bootstrap of {label}: the supplied catalog was padded with "
                f"raw_selection_use_observing_time={cat.raw_selection_use_observing_time} but the "
                f"sampled likelihood used {hbi_config.raw_selection_use_observing_time}"
            )
    counts_e, counts_s = bootstrap_counts(cat_a, n_replicates, seed)
    shifts = []
    for sample, model, cat, label in (
        (sample_a, model_a, cat_a, labels[0]),
        (sample_b, model_b, cat_b, labels[1]),
    ):
        evaluator = CatalogWeightEvaluator(
            cat, model, sample.names, batch_size=batch_size, backend=backend
        )
        shifts.append(
            bootstrap_model_shifts(
                sample, evaluator, counts_e, counts_s, label=label, replicate_chunk=replicate_chunk
            )
        )
    result = BootstrapEdgeMCError(
        n_replicates=int(n_replicates),
        seed=int(seed),
        model_a=shifts[0],
        model_b=shifts[1],
        min_ess=min_ess,
    )
    worst = min(float(np.min(s.reweighting_ess)) for s in shifts)
    if worst < min_ess:
        raise InsufficientReweightingESSError(
            f"bootstrap reweighting ESS fell to {worst:.3g} < {min_ess:g} "
            f"({[s.to_dict() for s in shifts]}); the posterior sample cannot represent the "
            "resampled likelihoods, so no empirical MC error is reported"
        )
    return result


def edge_mc_report(
    predicted: EdgeMCError,
    empirical: BootstrapEdgeMCError | None = None,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format_version": EDGE_MC_FORMAT,
        "predicted": predicted.to_dict(),
        "empirical": None if empirical is None else empirical.to_dict(),
    }
    if extra:
        payload.update(dict(extra))
    return json_ready(payload)
