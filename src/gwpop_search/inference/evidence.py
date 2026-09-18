"""JAXNS evidence backend through NumPyro's official nested-sampling wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .numpyro import build_numpyro_model
from .priors import PriorSpec


class EvidenceBackendUnavailableError(ImportError):
    pass


def _require_nested_sampling():
    try:
        import jax
        import jaxns
        import numpyro
        from numpyro.contrib.nested_sampling import NestedSampler
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise EvidenceBackendUnavailableError(
            "Nested-sampling evidence requires the evidence extra: "
            "pip install 'gwpop-search[evidence]'"
        ) from exc
    return jax, jaxns, numpyro, NestedSampler


@dataclass(frozen=True)
class NestedSamplingConfig:
    num_live_points: int | None = None
    max_samples: int = 100_000
    dlogz: float = 0.01
    num_posterior_samples: int = 2_000

    def __post_init__(self) -> None:
        if self.num_live_points is not None and self.num_live_points <= 0:
            raise ValueError("num_live_points must be positive when supplied")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self.dlogz <= 0.0:
            raise ValueError("dlogz must be positive")
        if self.num_posterior_samples <= 0:
            raise ValueError("num_posterior_samples must be positive")


@dataclass(frozen=True)
class EvidenceResult:
    log_evidence: float
    log_evidence_error: float
    posterior_samples: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, object]
    seed: int
    backend: str
    config: NestedSamplingConfig


@dataclass(frozen=True)
class EvidenceRepeatSummary:
    log_evidence_mean: float
    repeat_std: float
    mean_reported_error: float
    max_reported_error: float
    n_repeats: int
    estimates: tuple[float, ...]
    errors: tuple[float, ...]

    @property
    def conservative_error(self) -> float:
        return float(max(self.repeat_std, self.max_reported_error))


def run_numpyro_nested_model(
    model,
    *,
    seed: int,
    config: NestedSamplingConfig | None = None,
) -> EvidenceResult:
    """Run JAXNS through NumPyro and return evidence plus resampled posterior."""
    jax, jaxns, numpyro, NestedSampler = _require_nested_sampling()
    cfg = NestedSamplingConfig() if config is None else config

    constructor_kwargs: dict[str, object] = {
        "max_samples": int(cfg.max_samples),
    }
    if cfg.num_live_points is not None:
        constructor_kwargs["num_live_points"] = int(cfg.num_live_points)

    nested = NestedSampler(
        model,
        constructor_kwargs=constructor_kwargs,
        termination_kwargs={"dlogZ": float(cfg.dlogz)},
    )
    run_key, posterior_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    nested.run(run_key)

    # NumPyro currently keeps the upstream JAXNS result object internally while
    # exposing posterior samples publicly. Evidence itself lives on that result.
    results = getattr(nested, "_results", None)
    if results is None:
        raise RuntimeError("NumPyro NestedSampler completed without a result object")

    samples = {
        name: np.asarray(value)
        for name, value in nested.get_samples(
            posterior_key,
            num_samples=cfg.num_posterior_samples,
        ).items()
    }

    diagnostics = {
        "jax_version": str(jax.__version__),
        "jaxns_version": str(getattr(jaxns, "__version__", "unknown")),
        "numpyro_version": str(numpyro.__version__),
        "nested_ess": float(np.asarray(results.ESS)),
        "total_num_samples": int(np.asarray(results.total_num_samples)),
        "total_phantom_samples": int(np.asarray(results.total_phantom_samples)),
        "total_num_likelihood_evaluations": int(
            np.asarray(results.total_num_likelihood_evaluations)
        ),
        "log_efficiency": float(np.asarray(results.log_efficiency)),
        "termination_reason": int(np.asarray(results.termination_reason)),
    }

    return EvidenceResult(
        log_evidence=float(np.asarray(results.log_Z_mean)),
        log_evidence_error=float(np.asarray(results.log_Z_uncert)),
        posterior_samples=samples,
        diagnostics=diagnostics,
        seed=int(seed),
        backend="numpyro-jaxns",
        config=cfg,
    )


def run_hbi_evidence(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: NestedSamplingConfig | None = None,
    hbi_config=None,
) -> EvidenceResult:
    model = build_numpyro_model(
        posterior,
        selection,
        population_model,
        priors,
        hbi_config=hbi_config,
    )
    return run_numpyro_nested_model(model, seed=seed, config=config)


def summarize_evidence_repeats(
    results: list[EvidenceResult] | tuple[EvidenceResult, ...],
) -> EvidenceRepeatSummary:
    if not results:
        raise ValueError("at least one evidence result is required")
    estimates = tuple(float(item.log_evidence) for item in results)
    errors = tuple(float(item.log_evidence_error) for item in results)
    return EvidenceRepeatSummary(
        log_evidence_mean=float(np.mean(estimates)),
        repeat_std=float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        mean_reported_error=float(np.mean(errors)),
        max_reported_error=float(np.max(errors)),
        n_repeats=len(results),
        estimates=estimates,
        errors=errors,
    )


def save_evidence_result(path: str | Path, result: EvidenceResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            f"posterior__{name}": np.asarray(values)
            for name, values in result.posterior_samples.items()
        },
    )
    metadata = {
        "log_evidence": float(result.log_evidence),
        "log_evidence_error": float(result.log_evidence_error),
        "diagnostics": dict(result.diagnostics),
        "seed": int(result.seed),
        "backend": result.backend,
        "config": asdict(result.config),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2)
    )


def load_evidence_result(path: str | Path) -> EvidenceResult:
    path = Path(path)
    metadata = json.loads(
        path.with_suffix(path.suffix + ".json").read_text()
    )
    with np.load(path, allow_pickle=False) as archive:
        posterior = {
            name.removeprefix("posterior__"): archive[name]
            for name in archive.files
            if name.startswith("posterior__")
        }
    return EvidenceResult(
        log_evidence=float(metadata["log_evidence"]),
        log_evidence_error=float(metadata["log_evidence_error"]),
        posterior_samples=posterior,
        diagnostics=metadata["diagnostics"],
        seed=int(metadata["seed"]),
        backend=str(metadata["backend"]),
        config=NestedSamplingConfig(**metadata["config"]),
    )
