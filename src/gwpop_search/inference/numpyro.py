"""Lazy NumPyro NUTS runner with chain-granularity checkpoint/resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .priors import PriorSpec, serialize_prior_map


class NumPyroUnavailableError(ImportError):
    """Raised when a NumPyro execution path is requested without NumPyro installed."""


def _require_numpyro():
    try:
        import jax
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise NumPyroUnavailableError(
            "NumPyro inference requires the inference extra: "
            "pip install 'gwpop-search[inference]'"
        ) from exc
    return jax, numpyro, dist, MCMC, NUTS


@dataclass(frozen=True)
class NUTSConfig:
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    dense_mass: bool = False
    chain_method: str = "sequential"
    progress_bar: bool = True

    def __post_init__(self) -> None:
        if min(self.num_warmup, self.num_samples, self.num_chains) <= 0:
            raise ValueError("num_warmup, num_samples, and num_chains must be positive")
        if not 0.0 < self.target_accept_prob < 1.0:
            raise ValueError("target_accept_prob must lie strictly between 0 and 1")
        if self.max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be positive")
        if self.chain_method not in {"sequential", "parallel", "vectorized"}:
            raise ValueError(
                "chain_method must be one of 'sequential', 'parallel', or 'vectorized'"
            )


@dataclass(frozen=True)
class NUTSResult:
    samples: Mapping[str, np.ndarray]
    extra_fields: Mapping[str, np.ndarray]
    seed: int
    config: NUTSConfig


def _sample_prior(name, spec: PriorSpec, dist, numpyro):
    if spec.family == "uniform":
        distribution = dist.Uniform(spec.low, spec.high)
    elif spec.family == "log_uniform":
        distribution = dist.LogUniform(spec.low, spec.high)
    elif spec.family == "normal":
        distribution = dist.Normal(spec.loc, spec.scale)
    else:  # guarded by PriorSpec
        raise ValueError(spec.family)
    return numpyro.sample(name, distribution)


def _validated_priors(priors: Mapping[str, PriorSpec]) -> dict[str, PriorSpec]:
    result = {str(name): spec for name, spec in priors.items()}
    if not result:
        raise ValueError("at least one hyperprior is required")
    bad = [name for name, spec in result.items() if not isinstance(spec, PriorSpec)]
    if bad:
        raise TypeError(f"prior entries must be PriorSpec objects; invalid keys={bad}")
    return result


def build_numpyro_model(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    hbi_config=None,
):
    """Build a NumPyro model whose only likelihood factor is the common HBI engine."""
    _, numpyro, dist, _, _ = _require_numpyro()

    from gwpop_search.hbi import HBIConfig, RateTreatment, build_jax_shape_log_likelihood

    cfg = HBIConfig() if hbi_config is None else hbi_config
    if cfg.rate_treatment is not RateTreatment.SHAPE:
        raise ValueError("Phase-3 NumPyro wrapper currently supports shape likelihood only")

    prior_map = _validated_priors(priors)
    likelihood = build_jax_shape_log_likelihood(
        posterior,
        selection,
        population_model,
        config=cfg,
    )

    def model():
        hyperparameters = {
            name: _sample_prior(name, spec, dist, numpyro)
            for name, spec in prior_map.items()
        }
        numpyro.factor("catalog_log_likelihood", likelihood(hyperparameters))

    return model


def run_nuts(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: NUTSConfig | None = None,
    hbi_config=None,
) -> NUTSResult:
    """Run NUTS and return chain-grouped samples plus HMC diagnostics."""
    jax, _, _, MCMC, NUTS = _require_numpyro()
    cfg = NUTSConfig() if config is None else config

    model = build_numpyro_model(
        posterior,
        selection,
        population_model,
        priors,
        hbi_config=hbi_config,
    )
    kernel = NUTS(
        model,
        target_accept_prob=cfg.target_accept_prob,
        max_tree_depth=cfg.max_tree_depth,
        dense_mass=cfg.dense_mass,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        chain_method=cfg.chain_method,
        progress_bar=cfg.progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(int(seed)),
        extra_fields=("diverging", "num_steps", "accept_prob"),
    )

    samples = {
        name: np.asarray(value)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    extra_fields = {
        name: np.asarray(value)
        for name, value in mcmc.get_extra_fields(group_by_chain=True).items()
    }
    return NUTSResult(
        samples=samples,
        extra_fields=extra_fields,
        seed=int(seed),
        config=cfg,
    )


def _chain_seed(root_seed: int, chain_index: int) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{int(chain_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _hbi_config_dict(hbi_config) -> dict[str, object]:
    from gwpop_search.hbi import HBIConfig

    cfg = HBIConfig() if hbi_config is None else hbi_config
    return {
        "rate_treatment": cfg.rate_treatment.value,
        "raw_selection_use_observing_time": bool(
            cfg.raw_selection_use_observing_time
        ),
        "selection_chunk_size": (
            None
            if cfg.selection_chunk_size is None
            else int(cfg.selection_chunk_size)
        ),
    }


def _model_config(population_model) -> dict[str, object]:
    if hasattr(population_model, "to_config"):
        payload = dict(population_model.to_config())
    else:
        payload = {}
    payload.setdefault(
        "class",
        f"{type(population_model).__module__}.{type(population_model).__qualname__}",
    )
    return payload


def build_run_manifest(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: NUTSConfig,
    hbi_config=None,
) -> dict[str, object]:
    """Build the Phase-3 checkpoint identity without production data provenance."""
    return {
        "format_version": "gwpop-search-nuts-1.0",
        "root_seed": int(seed),
        "nuts_config": asdict(config),
        "hbi_config": _hbi_config_dict(hbi_config),
        "priors": serialize_prior_map(_validated_priors(priors)),
        "model": _model_config(population_model),
        "pe_basis": posterior.basis.identity,
        "selection_basis": selection.basis.identity,
        "event_names": list(posterior.event_names),
        "n_events": int(posterior.n_events),
        "n_pe_samples": int(posterior.n_samples_total),
        "n_selected": int(selection.n_selected),
        "selection_mode": selection.mode.value,
    }


def save_result(path: str | Path, result: NUTSResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        f"sample__{name}": np.asarray(value)
        for name, value in result.samples.items()
    }
    arrays.update(
        {
            f"extra__{name}": np.asarray(value)
            for name, value in result.extra_fields.items()
        }
    )
    np.savez_compressed(path, **arrays)

    metadata = {
        "seed": int(result.seed),
        "config": asdict(result.config),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2)
    )


def load_result(path: str | Path) -> NUTSResult:
    path = Path(path)
    metadata = json.loads(
        path.with_suffix(path.suffix + ".json").read_text()
    )
    with np.load(path, allow_pickle=False) as archive:
        samples = {
            name.removeprefix("sample__"): archive[name]
            for name in archive.files
            if name.startswith("sample__")
        }
        extra_fields = {
            name.removeprefix("extra__"): archive[name]
            for name in archive.files
            if name.startswith("extra__")
        }
    return NUTSResult(
        samples=samples,
        extra_fields=extra_fields,
        seed=int(metadata["seed"]),
        config=NUTSConfig(**metadata["config"]),
    )


def run_resumable_chains(
    run_dir: str | Path,
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: NUTSConfig | None = None,
    hbi_config=None,
) -> NUTSResult:
    """Run independent NUTS chains with durable completion at each chain boundary.

    This is deliberately chain-granularity checkpointing. If a job dies after a
    completed chain, that chain is loaded on resume and only missing chains run.
    Within-chain state checkpointing is deferred until it is needed in production.
    """
    cfg = NUTSConfig() if config is None else config
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        posterior,
        selection,
        population_model,
        priors,
        seed=seed,
        config=cfg,
        hbi_config=hbi_config,
    )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError(
                "resume manifest does not match the requested scientific/run configuration"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))

    single_chain = replace(cfg, num_chains=1, chain_method="sequential")
    results: list[NUTSResult] = []

    for chain_index in range(cfg.num_chains):
        chain_path = run_dir / f"chain_{chain_index:03d}.npz"
        expected_seed = _chain_seed(seed, chain_index)

        if (
            chain_path.exists()
            and chain_path.with_suffix(chain_path.suffix + ".json").exists()
        ):
            result = load_result(chain_path)
            if result.seed != expected_seed or result.config != single_chain:
                raise ValueError(
                    f"checkpoint metadata mismatch for chain {chain_index}"
                )
        else:
            result = run_nuts(
                posterior,
                selection,
                population_model,
                priors,
                seed=expected_seed,
                config=single_chain,
                hbi_config=hbi_config,
            )
            save_result(chain_path, result)

        results.append(result)

    sample_keys = tuple(results[0].samples)
    if any(tuple(result.samples) != sample_keys for result in results[1:]):
        raise ValueError("completed chains have inconsistent posterior sample sites")

    samples = {
        name: np.concatenate([result.samples[name] for result in results], axis=0)
        for name in sample_keys
    }

    common_extra = set(results[0].extra_fields)
    for result in results[1:]:
        common_extra &= set(result.extra_fields)
    extra_fields = {
        name: np.concatenate(
            [result.extra_fields[name] for result in results],
            axis=0,
        )
        for name in sorted(common_extra)
    }

    return NUTSResult(
        samples=samples,
        extra_fields=extra_fields,
        seed=int(seed),
        config=cfg,
    )
