"""Deterministic repeated-evidence campaigns over a finite model graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from gwpop_search.grammar import ModelGraph, ModelSpec
from gwpop_search.models import compile_model_spec
from gwpop_search.search import ModelEvidence, ModelPrior, score_model_graph

from .evidence import (
    EvidenceResult,
    NestedSamplingConfig,
    load_evidence_result,
    run_hbi_evidence,
    save_evidence_result,
    summarize_evidence_repeats,
)
from .model_spec import prior_specs_from_model_spec
from .numpyro import _code_identity


@dataclass(frozen=True)
class EvidenceCampaignConfig:
    repeats: int = 2
    nested_sampling: NestedSamplingConfig = field(
        default_factory=NestedSamplingConfig
    )

    def __post_init__(self) -> None:
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")


def evidence_seed(root_seed: int, model_hash: str, repeat_index: int) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{model_hash}:{int(repeat_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _prior_config(model_prior: ModelPrior) -> dict[str, object]:
    payload: dict[str, object] = {
        "class": f"{type(model_prior).__module__}.{type(model_prior).__qualname__}",
        "version": model_prior.version,
    }
    if is_dataclass(model_prior):
        payload["parameters"] = asdict(model_prior)
    return payload


def _hbi_config_dict(hbi_config) -> dict[str, object]:
    from gwpop_search.hbi import HBIConfig

    cfg = HBIConfig() if hbi_config is None else hbi_config
    return {
        "rate_treatment": cfg.rate_treatment.value,
        "raw_selection_use_observing_time": bool(
            cfg.raw_selection_use_observing_time
        ),
        "selection_chunk_size": cfg.selection_chunk_size,
    }


def build_evidence_campaign_manifest(
    graph: ModelGraph,
    posterior,
    selection,
    *,
    root_seed: int,
    config: EvidenceCampaignConfig,
    model_prior: ModelPrior,
    hbi_config=None,
    model_hashes: Iterable[str] | None = None,
    dataset_label: str = "unspecified",
    dataset_identity: str = "unspecified",
) -> dict[str, object]:
    selected = (
        [model.model_hash for model in graph.nodes]
        if model_hashes is None
        else list(model_hashes)
    )
    unknown = set(selected) - set(graph.by_hash)
    if unknown:
        raise ValueError(f"unknown selected model hash(es): {sorted(unknown)}")
    return {
        "format_version": "gwpop-search-evidence-campaign-1.0",
        "code": _code_identity(),
        "graph_root_hash": graph.root_hash,
        "selected_model_hashes": selected,
        "dataset_label": str(dataset_label),
        "pe_basis": posterior.basis.identity,
        "selection_basis": selection.basis.identity,
        "event_names": list(posterior.event_names),
        "n_events": int(posterior.n_events),
        "n_pe_samples": int(posterior.n_samples_total),
        "n_selected": int(selection.n_selected),
        "root_seed": int(root_seed),
        "campaign_config": {
            "repeats": int(config.repeats),
            "nested_sampling": asdict(config.nested_sampling),
        },
        "hbi_config": _hbi_config_dict(hbi_config),
        "model_prior": _prior_config(model_prior),
    }


def _write_json_once(path: Path, payload: dict[str, object], *, what: str) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                f"{what} mismatch in existing evidence directory {path.parent}"
            )
    else:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def _write_model_spec_once(model_dir: Path, spec: ModelSpec) -> None:
    _write_json_once(
        model_dir / "model_spec.json",
        spec.to_dict(),
        what="model spec",
    )


def build_model_evidence_manifest(
    spec: ModelSpec,
    posterior,
    selection,
    *,
    root_seed: int,
    config: EvidenceCampaignConfig,
    hbi_config=None,
    dataset_identity: str = "unspecified",
) -> dict[str, object]:
    """Pin inputs that can make cached evidence scientifically incompatible.

    Production callers should pass the frozen dataset-manifest hash as the
    dataset identity. Synthetic/development runs still pin basis, event list,
    counts, selection mode, code, HBI config, and evidence config.
    """
    return {
        "format_version": "gwpop-search-model-evidence-1.0",
        "code": _code_identity(),
        "model_hash": spec.model_hash,
        "dataset_identity": str(dataset_identity),
        "pe_basis": posterior.basis.identity,
        "selection_basis": selection.basis.identity,
        "event_names": list(posterior.event_names),
        "n_events": int(posterior.n_events),
        "n_pe_samples": int(posterior.n_samples_total),
        "n_selected": int(selection.n_selected),
        "selection_mode": selection.mode.value,
        "root_seed": int(root_seed),
        "campaign_config": {
            "repeats": int(config.repeats),
            "nested_sampling": asdict(config.nested_sampling),
        },
        "hbi_config": _hbi_config_dict(hbi_config),
    }


def run_model_evidence_repeats(
    model_dir: str | Path,
    spec: ModelSpec,
    posterior,
    selection,
    *,
    root_seed: int,
    config: EvidenceCampaignConfig,
    hbi_config=None,
    dataset_identity: str = "unspecified",
) -> tuple[list[EvidenceResult], dict[str, object]]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_model_spec_once(model_dir, spec)
    _write_json_once(
        model_dir / "manifest.json",
        build_model_evidence_manifest(
            spec,
            posterior,
            selection,
            root_seed=root_seed,
            config=config,
            hbi_config=hbi_config,
            dataset_identity=dataset_identity,
        ),
        what="evidence manifest",
    )

    model = compile_model_spec(spec)
    priors = prior_specs_from_model_spec(spec)
    results: list[EvidenceResult] = []

    for repeat_index in range(config.repeats):
        path = model_dir / f"evidence_{repeat_index:03d}.npz"
        seed = evidence_seed(root_seed, spec.model_hash, repeat_index)
        if path.exists() and path.with_suffix(path.suffix + ".json").exists():
            result = load_evidence_result(path)
            if result.seed != seed or result.config != config.nested_sampling:
                raise ValueError(
                    f"evidence checkpoint metadata mismatch for "
                    f"{spec.short_hash} repeat {repeat_index}"
                )
        else:
            result = run_hbi_evidence(
                posterior,
                selection,
                model,
                priors,
                seed=seed,
                config=config.nested_sampling,
                hbi_config=hbi_config,
            )
            save_evidence_result(path, result)
        results.append(result)

    repeat = summarize_evidence_repeats(results)
    summary = {
        "model_hash": spec.model_hash,
        "n_repeats": repeat.n_repeats,
        "log_evidence_mean": repeat.log_evidence_mean,
        "repeat_std": repeat.repeat_std,
        "mean_reported_error": repeat.mean_reported_error,
        "max_reported_error": repeat.max_reported_error,
        "conservative_error": repeat.conservative_error,
        "estimates": list(repeat.estimates),
        "errors": list(repeat.errors),
    }
    (model_dir / "evidence_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return results, summary


def run_graph_evidence_campaign(
    root: str | Path,
    graph: ModelGraph,
    posterior,
    selection,
    *,
    root_seed: int,
    config: EvidenceCampaignConfig,
    model_prior: ModelPrior,
    hbi_config=None,
    model_hashes: Iterable[str] | None = None,
    dataset_label: str = "unspecified",
) -> dict[str, object]:
    """Run/resume repeated evidence for selected nodes and score the resulting graph."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    manifest = build_evidence_campaign_manifest(
        graph,
        posterior,
        selection,
        root_seed=root_seed,
        config=config,
        model_prior=model_prior,
        hbi_config=hbi_config,
        model_hashes=model_hashes,
        dataset_label=dataset_label,
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "existing evidence campaign manifest does not match requested run"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))

    by_hash = graph.by_hash
    evidences: dict[str, ModelEvidence] = {}
    for model_hash in manifest["selected_model_hashes"]:
        spec = by_hash[model_hash]
        _, summary = run_model_evidence_repeats(
            root / "models" / model_hash,
            spec,
            posterior,
            selection,
            root_seed=root_seed,
            config=config,
            hbi_config=hbi_config,
            dataset_identity=dataset_identity,
        )
        evidences[model_hash] = ModelEvidence(
            model_hash=model_hash,
            log_evidence=float(summary["log_evidence_mean"]),
            log_evidence_error=float(summary["conservative_error"]),
        )

    scored = score_model_graph(
        graph,
        evidences,
        model_prior=model_prior,
    )
    payload = scored.to_dict()
    (root / "scored_graph.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2)
    )
    return payload
