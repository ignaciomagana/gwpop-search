"""Deterministic repeated dynesty evidence for single models and finite model graphs.

Layout of one model's evidence directory::

    model_dir/
      model_spec.json         the ModelSpec (written once, compared on reuse)
      manifest.json           gwpop-search-model-evidence-2.0 (written once, compared)
      repeat_000/ ...         one resumable dynesty run per repeat
      repeat_001/             (manifest.json, checkpoint.pkl, result.npz[.json])
      evidence_summary.json   gwpop-search-evidence-summary-2.0

Repeat ``r`` uses the seed ``evidence_seed(root_seed, model_hash, r)`` and the
fully resolved :class:`~gwpop_search.inference.dynesty_backend.DynestyConfig`
of :meth:`EvidenceCampaignConfig.dynesty_config_for` (``slices =
slices_multiplier * (3 + ndim)`` for the slice samplers). Completed repeats
are reused and interrupted ones resumed, each only under the backend's
strict identity checks (data digests, priors, dynesty configuration, seed,
code, software and runtime). Directories written by the JAXNS/NumPyro era
(``evidence_###.npz`` files or a 1.x manifest) are refused with
:class:`~gwpop_search.inference.evidence.LegacyEvidenceArtifactError`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import hashlib
import json
import numbers
from pathlib import Path
from typing import Iterable, Mapping

from gwpop_search.grammar import ModelGraph, ModelSpec
from gwpop_search.models import compile_model_spec
from gwpop_search.search import ModelEvidence, ModelPrior, score_model_graph

from .dynesty_backend import DynestyConfig, DynestyResult
from .evidence import (
    LegacyEvidenceArtifactError,
    run_diagnostics,
    run_hbi_evidence,
    sampler_backend_identity,
    summarize_evidence_repeats,
)
from .label_switching import IDENTITY, ModelParameterization
from .model_spec import prior_specs_from_model_spec
from .numpyro import _code_identity
from .priors import serialize_prior_map
from .synthetic import require_population_proxy_coverage

MODEL_EVIDENCE_FORMAT_VERSION = "gwpop-search-model-evidence-2.0"
EVIDENCE_SUMMARY_FORMAT_VERSION = "gwpop-search-evidence-summary-2.0"
EVIDENCE_CAMPAIGN_FORMAT_VERSION = "gwpop-search-evidence-campaign-2.0"

_SLICE_SAMPLERS = ("slice", "rslice")


def _default_dynesty() -> DynestyConfig:
    return DynestyConfig(nlive=1000, bound="multi", sample="rslice", dlogz=0.1)


@dataclass(frozen=True)
class EvidenceCampaignConfig:
    """Repeated independent dynesty runs for one model.

    ``dynesty`` is the run template; with ``slices_multiplier`` set (slice
    samplers only, ``dynesty.slices`` must then be ``None``) each model's runs
    use ``slices = slices_multiplier * (3 + ndim)``, i.e. that multiple of
    dynesty's own ``rslice`` default.
    """

    repeats: int = 2
    dynesty: DynestyConfig = field(default_factory=_default_dynesty)
    slices_multiplier: int | None = 2

    def __post_init__(self) -> None:
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, numbers.Integral):
            raise TypeError("repeats must be an integer")
        if int(self.repeats) <= 0:
            raise ValueError("repeats must be positive")
        object.__setattr__(self, "repeats", int(self.repeats))
        if not isinstance(self.dynesty, DynestyConfig):
            raise TypeError("dynesty must be a DynestyConfig")
        if self.slices_multiplier is not None:
            value = self.slices_multiplier
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError("slices_multiplier must be a positive integer or None")
            if self.dynesty.sample not in _SLICE_SAMPLERS:
                raise ValueError("slices_multiplier applies only to sample='slice'/'rslice'")
            if self.dynesty.slices is not None:
                raise ValueError(
                    "set either dynesty.slices (fixed) or slices_multiplier (per model), "
                    "not both"
                )
            object.__setattr__(self, "slices_multiplier", int(value))

    def dynesty_config_for(self, ndim: int) -> DynestyConfig:
        """The fully resolved per-run configuration for a model with ``ndim`` parameters."""
        ndim = int(ndim)
        if ndim <= 0:
            raise ValueError("ndim must be positive")
        if self.slices_multiplier is None:
            return self.dynesty
        return replace(self.dynesty, slices=self.slices_multiplier * (3 + ndim))

    def to_dict(self) -> dict[str, object]:
        return {
            "repeats": int(self.repeats),
            "dynesty": self.dynesty.to_dict(),
            "slices_multiplier": self.slices_multiplier,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvidenceCampaignConfig":
        payload = dict(payload)
        if "nested_sampling" in payload:
            raise LegacyEvidenceArtifactError(
                "JAXNS/NumPyro-era evidence configuration (nested_sampling block); "
                "re-freeze it for the dynesty backend"
            )
        unknown = sorted(set(payload) - {"repeats", "dynesty", "slices_multiplier"})
        if unknown:
            raise ValueError(f"unknown EvidenceCampaignConfig field(s): {unknown}")
        return cls(
            repeats=payload["repeats"],
            dynesty=DynestyConfig.from_dict(dict(payload["dynesty"])),
            slices_multiplier=payload.get("slices_multiplier"),
        )


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
        "raw_selection_use_observing_time": bool(cfg.raw_selection_use_observing_time),
        "selection_chunk_size": (
            None if cfg.selection_chunk_size is None else int(cfg.selection_chunk_size)
        ),
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
    canonicalize_exchangeable_components: bool = True,
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
        "format_version": EVIDENCE_CAMPAIGN_FORMAT_VERSION,
        "code": _code_identity(),
        "sampler_backend": sampler_backend_identity(),
        "graph_root_hash": graph.root_hash,
        "selected_model_hashes": selected,
        "dataset_label": str(dataset_label),
        "dataset_identity": str(dataset_identity),
        "pe_basis": posterior.basis.identity,
        "selection_basis": selection.basis.identity,
        "event_names": list(posterior.event_names),
        "n_events": int(posterior.n_events),
        "n_pe_samples": int(posterior.n_samples_total),
        "n_selected": int(selection.n_selected),
        "root_seed": int(root_seed),
        "campaign_config": config.to_dict(),
        "canonicalize_exchangeable_components": bool(canonicalize_exchangeable_components),
        "hbi_config": _hbi_config_dict(hbi_config),
        "model_prior": _prior_config(model_prior),
    }


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    tmp.replace(path)


def _write_json_once(path: Path, payload: dict[str, object], *, what: str) -> None:
    payload = json.loads(json.dumps(payload))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            differing = sorted(
                key for key in set(existing) | set(payload) if existing.get(key) != payload.get(key)
            )
            raise ValueError(
                f"{what} mismatch in existing evidence directory {path.parent}; "
                f"differing keys: {differing}"
            )
    else:
        _atomic_write_json(path, payload)


def _refuse_legacy_directory(model_dir: Path) -> None:
    legacy = sorted(path.name for path in model_dir.glob("evidence_[0-9][0-9][0-9].npz"))
    if legacy:
        raise LegacyEvidenceArtifactError(
            f"{model_dir} holds JAXNS/NumPyro-era evidence files {legacy[:4]}; dynesty "
            "evidence must be written to a new artifact root"
        )
    manifest = model_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if payload.get("format_version") != MODEL_EVIDENCE_FORMAT_VERSION:
            raise LegacyEvidenceArtifactError(
                f"{manifest} has format {payload.get('format_version')!r}; expected "
                f"{MODEL_EVIDENCE_FORMAT_VERSION} (JAXNS/NumPyro-era evidence directories "
                "cannot be resumed with dynesty)"
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
    parameterization: ModelParameterization = IDENTITY,
) -> dict[str, object]:
    """Pin every input that makes cached evidence scientifically incompatible.

    Production callers pass the frozen dataset-manifest hash as the dataset
    identity. Each repeat's own manifest additionally pins the data digests,
    code, software and runtime (see ``run_dynesty_population``).
    """
    priors = prior_specs_from_model_spec(spec)
    names = sorted(priors)
    return {
        "format_version": MODEL_EVIDENCE_FORMAT_VERSION,
        "sampler_backend": sampler_backend_identity(),
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
        "repeat_seeds": [
            evidence_seed(root_seed, spec.model_hash, index) for index in range(config.repeats)
        ],
        "campaign_config": config.to_dict(),
        "resolved_dynesty_config": config.dynesty_config_for(len(names)).to_dict(),
        "parameter_order": names,
        "priors": serialize_prior_map(priors),
        "parameterization": parameterization.to_dict(),
        "hbi_config": _hbi_config_dict(hbi_config),
    }


def repeat_run_dir(model_dir: str | Path, repeat_index: int) -> Path:
    return Path(model_dir) / f"repeat_{int(repeat_index):03d}"


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
    parameterization: ModelParameterization = IDENTITY,
) -> tuple[list[DynestyResult], dict[str, object]]:
    """Run/resume ``config.repeats`` independent dynesty runs of one model.

    Returns the per-repeat results (in repeat order) and the written
    ``evidence_summary.json`` payload (mean, repeat std, reported errors,
    ``conservative_error = max(repeat std, max logzerr)``, pairwise z and the
    per-run nested-sampling diagnostics).
    """
    if not isinstance(config, EvidenceCampaignConfig):
        raise TypeError("config must be an EvidenceCampaignConfig")
    priors = prior_specs_from_model_spec(spec)
    require_population_proxy_coverage(
        selection,
        priors=priors,
        context=f"evidence for model {spec.model_hash}",
    )
    parameterization.validate(priors)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    _refuse_legacy_directory(model_dir)
    _write_json_once(model_dir / "model_spec.json", spec.to_dict(), what="model spec")
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
            parameterization=parameterization,
        ),
        what="evidence manifest",
    )

    model = compile_model_spec(spec)
    run_config = config.dynesty_config_for(len(priors))
    results: list[DynestyResult] = []
    for repeat_index in range(config.repeats):
        result = run_hbi_evidence(
            posterior,
            selection,
            model,
            priors,
            seed=evidence_seed(root_seed, spec.model_hash, repeat_index),
            config=run_config,
            hbi_config=hbi_config,
            run_dir=repeat_run_dir(model_dir, repeat_index),
            parameterization=parameterization,
        )
        results.append(result)

    repeat = summarize_evidence_repeats(results)
    summary = {
        "format_version": EVIDENCE_SUMMARY_FORMAT_VERSION,
        "model_hash": spec.model_hash,
        "sampler_backend": sampler_backend_identity(),
        **repeat.to_dict(),
        "runs": [run_diagnostics(result, repeat=index) for index, result in enumerate(results)],
    }
    _atomic_write_json(model_dir / "evidence_summary.json", summary)
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
    dataset_identity: str = "unspecified",
    canonicalize_exchangeable_components: bool = True,
) -> dict[str, object]:
    """Run/resume repeated evidence for selected nodes and score the resulting graph."""
    from .label_switching import parameterization_for_spec

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
        dataset_identity=dataset_identity,
        canonicalize_exchangeable_components=canonicalize_exchangeable_components,
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("existing evidence campaign manifest does not match requested run")
    else:
        _atomic_write_json(manifest_path, manifest)

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
            parameterization=parameterization_for_spec(
                spec, canonicalize=canonicalize_exchangeable_components
            ),
        )
        evidences[model_hash] = ModelEvidence(
            model_hash=model_hash,
            log_evidence=float(summary["log_evidence_mean"]),
            log_evidence_error=float(summary["conservative_error"]),
        )

    scored = score_model_graph(graph, evidences, model_prior=model_prior)
    payload = scored.to_dict()
    _atomic_write_json(root / "scored_graph.json", payload)
    return payload
