"""Frozen-data deterministic production search runner."""

from __future__ import annotations

import json
from pathlib import Path

from gwpop_search.data import PosteriorCatalog, SelectionCatalog
from gwpop_search.grammar import load_model_graph
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.search import (
    ComplexityModelPrior,
    ModelEvidence,
    SearchExecutionConfig,
    UniformModelPrior,
    execute_search,
    score_model_graph,
)
from gwpop_search.store import ResultStore

from .config import ProductionCampaignConfig
from .manifest import DatasetManifest
from .validate import validate_production_freeze


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _single_artifact(
    manifest: DatasetManifest,
    role: str,
) -> str:
    matches = [item.path for item in manifest.artifacts if item.role == role]
    if len(matches) != 1:
        raise ValueError(
            f"production runner requires exactly one {role!r} artifact; "
            f"found {len(matches)}"
        )
    return matches[0]


def load_frozen_dataset(
    manifest: DatasetManifest,
    *,
    data_base_dir: str | Path = ".",
) -> tuple[PosteriorCatalog, SelectionCatalog]:
    base = Path(data_base_dir)
    posterior = PosteriorCatalog.from_hdf5(
        _resolve_path(base, _single_artifact(manifest, "pe"))
    )
    selection = SelectionCatalog.from_hdf5(
        _resolve_path(base, _single_artifact(manifest, "selection"))
    )

    if posterior.basis.identity != manifest.coordinate_basis_identity:
        raise ValueError(
            "PE coordinate basis does not match the frozen dataset manifest"
        )
    if selection.basis.identity != manifest.coordinate_basis_identity:
        raise ValueError(
            "selection coordinate basis does not match the frozen dataset manifest"
        )
    if tuple(posterior.event_names) != tuple(manifest.event_names):
        raise ValueError(
            "PE event ordering/list does not match the frozen dataset manifest"
        )
    if selection.basis.identity != posterior.basis.identity:
        raise ValueError("PE and selection coordinate bases differ")
    return posterior, selection


def model_prior_from_config(payload):
    payload = dict(payload)
    version = str(payload.get("version", ""))
    if version == "uniform-v1":
        return UniformModelPrior()
    if version == "axis-complexity-v1":
        return ComplexityModelPrior(
            penalty_per_axis=float(payload["penalty_per_axis"])
        )
    raise ValueError(f"unsupported frozen model prior version {version!r}")


def _evidence_from_evaluation(path: Path) -> ModelEvidence | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    fidelity = payload.get("fidelity")
    diagnostics = payload.get("diagnostics", {})

    if fidelity == "F3":
        evidence = diagnostics.get("evidence")
    elif fidelity == "F4":
        evidence = diagnostics.get("evidence", {}).get("evidence")
    else:
        return None
    if not isinstance(evidence, dict):
        return None

    return ModelEvidence(
        model_hash=str(payload["model_hash"]),
        log_evidence=float(evidence["log_evidence_mean"]),
        log_evidence_error=float(evidence["conservative_error"]),
    )


def collect_best_available_evidence(
    state_database: str | Path,
) -> dict[str, ModelEvidence]:
    """Prefer F4 evidence, falling back to F3 for models not promoted to F4."""
    store = ResultStore(state_database)
    rows = store.evaluations()
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["model_hash"]),
            0 if row["fidelity"] == "F3" else 1,
        ),
    )
    result: dict[str, ModelEvidence] = {}
    for row in rows:
        if row["fidelity"] not in {"F3", "F4"}:
            continue
        if row["status"] != "complete" or not bool(row["diagnostics_pass"]):
            continue
        artifact = row.get("artifact_path")
        if not artifact:
            continue
        evidence = _evidence_from_evaluation(
            Path(str(artifact)) / "evaluation.json"
        )
        if evidence is not None:
            result[evidence.model_hash] = evidence
    return result


def write_scientific_scoring(
    graph,
    *,
    state_database: str | Path,
    artifact_root: str | Path,
    model_prior,
) -> dict[str, object]:
    """Score the declared model set only when evidence coverage is complete."""
    artifact_root = Path(artifact_root)
    evidences = collect_best_available_evidence(state_database)
    coverage = {
        "format_version": "gwpop-search-evidence-coverage-1.0",
        "n_graph_models": len(graph.nodes),
        "n_models_with_evidence": len(evidences),
        "complete": len(evidences) == len(graph.nodes),
        "missing_model_hashes": [
            model.model_hash
            for model in graph.nodes
            if model.model_hash not in evidences
        ],
    }
    (artifact_root / "evidence_coverage.json").write_text(
        json.dumps(coverage, sort_keys=True, indent=2)
    )

    if not coverage["complete"]:
        return {
            "evidence_coverage": coverage,
            "scored_graph": None,
            "reason": (
                "posterior model probabilities are undefined over the declared "
                "graph until every node has F3/F4 evidence"
            ),
        }

    scored = score_model_graph(
        graph,
        evidences,
        model_prior=model_prior,
    )
    payload = scored.to_dict()
    (artifact_root / "scored_graph.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2)
    )
    return {
        "evidence_coverage": coverage,
        "scored_graph": payload,
        "reason": None,
    }


def run_production_search(
    manifest: DatasetManifest,
    graph_path: str | Path,
    campaign: ProductionCampaignConfig,
    *,
    data_base_dir: str | Path = ".",
    work_dir: str | Path = ".",
    require_current_commit: bool = True,
) -> dict[str, object]:
    """Validate the freeze, load canonical HDF5s, and run/resume F0--F4."""
    if campaign.agents_enabled:
        raise ValueError(
            "deterministic production runner requires agents_enabled=false; "
            "agent orchestration is a separate optional layer"
        )

    freeze = validate_production_freeze(
        manifest,
        graph_path,
        campaign,
        data_base_dir=data_base_dir,
        require_current_commit=require_current_commit,
    )
    if not freeze["valid"]:
        raise ValueError("production freeze validation failed")

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=data_base_dir,
    )
    graph = load_model_graph(graph_path)

    work_dir = Path(work_dir)
    artifact_root = _resolve_path(work_dir, campaign.artifact_root)
    state_database = _resolve_path(work_dir, campaign.state_database)
    artifact_root.mkdir(parents=True, exist_ok=True)

    evaluator = DeterministicHBIEvaluator(
        posterior,
        selection,
        config=campaign.fidelity,
        dataset_identity=manifest.manifest_hash,
    )
    execution = execute_search(
        graph,
        evaluator,
        state_database=state_database,
        artifact_root=artifact_root,
        config=SearchExecutionConfig(
            root_seed=campaign.seed_policy.root_seed,
            scheduler=campaign.scheduler,
            max_models_by_fidelity={
                "F3": campaign.budget.max_f3_models,
                "F4": campaign.budget.max_f4_models,
            },
            max_total_compute_cost=campaign.budget.max_gpu_hours,
        ),
    )

    scoring = write_scientific_scoring(
        graph,
        state_database=state_database,
        artifact_root=artifact_root,
        model_prior=model_prior_from_config(campaign.model_prior),
    )
    result = {
        "format_version": "gwpop-search-production-run-1.0",
        "campaign_id": campaign.campaign_id,
        "campaign_hash": campaign.campaign_hash,
        "dataset_manifest_hash": manifest.manifest_hash,
        "graph_hash": campaign.model_graph_hash,
        "execution": execution.to_dict(),
        "scientific_scoring": scoring,
    }
    (artifact_root / "production_run_summary.json").write_text(
        json.dumps(result, sort_keys=True, indent=2)
    )
    return result
