"""Evaluate one declared model specification at a frozen campaign's settings.

Execution plumbing for robustness reruns that are not nodes of the frozen
graph (a production model with one hyperprior widened, an exploratory depth-2
model). The evaluation is the one :func:`gwpop_search.search.execute_search`
and ``run-fidelity-evaluation`` perform for a graph model: the same
:class:`~gwpop_search.inference.fidelity.DeterministicHBIEvaluator` built from
the frozen catalog, the frozen selection, ``campaign.fidelity`` and the
dataset-manifest hash, the same seed ``evaluation_seed(campaign root seed,
model hash, fidelity)`` and the same run-directory layout
``<root>/<fidelity>/<model hash>``. Because the model hash is the SHA-256 of
the canonical specification (hyperpriors included), the spec of a graph model
reproduces that model's evaluation exactly, and a widened prior is a distinct
model with its own hash, seed and directory.

Nothing is written to a state database. The provenance of the request (the
spec file's SHA-256, the running code, the frozen inputs) is written next to
the run directory, never inside it, so the run directory stays exactly what
the production evaluator writes.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from gwpop_search.grammar import DEFAULT_COMPONENT_REGISTRY, ModelGraph, ModelSpec
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.search import Fidelity, evaluation_seed

from .config import ProductionCampaignConfig

MODEL_SPEC_EVALUATION_FORMAT = "gwpop-search-model-spec-evaluation-1.0"
PROVENANCE_SUFFIX = ".spec_evaluation.json"


def model_spec_run_dir(
    root: str | Path,
    model_hash: str,
    fidelity: Fidelity,
) -> Path:
    """``<root>/<fidelity>/<model hash>``: the production/preflight layout."""
    return Path(root) / Fidelity(fidelity).value / str(model_hash)


def model_spec_provenance_path(
    root: str | Path,
    model_hash: str,
    fidelity: Fidelity,
) -> Path:
    """Sibling of the run directory (the run directory itself stays pristine)."""
    return (
        Path(root)
        / Fidelity(fidelity).value
        / f"{model_hash}{PROVENANCE_SUFFIX}"
    )


def model_spec_diff(
    parent: ModelSpec,
    child: ModelSpec,
) -> dict[str, object]:
    """Structural and hyperprior differences of ``child`` relative to ``parent``."""
    parent_payload = parent.to_dict()
    child_payload = child.to_dict()
    blocks = {}
    for name in sorted(
        set(parent_payload["blocks"]) | set(child_payload["blocks"])
    ):
        a = parent_payload["blocks"].get(name)
        b = child_payload["blocks"].get(name)
        if a != b:
            blocks[name] = {"parent": a, "child": b}
    priors = {}
    for name in sorted(
        set(parent_payload["priors"]) | set(child_payload["priors"])
    ):
        a = parent_payload["priors"].get(name)
        b = child_payload["priors"].get(name)
        if a != b:
            priors[name] = {"parent": a, "child": b}
    return {
        "parent_model_hash": parent.model_hash,
        "child_model_hash": child.model_hash,
        "blocks": blocks,
        "priors": priors,
    }


def with_prior(
    spec: ModelSpec,
    name: str,
    family: str,
    parameters: Mapping[str, float],
) -> ModelSpec:
    """``spec`` with the declared hyperprior ``name`` replaced.

    Only an existing prior may be replaced: a typo would otherwise add an
    unused prior and silently yield a different model hash.
    """
    if name not in spec.priors:
        raise ValueError(
            f"prior {name!r} is not declared by model {spec.model_hash}; "
            f"declared: {sorted(spec.priors)}"
        )
    payload = spec.to_dict()
    payload["priors"][name] = {
        "family": str(family),
        "parameters": {str(k): float(v) for k, v in parameters.items()},
    }
    return ModelSpec.from_dict(payload)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_model_spec_evaluation(
    root: str | Path,
    posterior,
    selection,
    campaign: ProductionCampaignConfig,
    spec: ModelSpec,
    *,
    dataset_identity: str,
    fidelity: Fidelity = Fidelity.F3_EVIDENCE,
    graph: ModelGraph | None = None,
    spec_path: str | Path | None = None,
    derived_from: ModelSpec | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Evaluate ``spec`` exactly as the frozen campaign evaluates a graph model.

    ``dataset_identity`` must be the identity the campaign files its runs
    under (the dataset-manifest hash for the production catalog). ``graph``
    (the frozen graph) and ``derived_from`` (the spec this one was derived
    from) only annotate the provenance. ``dry_run`` validates the spec
    against the grammar and the frozen selection's population-proxy support
    and reports the hash, seed and run directory without sampling or writing.
    """
    from gwpop_search.inference.dynesty_backend import _code_identity
    from gwpop_search.inference.fidelity import LADDER_V2
    from gwpop_search.inference.model_spec import prior_specs_from_model_spec
    from gwpop_search.inference.synthetic import (
        require_population_proxy_coverage,
    )
    from gwpop_search.models import compile_model_spec

    fidelity = Fidelity(fidelity)
    if fidelity not in LADDER_V2:
        raise ValueError(f"{fidelity.value} is not a rung of fidelity ladder v2")
    DEFAULT_COMPONENT_REGISTRY.validate_model(spec)
    model_hash = spec.model_hash

    root = Path(root)
    artifact_root = Path(campaign.artifact_root)
    if artifact_root.is_absolute() and root.resolve() == artifact_root.resolve():
        raise ValueError(
            f"refusing to write into the campaign's production artifact root "
            f"{artifact_root}; pass a separate --root"
        )

    # The same checks the evaluator runs first, so a dry run refuses what the
    # evaluation would refuse.
    compile_model_spec(spec)
    priors = prior_specs_from_model_spec(spec)
    require_population_proxy_coverage(
        selection,
        priors=priors,
        context=f"model {model_hash}",
    )

    evaluator = DeterministicHBIEvaluator(
        posterior,
        selection,
        config=campaign.fidelity,
        dataset_identity=str(dataset_identity),
    )
    seed = evaluation_seed(campaign.seed_policy.root_seed, model_hash, fidelity)
    run_dir = model_spec_run_dir(root, model_hash, fidelity)
    provenance_path = model_spec_provenance_path(root, model_hash, fidelity)

    resolved_dynesty = None
    if fidelity is not Fidelity.F0_SANITY:
        resolved_dynesty = (
            campaign.fidelity.evidence_config(fidelity)
            .dynesty_config_for(len(priors))
            .to_dict()
        )

    spec_file = None
    if spec_path is not None:
        spec_path = Path(spec_path)
        spec_file = {
            "path": str(spec_path.resolve()),
            "sha256": _sha256_bytes(spec_path.read_bytes()),
        }

    in_graph = graph is not None and model_hash in graph.by_hash
    payload: dict[str, object] = {
        "format_version": MODEL_SPEC_EVALUATION_FORMAT,
        "state_database_written": False,
        "dry_run": bool(dry_run),
        "model_hash": model_hash,
        "canonical_spec_sha256": _sha256_bytes(
            spec.canonical_json().encode("utf-8")
        ),
        "spec_file": spec_file,
        "spec": spec.to_dict(),
        "n_parameters": len(priors),
        "fidelity": fidelity.value,
        "seed": int(seed),
        "seed_derivation": (
            "evaluation_seed(campaign.seed_policy.root_seed, model_hash, fidelity)"
        ),
        "campaign_root_seed": int(campaign.seed_policy.root_seed),
        "campaign_id": campaign.campaign_id,
        "campaign_hash": campaign.campaign_hash,
        "dataset_identity": str(dataset_identity),
        "fidelity_config_sha256": evaluator.fidelity_config_sha256,
        "evidence_config": (
            None
            if fidelity is Fidelity.F0_SANITY
            else campaign.fidelity.evidence_config(fidelity).to_dict()
        ),
        "resolved_dynesty_config": resolved_dynesty,
        "criteria": (
            None
            if fidelity is Fidelity.F0_SANITY
            else campaign.fidelity.criteria(fidelity).to_dict()
        ),
        "graph": (
            None
            if graph is None
            else {
                "root_hash": graph.root_hash,
                "model_in_graph": bool(in_graph),
                "depth": (
                    int(graph.depths[model_hash]) if in_graph else None
                ),
            }
        ),
        "derived_from": (
            None if derived_from is None else model_spec_diff(derived_from, spec)
        ),
        "code": _code_identity(),
        "run_dir": str(run_dir),
        "evaluation": str(run_dir / "evaluation.json"),
        "provenance": str(provenance_path),
    }
    if dry_run:
        return payload

    run_dir.mkdir(parents=True, exist_ok=True)
    record = evaluator.evaluate(
        spec,
        fidelity,
        seed=seed,
        run_dir=run_dir,
    )
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    diagnostics = evaluation.get("diagnostics", {})
    evidence = diagnostics.get("evidence") or {}
    payload.update(
        {
            "diagnostics_pass": bool(record.diagnostics_pass),
            "screen_value": record.screen_value,
            "log_evidence_mean": evidence.get("log_evidence_mean"),
            "log_evidence_conservative_error": evidence.get("conservative_error"),
            "failure": diagnostics.get("failure"),
            "failed_gates": [
                item.get("name")
                for item in diagnostics.get("checks", [])
                if item.get("stage", "gate") == "gate" and not item.get("passed")
            ],
            "compute_cost_hours": record.compute_cost,
        }
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = provenance_path.with_name(provenance_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    tmp.replace(provenance_path)
    return payload
