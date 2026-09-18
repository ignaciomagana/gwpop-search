"""Independent full-HBI evidence comparison for reviewed scout descendants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gwpop_search.grammar import ModelSpec
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.production import ProductionCampaignConfig
from gwpop_search.search import Fidelity, evaluation_seed


def scout_comparison_seed_root(
    campaign_seed: int,
    parent_hash: str,
    child_hash: str,
) -> int:
    digest = hashlib.sha256(
        (
            f"{int(campaign_seed)}:{parent_hash}:{child_hash}:"
            "scout-descendant-comparison-v1"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _write_manifest_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                "existing scout descendant comparison does not match request"
            )
    else:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def compare_scout_descendant_evidence(
    root: str | Path,
    posterior,
    selection,
    campaign: ProductionCampaignConfig,
    *,
    dataset_identity: str,
    parent: ModelSpec,
    child: ModelSpec,
    proposal_id: str,
) -> dict[str, object]:
    """Refit parent and child independently at the frozen F3 evidence fidelity."""
    if parent.model_hash == child.model_hash:
        raise ValueError("parent and child model hashes must differ")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    seed_root = scout_comparison_seed_root(
        campaign.seed_policy.root_seed,
        parent.model_hash,
        child.model_hash,
    )
    manifest = {
        "format_version": "gwpop-search-scout-descendant-comparison-1.0",
        "campaign_hash": campaign.campaign_hash,
        "dataset_identity": str(dataset_identity),
        "proposal_id": str(proposal_id),
        "parent_model_hash": parent.model_hash,
        "child_model_hash": child.model_hash,
        "seed_root": int(seed_root),
        "fidelity": Fidelity.F3_EVIDENCE.value,
    }
    _write_manifest_once(root / "manifest.json", manifest)

    evaluator = DeterministicHBIEvaluator(
        posterior,
        selection,
        config=campaign.fidelity,
        dataset_identity=dataset_identity,
    )
    records = {}
    payloads = {}
    for label, model in (("parent", parent), ("child", child)):
        run_dir = root / label
        seed = evaluation_seed(
            seed_root,
            model.model_hash,
            Fidelity.F3_EVIDENCE,
        )
        record = evaluator.evaluate(
            model,
            Fidelity.F3_EVIDENCE,
            seed=seed,
            run_dir=run_dir,
        )
        records[label] = record
        payloads[label] = json.loads(
            (run_dir / "evaluation.json").read_text()
        )

    both_valid = bool(
        records["parent"].diagnostics_pass
        and records["child"].diagnostics_pass
    )
    log_bf = (
        None
        if not both_valid
        else float(
            records["child"].screen_value
            - records["parent"].screen_value
        )
    )
    summary = {
        "format_version": "gwpop-search-scout-descendant-comparison-summary-1.0",
        "proposal_id": str(proposal_id),
        "parent_model_hash": parent.model_hash,
        "child_model_hash": child.model_hash,
        "both_numerically_valid": both_valid,
        "log_bayes_factor_child_over_parent": log_bf,
        "parent": payloads["parent"],
        "child": payloads["child"],
        "interpretation": (
            "independent_full_hbi_refit"
            if both_valid
            else "comparison_blocked_by_numerical_failure"
        ),
    }
    (root / "comparison_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
