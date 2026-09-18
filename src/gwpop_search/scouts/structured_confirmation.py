"""Confirm an injected scout descendant with independent F3 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gwpop_search.data import PosteriorCatalog, SelectionCatalog
from gwpop_search.grammar import ModelSpec, save_model_spec
from gwpop_search.inference.fidelity import FidelityRunConfig

from .campaign import assess_structured_scout_campaign, structured_scout_seed
from .comparison import compare_scout_descendant_evidence_config
from .review import review_scout_proposal, write_scout_review


def _plan_identity(plan: dict[str, object]) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_confirmation_run(
    campaign_summary: dict[str, object],
    *,
    run_index: int | None,
) -> int:
    expected = str(campaign_summary["expected_mutation_id"])
    if expected == "null":
        raise ValueError("null scout campaigns have no injected descendant to confirm")

    rows = list(campaign_summary["runs"])
    eligible = [
        row
        for row in rows
        if bool(row["complete"])
        and bool(row["numerical_pass"])
        and bool(row["expected_proposed"])
    ]
    if run_index is not None:
        matches = [
            row
            for row in eligible
            if int(row["run_index"]) == int(run_index)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"run_index={run_index} is not an eligible injected-proposal run"
            )
        return int(matches[0]["run_index"])
    if not eligible:
        raise ValueError(
            "structured scout campaign has no numerically valid run that "
            "proposed the injected mutation"
        )
    return min(int(row["run_index"]) for row in eligible)


def confirm_structured_scout_descendant(
    campaign_root: str | Path,
    output_root: str | Path,
    fidelity_config: FidelityRunConfig,
    *,
    run_index: int | None = None,
) -> dict[str, object]:
    """Materialize and independently F3-confirm one injected scout proposal."""
    campaign_root = Path(campaign_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    plan = json.loads((campaign_root / "campaign_plan.json").read_text())
    summary_path = campaign_root / "campaign_summary.json"
    campaign_summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else assess_structured_scout_campaign(campaign_root)
    )
    selected_index = _select_confirmation_run(
        campaign_summary,
        run_index=run_index,
    )
    expected_mutation = str(campaign_summary["expected_mutation_id"])
    run_plan = [
        row
        for row in plan["runs"]
        if int(row["run_index"]) == selected_index
    ]
    if len(run_plan) != 1:
        raise ValueError("structured campaign plan has inconsistent run index")
    run_plan = run_plan[0]

    run_dir = campaign_root / f"run_{selected_index:03d}"
    scout_summary = json.loads(
        (run_dir / "scout" / "scout_summary.json").read_text()
    )
    proposals = [
        item
        for item in scout_summary["validated_proposals"]
        if str(item["mutation_id"]) == expected_mutation
    ]
    if len(proposals) != 1:
        raise ValueError(
            "expected exactly one validated proposal for injected mutation "
            f"{expected_mutation!r}; found {len(proposals)}"
        )
    proposal_id = str(proposals[0]["proposal_id"])

    parent = ModelSpec.from_dict(dict(plan["base_model_spec"]))
    review, child = review_scout_proposal(
        scout_summary,
        parent,
        proposal_id=proposal_id,
        decision="accepted",
        note=(
            "accepted only for structured-injection engineering confirmation; "
            "not a production scientific review"
        ),
    )
    if child is None or review.mutation_id != expected_mutation:
        raise RuntimeError("structured confirmation materialized wrong descendant")

    write_scout_review(output_root / "review.json", review)
    save_model_spec(output_root / "parent.json", parent)
    save_model_spec(output_root / "child.json", child)

    posterior = PosteriorCatalog.from_hdf5(run_dir / "pe.h5")
    selection = SelectionCatalog.from_hdf5(run_dir / "selection.h5")
    data_seed = int(run_plan["data_seed"])
    root_seed = structured_scout_seed(
        int(plan["root_seed"]),
        selected_index,
        "descendant-comparison",
    )
    dataset_identity = (
        f"structured-confirmation:{expected_mutation}:{data_seed}"
    )
    comparison = compare_scout_descendant_evidence_config(
        output_root / "comparison",
        posterior,
        selection,
        fidelity_config,
        root_seed=root_seed,
        comparison_identity=_plan_identity(plan),
        dataset_identity=dataset_identity,
        parent=parent,
        child=child,
        proposal_id=proposal_id,
    )

    result = {
        "format_version": "gwpop-search-structured-descendant-confirmation-1.0",
        "campaign_plan_identity": _plan_identity(plan),
        "campaign_root": str(campaign_root.resolve()),
        "run_index": int(selected_index),
        "data_seed": data_seed,
        "expected_mutation_id": expected_mutation,
        "proposal_id": proposal_id,
        "parent_model_hash": parent.model_hash,
        "child_model_hash": child.model_hash,
        "review_path": str((output_root / "review.json").resolve()),
        "comparison_root": str((output_root / "comparison").resolve()),
        "comparison": comparison,
        "engineering_confirmation_passed": bool(
            comparison["both_numerically_valid"]
            and comparison["log_bayes_factor_child_over_parent"] is not None
        ),
        "interpretation": (
            "injected_descendant_independent_f3_confirmation_only; "
            "not a GWTC-5 discovery claim"
        ),
    }
    (output_root / "confirmation_summary.json").write_text(
        json.dumps(result, sort_keys=True, indent=2)
    )
    return result
