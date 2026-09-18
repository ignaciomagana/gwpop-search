"""Complete valid F3 evidence over every node in a frozen model graph."""

from __future__ import annotations

import json
import math
from pathlib import Path

from gwpop_search.grammar import ModelGraph
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.search import (
    Fidelity,
    SearchBudgetExceeded,
    evaluation_run_id,
    evaluation_seed,
)
from gwpop_search.store import ResultStore

from .config import ProductionCampaignConfig
from .runner import (
    collect_best_available_evidence,
    model_prior_from_config,
    write_scientific_scoring,
)


def _total_compute_cost(store: ResultStore) -> float:
    return float(
        math.fsum(
            float(row["compute_cost"])
            for row in store.evaluations()
        )
    )


def complete_graph_evidence(
    graph: ModelGraph,
    posterior,
    selection,
    campaign: ProductionCampaignConfig,
    *,
    dataset_identity: str,
    state_database: str | Path,
    artifact_root: str | Path,
) -> dict[str, object]:
    """Run/resume F3 for all graph nodes lacking valid F3/F4 evidence.

    Screening can prioritize which nodes reach evidence first, but it cannot
    define a normalized posterior over the declared model space. This function
    is the explicit completion stage required before full model probabilities.
    """
    store = ResultStore(state_database)
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    evaluator = DeterministicHBIEvaluator(
        posterior,
        selection,
        config=campaign.fidelity,
        dataset_identity=dataset_identity,
    )

    for model in graph.nodes:
        store.register_model(model)

    initially_valid = collect_best_available_evidence(state_database)
    evaluated = []
    reused = []
    blocked = []
    total_cost = _total_compute_cost(store)

    for model in sorted(graph.nodes, key=lambda item: item.model_hash):
        model_hash = model.model_hash
        if model_hash in initially_valid:
            reused.append(model_hash)
            continue

        seed = evaluation_seed(
            campaign.seed_policy.root_seed,
            model_hash,
            Fidelity.F3_EVIDENCE,
        )
        rows = [
            row
            for row in store.evaluations(
                model_hash=model_hash,
                fidelity=Fidelity.F3_EVIDENCE.value,
            )
            if int(row["seed"]) == int(seed)
        ]
        if rows:
            if len(rows) > 1:
                raise RuntimeError(
                    f"multiple F3 evaluations for {model_hash} seed={seed}"
                )
            row = rows[0]
            blocked.append(
                {
                    "model_hash": model_hash,
                    "reason": (
                        "existing_frozen_f3_is_not_valid_scientific_evidence"
                    ),
                    "status": row["status"],
                    "diagnostics_pass": bool(row["diagnostics_pass"]),
                    "artifact_path": row.get("artifact_path"),
                }
            )
            continue

        if total_cost >= campaign.budget.max_gpu_hours:
            raise SearchBudgetExceeded(
                "frozen production GPU-hour budget exhausted before "
                "evidence completion"
            )

        run_dir = artifact_root / Fidelity.F3_EVIDENCE.value / model_hash
        run_dir.mkdir(parents=True, exist_ok=True)
        record = evaluator.evaluate(
            model,
            Fidelity.F3_EVIDENCE,
            seed=seed,
            run_dir=run_dir,
        )
        run_id = evaluation_run_id(
            model_hash,
            Fidelity.F3_EVIDENCE,
            seed,
        )
        store.record_evaluation(
            run_id,
            record,
            seed=seed,
            run_config={
                "executor": "evidence-completion-v1",
                "fidelity": Fidelity.F3_EVIDENCE.value,
            },
            artifact_path=str(run_dir),
        )
        total_cost = _total_compute_cost(store)
        evaluated.append(model_hash)

        if total_cost > campaign.budget.max_gpu_hours:
            raise SearchBudgetExceeded(
                "frozen production GPU-hour budget was exceeded by the "
                f"completed evidence evaluation: {total_cost:.6g} > "
                f"{campaign.budget.max_gpu_hours:.6g}"
            )
        if not record.diagnostics_pass:
            blocked.append(
                {
                    "model_hash": model_hash,
                    "reason": "new_f3_failed_numerical_evidence_gate",
                    "status": record.status,
                    "diagnostics_pass": False,
                    "artifact_path": str(run_dir),
                }
            )

    scoring = write_scientific_scoring(
        graph,
        state_database=state_database,
        artifact_root=artifact_root,
        model_prior=model_prior_from_config(campaign.model_prior),
    )
    valid = collect_best_available_evidence(state_database)
    summary = {
        "format_version": "gwpop-search-evidence-completion-1.0",
        "campaign_id": campaign.campaign_id,
        "dataset_identity": str(dataset_identity),
        "n_graph_models": len(graph.nodes),
        "n_valid_evidence_initial": len(initially_valid),
        "n_reused": len(reused),
        "n_evaluated": len(evaluated),
        "n_blocked_invalid": len(blocked),
        "blocked_invalid": blocked,
        "n_valid_evidence_final": len(valid),
        "total_compute_cost": total_cost,
        "full_model_posterior_available": bool(
            scoring["evidence_coverage"]["complete"]
        ),
        "scientific_scoring": scoring,
    }
    (artifact_root / "evidence_completion_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
