"""Complete valid F3 evidence over every node in a frozen model graph.

Completion runs the identical F3 rung (seed ``evaluation_seed(root, model,
"F3")``, dynesty repeats of the frozen fidelity configuration) for every
graph node that lacks valid evidence at the *required* fidelity, so the
complete evidence set does not depend on which models the search beam
promoted. The requirement is ``("F3",)`` by default because the calibrated
null statistic (D4, ``f3_completion``) reads F3 evidence only: an F4 row must
not stand in for a missing or failed F3, or the statistic would have no
defined value for that model and completion could never repair it. A frozen
F3 that failed its gates (including typed numerical failures such as no
finite prior support) is never re-run; it blocks the normalized graph
posterior. Rows are recorded with executor ``evidence-completion-v2``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from gwpop_search.grammar import ModelGraph
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.search import (
    COMPLETION_EXECUTOR_VERSION,
    Fidelity,
    SearchBudgetExceeded,
    evaluation_run_config,
    evaluation_run_id,
    evaluation_seed,
    require_fidelity_config_identity,
    require_v2_state,
)
from gwpop_search.store import ResultStore

EVIDENCE_COMPLETION_FORMAT_VERSION = "gwpop-search-evidence-completion-1.2"
# The evidence the completed set must contain (D4: the null statistic is F3).
REQUIRED_EVIDENCE_FIDELITIES = ("F3",)

from .config import ProductionCampaignConfig
from .runner import (
    collect_best_available_evidence,
    model_prior_from_config,
    write_scientific_scoring,
)


def _evaluation_failure(artifact_path) -> dict[str, object] | None:
    """The typed failure block of an evaluation artifact, if it records one."""
    if not artifact_path:
        return None
    path = Path(str(artifact_path)) / "evaluation.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return None
    failure = (payload.get("diagnostics") or {}).get("failure")
    return failure if isinstance(failure, dict) else None


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
    root_seed: int | None = None,
    required_fidelities: tuple[str, ...] = REQUIRED_EVIDENCE_FIDELITIES,
) -> dict[str, object]:
    """Run/resume F3 for all graph nodes lacking valid evidence.

    Screening can prioritize which nodes reach evidence first, but it cannot
    define a normalized posterior over the declared model space. This function
    is the explicit completion stage required before full model probabilities.

    ``required_fidelities`` is the evidence a node must already have to count
    as complete; it defaults to ``("F3",)`` so that a passed F4 never masks a
    missing or failed F3 (see the module docstring).
    """
    required_fidelities = tuple(str(item) for item in required_fidelities)
    if len(graph.nodes) > campaign.budget.max_f3_models:
        raise SearchBudgetExceeded(
            f"declared graph has {len(graph.nodes)} models but the frozen "
            f"F3 budget allows only {campaign.budget.max_f3_models}; "
            "full model-posterior evidence completion is not permitted by "
            "this campaign"
        )

    seed_root = (
        campaign.seed_policy.root_seed
        if root_seed is None
        else int(root_seed)
    )
    store = ResultStore(state_database)
    require_v2_state(store)
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

    config_sha256 = getattr(evaluator, "fidelity_config_sha256", None)
    initially_valid = collect_best_available_evidence(
        state_database,
        fidelities=required_fidelities,
        fidelity_config_sha256=config_sha256,
    )
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
            seed_root,
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
            require_fidelity_config_identity(row, config_sha256)
            blocked.append(
                {
                    "model_hash": model_hash,
                    "reason": (
                        "existing_frozen_f3_is_not_valid_scientific_evidence"
                    ),
                    "status": row["status"],
                    "diagnostics_pass": bool(row["diagnostics_pass"]),
                    "artifact_path": row.get("artifact_path"),
                    "failure": _evaluation_failure(row.get("artifact_path")),
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
            run_config=evaluation_run_config(
                Fidelity.F3_EVIDENCE,
                executor=COMPLETION_EXECUTOR_VERSION,
                fidelity_config_sha256=config_sha256,
            ),
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
                    "failure": _evaluation_failure(str(run_dir)),
                }
            )

    scoring = write_scientific_scoring(
        graph,
        state_database=state_database,
        artifact_root=artifact_root,
        model_prior=model_prior_from_config(campaign.model_prior),
        fidelities=required_fidelities,
        fidelity_config_sha256=config_sha256,
    )
    valid = collect_best_available_evidence(
        state_database,
        fidelities=required_fidelities,
        fidelity_config_sha256=config_sha256,
    )
    summary = {
        "format_version": EVIDENCE_COMPLETION_FORMAT_VERSION,
        "executor": COMPLETION_EXECUTOR_VERSION,
        "sampler_backend": dict(campaign.sampler_backend),
        "required_evidence_fidelities": list(required_fidelities),
        "fidelity_config_sha256": config_sha256,
        "campaign_id": campaign.campaign_id,
        "dataset_identity": str(dataset_identity),
        "evidence_seed_root": int(seed_root),
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
