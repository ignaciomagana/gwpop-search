"""Multi-seed structured-injection campaigns for HSGP scout validation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

from gwpop_search.data import PosteriorCatalog, SelectionCatalog
from gwpop_search.grammar import ModelSpec, baseline_model_spec
from gwpop_search.inference.numpyro import _code_identity
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS

from .config import ScoutCampaignConfig
from .inference import run_conditional_hsgp_scout
from .synthetic import (
    StructuredScoutInjection,
    generate_structured_scout_dataset,
)


def structured_scout_seed(
    root_seed: int,
    run_index: int,
    label: str,
) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{int(run_index)}:{label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def build_structured_scout_campaign_plan(
    *,
    n_runs: int,
    root_seed: int,
    injection: StructuredScoutInjection,
    survey_config: SyntheticSurveyConfig,
    scout_config: ScoutCampaignConfig,
    base_spec: ModelSpec,
    base_hyperparameters: Mapping[str, float],
) -> dict[str, object]:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    runs = []
    for index in range(int(n_runs)):
        runs.append(
            {
                "run_index": index,
                "data_seed": structured_scout_seed(
                    root_seed,
                    index,
                    "data",
                ),
                "sampler_seed": structured_scout_seed(
                    root_seed,
                    index,
                    "sampler",
                ),
            }
        )
    return {
        "format_version": "gwpop-search-structured-scout-campaign-1.0",
        "code": _code_identity(),
        "n_runs": int(n_runs),
        "root_seed": int(root_seed),
        "injection": asdict(injection),
        "survey_config": asdict(survey_config),
        "scout_config": scout_config.to_dict(),
        "base_model_hash": base_spec.model_hash,
        "base_model_spec": base_spec.to_dict(),
        "base_hyperparameters": {
            str(name): float(value)
            for name, value in sorted(base_hyperparameters.items())
        },
        "runs": runs,
    }


def _write_plan_once(path: Path, plan: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "existing structured scout campaign plan does not match request"
            )
    else:
        path.write_text(json.dumps(plan, sort_keys=True, indent=2))


def _load_or_generate_dataset(
    run_dir: Path,
    *,
    data_seed: int,
    injection: StructuredScoutInjection,
    survey_config: SyntheticSurveyConfig,
    base_hyperparameters: Mapping[str, float],
):
    pe_path = run_dir / "pe.h5"
    selection_path = run_dir / "selection.h5"
    truth_path = run_dir / "truth.json"

    if pe_path.exists() or selection_path.exists() or truth_path.exists():
        if not (pe_path.exists() and selection_path.exists() and truth_path.exists()):
            raise ValueError(
                f"incomplete structured dataset checkpoint in {run_dir}"
            )
        posterior = PosteriorCatalog.from_hdf5(pe_path)
        selection = SelectionCatalog.from_hdf5(selection_path)
        truth = json.loads(truth_path.read_text())
        if int(truth["seed"]) != int(data_seed):
            raise ValueError("structured dataset seed mismatch on resume")
        if truth["injection"] != asdict(injection):
            raise ValueError("structured dataset injection mismatch on resume")
        return posterior, selection, truth

    dataset = generate_structured_scout_dataset(
        seed=data_seed,
        injection=injection,
        survey_config=survey_config,
        hyperparameters=base_hyperparameters,
    )
    dataset.posterior.to_hdf5(pe_path)
    dataset.selection.to_hdf5(selection_path)
    truth = {
        "format_version": "gwpop-search-structured-scout-truth-1.0",
        "seed": int(data_seed),
        "injection": asdict(injection),
        "truth_hyperparameters": dict(dataset.truth_hyperparameters),
        "event_truths": {
            name: values.tolist()
            for name, values in dataset.event_truths.items()
        },
    }
    truth_path.write_text(json.dumps(truth, sort_keys=True, indent=2))
    return dataset.posterior, dataset.selection, truth


def assess_structured_scout_campaign(root: str | Path) -> dict[str, object]:
    root = Path(root)
    plan = json.loads((root / "campaign_plan.json").read_text())
    expected = str(plan["injection"]["mutation_id"])

    rows = []
    for run in plan["runs"]:
        index = int(run["run_index"])
        path = root / f"run_{index:03d}" / "scout" / "scout_summary.json"
        if not path.exists():
            rows.append(
                {
                    "run_index": index,
                    "complete": False,
                    "numerical_pass": False,
                    "proposal_mutation_ids": [],
                    "expected_proposed": False,
                }
            )
            continue
        summary = json.loads(path.read_text())
        proposals = [
            str(item["mutation_id"])
            for item in summary["validated_proposals"]
        ]
        expected_proposed = (
            False
            if expected == "null"
            else expected in proposals
        )
        rows.append(
            {
                "run_index": index,
                "complete": True,
                "numerical_pass": bool(summary["numerical"]["passed"]),
                "proposal_mutation_ids": proposals,
                "expected_proposed": bool(expected_proposed),
            }
        )

    complete = [row for row in rows if row["complete"]]
    numerical = [row for row in complete if row["numerical_pass"]]
    n_expected = sum(row["expected_proposed"] for row in numerical)
    n_any = sum(bool(row["proposal_mutation_ids"]) for row in numerical)
    n_off_target = sum(
        len(
            [
                mutation
                for mutation in row["proposal_mutation_ids"]
                if expected == "null" or mutation != expected
            ]
        )
        for row in numerical
    )

    result = {
        "format_version": "gwpop-search-structured-scout-assessment-1.0",
        "expected_mutation_id": expected,
        "n_runs": len(rows),
        "n_complete": len(complete),
        "n_numerical_pass": len(numerical),
        "n_expected_proposed": int(n_expected),
        "expected_proposal_fraction_among_numerical_pass": (
            None
            if not numerical or expected == "null"
            else float(n_expected / len(numerical))
        ),
        "n_runs_with_any_proposal": int(n_any),
        "any_proposal_fraction_among_numerical_pass": (
            None if not numerical else float(n_any / len(numerical))
        ),
        "n_off_target_proposals": int(n_off_target),
        "runs": rows,
        "interpretation": (
            "diagnostic_only; acceptance thresholds require a declared "
            "multi-seed validation design"
        ),
    }
    (root / "campaign_summary.json").write_text(
        json.dumps(result, sort_keys=True, indent=2)
    )
    return result


def run_structured_scout_campaign(
    root: str | Path,
    *,
    n_runs: int,
    root_seed: int,
    injection: StructuredScoutInjection,
    survey_config: SyntheticSurveyConfig | None = None,
    scout_config: ScoutCampaignConfig,
    base_spec: ModelSpec | None = None,
    base_hyperparameters: Mapping[str, float] | None = None,
) -> dict[str, object]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    survey_config = (
        SyntheticSurveyConfig()
        if survey_config is None
        else survey_config
    )
    base_spec = baseline_model_spec() if base_spec is None else base_spec
    base_hyperparameters = dict(
        DEFAULT_BASELINE_HYPERPARAMETERS
        if base_hyperparameters is None
        else base_hyperparameters
    )

    plan = build_structured_scout_campaign_plan(
        n_runs=n_runs,
        root_seed=root_seed,
        injection=injection,
        survey_config=survey_config,
        scout_config=scout_config,
        base_spec=base_spec,
        base_hyperparameters=base_hyperparameters,
    )
    _write_plan_once(root / "campaign_plan.json", plan)

    for run in plan["runs"]:
        index = int(run["run_index"])
        run_dir = root / f"run_{index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        posterior, selection, _ = _load_or_generate_dataset(
            run_dir,
            data_seed=int(run["data_seed"]),
            injection=injection,
            survey_config=survey_config,
            base_hyperparameters=base_hyperparameters,
        )
        run_conditional_hsgp_scout(
            run_dir / "scout",
            posterior,
            selection,
            base_spec=base_spec,
            base_hyperparameters=base_hyperparameters,
            hsgp_config=scout_config.hsgp,
            seed=int(run["sampler_seed"]),
            config=scout_config.run,
            dataset_identity=(
                f"structured:{injection.injection_label}:"
                f"{int(run['data_seed'])}"
            ),
        )

    return assess_structured_scout_campaign(root)
