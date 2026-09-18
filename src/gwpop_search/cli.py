"""Command-line entry point for reproducible gwpop-search campaigns."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import __version__


def _survey_config(args: argparse.Namespace):
    from .inference.synthetic import SyntheticSurveyConfig

    return SyntheticSurveyConfig(
        n_events=args.n_events,
        posterior_samples_per_event=args.pe_samples,
        n_injections=args.n_injections,
    )


def _nuts_config(args: argparse.Namespace):
    from .inference.numpyro import NUTSConfig

    return NUTSConfig(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        progress_bar=not args.no_progress,
    )


def _run_synthetic_recovery(args: argparse.Namespace) -> None:
    from .inference.recovery import run_synthetic_baseline_recovery

    _, summary = run_synthetic_baseline_recovery(
        Path(args.run_dir),
        data_seed=args.data_seed,
        sampler_seed=args.sampler_seed,
        survey_config=_survey_config(args),
        nuts_config=_nuts_config(args),
        selection_chunk_size=args.selection_chunk_size,
    )
    diagnostic = summary["diagnostics"]
    print(
        "synthetic recovery complete: "
        f"events={diagnostic['n_events']} "
        f"selected_injections={diagnostic['n_selected_injections']} "
        f"divergences={diagnostic['n_divergent']} "
        f"max_r_hat={diagnostic['max_r_hat']}"
    )


def _run_synthetic_campaign(args: argparse.Namespace) -> None:
    from .inference.campaign import (
        RecoveryAcceptanceCriteria,
        run_recovery_campaign,
    )

    criteria = RecoveryAcceptanceCriteria(min_runs=args.n_runs)
    summary = run_recovery_campaign(
        Path(args.root),
        n_runs=args.n_runs,
        root_seed=args.root_seed,
        survey_config=_survey_config(args),
        nuts_config=_nuts_config(args),
        selection_chunk_size=args.selection_chunk_size,
        criteria=criteria,
    )
    print(
        "synthetic campaign complete: "
        f"runs={summary['n_runs']} "
        f"numerical_pass={summary['n_numerical_pass']} "
        f"gate={summary['phase3_numerical_gate_passed']}"
    )


def _assess_synthetic_campaign(args: argparse.Namespace) -> None:
    from .inference.campaign import (
        RecoveryAcceptanceCriteria,
        assess_recovery_campaign,
    )

    summary = assess_recovery_campaign(
        Path(args.root),
        criteria=RecoveryAcceptanceCriteria(min_runs=args.min_runs),
    )
    print(
        "synthetic campaign assessment: "
        f"runs={summary['n_runs']} "
        f"numerical_pass={summary['n_numerical_pass']} "
        f"gate={summary['phase3_numerical_gate_passed']}"
    )


def _add_stress_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stop-fidelity",
        choices=("F2", "F3", "F4"),
        default="F3",
    )
    parser.add_argument(
        "--max-gpu-hours-per-scenario",
        type=float,
        default=250.0,
    )
    parser.add_argument("--max-f3-models", type=int, default=20)
    parser.add_argument("--max-f4-models", type=int, default=8)


def _add_common_recovery_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-events", type=int, default=48)
    parser.add_argument("--pe-samples", type=int, default=256)
    parser.add_argument("--n-injections", type=int, default=20_000)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--selection-chunk-size", type=int, default=4096)
    parser.add_argument("--no-progress", action="store_true")


def _inspect_model_graph(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph

    graph = load_model_graph(Path(args.graph))
    incoming: dict[str, list[str]] = {model.model_hash: [] for model in graph.nodes}
    for edge in graph.edges:
        incoming[edge.child_hash].append(edge.mutation_id)
    payload = {
        "root_hash": graph.root_hash,
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "nodes": [
            {
                "model_hash": model.model_hash,
                "depth": int(graph.depths[model.model_hash]),
                "incoming_mutations": sorted(incoming[model.model_hash]),
                "mass": model.mass.to_dict(),
                "pairing": model.pairing.to_dict(),
                "chieff": model.chieff.to_dict(),
                "redshift": model.redshift.to_dict(),
                "prior_names": sorted(model.priors),
            }
            for model in sorted(graph.nodes, key=lambda item: item.model_hash)
        ],
    }
    print(json.dumps(payload, sort_keys=True, indent=2))


def _extract_model(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph, save_model_spec

    graph = load_model_graph(Path(args.graph))
    if args.model_hash not in graph.by_hash:
        raise ValueError(f"unknown graph model hash {args.model_hash!r}")
    model = graph.by_hash[args.model_hash]
    save_model_spec(Path(args.output), model)
    print(
        "model extracted: "
        f"hash={model.model_hash} depth={graph.depths[model.model_hash]} "
        f"output={args.output}"
    )


def _enumerate_models(args: argparse.Namespace) -> None:
    from .grammar import (
        baseline_model_spec,
        enumerate_model_graph,
        save_model_graph,
    )

    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=args.max_depth,
        max_models=args.max_models,
    )
    save_model_graph(Path(args.output), graph)
    print(
        "model graph written: "
        f"nodes={len(graph.nodes)} edges={len(graph.edges)} "
        f"root={graph.root_hash[:16]}"
    )


def _validate_model_spec(args: argparse.Namespace) -> None:
    from .grammar import DEFAULT_COMPONENT_REGISTRY, load_model_spec
    from .models import DeclarativeGwcatChiEffModel

    spec = load_model_spec(Path(args.spec))
    DEFAULT_COMPONENT_REGISTRY.validate_model(spec)
    DeclarativeGwcatChiEffModel(spec)
    print(f"valid model spec: {spec.model_hash}")


def _read_json_mapping(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _run_holdout_validation(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )
    from .validation import HoldoutCampaignConfig, run_holdout_campaign

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    freeze = validate_production_freeze(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    if not freeze["valid"]:
        raise ValueError("production freeze validation failed")

    graph = load_model_graph(Path(args.graph))
    missing = [
        model_hash
        for model_hash in args.model_hash
        if model_hash not in graph.by_hash
    ]
    if missing:
        raise ValueError(
            f"holdout validation references unknown model hash(es) {missing}"
        )

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    summary = run_holdout_campaign(
        Path(args.root),
        posterior,
        selection,
        campaign,
        dataset_identity=manifest.manifest_hash,
        models=tuple(graph.by_hash[item] for item in args.model_hash),
        config=HoldoutCampaignConfig(
            n_folds=args.n_folds,
            seed=args.fold_seed,
        ),
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


def _diagnose_frozen_selection_null(args: argparse.Namespace) -> None:
    from .grammar import load_model_spec
    from .nulls import frozen_selection_resampling_probabilities
    from .production import load_dataset_manifest, load_frozen_dataset

    manifest = load_dataset_manifest(Path(args.manifest))
    _, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    model = load_model_spec(Path(args.model))
    hyperparameters = {
        str(name): float(value)
        for name, value in _read_json_mapping(
            args.hyperparameters_json
        ).items()
    }
    _, diagnostics = frozen_selection_resampling_probabilities(
        selection,
        model,
        hyperparameters,
    )
    diagnostics = {
        "format_version": "gwpop-search-frozen-selection-null-preflight-1.0",
        "dataset_manifest_hash": manifest.manifest_hash,
        "model_hash": model.model_hash,
        **diagnostics,
    }
    print(json.dumps(diagnostics, sort_keys=True, indent=2))


def _write_exact_null_config(args: argparse.Namespace) -> None:
    from .inference.synthetic import SyntheticSurveyConfig
    from .models import DEFAULT_BASELINE_HYPERPARAMETERS
    from .nulls import (
        ExactNullCampaignConfig,
        save_exact_null_campaign_config,
    )
    from .production import load_dataset_manifest

    truth = (
        dict(DEFAULT_BASELINE_HYPERPARAMETERS)
        if args.truth_hyperparameters_json is None
        else {
            str(name): float(value)
            for name, value in _read_json_mapping(
                args.truth_hyperparameters_json
            ).items()
        }
    )
    if args.data_mode == "frozen_selection_resample":
        if args.manifest is None:
            raise ValueError(
                "--manifest is required for frozen_selection_resample"
            )
        manifest = load_dataset_manifest(Path(args.manifest))
        n_events = len(manifest.event_names)
        if args.n_events is not None and args.n_events != n_events:
            raise ValueError(
                "--n-events disagrees with the frozen dataset manifest"
            )
    else:
        n_events = 64 if args.n_events is None else args.n_events

    config = ExactNullCampaignConfig(
        n_nulls=args.n_nulls,
        root_seed=args.root_seed,
        survey=SyntheticSurveyConfig(
            n_events=n_events,
            posterior_samples_per_event=args.pe_samples,
            n_injections=args.n_injections,
        ),
        truth_hyperparameters=truth,
        data_mode=args.data_mode,
        min_resampling_ess=args.min_resampling_ess,
        max_gpu_hours_per_null=args.max_gpu_hours_per_null,
    )
    save_exact_null_campaign_config(Path(args.output), config)
    print(
        "exact null calibration config written: "
        f"{args.output} n_nulls={config.n_nulls}"
    )


def _run_exact_null_calibration(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph
    from .nulls import (
        load_exact_null_campaign_config,
        run_exact_null_campaign,
    )
    from .production import (
        load_dataset_manifest,
        load_production_campaign,
        validate_production_freeze,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    freeze = validate_production_freeze(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    if not freeze["valid"]:
        raise ValueError("production freeze validation failed")

    observed = args.observed_state_database
    if observed is None:
        candidate = Path(args.work_dir) / campaign.state_database
        observed = str(candidate) if candidate.is_file() else None

    null_config = load_exact_null_campaign_config(
        Path(args.null_config)
    )
    production_posterior = None
    production_selection = None
    production_dataset_identity = None
    if null_config.data_mode == "frozen_selection_resample":
        from .production import load_frozen_dataset

        production_posterior, production_selection = load_frozen_dataset(
            manifest,
            data_base_dir=Path(args.base_dir),
        )
        production_dataset_identity = manifest.manifest_hash

    summary = run_exact_null_campaign(
        Path(args.root),
        load_model_graph(Path(args.graph)),
        campaign,
        null_config,
        observed_state_database=observed,
        production_posterior=production_posterior,
        production_selection=production_selection,
        production_dataset_identity=production_dataset_identity,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


def _nearby_baseline_config_from_args(args: argparse.Namespace):
    from .search import Fidelity
    from .validation import NearbyBaselineConfig

    return NearbyBaselineConfig(
        stop_fidelity=Fidelity(args.stop_fidelity),
        max_gpu_hours_per_scenario=args.max_gpu_hours_per_scenario,
        max_f3_models=args.max_f3_models,
        max_f4_models=args.max_f4_models,
    )


def _write_nearby_baseline_config(args: argparse.Namespace) -> None:
    from .grammar import load_model_spec
    from .validation import (
        NearbyBaselineScenario,
        NearbyBaselineSuiteSpec,
        save_nearby_baseline_suite_spec,
    )

    spec = NearbyBaselineSuiteSpec(
        scenarios=(
            NearbyBaselineScenario(
                scenario_id=args.scenario_id,
                root_spec=load_model_spec(Path(args.root_model)),
                max_depth=args.max_depth,
                max_models=args.max_models,
                note=args.note or "",
            ),
        ),
        config=_nearby_baseline_config_from_args(args),
    )
    save_nearby_baseline_suite_spec(Path(args.output), spec)
    print(
        "nearby-baseline config written: "
        f"{args.output} scenario={args.scenario_id}"
    )


def _run_nearby_baseline_suite(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )
    from .validation import (
        load_nearby_baseline_suite_spec,
        run_nearby_baseline_suite,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    freeze = validate_production_freeze(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    if not freeze["valid"]:
        raise ValueError("production freeze validation failed")

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    reference_graph = load_model_graph(Path(args.graph))
    suite = load_nearby_baseline_suite_spec(Path(args.nearby_config))

    reference = args.reference_state_database
    if reference is None:
        candidate = Path(args.work_dir) / campaign.state_database
        reference = str(candidate) if candidate.is_file() else None

    summary = run_nearby_baseline_suite(
        Path(args.root),
        posterior,
        selection,
        reference_graph,
        campaign,
        base_dataset_identity=manifest.manifest_hash,
        suite=suite,
        reference_state_database=reference,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


def _stress_config_from_args(args: argparse.Namespace):
    from .search import Fidelity
    from .validation import EventStressConfig

    return EventStressConfig(
        stop_fidelity=Fidelity(args.stop_fidelity),
        max_gpu_hours_per_scenario=args.max_gpu_hours_per_scenario,
        max_f3_models=args.max_f3_models,
        max_f4_models=args.max_f4_models,
    )


def _write_loo_stress_config(args: argparse.Namespace) -> None:
    from .production import load_dataset_manifest
    from .validation import (
        EventStressSuiteSpec,
        leave_one_out_scenarios,
        save_event_stress_suite_spec,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    spec = EventStressSuiteSpec(
        scenarios=leave_one_out_scenarios(manifest.event_names),
        config=_stress_config_from_args(args),
    )
    save_event_stress_suite_spec(Path(args.output), spec)
    print(
        "leave-one-out stress config written: "
        f"{args.output} scenarios={len(spec.scenarios)}"
    )


def _write_event_drop_stress_config(args: argparse.Namespace) -> None:
    from .validation import (
        EventDropScenario,
        EventStressSuiteSpec,
        save_event_stress_suite_spec,
    )

    spec = EventStressSuiteSpec(
        scenarios=(
            EventDropScenario(
                scenario_id=args.scenario_id,
                drop_events=tuple(args.drop_event),
                category=args.category,
                note=args.note or "",
            ),
        ),
        config=_stress_config_from_args(args),
    )
    save_event_stress_suite_spec(Path(args.output), spec)
    print(
        "event-drop stress config written: "
        f"{args.output} scenario={args.scenario_id}"
    )


def _run_event_stress_suite(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )
    from .validation import (
        load_event_stress_suite_spec,
        run_event_drop_stress_suite,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    freeze = validate_production_freeze(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    if not freeze["valid"]:
        raise ValueError("production freeze validation failed")

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    graph = load_model_graph(Path(args.graph))
    spec = load_event_stress_suite_spec(Path(args.stress_config))

    reference = args.reference_state_database
    if reference is None:
        candidate = Path(args.work_dir) / campaign.state_database
        reference = str(candidate) if candidate.is_file() else None

    summary = run_event_drop_stress_suite(
        Path(args.root),
        posterior,
        selection,
        graph,
        campaign,
        base_dataset_identity=manifest.manifest_hash,
        scenarios=spec.scenarios,
        config=spec.config,
        reference_state_database=reference,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


def _export_scout_baseline(args: argparse.Namespace) -> None:
    from .grammar import load_model_spec
    from .scouts import export_scout_baseline_hyperparameters

    provenance = export_scout_baseline_hyperparameters(
        Path(args.evaluation),
        load_model_spec(Path(args.model)),
        Path(args.output),
    )
    print(json.dumps(provenance, sort_keys=True, indent=2))


def _review_scout_proposal(args: argparse.Namespace) -> None:
    from .grammar import load_model_spec, save_model_spec
    from .scouts import review_scout_proposal, write_scout_review

    summary = json.loads(Path(args.scout_summary).read_text())
    parent = load_model_spec(Path(args.parent_model))
    review, child = review_scout_proposal(
        summary,
        parent,
        proposal_id=args.proposal_id,
        decision=args.decision,
        note=args.note or "",
    )
    write_scout_review(Path(args.review_output), review)

    if review.decision == "accepted":
        if args.child_output is None:
            raise ValueError(
                "--child-output is required when accepting a scout proposal"
            )
        save_model_spec(Path(args.child_output), child)
        print(
            "scout proposal accepted: "
            f"mutation={review.mutation_id} child={review.child_model_hash}"
        )
    else:
        if args.child_output is not None:
            raise ValueError(
                "--child-output is invalid when rejecting a scout proposal"
            )
        print(
            "scout proposal rejected: "
            f"mutation={review.mutation_id}"
        )


def _compare_scout_descendant(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph, load_model_spec
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )
    from .scouts import (
        compare_scout_descendant_evidence,
        load_scout_review,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    freeze = validate_production_freeze(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    if not freeze["valid"]:
        raise ValueError("production freeze validation failed")

    graph = load_model_graph(Path(args.graph))
    parent = load_model_spec(Path(args.parent_model))
    child = load_model_spec(Path(args.child_model))
    review = load_scout_review(Path(args.review))
    if review.decision != "accepted":
        raise ValueError("scout comparison requires an accepted review record")
    if review.parent_model_hash != parent.model_hash:
        raise ValueError("review parent hash does not match parent model")
    if review.child_model_hash != child.model_hash:
        raise ValueError("review child hash does not match child model")
    if parent.model_hash not in graph.by_hash:
        raise ValueError(
            "reviewed scout parent is not a node in the frozen reference graph"
        )

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    summary = compare_scout_descendant_evidence(
        Path(args.root),
        posterior,
        selection,
        campaign,
        dataset_identity=manifest.manifest_hash,
        parent=parent,
        child=child,
        proposal_id=review.proposal_id,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


def _run_structured_scout_campaign(args: argparse.Namespace) -> None:
    from .grammar import baseline_model_spec, load_model_spec
    from .inference.synthetic import SyntheticSurveyConfig
    from .models import DEFAULT_BASELINE_HYPERPARAMETERS
    from .scouts import (
        StructuredScoutInjection,
        load_scout_campaign_config,
        run_structured_scout_campaign,
    )

    base_spec = (
        baseline_model_spec()
        if args.base_model is None
        else load_model_spec(Path(args.base_model))
    )
    base_hyperparameters = (
        dict(DEFAULT_BASELINE_HYPERPARAMETERS)
        if args.base_hyperparameters_json is None
        else {
            str(name): float(value)
            for name, value in _read_json_mapping(
                args.base_hyperparameters_json
            ).items()
        }
    )
    summary = run_structured_scout_campaign(
        Path(args.root),
        n_runs=args.n_runs,
        root_seed=args.root_seed,
        injection=StructuredScoutInjection(
            mutation_id=args.mutation_id,
            strength=args.strength,
        ),
        survey_config=SyntheticSurveyConfig(
            n_events=args.n_events,
            posterior_samples_per_event=args.pe_samples,
            n_injections=args.n_injections,
        ),
        scout_config=load_scout_campaign_config(Path(args.scout_config)),
        base_spec=base_spec,
        base_hyperparameters=base_hyperparameters,
    )
    print(
        "structured scout campaign complete: "
        f"runs={summary['n_runs']} "
        f"numerical_pass={summary['n_numerical_pass']} "
        f"expected_reachable={summary['expected_mutation_reachable']} "
        f"expected_proposed={summary['n_expected_proposed']}"
    )


def _assess_structured_scout_campaign(args: argparse.Namespace) -> None:
    from .scouts import assess_structured_scout_campaign

    summary = assess_structured_scout_campaign(Path(args.root))
    print(json.dumps(summary, sort_keys=True, indent=2))


def _write_default_scout_config(args: argparse.Namespace) -> None:
    from .scouts import (
        default_scout_campaign_config,
        save_scout_campaign_config,
    )

    config = default_scout_campaign_config(args.target, args.covariate)
    save_scout_campaign_config(Path(args.output), config)
    print(
        "default HSGP scout config written: "
        f"{args.output} target={args.target} covariate={args.covariate}"
    )


def _run_hsgp_scout(args: argparse.Namespace) -> None:
    from .grammar import load_model_spec
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
    )
    from .scouts import (
        load_scout_campaign_config,
        run_conditional_hsgp_scout,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    base_spec = load_model_spec(Path(args.base_model))
    raw_hyperparameters = _read_json_mapping(args.base_hyperparameters_json)
    base_hyperparameters = {
        str(name): float(value)
        for name, value in raw_hyperparameters.items()
    }
    scout = load_scout_campaign_config(Path(args.scout_config))

    _, summary = run_conditional_hsgp_scout(
        Path(args.run_dir),
        posterior,
        selection,
        base_spec=base_spec,
        base_hyperparameters=base_hyperparameters,
        hsgp_config=scout.hsgp,
        seed=args.seed,
        config=scout.run,
        dataset_identity=manifest.manifest_hash,
    )
    print(
        "HSGP scout complete: "
        f"numerical_pass={summary['numerical']['passed']} "
        f"raw_proposals={len(summary['raw_proposals'])} "
        f"validated_proposals={len(summary['validated_proposals'])}"
    )


def _canonicalize_gwcat_v2(args: argparse.Namespace) -> None:
    from .data import canonicalize_gwcat_v2_pair

    report = canonicalize_gwcat_v2_pair(
        Path(args.pe_export),
        Path(args.selection_export),
        Path(args.output_dir),
        required_spin_basis=args.spin_basis,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


def _freeze_dataset(args: argparse.Namespace) -> None:
    from .production import (
        build_dataset_manifest_from_canonical_files,
        save_dataset_manifest,
    )

    output = Path(args.output)
    base = output.parent.resolve()
    pe = Path(args.pe).resolve()
    selection = Path(args.selection).resolve()
    metadata = (
        {}
        if args.metadata_json is None
        else _read_json_mapping(args.metadata_json)
    )
    manifest = build_dataset_manifest_from_canonical_files(
        pe,
        selection,
        dataset_id=args.dataset_id,
        event_selection=_read_json_mapping(args.event_selection_json),
        waveform_policy=_read_json_mapping(args.waveform_policy_json),
        stored_pe_path=os.path.relpath(pe, base),
        stored_selection_path=os.path.relpath(selection, base),
        metadata=metadata,
    )
    save_dataset_manifest(output, manifest)
    print(
        f"dataset manifest frozen: {output} "
        f"sha256={manifest.manifest_hash}"
    )


def _write_default_fidelity_config(args: argparse.Namespace) -> None:
    from .inference.fidelity import (
        FidelityRunConfig,
        save_fidelity_run_config,
    )

    save_fidelity_run_config(Path(args.output), FidelityRunConfig())
    print(f"default fidelity config written: {args.output}")


def _freeze_production_campaign(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph
    from .inference.fidelity import load_fidelity_run_config
    from .inference.numpyro import _code_identity
    from .production import (
        SearchBudget,
        SeedPolicy,
        build_production_campaign,
        load_dataset_manifest,
        save_production_campaign,
    )
    from .search import SchedulerConfig

    if args.model_prior == "axis-complexity":
        if args.model_prior_penalty is None:
            raise ValueError(
                "--model-prior-penalty is required for axis-complexity"
            )
        model_prior = {
            "version": "axis-complexity-v1",
            "penalty_per_axis": float(args.model_prior_penalty),
        }
    else:
        if args.model_prior_penalty is not None:
            raise ValueError(
                "--model-prior-penalty is invalid for a uniform model prior"
            )
        model_prior = {"version": "uniform-v1"}

    git_commit = args.git_commit or str(_code_identity()["git_commit"])
    campaign = build_production_campaign(
        load_dataset_manifest(Path(args.manifest)),
        load_model_graph(Path(args.graph)),
        campaign_id=args.campaign_id,
        git_commit=git_commit,
        model_prior=model_prior,
        fidelity=load_fidelity_run_config(Path(args.fidelity_config)),
        scheduler=SchedulerConfig(
            beam_width=args.beam_width,
            exploration_quota=args.exploration_quota,
            seed=args.scheduler_seed,
        ),
        seed_policy=SeedPolicy(root_seed=args.root_seed),
        budget=SearchBudget(
            max_gpu_hours=args.max_gpu_hours,
            max_f3_models=args.max_f3_models,
            max_f4_models=args.max_f4_models,
            max_null_replays=args.max_null_replays,
        ),
        artifact_root=args.artifact_root,
        state_database=args.state_database,
        agents_enabled=False,
    )
    save_production_campaign(Path(args.output), campaign)
    print(
        f"production campaign frozen: {args.output} "
        f"sha256={campaign.campaign_hash}"
    )


def _complete_production_evidence(args: argparse.Namespace) -> None:
    from .grammar import load_model_graph
    from .production import (
        complete_graph_evidence,
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    freeze = validate_production_freeze(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    if not freeze["valid"]:
        raise ValueError("production freeze validation failed")

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    graph = load_model_graph(Path(args.graph))
    work_dir = Path(args.work_dir)
    state_database = work_dir / campaign.state_database
    artifact_root = work_dir / campaign.artifact_root

    summary = complete_graph_evidence(
        graph,
        posterior,
        selection,
        campaign,
        dataset_identity=manifest.manifest_hash,
        state_database=state_database,
        artifact_root=artifact_root,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


def _run_production_search(args: argparse.Namespace) -> None:
    from .production import (
        load_dataset_manifest,
        load_production_campaign,
        run_production_search,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    result = run_production_search(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        work_dir=Path(args.work_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


def _validate_dataset_freeze(args: argparse.Namespace) -> None:
    from .production import (
        load_dataset_manifest,
        validate_dataset_manifest_files,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    result = validate_dataset_manifest_files(
        manifest,
        base_dir=Path(args.base_dir),
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def _validate_production_freeze(args: argparse.Namespace) -> None:
    from .production import (
        load_dataset_manifest,
        load_production_campaign,
        validate_production_freeze,
    )

    manifest = load_dataset_manifest(Path(args.manifest))
    campaign = load_production_campaign(Path(args.campaign))
    result = validate_production_freeze(
        manifest,
        Path(args.graph),
        campaign,
        data_base_dir=Path(args.base_dir),
        require_current_commit=not args.ignore_current_commit,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gwpop-search",
        description="Systematic gravitational-wave population-model search.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    synthetic = subparsers.add_parser(
        "synthetic-recovery",
        help="run/resume one Phase-3 baseline BBH synthetic NUTS recovery",
    )
    synthetic.add_argument("--run-dir", required=True)
    synthetic.add_argument("--data-seed", type=int, default=20260917)
    synthetic.add_argument("--sampler-seed", type=int, default=20260918)
    _add_common_recovery_arguments(synthetic)
    synthetic.set_defaults(func=_run_synthetic_recovery)

    campaign = subparsers.add_parser(
        "synthetic-campaign",
        help="run/resume a deterministic multi-seed Phase-3 recovery campaign",
    )
    campaign.add_argument("--root", required=True)
    campaign.add_argument("--n-runs", type=int, default=4)
    campaign.add_argument("--root-seed", type=int, default=20260917)
    _add_common_recovery_arguments(campaign)
    campaign.set_defaults(func=_run_synthetic_campaign)

    assess = subparsers.add_parser(
        "assess-synthetic-campaign",
        help="assess completed Phase-3 recovery runs without launching inference",
    )
    assess.add_argument("--root", required=True)
    assess.add_argument("--min-runs", type=int, default=4)
    assess.set_defaults(func=_assess_synthetic_campaign)

    inspect_graph = subparsers.add_parser(
        "inspect-model-graph",
        help="print a concise structural inventory of a frozen model graph",
    )
    inspect_graph.add_argument("--graph", required=True)
    inspect_graph.set_defaults(func=_inspect_model_graph)

    extract_model = subparsers.add_parser(
        "extract-model",
        help="materialize one exact model spec from a frozen graph by hash",
    )
    extract_model.add_argument("--graph", required=True)
    extract_model.add_argument("--model-hash", required=True)
    extract_model.add_argument("--output", required=True)
    extract_model.set_defaults(func=_extract_model)

    enumerate_parser = subparsers.add_parser(
        "enumerate-models",
        help="write the deterministic initial declarative model graph",
    )
    enumerate_parser.add_argument("--output", required=True)
    enumerate_parser.add_argument("--max-depth", type=int, default=2)
    enumerate_parser.add_argument("--max-models", type=int, default=40)
    enumerate_parser.set_defaults(func=_enumerate_models)

    validate_parser = subparsers.add_parser(
        "validate-model",
        help="validate and compile a declarative JSON/YAML model spec",
    )
    validate_parser.add_argument("--spec", required=True)
    validate_parser.set_defaults(func=_validate_model_spec)

    holdout = subparsers.add_parser(
        "run-holdout-validation",
        help="run/resume strict K-fold held-out detected-event prediction",
    )
    holdout.add_argument("--manifest", required=True)
    holdout.add_argument("--graph", required=True)
    holdout.add_argument("--campaign", required=True)
    holdout.add_argument(
        "--model-hash",
        action="append",
        required=True,
        help="declared graph model to validate; repeat for multiple models",
    )
    holdout.add_argument("--n-folds", type=int, default=5)
    holdout.add_argument("--fold-seed", type=int)
    holdout.add_argument("--root", required=True)
    holdout.add_argument("--base-dir", default=".")
    holdout.add_argument("--ignore-current-commit", action="store_true")
    holdout.set_defaults(func=_run_holdout_validation)

    null_preflight = subparsers.add_parser(
        "diagnose-frozen-selection-null",
        help="preflight estimator-ready selection support for production nulls",
    )
    null_preflight.add_argument("--manifest", required=True)
    null_preflight.add_argument("--base-dir", default=".")
    null_preflight.add_argument("--model", required=True)
    null_preflight.add_argument("--hyperparameters-json", required=True)
    null_preflight.set_defaults(func=_diagnose_frozen_selection_null)

    null_template = subparsers.add_parser(
        "write-null-calibration-config",
        help="write a frozen exact-search baseline-null campaign configuration",
    )
    null_template.add_argument("--n-nulls", type=int, default=100)
    null_template.add_argument("--root-seed", type=int, default=20260918)
    null_template.add_argument("--n-events", type=int)
    null_template.add_argument("--pe-samples", type=int, default=256)
    null_template.add_argument("--n-injections", type=int, default=20_000)
    null_template.add_argument("--truth-hyperparameters-json")
    null_template.add_argument("--manifest")
    null_template.add_argument(
        "--data-mode",
        choices=("frozen_selection_resample", "synthetic_survey"),
        default="frozen_selection_resample",
    )
    null_template.add_argument(
        "--min-resampling-ess",
        type=float,
        default=200.0,
    )
    null_template.add_argument(
        "--max-gpu-hours-per-null",
        type=float,
        default=12.0,
    )
    null_template.add_argument("--output", required=True)
    null_template.set_defaults(func=_write_exact_null_config)

    null_run = subparsers.add_parser(
        "run-null-search-calibration",
        help="run/resume exact deterministic-search baseline-null replays",
    )
    null_run.add_argument("--manifest", required=True)
    null_run.add_argument("--graph", required=True)
    null_run.add_argument("--campaign", required=True)
    null_run.add_argument("--null-config", required=True)
    null_run.add_argument("--root", required=True)
    null_run.add_argument("--base-dir", default=".")
    null_run.add_argument("--work-dir", default=".")
    null_run.add_argument("--observed-state-database")
    null_run.add_argument("--ignore-current-commit", action="store_true")
    null_run.set_defaults(func=_run_exact_null_calibration)

    nearby_template = subparsers.add_parser(
        "write-nearby-baseline-config",
        help="write one explicit alternative-root model robustness scenario",
    )
    nearby_template.add_argument("--scenario-id", required=True)
    nearby_template.add_argument("--root-model", required=True)
    nearby_template.add_argument("--max-depth", type=int, default=1)
    nearby_template.add_argument("--max-models", type=int, default=20)
    nearby_template.add_argument("--note")
    nearby_template.add_argument("--output", required=True)
    _add_stress_arguments(nearby_template)
    nearby_template.set_defaults(func=_write_nearby_baseline_config)

    nearby_run = subparsers.add_parser(
        "run-nearby-baseline-suite",
        help="run/resume explicit nearby-baseline search robustness scenarios",
    )
    nearby_run.add_argument("--manifest", required=True)
    nearby_run.add_argument("--graph", required=True)
    nearby_run.add_argument("--campaign", required=True)
    nearby_run.add_argument("--nearby-config", required=True)
    nearby_run.add_argument("--base-dir", default=".")
    nearby_run.add_argument("--work-dir", default=".")
    nearby_run.add_argument("--root", required=True)
    nearby_run.add_argument("--reference-state-database")
    nearby_run.add_argument("--ignore-current-commit", action="store_true")
    nearby_run.set_defaults(func=_run_nearby_baseline_suite)

    loo_stress = subparsers.add_parser(
        "write-loo-stress-config",
        help="write explicit leave-one-out scenarios for every frozen event",
    )
    loo_stress.add_argument("--manifest", required=True)
    loo_stress.add_argument("--output", required=True)
    _add_stress_arguments(loo_stress)
    loo_stress.set_defaults(func=_write_loo_stress_config)

    drop_stress = subparsers.add_parser(
        "write-event-drop-stress-config",
        help="write one explicit custom/loud-event drop stress scenario",
    )
    drop_stress.add_argument("--scenario-id", required=True)
    drop_stress.add_argument(
        "--drop-event",
        action="append",
        required=True,
    )
    drop_stress.add_argument(
        "--category",
        choices=("leave_one_out", "loud_event", "custom"),
        default="custom",
    )
    drop_stress.add_argument("--note")
    drop_stress.add_argument("--output", required=True)
    _add_stress_arguments(drop_stress)
    drop_stress.set_defaults(func=_write_event_drop_stress_config)

    run_stress = subparsers.add_parser(
        "run-event-stress-suite",
        help="run/resume event-drop searches under the frozen production stack",
    )
    run_stress.add_argument("--manifest", required=True)
    run_stress.add_argument("--graph", required=True)
    run_stress.add_argument("--campaign", required=True)
    run_stress.add_argument("--stress-config", required=True)
    run_stress.add_argument("--base-dir", default=".")
    run_stress.add_argument("--work-dir", default=".")
    run_stress.add_argument("--root", required=True)
    run_stress.add_argument("--reference-state-database")
    run_stress.add_argument("--ignore-current-commit", action="store_true")
    run_stress.set_defaults(func=_run_event_stress_suite)

    export_scout = subparsers.add_parser(
        "export-scout-baseline",
        help="export frozen median hyperparameters from a valid F3/F4 model fit",
    )
    export_scout.add_argument("--evaluation", required=True)
    export_scout.add_argument("--model", required=True)
    export_scout.add_argument("--output", required=True)
    export_scout.set_defaults(func=_export_scout_baseline)

    review_scout = subparsers.add_parser(
        "review-scout-proposal",
        help="explicitly accept/reject one numerically validated HSGP proposal",
    )
    review_scout.add_argument("--scout-summary", required=True)
    review_scout.add_argument("--parent-model", required=True)
    review_scout.add_argument("--proposal-id", required=True)
    review_scout.add_argument(
        "--decision",
        choices=("accepted", "rejected"),
        required=True,
    )
    review_scout.add_argument("--note")
    review_scout.add_argument("--review-output", required=True)
    review_scout.add_argument("--child-output")
    review_scout.set_defaults(func=_review_scout_proposal)

    compare_scout = subparsers.add_parser(
        "compare-scout-descendant",
        help="independently refit/evidence-compare an accepted scout child",
    )
    compare_scout.add_argument("--manifest", required=True)
    compare_scout.add_argument("--graph", required=True)
    compare_scout.add_argument("--campaign", required=True)
    compare_scout.add_argument("--review", required=True)
    compare_scout.add_argument("--parent-model", required=True)
    compare_scout.add_argument("--child-model", required=True)
    compare_scout.add_argument("--root", required=True)
    compare_scout.add_argument("--base-dir", default=".")
    compare_scout.add_argument("--ignore-current-commit", action="store_true")
    compare_scout.set_defaults(func=_compare_scout_descendant)

    structured_scout = subparsers.add_parser(
        "structured-scout-campaign",
        help="run/resume a multi-seed structured-injection HSGP validation campaign",
    )
    structured_scout.add_argument("--root", required=True)
    structured_scout.add_argument("--n-runs", type=int, default=8)
    structured_scout.add_argument("--root-seed", type=int, default=20260918)
    structured_scout.add_argument(
        "--mutation-id",
        choices=(
            "null",
            "pairing.beta.linear_m1",
            "chieff.mean.linear_m1",
            "chieff.mean.linear_q",
            "chieff.mean.linear_z",
            "chieff.width.linear_m1",
            "chieff.width.linear_q",
            "chieff.width.linear_z",
        ),
        required=True,
    )
    structured_scout.add_argument("--strength", type=float, required=True)
    structured_scout.add_argument("--scout-config", required=True)
    structured_scout.add_argument("--n-events", type=int, default=64)
    structured_scout.add_argument("--pe-samples", type=int, default=256)
    structured_scout.add_argument("--n-injections", type=int, default=20_000)
    structured_scout.add_argument("--base-model")
    structured_scout.add_argument("--base-hyperparameters-json")
    structured_scout.set_defaults(func=_run_structured_scout_campaign)

    assess_structured = subparsers.add_parser(
        "assess-structured-scout-campaign",
        help="assess completed structured HSGP scout runs without inference",
    )
    assess_structured.add_argument("--root", required=True)
    assess_structured.set_defaults(func=_assess_structured_scout_campaign)

    scout_template = subparsers.add_parser(
        "write-default-scout-config",
        help="write a reviewable conditional-HSGP scout configuration",
    )
    scout_template.add_argument(
        "--target",
        choices=("q", "chi_eff"),
        required=True,
    )
    scout_template.add_argument(
        "--covariate",
        choices=("m1_source", "q", "z"),
        required=True,
    )
    scout_template.add_argument("--output", required=True)
    scout_template.set_defaults(func=_write_default_scout_config)

    scout_run = subparsers.add_parser(
        "run-hsgp-scout",
        help="run/resume one conditional-HSGP scout on a frozen dataset",
    )
    scout_run.add_argument("--manifest", required=True)
    scout_run.add_argument("--base-dir", default=".")
    scout_run.add_argument("--base-model", required=True)
    scout_run.add_argument("--base-hyperparameters-json", required=True)
    scout_run.add_argument("--scout-config", required=True)
    scout_run.add_argument("--run-dir", required=True)
    scout_run.add_argument("--seed", type=int, required=True)
    scout_run.set_defaults(func=_run_hsgp_scout)

    canonicalize = subparsers.add_parser(
        "canonicalize-gwcat-v2",
        help=(
            "convert reviewed gwcat-v2 PE/selection exports to canonical "
            "gwpop-search HDF5s with an audit report"
        ),
    )
    canonicalize.add_argument("--pe-export", required=True)
    canonicalize.add_argument("--selection-export", required=True)
    canonicalize.add_argument(
        "--spin-basis",
        choices=("chieff", "chieff_chip", "component"),
        required=True,
    )
    canonicalize.add_argument("--output-dir", required=True)
    canonicalize.set_defaults(func=_canonicalize_gwcat_v2)

    freeze_dataset = subparsers.add_parser(
        "freeze-dataset",
        help="hash canonical PE/selection HDF5s into a frozen dataset manifest",
    )
    freeze_dataset.add_argument("--pe", required=True)
    freeze_dataset.add_argument("--selection", required=True)
    freeze_dataset.add_argument("--dataset-id", required=True)
    freeze_dataset.add_argument("--event-selection-json", required=True)
    freeze_dataset.add_argument("--waveform-policy-json", required=True)
    freeze_dataset.add_argument("--metadata-json")
    freeze_dataset.add_argument("--output", required=True)
    freeze_dataset.set_defaults(func=_freeze_dataset)

    fidelity_template = subparsers.add_parser(
        "write-default-fidelity-config",
        help="write the explicit default F0-F4 numerical configuration for review",
    )
    fidelity_template.add_argument("--output", required=True)
    fidelity_template.set_defaults(func=_write_default_fidelity_config)

    freeze_campaign = subparsers.add_parser(
        "freeze-production-campaign",
        help="freeze graph/data hashes, numerical settings, scheduler, prior, and budget",
    )
    freeze_campaign.add_argument("--manifest", required=True)
    freeze_campaign.add_argument("--graph", required=True)
    freeze_campaign.add_argument("--fidelity-config", required=True)
    freeze_campaign.add_argument("--campaign-id", required=True)
    freeze_campaign.add_argument("--model-prior", choices=("axis-complexity", "uniform"), required=True)
    freeze_campaign.add_argument("--model-prior-penalty", type=float)
    freeze_campaign.add_argument("--beam-width", type=int, required=True)
    freeze_campaign.add_argument("--exploration-quota", type=int, required=True)
    freeze_campaign.add_argument("--scheduler-seed", type=int, required=True)
    freeze_campaign.add_argument("--root-seed", type=int, required=True)
    freeze_campaign.add_argument("--max-gpu-hours", type=float, required=True)
    freeze_campaign.add_argument("--max-f3-models", type=int, required=True)
    freeze_campaign.add_argument("--max-f4-models", type=int, required=True)
    freeze_campaign.add_argument("--max-null-replays", type=int, required=True)
    freeze_campaign.add_argument("--artifact-root", required=True)
    freeze_campaign.add_argument("--state-database", required=True)
    freeze_campaign.add_argument("--git-commit")
    freeze_campaign.add_argument("--output", required=True)
    freeze_campaign.set_defaults(func=_freeze_production_campaign)

    complete_evidence = subparsers.add_parser(
        "complete-production-evidence",
        help="run/resume valid F3 evidence for every declared graph model",
    )
    complete_evidence.add_argument("--manifest", required=True)
    complete_evidence.add_argument("--graph", required=True)
    complete_evidence.add_argument("--campaign", required=True)
    complete_evidence.add_argument("--base-dir", default=".")
    complete_evidence.add_argument("--work-dir", default=".")
    complete_evidence.add_argument(
        "--ignore-current-commit",
        action="store_true",
    )
    complete_evidence.set_defaults(func=_complete_production_evidence)

    run_production = subparsers.add_parser(
        "run-production-search",
        help="validate and run/resume the frozen deterministic F0-F4 search",
    )
    run_production.add_argument("--manifest", required=True)
    run_production.add_argument("--graph", required=True)
    run_production.add_argument("--campaign", required=True)
    run_production.add_argument("--base-dir", default=".")
    run_production.add_argument("--work-dir", default=".")
    run_production.add_argument("--ignore-current-commit", action="store_true")
    run_production.set_defaults(func=_run_production_search)

    dataset_freeze = subparsers.add_parser(
        "validate-dataset-manifest",
        help="verify every frozen dataset artifact checksum and size",
    )
    dataset_freeze.add_argument("--manifest", required=True)
    dataset_freeze.add_argument("--base-dir", default=".")
    dataset_freeze.set_defaults(func=_validate_dataset_freeze)

    production_freeze = subparsers.add_parser(
        "validate-production-freeze",
        help="cross-check frozen data, model graph, campaign config, and code revision",
    )
    production_freeze.add_argument("--manifest", required=True)
    production_freeze.add_argument("--graph", required=True)
    production_freeze.add_argument("--campaign", required=True)
    production_freeze.add_argument("--base-dir", default=".")
    production_freeze.add_argument("--ignore-current-commit", action="store_true")
    production_freeze.set_defaults(func=_validate_production_freeze)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
