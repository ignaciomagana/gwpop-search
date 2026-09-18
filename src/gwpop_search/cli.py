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

    structured_scout = subparsers.add_parser(
        "structured-scout-campaign",
        help="run/resume a multi-seed structured-injection HSGP validation campaign",
    )
    structured_scout.add_argument("--root", required=True)
    structured_scout.add_argument("--n-runs", type=int, default=4)
    structured_scout.add_argument("--root-seed", type=int, default=20260918)
    structured_scout.add_argument(
        "--mutation-id",
        choices=(
            "null",
            "pairing.beta.linear_m1",
            "chieff.mean.linear_q",
            "chieff.width.linear_q",
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
