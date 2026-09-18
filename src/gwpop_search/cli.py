"""Command-line entry point for reproducible gwpop-search campaigns."""

from __future__ import annotations

import argparse
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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
