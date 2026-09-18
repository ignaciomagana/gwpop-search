"""Command-line entry point for reproducible gwpop-search campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__


def _run_synthetic_recovery(args: argparse.Namespace) -> None:
    from .inference.numpyro import NUTSConfig
    from .inference.recovery import run_synthetic_baseline_recovery
    from .inference.synthetic import SyntheticSurveyConfig

    survey = SyntheticSurveyConfig(
        n_events=args.n_events,
        posterior_samples_per_event=args.pe_samples,
        n_injections=args.n_injections,
    )
    nuts = NUTSConfig(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        progress_bar=not args.no_progress,
    )
    _, summary = run_synthetic_baseline_recovery(
        Path(args.run_dir),
        data_seed=args.data_seed,
        sampler_seed=args.sampler_seed,
        survey_config=survey,
        nuts_config=nuts,
        selection_chunk_size=args.selection_chunk_size,
    )
    diagnostic = summary["diagnostics"]
    print(
        "synthetic recovery complete: "
        f"events={diagnostic['n_events']} "
        f"selected_injections={diagnostic['n_selected_injections']} "
        f"divergences={diagnostic['n_divergent']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gwpop-search",
        description="Systematic gravitational-wave population-model search.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    synthetic = subparsers.add_parser(
        "synthetic-recovery",
        help="run/resume the Phase-3 baseline BBH synthetic NUTS recovery",
    )
    synthetic.add_argument("--run-dir", required=True)
    synthetic.add_argument("--data-seed", type=int, default=20260917)
    synthetic.add_argument("--sampler-seed", type=int, default=20260918)
    synthetic.add_argument("--n-events", type=int, default=48)
    synthetic.add_argument("--pe-samples", type=int, default=256)
    synthetic.add_argument("--n-injections", type=int, default=20_000)
    synthetic.add_argument("--num-warmup", type=int, default=1000)
    synthetic.add_argument("--num-samples", type=int, default=1000)
    synthetic.add_argument("--num-chains", type=int, default=4)
    synthetic.add_argument("--target-accept", type=float, default=0.9)
    synthetic.add_argument("--max-tree-depth", type=int, default=10)
    synthetic.add_argument("--selection-chunk-size", type=int, default=4096)
    synthetic.add_argument("--no-progress", action="store_true")
    synthetic.set_defaults(func=_run_synthetic_recovery)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
