"""Command-line entry point for reproducible gwpop-search campaigns."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__

_LEGACY_DEFAULT_N_INJECTIONS = 20_000


def _survey_config(args: argparse.Namespace):
    from .inference.synthetic import (
        INJECTION_DRAW_POPULATION_PROXY,
        RECOMMENDED_POPULATION_PROXY_N_INJECTIONS,
        SyntheticSurveyConfig,
    )

    proxy = args.injection_draw == INJECTION_DRAW_POPULATION_PROXY
    n_injections = args.n_injections
    if n_injections is None:
        # The legacy box keeps its historical default so existing campaign
        # plans reproduce; population_proxy defaults to the recommended size.
        n_injections = (
            RECOMMENDED_POPULATION_PROXY_N_INJECTIONS
            if proxy
            else _LEGACY_DEFAULT_N_INJECTIONS
        )
    elif proxy and n_injections < RECOMMENDED_POPULATION_PROXY_N_INJECTIONS:
        print(
            "warning: --n-injections "
            f"{n_injections} is below the recommended "
            f"{RECOMMENDED_POPULATION_PROXY_N_INJECTIONS} for population_proxy "
            "(see docs/phase3_recovery.md)",
            file=sys.stderr,
        )
    return SyntheticSurveyConfig(
        n_events=args.n_events,
        posterior_samples_per_event=args.pe_samples,
        n_injections=n_injections,
        injection_draw=args.injection_draw,
        observation_model=args.observation_model,
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


def _fingerprint_recovery_checkpoint(args: argparse.Namespace) -> None:
    from .inference.campaign import checkpoint_fingerprints

    payload = checkpoint_fingerprints(Path(args.run_dir))
    print(json.dumps(payload, sort_keys=True, indent=2))


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


# ---------------------------------------------------------------------------
# Phase-3 v2 (dynesty) recovery campaign and Phase-3c evidence check
# ---------------------------------------------------------------------------


def _ns_survey_config(args: argparse.Namespace):
    from .inference.synthetic import SyntheticSurveyConfig

    return SyntheticSurveyConfig(
        n_events=args.n_events,
        posterior_samples_per_event=args.pe_samples,
        n_injections=args.n_injections,
        injection_draw=args.injection_draw,
        observation_model=args.observation_model,
    )


def _ns_dynesty_config(args: argparse.Namespace, *, slices: int | None):
    from .inference.dynesty_backend import DynestyConfig

    return DynestyConfig(
        nlive=args.nlive,
        bound=args.bound,
        sample=args.sample,
        slices=slices,
        dlogz=args.dlogz,
        batch_size=args.batch_size,
        maxiter=args.maxiter,
        maxcall=args.maxcall,
        checkpoint_every=args.checkpoint_every,
        num_posterior_samples=args.num_posterior_samples,
    )


def _add_ns_arguments(parser: argparse.ArgumentParser) -> None:
    """Survey-v2 and dynesty options; defaults are the Phase-3 v2 plan."""
    parser.add_argument("--root", required=True)
    parser.add_argument("--root-seed", type=int, default=20260917)
    parser.add_argument("--n-events", type=int, default=48)
    parser.add_argument("--pe-samples", type=int, default=1024)
    parser.add_argument("--n-injections", type=int, default=100_000)
    parser.add_argument(
        "--injection-draw",
        choices=("uniform_detector_box", "population_proxy"),
        default="population_proxy",
    )
    parser.add_argument(
        "--observation-model",
        choices=("truth_centered", "noisy_observation"),
        default="noisy_observation",
    )
    parser.add_argument("--nlive", type=int, default=1000)
    parser.add_argument(
        "--bound", choices=("none", "single", "multi", "balls", "cubes"), default="multi"
    )
    parser.add_argument(
        "--sample", choices=("unif", "rwalk", "slice", "rslice"), default="rslice"
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=None,
        help="slice-sampler slices (default: 2*(3+ndim) of each model)",
    )
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="dynesty queue size and fixed device batch",
    )
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--num-posterior-samples", type=int, default=4000)
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=300.0,
        help="seconds between dynesty checkpoints (I/O only; not part of the plan)",
    )
    parser.add_argument(
        "--importance-draws",
        type=int,
        default=16384,
        help=(
            "pooled posterior draws the importance diagnostics evaluate; the default is at "
            "least the pooled draw count of every planned fit, so the gated quantiles carry "
            "no subsample Monte-Carlo noise"
        ),
    )
    parser.add_argument("--rhat-draws-per-run", type=int, default=2000)


def _run_synthetic_campaign_ns(args: argparse.Namespace) -> None:
    from .inference.phase3_ns import (
        NSRecoveryAcceptanceCriteria,
        phase3_hbi_config,
        run_ns_recovery_campaign,
        slices_for_ndim,
    )
    from .inference.priors import BASELINE_SYNTHETIC_PRIORS

    slices = args.slices
    if slices is None and args.sample in ("slice", "rslice"):
        slices = slices_for_ndim(len(BASELINE_SYNTHETIC_PRIORS))
    summary = run_ns_recovery_campaign(
        Path(args.root),
        n_runs=args.n_runs,
        root_seed=args.root_seed,
        repeats=args.repeats,
        survey_config=_ns_survey_config(args),
        dynesty_config=_ns_dynesty_config(args, slices=slices),
        hbi_config=phase3_hbi_config(),
        criteria=NSRecoveryAcceptanceCriteria(
            min_runs=args.min_runs, min_repeats=args.min_repeats
        ),
        importance_draws=args.importance_draws,
        rhat_draws_per_run=args.rhat_draws_per_run,
    )
    print(
        "dynesty synthetic campaign complete: "
        f"runs={summary['n_runs']} "
        f"numerical_pass={summary['n_numerical_pass']} "
        f"gate={summary['phase3_numerical_gate_passed']}"
    )


def _assess_synthetic_campaign_ns(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from .inference.phase3_ns import (
        CAMPAIGN_PLAN_NAME,
        NSRecoveryAcceptanceCriteria,
        assess_ns_recovery_campaign,
    )

    root = Path(args.root)
    criteria = None
    if args.min_runs is not None or args.min_repeats is not None:
        plan_path = root / CAMPAIGN_PLAN_NAME
        criteria = (
            NSRecoveryAcceptanceCriteria.from_dict(json.loads(plan_path.read_text())["criteria"])
            if plan_path.exists()
            else NSRecoveryAcceptanceCriteria()
        )
        overrides = {}
        if args.min_runs is not None:
            overrides["min_runs"] = args.min_runs
        if args.min_repeats is not None:
            overrides["min_repeats"] = args.min_repeats
        criteria = replace(criteria, **overrides)
    summary = assess_ns_recovery_campaign(root, criteria=criteria)
    print(
        "dynesty synthetic campaign assessment: "
        f"runs={summary['n_runs']} "
        f"numerical_pass={summary['n_numerical_pass']} "
        f"gate={summary['phase3_numerical_gate_passed']}"
    )


def _fingerprint_ns_run(args: argparse.Namespace) -> None:
    from .inference.phase3_ns import compare_ns_fingerprints, ns_fingerprint_report

    report = ns_fingerprint_report(Path(args.run_dir))
    comparison = None
    if args.compare_to:
        before = json.loads(Path(args.compare_to).read_text())
        comparison = compare_ns_fingerprints(
            before.get("fingerprints", before), report["fingerprints"]
        )
        report["comparison"] = comparison
    text = json.dumps(report, sort_keys=True, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    if comparison is not None and not comparison["passed"]:
        raise SystemExit(1)


def _evidence_check_dynesty(args: argparse.Namespace):
    """Dynesty base configuration and slices rule of the evidence check.

    Without ``--slices`` a slice sampler uses ``2*(3+ndim)`` of each model.
    """
    from .inference.phase3_evidence import SLICES_RULE_FIXED, SLICES_RULE_PER_MODEL

    per_model = args.slices is None and args.sample in ("slice", "rslice")
    rule = SLICES_RULE_PER_MODEL if per_model else SLICES_RULE_FIXED
    return _ns_dynesty_config(args, slices=args.slices), rule


def _run_synthetic_evidence_check(args: argparse.Namespace) -> None:
    from .inference.phase3_evidence import default_injection_strengths, run_evidence_check
    from .inference.phase3_ns import phase3_hbi_config

    strengths = default_injection_strengths()
    if args.chi_mu_q_slope is not None:
        strengths["chieff.mean.linear_q"] = args.chi_mu_q_slope
    if args.beta_q_m1_slope is not None:
        strengths["pairing.beta.linear_m1"] = args.beta_q_m1_slope
    dynesty_config, slices_rule = _evidence_check_dynesty(args)
    summary = run_evidence_check(
        Path(args.root),
        n_catalogs=args.n_catalogs,
        root_seed=args.root_seed,
        repeats=args.repeats,
        survey_config=_ns_survey_config(args),
        dynesty_config=dynesty_config,
        slices_rule=slices_rule,
        hbi_config=phase3_hbi_config(),
        injection_strengths=strengths,
        importance_draws=args.importance_draws,
        rhat_draws_per_run=args.rhat_draws_per_run,
        sddr_bootstrap=args.sddr_bootstrap,
    )
    for case in [*summary["null_cases"], *summary["injected_cases"]]:
        print(
            f"{case['kind']:8s} {case['catalog']:34s} {case['atom']:24s} "
            f"lnBF={case['ln_bf']:+.3f} sigma={case['ln_bf_sigma']} "
            f"sddr={case['sddr']['ln_bf']}"
        )
    print(f"evidence check passed={summary['evidence_check_passed']}")


def _add_stress_arguments(parser: argparse.ArgumentParser) -> None:
    # fidelity ladder v2: suites stop at an evidence rung (F2 is not in the ladder)
    parser.add_argument(
        "--stop-fidelity",
        choices=("F3", "F4"),
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
    parser.add_argument(
        "--n-injections",
        type=int,
        default=None,
        help=(
            "selection draws N_draw (default: 20000 for uniform_detector_box, "
            "the recommended 100000 for population_proxy)"
        ),
    )
    parser.add_argument(
        "--injection-draw",
        choices=("uniform_detector_box", "population_proxy"),
        default="uniform_detector_box",
        help=(
            "selection-injection distribution: the legacy uniform detector-frame "
            "box, or draws from the baseline population at the default proxy "
            "hyperparameters mixed with a defensive q -> 1 pairing component "
            "(stored draw density = that exact mixture density)"
        ),
    )
    parser.add_argument(
        "--observation-model",
        choices=("truth_centered", "noisy_observation"),
        default="truth_centered",
        help=(
            "truth_centered: legacy zero-noise PE and detection on true "
            "parameters; noisy_observation: one noise realisation per event and "
            "injection, detection on the observed data and PE drawn from the "
            "posterior given those data (DAG-consistent)"
        ),
    )
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
        mutations_for_profile,
        save_model_graph,
    )

    graph = enumerate_model_graph(
        baseline_model_spec(args.hyperprior_profile),
        mutations=mutations_for_profile(args.hyperprior_profile),
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
    from .analysis.posterior_gates import PosteriorGateCriteria
    from .inference.dynesty_backend import DynestyConfig

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
            repeats=args.posterior_repeats,
            posterior_config=DynestyConfig(
                nlive=args.posterior_nlive,
                bound="multi",
                sample="rslice",
                slices=args.posterior_slices,
                dlogz=args.posterior_dlogz,
                maxcall=args.posterior_maxcall,
                batch_size=args.posterior_batch_size,
            ),
            criteria=(
                PosteriorGateCriteria.f4()
                if args.gate_profile == "f4"
                else PosteriorGateCriteria.f3()
            ),
            predictive_draws=args.predictive_draws,
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
        PE_SCALE_POLICY_DECLARED_FIXED,
        PE_SCALE_POLICY_MATCH_OBSERVED,
        ExactNullCampaignConfig,
        measure_observed_pe_scales,
        save_exact_null_campaign_config,
    )
    from .production import load_dataset_manifest, load_frozen_dataset

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
    pe_samples = args.pe_samples
    pe_scale_policy = args.pe_scale_policy
    if pe_scale_policy is None:
        pe_scale_policy = (
            PE_SCALE_POLICY_MATCH_OBSERVED
            if args.data_mode == "frozen_selection_resample"
            else PE_SCALE_POLICY_DECLARED_FIXED
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
        if pe_scale_policy == PE_SCALE_POLICY_MATCH_OBSERVED:
            # The null PE sample count is the observed one (D4: identical F3
            # Monte-Carlo regime), so it is read from the frozen catalog and
            # frozen into the configuration rather than chosen.
            if args.base_dir is None:
                raise ValueError(
                    "--base-dir is required with --pe-scale-policy match_observed: "
                    "the null PE sample count is read from the frozen catalog"
                )
            posterior, _ = load_frozen_dataset(
                manifest,
                data_base_dir=Path(args.base_dir),
            )
            observed_samples = measure_observed_pe_scales(
                posterior
            ).uniform_samples_per_event()
            if pe_samples is not None and int(pe_samples) != observed_samples:
                raise ValueError(
                    f"--pe-samples {int(pe_samples)} disagrees with the frozen "
                    f"catalog's {observed_samples} samples per event; "
                    "--pe-scale-policy match_observed requires the observed count"
                )
            pe_samples = observed_samples
    else:
        n_events = 64 if args.n_events is None else args.n_events
    if pe_samples is None:
        pe_samples = 256

    config = ExactNullCampaignConfig(
        n_nulls=args.n_nulls,
        root_seed=args.root_seed,
        survey=SyntheticSurveyConfig(
            n_events=n_events,
            posterior_samples_per_event=int(pe_samples),
            n_injections=args.n_injections,
            observation_model="noisy_observation",
        ),
        truth_hyperparameters=truth,
        data_mode=args.data_mode,
        min_resampling_ess=args.min_resampling_ess,
        min_resampling_ess_per_event=args.min_resampling_ess_per_event,
        pe_scale_policy=pe_scale_policy,
        max_gpu_hours_per_null=args.max_gpu_hours_per_null,
        statistic=args.statistic,
    )
    save_exact_null_campaign_config(Path(args.output), config)
    print(
        "exact null calibration config written: "
        f"{args.output} n_nulls={config.n_nulls} statistic={config.statistic} "
        f"pe_scale_policy={config.pe_scale_policy} "
        f"pe_samples={config.survey.posterior_samples_per_event} "
        f"min_resampling_ess>=max({config.min_resampling_ess:g}, "
        f"{config.min_resampling_ess_per_event:g}x{config.survey.n_events}) "
        f"max_gpu_hours_per_null={config.max_gpu_hours_per_null}"
    )


def _load_exact_null_cli_context(
    args: argparse.Namespace,
    *,
    load_data: bool,
):
    from .grammar import load_model_graph
    from .nulls import load_exact_null_campaign_config
    from .production import (
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

    graph = load_model_graph(Path(args.graph))
    config = load_exact_null_campaign_config(Path(args.null_config))
    posterior = None
    selection = None
    dataset_identity = None
    if config.data_mode == "frozen_selection_resample":
        dataset_identity = manifest.manifest_hash
        if load_data:
            posterior, selection = load_frozen_dataset(
                manifest,
                data_base_dir=Path(args.base_dir),
            )
    return (
        manifest,
        campaign,
        graph,
        config,
        posterior,
        selection,
        dataset_identity,
    )


def _prepare_exact_null_calibration(args: argparse.Namespace) -> None:
    from .nulls import prepare_exact_null_campaign

    (
        _,
        campaign,
        graph,
        config,
        posterior,
        selection,
        dataset_identity,
    ) = _load_exact_null_cli_context(args, load_data=True)
    plan = prepare_exact_null_campaign(
        Path(args.root),
        graph,
        campaign,
        config,
        production_posterior=posterior,
        production_selection=selection,
        production_dataset_identity=dataset_identity,
    )
    print(json.dumps(plan, sort_keys=True, indent=2))
    precheck_path = Path(args.root) / "null_data_precheck.json"
    if precheck_path.is_file():
        precheck = json.loads(precheck_path.read_text())
        resampling = precheck["resampling"]
        print(
            "null data precheck: "
            f"resampling_ess={resampling['resampling_ess']:.6g} "
            f"(required {resampling['required_resampling_ess']:.6g} for "
            f"{precheck['n_events']} events, "
            f"{resampling['resampling_ess_per_event']:.4g} per event) "
            f"pe_scale_policy={precheck['pe_scale_policy']} "
            f"max_observed/declared_pe_width_ratio="
            f"{precheck['pe_scales']['max_width_ratio']:.4g} -> {precheck_path}"
        )


def _run_exact_null_index(args: argparse.Namespace) -> None:
    from .nulls import run_exact_null_index

    (
        _,
        campaign,
        graph,
        config,
        posterior,
        selection,
        dataset_identity,
    ) = _load_exact_null_cli_context(args, load_data=True)
    result = run_exact_null_index(
        Path(args.root),
        graph,
        campaign,
        config,
        null_index=args.null_index,
        production_posterior=posterior,
        production_selection=selection,
        production_dataset_identity=dataset_identity,
    )
    print(json.dumps(result.to_dict(), sort_keys=True, indent=2))


def _run_null_model_evaluation(args: argparse.Namespace) -> None:
    """Pre-compute one model's evidence rung for one exact-null replay.

    Execution-only: the record is printed and written into the replay's own
    artifact tree, never to a state database (the replay writes its own rows
    when it reloads this run). The seed, dataset identity and run directory
    are the replay's; there is deliberately no seed override.
    """
    from .nulls import run_null_model_evaluation

    (
        _,
        campaign,
        graph,
        config,
        posterior,
        selection,
        dataset_identity,
    ) = _load_exact_null_cli_context(args, load_data=True)
    payload = run_null_model_evaluation(
        Path(args.root),
        graph,
        campaign,
        config,
        null_index=args.null_index,
        model_hash=args.model_hash,
        production_posterior=posterior,
        production_selection=selection,
        production_dataset_identity=dataset_identity,
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


def _run_stress_model_evaluation(args: argparse.Namespace) -> None:
    """Pre-compute one model's F3 evidence for one event-drop scenario.

    The event-drop counterpart of ``run-null-model-evaluation``: nothing is
    written to a state database and no ``stress_plan.json`` is written, so the
    suite still freezes the full scenario list itself.
    """
    from .grammar import load_model_graph
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )
    from .validation import (
        load_event_stress_suite_spec,
        run_stress_model_evaluation,
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

    spec = load_event_stress_suite_spec(Path(args.stress_config))
    scenarios = {item.scenario_id: item for item in spec.scenarios}
    if args.scenario_id not in scenarios:
        raise ValueError(
            f"unknown scenario id {args.scenario_id!r}; the suite declares "
            f"{sorted(scenarios)}"
        )

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    payload = run_stress_model_evaluation(
        Path(args.root),
        posterior,
        selection,
        load_model_graph(Path(args.graph)),
        campaign,
        base_dataset_identity=manifest.manifest_hash,
        scenario=scenarios[args.scenario_id],
        model_hash=args.model_hash,
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


def _list_nearby_scenario_models(args: argparse.Namespace) -> None:
    """Print the models a nearby-baseline scenario evaluates (no data, no sampling).

    Default output: one model hash per line, in the order the suite evaluates
    them (sorted by hash; every graph model is an F3 candidate, and the ones
    that pass F0 inside the suite are evaluated at F3). ``--json`` prints the
    full listing (depth, root mutation ids, F3 budget check).
    """
    from .validation import load_nearby_baseline_suite_spec, nearby_scenario_models

    suite = load_nearby_baseline_suite_spec(Path(args.nearby_config))
    scenarios = list(suite.scenarios)
    if args.scenario_id is not None:
        scenarios = [item for item in scenarios if item.scenario_id == args.scenario_id]
        if not scenarios:
            raise ValueError(
                f"unknown scenario id {args.scenario_id!r}; the suite declares "
                f"{sorted(item.scenario_id for item in suite.scenarios)}"
            )
    listings = [nearby_scenario_models(item, suite.config) for item in scenarios]
    if args.json:
        print(json.dumps({"scenarios": listings}, sort_keys=True, indent=2))
        return
    for listing in listings:
        for row in listing["models"]:
            if len(listings) > 1:
                print(f"{listing['scenario_id']} {row['model_hash']}")
            else:
                print(row["model_hash"])


def _run_nearby_model_evaluation(args: argparse.Namespace) -> None:
    """Pre-compute one model's F3 evidence for one nearby-baseline scenario.

    The alternative-root counterpart of ``run-stress-model-evaluation``:
    nothing is written to a state database and no ``nearby_baseline_plan.json``
    is written; ``run-nearby-baseline-suite`` later reloads the run.
    """
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )
    from .validation import (
        load_nearby_baseline_suite_spec,
        run_nearby_model_evaluation,
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

    suite = load_nearby_baseline_suite_spec(Path(args.nearby_config))
    scenarios = {item.scenario_id: item for item in suite.scenarios}
    if args.scenario_id not in scenarios:
        raise ValueError(
            f"unknown scenario id {args.scenario_id!r}; the suite declares "
            f"{sorted(scenarios)}"
        )

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    payload = run_nearby_model_evaluation(
        Path(args.root),
        posterior,
        selection,
        campaign,
        base_dataset_identity=manifest.manifest_hash,
        scenario=scenarios[args.scenario_id],
        model_hash=args.model_hash,
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


def _export_model_spec(args: argparse.Namespace) -> None:
    """Write the declarative spec of one frozen-graph model (optionally re-priored).

    ``--set-prior NAME FAMILY A B`` replaces one declared hyperprior
    (uniform/log_uniform: A=low B=high; normal: A=loc B=scale), so a
    prior-widened variant is derived from the exact production spec.
    """
    from .grammar import load_model_graph, save_model_spec
    from .production import model_spec_diff, with_prior

    graph = load_model_graph(Path(args.graph))
    if args.model_hash not in graph.by_hash:
        raise ValueError(f"unknown graph model hash {args.model_hash!r}")
    parent = graph.by_hash[args.model_hash]
    spec = parent
    for name, family, first, second in args.set_prior or []:
        family = str(family).strip().lower()
        keys = ("loc", "scale") if family == "normal" else ("low", "high")
        spec = with_prior(
            spec,
            name,
            family,
            {keys[0]: float(first), keys[1]: float(second)},
        )
    if args.output is None:
        print(json.dumps(spec.to_dict(), sort_keys=True, indent=2))
        return
    save_model_spec(Path(args.output), spec)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_hash": spec.model_hash,
                "derived_from": model_spec_diff(parent, spec),
            },
            sort_keys=True,
            indent=2,
        )
    )


def _run_model_spec_evaluation(args: argparse.Namespace) -> None:
    """Evaluate one model spec JSON at the frozen campaign's settings.

    The same evaluator, seed derivation, gates and artifact layout as a
    production (or ``run-fidelity-evaluation``) evaluation of a graph model;
    nothing is written to a state database.
    """
    from .grammar import load_model_graph, load_model_spec
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        run_model_spec_evaluation,
        validate_production_freeze,
    )
    from .search import Fidelity

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
    spec = load_model_spec(Path(args.model_spec))
    derived_from = None
    if args.derived_from is not None:
        if args.derived_from not in graph.by_hash:
            raise ValueError(f"unknown graph model hash {args.derived_from!r}")
        derived_from = graph.by_hash[args.derived_from]

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    payload = run_model_spec_evaluation(
        Path(args.root),
        posterior,
        selection,
        campaign,
        spec,
        dataset_identity=manifest.manifest_hash,
        fidelity=Fidelity(args.fidelity),
        graph=graph,
        spec_path=Path(args.model_spec),
        derived_from=derived_from,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


def _finalize_exact_null_calibration(args: argparse.Namespace) -> None:
    from .nulls import finalize_exact_null_campaign

    (
        manifest,
        campaign,
        graph,
        config,
        _,
        _,
        dataset_identity,
    ) = _load_exact_null_cli_context(args, load_data=False)

    observed = args.observed_state_database
    if observed is None:
        candidate = Path(args.work_dir) / campaign.state_database
        observed = str(candidate) if candidate.is_file() else None

    summary = finalize_exact_null_campaign(
        Path(args.root),
        graph,
        campaign,
        config,
        observed_state_database=observed,
        production_dataset_identity=dataset_identity,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


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


def _write_v2_alt_root_config(args: argparse.Namespace) -> None:
    from .grammar import load_model_spec
    from .validation import (
        NearbyBaselineSuiteSpec,
        save_nearby_baseline_suite_spec,
        v2_alt_root_scenario,
        v2_chi_eff_atom_mutations,
    )

    if (args.root_model is None) == (args.alt_root is None):
        raise ValueError("pass exactly one of --root-model or --alt-root")
    if args.alt_root is not None:
        from .grammar.v2 import V2_PROFILE, v2_alternative_roots

        root_spec = v2_alternative_roots()[args.alt_root]
        catalogue = args.mutation_catalogue or V2_PROFILE
    else:
        root_spec = load_model_spec(Path(args.root_model))
        catalogue = args.mutation_catalogue or "default"
    chi_eff = {}
    for item in args.chi_eff_atom or []:
        label, sep, mutation_id = str(item).partition("=")
        if not sep or not label or not mutation_id:
            raise ValueError(f"--chi-eff-atom expects LABEL=MUTATION_ID; got {item!r}")
        if label in chi_eff:
            raise ValueError(f"--chi-eff-atom {label} given twice")
        chi_eff[label] = tuple(x for x in mutation_id.split(",") if x) if "," in mutation_id else mutation_id
    if not chi_eff and catalogue == "gwtc5-v2":
        # the v2 grammar's own C1-C6 / S1-S4 atoms (each one mutation of R0)
        chi_eff = v2_chi_eff_atom_mutations()
    candidates = [tuple(x for x in str(item).split(",") if x) for item in args.candidate or []]
    scenario = v2_alt_root_scenario(
        args.scenario_id,
        root_spec,
        candidate_paths=candidates,
        chi_eff_mutation_ids=chi_eff,
        mutation_catalogue_name=catalogue,
        note=args.note or "",
    )
    spec = NearbyBaselineSuiteSpec(
        scenarios=(scenario,),
        config=_nearby_baseline_config_from_args(args),
    )
    save_nearby_baseline_suite_spec(Path(args.output), spec)
    print(
        "v2 D5 alternative-root config written: "
        f"{args.output} scenario={args.scenario_id} paths={len(scenario.mutation_paths)} "
        f"models<={scenario.max_models}"
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


def _confirm_structured_scout_descendant(args: argparse.Namespace) -> None:
    from .inference.fidelity import load_fidelity_run_config
    from .scouts import confirm_structured_scout_descendant

    summary = confirm_structured_scout_descendant(
        Path(args.campaign_root),
        Path(args.output_root),
        load_fidelity_run_config(Path(args.fidelity_config)),
        run_index=args.run_index,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


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
    from .data import GwcatV2DataPolicy, canonicalize_gwcat_v2_pair

    policy = (
        None if args.v2_policy is None else GwcatV2DataPolicy.from_json(args.v2_policy)
    )
    report = canonicalize_gwcat_v2_pair(
        Path(args.pe_export),
        Path(args.selection_export),
        Path(args.output_dir),
        required_spin_basis=args.spin_basis,
        required_selection_spin_basis=args.selection_spin_basis,
        policy=policy,
        spin_prior_allow_list=args.spin_prior_allow_list,
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
    from dataclasses import replace

    from .inference.fidelity import (
        FidelityRunConfig,
        fidelity_config_sha256,
        save_fidelity_run_config,
    )

    default = FidelityRunConfig()

    def rung(evidence, *, nlive, repeats, maxcall):
        dynesty = evidence.dynesty
        updates = {"batch_size": args.batch_size}
        if nlive is not None:
            updates["nlive"] = nlive
        if maxcall is not None:
            updates["maxcall"] = maxcall
        return replace(
            evidence,
            repeats=evidence.repeats if repeats is None else repeats,
            dynesty=replace(dynesty, **updates),
        )

    config = replace(
        default,
        f0=replace(
            default.f0,
            prior_draws=args.f0_prior_draws,
            batch_size=args.batch_size,
        ),
        f3_evidence=rung(
            default.f3_evidence,
            nlive=args.f3_nlive,
            repeats=args.f3_repeats,
            maxcall=args.f3_maxcall,
        ),
        f4_evidence=rung(
            default.f4_evidence,
            nlive=args.f4_nlive,
            repeats=args.f4_repeats,
            maxcall=args.f4_maxcall,
        ),
    )
    save_fidelity_run_config(Path(args.output), config)
    print(
        f"fidelity config 2.0 written: {args.output} "
        f"sha256={fidelity_config_sha256(config)}"
    )


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

    from .production.freeze import verify_graph_file

    git_commit = args.git_commit or str(_code_identity()["git_commit"])
    campaign = build_production_campaign(
        load_dataset_manifest(Path(args.manifest)),
        load_model_graph(Path(args.graph)),
        graph_file_sha256=str(verify_graph_file(Path(args.graph))["file_sha256"]),
        campaign_id=args.campaign_id,
        git_commit=git_commit,
        model_prior=model_prior,
        fidelity=load_fidelity_run_config(Path(args.fidelity_config)),
        scheduler=SchedulerConfig(
            beam_width=args.beam_width,
            exploration_quota=args.exploration_quota,
            seed=args.scheduler_seed,
            ladder=tuple(
                item.strip() for item in args.ladder.split(",") if item.strip()
            ),
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
        require_root_profile=args.require_root_profile,
    )
    save_production_campaign(Path(args.output), campaign)
    from .production.validate import root_hyperprior_profile_for_hash

    profile = root_hyperprior_profile_for_hash(campaign.model_graph_root_hash)
    print(
        f"production campaign frozen: {args.output} "
        f"sha256={campaign.campaign_hash} "
        f"ladder={','.join(campaign.scheduler.ladder)} "
        f"root_hyperprior_profile={profile} "
        f"sampler={campaign.sampler_backend['name']}=={campaign.sampler_backend['version']}"
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


# ---------------------------------------------------------------------------
# Track B: fidelity ladder v2 (dynesty) diagnostics subcommands
# ---------------------------------------------------------------------------


def _run_fidelity_evaluation(args: argparse.Namespace) -> None:
    """Evaluate graph models at one ladder rung outside the state database.

    Preflight/diagnosis only: the records are printed, never written to a
    production state database. The default seed is the production seed
    ``evaluation_seed(campaign root seed, model, fidelity)``.
    """
    from .grammar import load_model_graph
    from .inference.fidelity import DeterministicHBIEvaluator
    from .production import (
        load_dataset_manifest,
        load_frozen_dataset,
        load_production_campaign,
        validate_production_freeze,
    )
    from .search import Fidelity, evaluation_seed

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
    if args.all_models:
        if args.model_hash:
            raise ValueError("--all-models and --model-hash are mutually exclusive")
        hashes = [model.model_hash for model in graph.nodes]
    else:
        if not args.model_hash:
            raise ValueError("pass --model-hash (repeatable) or --all-models")
        hashes = list(args.model_hash)
    unknown = [item for item in hashes if item not in graph.by_hash]
    if unknown:
        raise ValueError(f"unknown graph model hash(es) {unknown}")

    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=Path(args.base_dir),
    )
    evaluator = DeterministicHBIEvaluator(
        posterior,
        selection,
        config=campaign.fidelity,
        dataset_identity=manifest.manifest_hash,
    )
    fidelity = Fidelity(args.fidelity)
    rows = []
    for model_hash in hashes:
        seed = (
            evaluation_seed(campaign.seed_policy.root_seed, model_hash, fidelity)
            if args.seed is None
            else int(args.seed)
        )
        run_dir = Path(args.output_root) / fidelity.value / model_hash
        record = evaluator.evaluate(
            graph.by_hash[model_hash],
            fidelity,
            seed=seed,
            run_dir=run_dir,
        )
        rows.append(
            {
                "model_hash": model_hash,
                "fidelity": fidelity.value,
                "seed": seed,
                "diagnostics_pass": record.diagnostics_pass,
                "screen_value": record.screen_value,
                "compute_cost_hours": record.compute_cost,
                "evaluation": str(run_dir / "evaluation.json"),
            }
        )
    print(
        json.dumps(
            {
                "format_version": "gwpop-search-fidelity-preflight-1.0",
                "state_database_written": False,
                "evaluations": rows,
            },
            sort_keys=True,
            indent=2,
        )
    )


def _summarize_fidelity_evaluation(args: argparse.Namespace) -> None:
    from .inference.fidelity import read_evaluation

    payload = read_evaluation(Path(args.evaluation))
    diagnostics = payload["diagnostics"]
    checks = diagnostics.get("checks", [])
    summary = {
        "model_hash": payload["model_hash"],
        "fidelity": payload["fidelity"],
        "passed": diagnostics.get("passed"),
        "screen_value": payload.get("screen_value"),
        "screen_value_semantics": payload.get("screen_value_semantics"),
        "failure": diagnostics.get("failure"),
        "failed_gates": [
            item
            for item in checks
            if item.get("stage", "gate") == "gate" and not item.get("passed")
        ],
        "n_gates": sum(1 for item in checks if item.get("stage", "gate") == "gate"),
        # Advisory diagnostics the backend could not produce (e.g. the F4
        # insertion-index test, which needs dynesty birth iterations): reported
        # explicitly so an operator never reads a missing advisory as a pass.
        "unavailable_advisories": [
            {
                "name": item.get("name"),
                "reason_code": item.get("reason_code"),
                "note": item.get("note"),
            }
            for item in checks
            if item.get("stage") == "advisory" and item.get("available") is False
        ],
        "evidence": diagnostics.get("evidence"),
        "posterior_median": diagnostics.get("posterior_median"),
    }
    if args.all_checks:
        summary["checks"] = checks
    print(json.dumps(summary, sort_keys=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    from .grammar import HYPERPRIOR_PROFILES
    from .nulls import (
        DEFAULT_MIN_RESAMPLING_ESS_PER_EVENT,
        PE_SCALE_POLICIES,
    )

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

    fingerprint = subparsers.add_parser(
        "fingerprint-recovery-checkpoint",
        help="hash completed Phase-3 chain checkpoints for resume-integrity review",
    )
    fingerprint.add_argument("--run-dir", required=True)
    fingerprint.set_defaults(func=_fingerprint_recovery_checkpoint)

    assess = subparsers.add_parser(
        "assess-synthetic-campaign",
        help="assess completed Phase-3 recovery runs without launching inference",
    )
    assess.add_argument("--root", required=True)
    assess.add_argument("--min-runs", type=int, default=4)
    assess.set_defaults(func=_assess_synthetic_campaign)

    # -- Phase-3 v2 on dynesty (NS recovery campaign, fingerprints, 3c evidence check) --
    campaign_ns = subparsers.add_parser(
        "synthetic-campaign-ns",
        help="run/resume the Phase-3 v2 multi-catalog recovery campaign on dynesty",
    )
    _add_ns_arguments(campaign_ns)
    campaign_ns.add_argument("--n-runs", type=int, default=4, help="number of catalogs")
    campaign_ns.add_argument(
        "--repeats", type=int, default=4, help="independent dynesty runs per catalog"
    )
    campaign_ns.add_argument("--min-runs", type=int, default=4)
    campaign_ns.add_argument("--min-repeats", type=int, default=4)
    campaign_ns.set_defaults(func=_run_synthetic_campaign_ns)

    assess_ns = subparsers.add_parser(
        "assess-synthetic-campaign-ns",
        help="assess a Phase-3 v2 dynesty campaign without launching inference",
    )
    assess_ns.add_argument("--root", required=True)
    assess_ns.add_argument(
        "--min-runs", type=int, default=None, help="override the plan's min_runs"
    )
    assess_ns.add_argument(
        "--min-repeats", type=int, default=None, help="override the plan's min_repeats"
    )
    assess_ns.set_defaults(func=_assess_synthetic_campaign_ns)

    fingerprint_ns = subparsers.add_parser(
        "fingerprint-ns-run",
        help="hash completed dynesty run artifacts (manifest, result) for resume review",
    )
    fingerprint_ns.add_argument(
        "--run-dir", required=True, help="a dynesty run directory or any directory above them"
    )
    fingerprint_ns.add_argument("--output", default=None, help="also write the JSON here")
    fingerprint_ns.add_argument(
        "--compare-to",
        default=None,
        help="earlier fingerprint JSON; exit 1 if any earlier file changed or disappeared",
    )
    fingerprint_ns.set_defaults(func=_fingerprint_ns_run)

    evidence_check = subparsers.add_parser(
        "synthetic-evidence-check",
        help="Phase-3c Bayes-factor sanity mini-campaign (nested atoms, SDDR cross-check)",
    )
    _add_ns_arguments(evidence_check)
    evidence_check.add_argument("--n-catalogs", type=int, default=4)
    evidence_check.add_argument(
        "--repeats", type=int, default=2, help="independent dynesty runs per model"
    )
    evidence_check.add_argument(
        "--chi-mu-q-slope",
        type=float,
        default=None,
        help="injected chieff.mean.linear_q slope (default 0.9 x prior upper bound = 0.54)",
    )
    evidence_check.add_argument(
        "--beta-q-m1-slope",
        type=float,
        default=None,
        help="injected pairing.beta.linear_m1 slope (default 0.9 x prior upper bound = 0.27)",
    )
    evidence_check.add_argument("--sddr-bootstrap", type=int, default=200)
    evidence_check.set_defaults(func=_run_synthetic_evidence_check)
    # -- end of the Phase-3 v2 dynesty block --

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
    enumerate_parser.add_argument(
        "--hyperprior-profile",
        default="phase3",
        help=(
            "registered root hyperprior profile (grammar.HYPERPRIOR_PROFILES); "
            "'phase3' reproduces the original graph, 'gwtc5-v1' is GWTC-5 production"
        ),
    )
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
    holdout.add_argument("--posterior-repeats", type=int, default=2)
    holdout.add_argument("--posterior-nlive", type=int, default=1000)
    holdout.add_argument(
        "--posterior-slices",
        type=int,
        help="rslice slices (default: 2*(3+ndim) per model)",
    )
    holdout.add_argument("--posterior-dlogz", type=float, default=0.1)
    holdout.add_argument("--posterior-maxcall", type=int)
    holdout.add_argument("--posterior-batch-size", type=int, default=64)
    holdout.add_argument(
        "--gate-profile",
        choices=("f4", "f3"),
        default="f4",
        help="posterior + importance thresholds of decision D3 (evidence precision off)",
    )
    holdout.add_argument(
        "--predictive-draws",
        type=int,
        help="score with this many pooled draws (default: every weighted point)",
    )
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
    null_template.add_argument(
        "--pe-samples",
        type=int,
        help=(
            "PE samples per null event; read from the frozen catalog under "
            "--pe-scale-policy match_observed (default 256 otherwise)"
        ),
    )
    null_template.add_argument("--n-injections", type=int, default=20_000)
    null_template.add_argument("--truth-hyperparameters-json")
    null_template.add_argument("--manifest")
    null_template.add_argument(
        "--base-dir",
        help=(
            "frozen data base directory; required by --pe-scale-policy "
            "match_observed, which measures the observed PE precision"
        ),
    )
    null_template.add_argument(
        "--pe-scale-policy",
        choices=PE_SCALE_POLICIES,
        default=None,
        help=(
            "match_observed: null PE precision and sample count come from the "
            "frozen catalog (per-event widths rank-matched on the detection "
            "statistic proxy), so nulls and the observed run share the F3 "
            "Monte-Carlo regime; declared_fixed: keep the configured scalars "
            "and record the measured mismatch (the default is match_observed "
            "for frozen_selection_resample and declared_fixed for the "
            "engineering synthetic_survey mode, which has no frozen catalog)"
        ),
    )
    null_template.add_argument(
        "--data-mode",
        choices=("frozen_selection_resample", "synthetic_survey"),
        default="frozen_selection_resample",
    )
    null_template.add_argument(
        "--min-resampling-ess",
        type=float,
        default=200.0,
        help="absolute floor on the frozen-selection resampling ESS",
    )
    null_template.add_argument(
        "--min-resampling-ess-per-event",
        type=float,
        default=DEFAULT_MIN_RESAMPLING_ESS_PER_EVENT,
        help=(
            "resampling ESS required per event; the gate is "
            "max(--min-resampling-ess, this x n_events), so a null catalog is "
            "drawn from an effective pool much larger than itself"
        ),
    )
    null_template.add_argument(
        "--max-gpu-hours-per-null",
        type=float,
        required=True,
        help=(
            "per-null compute ceiling (wall-clock hours of the replay's "
            "evaluations); freeze it from cost calibration"
        ),
    )
    null_template.add_argument(
        "--statistic",
        choices=("f3_completion",),
        default="f3_completion",
        help=(
            "calibrated statistic: max edge ln BF / ln posterior odds from F3 "
            "evidence after full-graph completion (nulls run F0 + F3 only)"
        ),
    )
    null_template.add_argument("--output", required=True)
    null_template.set_defaults(func=_write_exact_null_config)

    null_prepare = subparsers.add_parser(
        "prepare-null-search-calibration",
        help="freeze the exact-null plan before serial or Slurm-array replay",
    )
    null_prepare.add_argument("--manifest", required=True)
    null_prepare.add_argument("--graph", required=True)
    null_prepare.add_argument("--campaign", required=True)
    null_prepare.add_argument("--null-config", required=True)
    null_prepare.add_argument("--root", required=True)
    null_prepare.add_argument("--base-dir", default=".")
    null_prepare.add_argument("--ignore-current-commit", action="store_true")
    null_prepare.set_defaults(func=_prepare_exact_null_calibration)

    null_index = subparsers.add_parser(
        "run-null-search-index",
        help="run/resume one deterministic exact-null replay by index",
    )
    null_index.add_argument("--manifest", required=True)
    null_index.add_argument("--graph", required=True)
    null_index.add_argument("--campaign", required=True)
    null_index.add_argument("--null-config", required=True)
    null_index.add_argument("--root", required=True)
    null_index.add_argument("--null-index", type=int, required=True)
    null_index.add_argument("--base-dir", default=".")
    null_index.add_argument("--ignore-current-commit", action="store_true")
    null_index.set_defaults(func=_run_exact_null_index)

    null_model_eval = subparsers.add_parser(
        "run-null-model-evaluation",
        help=(
            "pre-compute one model's evidence rung inside one exact-null replay's "
            "artifact tree (execution-only; nothing is recorded as production "
            "state, and run-null-search-index later reloads it)"
        ),
    )
    null_model_eval.add_argument("--manifest", required=True)
    null_model_eval.add_argument("--graph", required=True)
    null_model_eval.add_argument("--campaign", required=True)
    null_model_eval.add_argument("--null-config", required=True)
    null_model_eval.add_argument("--root", required=True)
    null_model_eval.add_argument("--null-index", type=int, required=True)
    null_model_eval.add_argument("--model-hash", required=True)
    null_model_eval.add_argument("--base-dir", default=".")
    null_model_eval.add_argument("--ignore-current-commit", action="store_true")
    null_model_eval.set_defaults(func=_run_null_model_evaluation)

    null_finalize = subparsers.add_parser(
        "finalize-null-search-calibration",
        help="aggregate a complete indexed exact-null campaign and calibrate",
    )
    null_finalize.add_argument("--manifest", required=True)
    null_finalize.add_argument("--graph", required=True)
    null_finalize.add_argument("--campaign", required=True)
    null_finalize.add_argument("--null-config", required=True)
    null_finalize.add_argument("--root", required=True)
    null_finalize.add_argument("--base-dir", default=".")
    null_finalize.add_argument("--work-dir", default=".")
    null_finalize.add_argument("--observed-state-database")
    null_finalize.add_argument("--ignore-current-commit", action="store_true")
    null_finalize.set_defaults(func=_finalize_exact_null_calibration)

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

    v2_alt = subparsers.add_parser(
        "write-v2-alt-root-config",
        help=(
            "write a v2 D5 alternative-root scenario restricted to the root, the "
            "candidate edges and all 10 chi_eff atoms (writes a config only; runs nothing)"
        ),
    )
    v2_alt.add_argument("--scenario-id", required=True)
    v2_alt.add_argument("--root-model", help="alternative root model spec JSON")
    v2_alt.add_argument(
        "--alt-root",
        choices=("A1", "A2"),
        help="the v2 grammar's alternative root (A1 = R0 + beta per component, "
        "A2 = R0 + kappa(m1)) instead of --root-model; implies --mutation-catalogue gwtc5-v2",
    )
    v2_alt.add_argument(
        "--candidate",
        action="append",
        help="candidate mutation path: 'ID' (depth 1) or 'A,B' (B applied to root+A); repeatable",
    )
    v2_alt.add_argument(
        "--chi-eff-atom",
        action="append",
        help="LABEL=MUTATION_ID (or LABEL=A,B for a two-step atom) for each of C1-C6, "
        "S1-S4 (all ten required; default with the gwtc5-v2 catalogue: the v2 atoms)",
    )
    v2_alt.add_argument(
        "--mutation-catalogue",
        default=None,
        help="named mutation catalogue (default: 'gwtc5-v2' with --alt-root, else 'default')",
    )
    v2_alt.add_argument("--note")
    v2_alt.add_argument("--output", required=True)
    _add_stress_arguments(v2_alt)
    v2_alt.set_defaults(func=_write_v2_alt_root_config)

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

    nearby_list = subparsers.add_parser(
        "list-nearby-scenario-models",
        help=(
            "print the model hashes a nearby-baseline scenario evaluates, in the "
            "suite's order (reads only the nearby config; no data, no sampling)"
        ),
    )
    nearby_list.add_argument("--nearby-config", required=True)
    nearby_list.add_argument("--scenario-id")
    nearby_list.add_argument("--json", action="store_true")
    nearby_list.set_defaults(func=_list_nearby_scenario_models)

    nearby_model_eval = subparsers.add_parser(
        "run-nearby-model-evaluation",
        help=(
            "pre-compute one model's F3 evidence inside one nearby-baseline "
            "scenario's artifact tree (execution-only; nothing is recorded as "
            "production state, and run-nearby-baseline-suite later reloads it)"
        ),
    )
    nearby_model_eval.add_argument("--manifest", required=True)
    nearby_model_eval.add_argument("--graph", required=True)
    nearby_model_eval.add_argument("--campaign", required=True)
    nearby_model_eval.add_argument("--nearby-config", required=True)
    nearby_model_eval.add_argument("--scenario-id", required=True)
    nearby_model_eval.add_argument("--model-hash", required=True)
    nearby_model_eval.add_argument(
        "--root",
        required=True,
        help="the SUITE root; the scenario tree is <root>/<scenario-id>",
    )
    nearby_model_eval.add_argument("--base-dir", default=".")
    nearby_model_eval.add_argument("--ignore-current-commit", action="store_true")
    nearby_model_eval.set_defaults(func=_run_nearby_model_evaluation)

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

    stress_model_eval = subparsers.add_parser(
        "run-stress-model-evaluation",
        help=(
            "pre-compute one model's F3 evidence inside one event-drop scenario's "
            "artifact tree (execution-only; nothing is recorded as production "
            "state, and run-event-stress-suite later reloads it)"
        ),
    )
    stress_model_eval.add_argument("--manifest", required=True)
    stress_model_eval.add_argument("--graph", required=True)
    stress_model_eval.add_argument("--campaign", required=True)
    stress_model_eval.add_argument("--stress-config", required=True)
    stress_model_eval.add_argument("--scenario-id", required=True)
    stress_model_eval.add_argument("--model-hash", required=True)
    stress_model_eval.add_argument(
        "--root",
        required=True,
        help="the SUITE root; the scenario tree is <root>/<scenario-id>",
    )
    stress_model_eval.add_argument("--base-dir", default=".")
    stress_model_eval.add_argument("--ignore-current-commit", action="store_true")
    stress_model_eval.set_defaults(func=_run_stress_model_evaluation)

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

    confirm_structured = subparsers.add_parser(
        "confirm-structured-scout-descendant",
        help=(
            "materialize and independently F3-confirm an injected scout "
            "descendant on the same structured mock"
        ),
    )
    confirm_structured.add_argument("--campaign-root", required=True)
    confirm_structured.add_argument("--fidelity-config", required=True)
    confirm_structured.add_argument("--output-root", required=True)
    confirm_structured.add_argument("--run-index", type=int)
    confirm_structured.set_defaults(func=_confirm_structured_scout_descendant)

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
    canonicalize.add_argument(
        "--selection-spin-basis",
        choices=("chieff", "chieff_chip", "component", "chieff_reference"),
        default=None,
        help=(
            "explicit selection-export basis (default: same as --spin-basis); "
            "chieff_reference is accepted only with --spin-basis chieff and a PE "
            "export whose chi_eff prior ceilings all equal the reference ceiling"
        ),
    )
    canonicalize.add_argument("--output-dir", required=True)
    canonicalize.add_argument(
        "--v2-policy",
        default=None,
        help=(
            "JSON GwcatV2DataPolicy (OD-7 spin-prior allow-list, z_max and G17 "
            "allow-list, exact priors, sky-marginal cumulative mixture); every "
            "check must pass and the policy hash is recorded in the report"
        ),
    )
    canonicalize.add_argument(
        "--spin-prior-allow-list",
        default=None,
        help=(
            "without --v2-policy: a .json {event: kind} or text 'NAME [KIND]' file "
            "of events allowed a non-own_analytic spin prior in a reference pair"
        ),
    )
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
        help=(
            "write the fidelity ladder v2 (F0 -> F3 -> F4, dynesty) configuration "
            "(format 2.0) for review"
        ),
    )
    fidelity_template.add_argument("--output", required=True)
    fidelity_template.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="dynesty queue size = fixed device batch of the likelihood (F0/F3/F4)",
    )
    fidelity_template.add_argument("--f0-prior-draws", type=int, default=4096)
    fidelity_template.add_argument("--f3-nlive", type=int)
    fidelity_template.add_argument("--f3-repeats", type=int)
    fidelity_template.add_argument(
        "--f3-maxcall",
        type=int,
        help="per-run dynesty maxcall at F3 (default: none; a budget stop fails the gate)",
    )
    fidelity_template.add_argument("--f4-nlive", type=int)
    fidelity_template.add_argument("--f4-repeats", type=int)
    fidelity_template.add_argument("--f4-maxcall", type=int)
    fidelity_template.set_defaults(func=_write_default_fidelity_config)

    freeze_campaign = subparsers.add_parser(
        "freeze-production-campaign",
        help="freeze graph/data hashes, numerical settings, scheduler, prior, and budget",
    )
    freeze_campaign.add_argument("--manifest", required=True)
    freeze_campaign.add_argument("--graph", required=True)
    freeze_campaign.add_argument("--fidelity-config", required=True)
    freeze_campaign.add_argument("--campaign-id", required=True)
    freeze_campaign.add_argument(
        "--model-prior", choices=("axis-complexity", "uniform"), required=True
    )
    freeze_campaign.add_argument("--model-prior-penalty", type=float)
    freeze_campaign.add_argument("--beam-width", type=int, required=True)
    freeze_campaign.add_argument("--exploration-quota", type=int, required=True)
    freeze_campaign.add_argument("--scheduler-seed", type=int, required=True)
    freeze_campaign.add_argument(
        "--ladder",
        default="F0,F3,F4",
        help="comma-separated fidelity ladder (default F0,F3,F4)",
    )
    freeze_campaign.add_argument("--root-seed", type=int, required=True)
    freeze_campaign.add_argument("--max-gpu-hours", type=float, required=True)
    freeze_campaign.add_argument("--max-f3-models", type=int, required=True)
    freeze_campaign.add_argument("--max-f4-models", type=int, required=True)
    freeze_campaign.add_argument("--max-null-replays", type=int, required=True)
    freeze_campaign.add_argument("--artifact-root", required=True)
    freeze_campaign.add_argument("--state-database", required=True)
    freeze_campaign.add_argument(
        "--require-root-profile",
        choices=HYPERPRIOR_PROFILES,
        help=(
            "refuse the freeze unless the graph root is this registered root "
            "hyperprior profile (GWTC-5 production: gwtc5-v1)"
        ),
    )
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
        help="validate and run/resume the frozen deterministic ladder search (F0 -> F3 -> F4)",
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
        help=(
            "cross-check frozen data, model graph, campaign config, code revision and "
            "the dynesty version pin"
        ),
    )
    production_freeze.add_argument("--manifest", required=True)
    production_freeze.add_argument("--graph", required=True)
    production_freeze.add_argument("--campaign", required=True)
    production_freeze.add_argument("--base-dir", default=".")
    production_freeze.add_argument("--ignore-current-commit", action="store_true")
    production_freeze.set_defaults(func=_validate_production_freeze)

    # --- Track B: fidelity ladder v2 (dynesty) diagnostics subcommands ---
    fidelity_eval = subparsers.add_parser(
        "run-fidelity-evaluation",
        help=(
            "evaluate graph models at one ladder rung (F0/F3/F4) outside the state "
            "database (preflight/diagnosis; nothing is recorded as production state)"
        ),
    )
    fidelity_eval.add_argument("--manifest", required=True)
    fidelity_eval.add_argument("--graph", required=True)
    fidelity_eval.add_argument("--campaign", required=True)
    fidelity_eval.add_argument("--fidelity", choices=("F0", "F3", "F4"), required=True)
    fidelity_eval.add_argument("--model-hash", action="append", default=[])
    fidelity_eval.add_argument("--all-models", action="store_true")
    fidelity_eval.add_argument(
        "--seed",
        type=int,
        help="override the production evaluation seed (default: campaign seed policy)",
    )
    fidelity_eval.add_argument("--output-root", required=True)
    fidelity_eval.add_argument("--base-dir", default=".")
    fidelity_eval.add_argument("--ignore-current-commit", action="store_true")
    fidelity_eval.set_defaults(func=_run_fidelity_evaluation)

    spec_export = subparsers.add_parser(
        "export-model-spec",
        help=(
            "write the declarative spec of one frozen-graph model by hash, "
            "optionally with declared hyperpriors replaced (--set-prior)"
        ),
    )
    spec_export.add_argument("--graph", required=True)
    spec_export.add_argument("--model-hash", required=True)
    spec_export.add_argument(
        "--set-prior",
        nargs=4,
        action="append",
        metavar=("NAME", "FAMILY", "A", "B"),
        help=(
            "replace a declared prior: uniform/log_uniform A=low B=high, "
            "normal A=loc B=scale (repeatable)"
        ),
    )
    spec_export.add_argument("--output", help="default: print the spec to stdout")
    spec_export.set_defaults(func=_export_model_spec)

    spec_eval = subparsers.add_parser(
        "run-model-spec-evaluation",
        help=(
            "evaluate one model spec JSON at the frozen campaign's fidelity "
            "settings, seed derivation and gates on the frozen dataset "
            "(nothing is recorded as production state)"
        ),
    )
    spec_eval.add_argument("--manifest", required=True)
    spec_eval.add_argument("--graph", required=True)
    spec_eval.add_argument("--campaign", required=True)
    spec_eval.add_argument("--model-spec", required=True)
    spec_eval.add_argument(
        "--root",
        required=True,
        help="output root; the run directory is <root>/<fidelity>/<model hash>",
    )
    spec_eval.add_argument("--fidelity", choices=("F0", "F3", "F4"), default="F3")
    spec_eval.add_argument(
        "--derived-from",
        help="graph model hash the spec was derived from (provenance only)",
    )
    spec_eval.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print hash/seed/run dir without sampling or writing",
    )
    spec_eval.add_argument("--base-dir", default=".")
    spec_eval.add_argument("--ignore-current-commit", action="store_true")
    spec_eval.set_defaults(func=_run_model_spec_evaluation)

    fidelity_summary = subparsers.add_parser(
        "summarize-fidelity-evaluation",
        help="print the gate outcome of one evaluation.json (format 2.0)",
    )
    fidelity_summary.add_argument("--evaluation", required=True)
    fidelity_summary.add_argument("--all-checks", action="store_true")
    fidelity_summary.set_defaults(func=_summarize_fidelity_evaluation)
    # --- Track C: robustness and model-comparison analyses (gwpop_search.analysis.cli) ---
    from .analysis.cli import register_analysis_subcommands

    register_analysis_subcommands(subparsers)
    # --- end Track C ---

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
