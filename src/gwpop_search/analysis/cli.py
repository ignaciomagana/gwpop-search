"""Command-line subcommands of the analysis package.

Every command operates on saved dynesty results (``result.npz`` + sidecar,
found recursively under ``--results`` paths and grouped by the model hash in
their likelihood identity) and on the frozen data (``--manifest``/``--base-dir``
of a production freeze, or canonical ``--pe``/``--selection`` HDF5 files). The
model specifications come from ``--graph``. Outputs are versioned JSON (and
Markdown for the model-comparison report).

* ``classify-graph-edges``       — nesting class of every edge (+ numeric check)
* ``analyze-edge-mc-error``      — per-model posterior-averaged weights, per-edge
  MC error/bias, ``Sigma_MC``; optional bootstrap-reweighting estimate
* ``analyze-prior-sensitivity``  — width variants per edge, model-prior variants
* ``analyze-sddr``               — Savage–Dickey cross-checks of eligible edges
* ``analyze-model-comparison``   — the full report with claim criteria
* ``run-psis-loo-influence``     — PSIS-LOO influence, optional exact-refit config
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _enable_x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


def _float_list(text: str) -> list[float]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("ln"):
            out.append(math.log(float(item[2:])))
        else:
            out.append(float(item))
    return out


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data (a production freeze or canonical HDF5 files)")
    group.add_argument("--manifest", help="frozen dataset manifest (production)")
    group.add_argument("--base-dir", default=".", help="base directory of the manifest artifacts")
    group.add_argument("--pe", help="canonical PE catalog HDF5 (instead of --manifest)")
    group.add_argument("--selection", help="canonical selection catalog HDF5 (instead of --manifest)")


def _add_results_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--graph", required=True, help="model graph JSON")
    parser.add_argument(
        "--results",
        action="append",
        required=required,
        help="dynesty result.npz file or directory searched recursively; repeatable",
    )
    parser.add_argument(
        "--model-hash",
        action="append",
        help="restrict to these model hashes (default: every model with results)",
    )


def _load_data(args):
    from gwpop_search.data import PosteriorCatalog, SelectionCatalog

    if args.manifest:
        if args.pe or args.selection:
            raise ValueError("pass either --manifest or --pe/--selection, not both")
        from gwpop_search.production import load_dataset_manifest, load_frozen_dataset

        manifest = load_dataset_manifest(Path(args.manifest))
        return load_frozen_dataset(manifest, data_base_dir=Path(args.base_dir))
    if not (args.pe and args.selection):
        raise ValueError("data required: --manifest, or both --pe and --selection")
    return PosteriorCatalog.from_hdf5(args.pe), SelectionCatalog.from_hdf5(args.selection)


def _load_graph_and_results(args):
    from gwpop_search.grammar import load_model_graph

    from ._common import discover_dynesty_results

    graph = load_model_graph(Path(args.graph))
    grouped = discover_dynesty_results(args.results or [], model_hashes=args.model_hash)
    unknown = sorted(set(grouped) - set(graph.by_hash))
    if unknown:
        raise ValueError(f"results for model hashes that are not graph nodes: {unknown}")
    return graph, grouped


def _write(path, payload) -> None:
    from ._common import write_json

    write_json(Path(path), payload)
    print(f"written: {path}")


def _edges_with(graph, grouped, *, need_both: bool = True):
    for edge in graph.edges:
        have = (edge.parent_hash in grouped, edge.child_hash in grouped)
        if (all(have) if need_both else any(have)):
            yield edge


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _classify_graph_edges(args) -> None:
    from gwpop_search.grammar import load_model_graph

    from .sddr import classify_graph_edges, edge_classification_report, verify_nesting

    graph = load_model_graph(Path(args.graph))
    report = edge_classification_report(graph)
    if args.verify:
        _enable_x64()
        by_hash = graph.by_hash
        checks = []
        for row in classify_graph_edges(graph):
            if row.embeddings:
                checks.append(verify_nesting(row, by_hash[row.parent_hash], by_hash[row.child_hash]))
        report["numerical_verification"] = checks
    _write(args.output, report)


def _analyze_edge_mc_error(args) -> None:
    _enable_x64()
    from gwpop_search.models import compile_model_spec

    from ._common import pool_dynesty_results
    from .edge_mc_error import (
        InsufficientReweightingESSError,
        bootstrap_edge_mc_error,
        compute_model_mc_weights,
        edge_mc_error,
        edge_mc_report,
        mc_covariance_matrix,
        save_model_mc_weights,
    )
    from .terms import pad_catalog

    posterior, selection = _load_data(args)
    graph, grouped = _load_graph_and_results(args)
    out = Path(args.output_dir)
    weights, samples, models, catalogs = {}, {}, {}, {}
    for model_hash, results in sorted(grouped.items()):
        spec = graph.by_hash[model_hash]
        models[model_hash] = compile_model_spec(spec)
        samples[model_hash] = pool_dynesty_results(
            results, n_draws=None if args.all_points else args.n_draws, seed=args.seed
        )
        catalogs[model_hash] = pad_catalog(posterior, selection, models[model_hash])
        weights[model_hash] = compute_model_mc_weights(
            samples[model_hash], posterior, selection, models[model_hash], label=model_hash,
            batch_size=args.batch_size, catalog=catalogs[model_hash],
        )
        save_model_mc_weights(out / "mc_weights" / f"{model_hash}.npz", weights[model_hash])
    edges = []
    for edge in _edges_with(graph, grouped):
        predicted = edge_mc_error(weights[edge.child_hash], weights[edge.parent_hash])
        empirical, refusal = None, None
        if args.bootstrap_replicates:
            boot_samples = {
                h: pool_dynesty_results(grouped[h], n_draws=args.bootstrap_draws, seed=args.seed)
                for h in (edge.child_hash, edge.parent_hash)
            }
            try:
                empirical = bootstrap_edge_mc_error(
                    boot_samples[edge.child_hash], boot_samples[edge.parent_hash], posterior, selection,
                    models[edge.child_hash], models[edge.parent_hash],
                    n_replicates=args.bootstrap_replicates, seed=args.seed, min_ess=args.min_ess,
                    labels=(edge.child_hash, edge.parent_hash), batch_size=args.batch_size,
                    catalogs=(catalogs[edge.child_hash], catalogs[edge.parent_hash]),
                )
            except InsufficientReweightingESSError as exc:
                refusal = str(exc)
        edges.append(
            edge_mc_report(
                predicted, empirical,
                extra={"parent_hash": edge.parent_hash, "child_hash": edge.child_hash,
                       "mutation_id": edge.mutation_id, "orientation": "ln BF = ln Z_child - ln Z_parent",
                       "empirical_refused": refusal},
            )
        )
    keys = sorted(weights)
    payload = {
        "format_version": "gwpop-search-edge-mc-error-campaign-1.0",
        "graph_root_hash": graph.root_hash,
        "n_draws": None if args.all_points else args.n_draws,
        "models": {h: weights[h].summary() for h in keys},
        "sigma_mc": {"model_hashes": keys, "matrix": mc_covariance_matrix([weights[h] for h in keys]).tolist()},
        "edges": edges,
    }
    _write(out / "edge_mc_error.json", payload)


def _analyze_prior_sensitivity(args) -> None:
    from ._common import pool_dynesty_results
    from .model_comparison import evidence_summaries_from_results
    from .prior_sensitivity import edge_prior_sensitivity, model_prior_variants, prior_sensitivity_report
    from .structure import find_alias_groups

    graph, grouped = _load_graph_and_results(args)
    evidences = evidence_summaries_from_results(grouped)
    pooled = {h: pool_dynesty_results(r) for h, r in grouped.items()}
    edges = []
    for edge in _edges_with(graph, grouped):
        lnbf = evidences[edge.child_hash].log_evidence - evidences[edge.parent_hash].log_evidence
        row = edge_prior_sensitivity(
            graph.by_hash[edge.parent_hash], graph.by_hash[edge.child_hash],
            pooled[edge.parent_hash], pooled[edge.child_hash], log_bayes_factor=lnbf, factor=args.factor,
            edge_fraction=args.edge_fraction, max_edge_mass=args.max_edge_mass,
        )
        row["mutation_id"] = edge.mutation_id
        edges.append(row)
    variants = []
    _enable_x64()
    aliases = find_alias_groups(graph.nodes)
    heads = {aliases.canonical[h] for h in graph.by_hash}
    if heads <= set(evidences):
        specs = {h: graph.by_hash[h] for h in heads}
        variants = model_prior_variants(
            graph.by_hash[graph.root_hash], specs, {h: evidences[h].log_evidence for h in heads},
            lambdas=_float_list(args.lambdas),
        )
    _write(args.output, prior_sensitivity_report(
        edges, variants,
        extra={"graph_root_hash": graph.root_hash,
               "model_prior_variants_note": None if variants else
               "model-prior variants need evidence for every effective model of the graph"},
    ))


def _analyze_sddr(args) -> None:
    from gwpop_search.inference import prior_specs_from_model_spec

    from ._common import pool_dynesty_results
    from .model_comparison import evidence_summaries_from_results
    from .sddr import SDDR_FORMAT, classify_graph_edges, sddr_edge_check

    graph, grouped = _load_graph_and_results(args)
    evidences = evidence_summaries_from_results(grouped)
    rows = []
    for nesting in classify_graph_edges(graph):
        larger = nesting.child_hash if nesting.larger_model == "child" else nesting.parent_hash
        smaller = nesting.parent_hash if nesting.larger_model == "child" else nesting.child_hash
        if not nesting.sddr_eligible:
            rows.append({**nesting.to_dict(), "status": "not_applicable", "reason": nesting.reason})
            continue
        if larger not in grouped:
            rows.append({**nesting.to_dict(), "status": "missing", "reason": "no results for the larger model"})
            continue
        lnbf = sigma = None
        if nesting.parent_hash in evidences and nesting.child_hash in evidences:
            p, c = evidences[nesting.parent_hash], evidences[nesting.child_hash]
            lnbf = c.log_evidence - p.log_evidence
            sigma = math.hypot(p.ns_error(args.ns_error_mode), c.ns_error(args.ns_error_mode))
        check = sddr_edge_check(
            nesting, pool_dynesty_results(grouped[larger]),
            prior_specs_from_model_spec(graph.by_hash[larger]),
            prior_specs_from_model_spec(graph.by_hash[smaller]),
            log_bf_ns=lnbf, sigma_ns=sigma, n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        rows.append({**nesting.to_dict(), **check.to_dict()})
    _write(args.output, {"format_version": SDDR_FORMAT, "graph_root_hash": graph.root_hash, "edges": rows})


def _read_many(paths) -> list[dict]:
    from ._common import read_json

    return [read_json(p) for p in paths or []]


def _model_prior(args):
    from gwpop_search.production.runner import model_prior_from_config
    from gwpop_search.search import ComplexityModelPrior

    if args.penalty is not None:
        return ComplexityModelPrior(penalty_per_axis=float(args.penalty))
    if args.model_prior_json:
        return model_prior_from_config(json.loads(Path(args.model_prior_json).read_text()))
    if args.campaign:
        from gwpop_search.production import load_production_campaign

        return model_prior_from_config(load_production_campaign(Path(args.campaign)).model_prior)
    raise ValueError("model prior required: --penalty, --model-prior-json or --campaign")


def _analyze_model_comparison(args) -> None:
    _enable_x64()
    from gwpop_search.grammar import load_model_graph

    from ._common import discover_dynesty_results, read_json
    from .edge_mc_error import load_model_mc_weights
    from .model_comparison import (
        ClaimCriteria,
        ComparisonConfig,
        EvidenceSummary,
        NullCalibration,
        build_model_comparison,
        evidence_summaries_from_results,
        nearby_values_from_summaries,
        render_markdown,
        stress_values_from_summaries,
    )

    graph = load_model_graph(Path(args.graph))
    gates = None if not args.gates else {str(k): bool(v) for k, v in read_json(args.gates).items()}
    if args.evidence:
        payload = read_json(args.evidence)
        evidences = {
            h: EvidenceSummary.from_dict({**row, "gates_passed": (gates or {}).get(h, row.get("gates_passed"))}, h)
            for h, row in payload.items()
            if h != "format_version"
        }
    else:
        if not args.results:
            raise ValueError("evidence required: --results or --evidence")
        grouped = discover_dynesty_results(args.results, model_hashes=args.model_hash)
        evidences = evidence_summaries_from_results(grouped, gates=gates)
    mc_weights = None
    if args.mc_dir:
        mc_weights = {}
        for path in sorted(Path(args.mc_dir).glob("*.npz")):
            mc_weights[path.stem] = load_model_mc_weights(path)
    sddr = None
    if args.sddr:
        rows = read_json(args.sddr)["edges"]
        sddr = {f"{row['parent_hash']}->{row['child_hash']}": row for row in rows}
        mutation_ids = [row["mutation_id"] for row in rows]
        sddr.update({row["mutation_id"]: row for row in rows if mutation_ids.count(row["mutation_id"]) == 1})
    prior_sensitivity = None
    if args.prior_sensitivity:
        data = read_json(args.prior_sensitivity)
        prior_sensitivity = {(row["parent_hash"], row["child_hash"]): row for row in data["edges"]}
    nulls = []
    for payload in _read_many(args.null_calibration):
        rows = payload.get("nulls", [payload])
        nulls.extend(NullCalibration.from_dict(row) for row in rows)
    stress = stress_values_from_summaries(_read_many(args.stress_summary)) if args.stress_summary else None
    nearby = nearby_values_from_summaries(_read_many(args.nearby_summary)) if args.nearby_summary else None
    loo = read_json(args.loo) if args.loo else None
    config = ComparisonConfig(
        ns_error_mode=args.ns_error_mode,
        kappa_hat=args.kappa_hat,
        n_propagation=args.n_propagation,
        seed=args.seed,
        model_prior_lambdas=tuple(_float_list(args.lambdas)),
        criteria=ClaimCriteria(alpha=args.alpha),
    )
    report = build_model_comparison(
        graph, evidences, _model_prior(args), mc_weights=mc_weights, sddr=sddr,
        prior_sensitivity=prior_sensitivity, null_calibrations=nulls or None, stress=stress, nearby=nearby,
        loo=loo, config=config,
    )
    _write(args.output, report)
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(report))
        print(f"written: {args.markdown}")


def _run_psis_loo_influence(args) -> None:
    _enable_x64()
    from gwpop_search.models import compile_model_spec

    from .psis_loo import PSISLOOConfig, check_same_catalog, evaluate_model_loo_terms, psis_loo_model, psis_loo_report
    from .structure import find_alias_groups, membership_by_atom, normalized_log_prior

    posterior, selection = _load_data(args)
    graph, grouped = _load_graph_and_results(args)
    check_same_catalog(grouped)
    config = PSISLOOConfig(k_threshold=args.k_threshold, min_loo_ess=args.min_loo_ess,
                           max_complement_mass=args.max_complement_mass)
    results = {}
    for model_hash, runs in sorted(grouped.items()):
        terms = evaluate_model_loo_terms(runs, posterior, selection, compile_model_spec(graph.by_hash[model_hash]),
                                         label=model_hash, batch_size=args.batch_size)
        results[model_hash] = psis_loo_model(terms, config=config, forced_events=args.force_event or ())
    edges = [e.to_dict() for e in _edges_with(graph, grouped)]
    log_priors = membership = None
    aliases = find_alias_groups(graph.nodes)
    heads = [h for h in (m.model_hash for m in graph.nodes) if aliases.canonical[h] == h]
    if args.penalty is not None and set(heads) <= set(results):
        from gwpop_search.search import ComplexityModelPrior

        specs = {h: graph.by_hash[h] for h in heads}
        log_priors = normalized_log_prior(list(specs.values()), graph.by_hash[graph.root_hash],
                                          ComplexityModelPrior(float(args.penalty)))
        membership = membership_by_atom(graph.by_hash[graph.root_hash], specs)
        results = {h: results[h] for h in heads}
    report = psis_loo_report(results, edges=edges, log_model_priors=log_priors,
                             structure_membership=membership, extra={"graph_root_hash": graph.root_hash})
    _write(args.output, report)
    if args.write_exact_refit_config:
        from gwpop_search.search import Fidelity
        from gwpop_search.validation import (
            EventStressConfig,
            EventStressSuiteSpec,
            save_event_stress_suite_spec,
            scenarios_from_psis_flags,
        )

        scenarios = scenarios_from_psis_flags(report)
        if not scenarios:
            print("no flagged events: no exact-refit config written")
            return
        spec = EventStressSuiteSpec(
            scenarios=scenarios,
            config=EventStressConfig(stop_fidelity=Fidelity(args.refit_stop_fidelity)),
        )
        save_event_stress_suite_spec(Path(args.write_exact_refit_config), spec)
        print(f"written: {args.write_exact_refit_config} ({len(scenarios)} exact-refit scenario(s))")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_analysis_subcommands(subparsers) -> None:
    classify = subparsers.add_parser(
        "classify-graph-edges",
        help="classify every graph edge as exact-SDDR / approximate / evidence-only",
    )
    classify.add_argument("--graph", required=True)
    classify.add_argument("--verify", action="store_true", help="numerically verify every embedding")
    classify.add_argument("--output", required=True)
    classify.set_defaults(func=_classify_graph_edges)

    mc = subparsers.add_parser(
        "analyze-edge-mc-error",
        help="Monte-Carlo-likelihood error/bias of ln Z and ln BF from saved dynesty results",
    )
    _add_data_arguments(mc)
    _add_results_arguments(mc)
    mc.add_argument("--n-draws", type=int, default=4096,
                    help="pooled systematic-resampling draws for the weight averages")
    mc.add_argument("--all-points", action="store_true",
                    help="average over every weighted nested-sampling point instead of --n-draws draws")
    mc.add_argument("--bootstrap-replicates", type=int, default=0,
                    help="bootstrap-reweighting replicates per edge (0 = formula only)")
    mc.add_argument("--bootstrap-draws", type=int, default=2048)
    mc.add_argument("--min-ess", type=float, default=100.0)
    mc.add_argument("--batch-size", type=int, default=16)
    mc.add_argument("--seed", type=int, default=0)
    mc.add_argument("--output-dir", required=True)
    mc.set_defaults(func=_analyze_edge_mc_error)

    prior = subparsers.add_parser(
        "analyze-prior-sensitivity",
        help="prior-width variants per edge and model-prior penalty variants",
    )
    _add_results_arguments(prior)
    prior.add_argument("--factor", type=float, default=2.0)
    prior.add_argument("--edge-fraction", type=float, default=0.05)
    prior.add_argument("--max-edge-mass", type=float, default=1e-3)
    prior.add_argument("--lambdas", default="0,ln2,ln4")
    prior.add_argument("--output", required=True)
    prior.set_defaults(func=_analyze_prior_sensitivity)

    sddr = subparsers.add_parser("analyze-sddr", help="Savage-Dickey cross-checks of eligible edges")
    _add_results_arguments(sddr)
    sddr.add_argument("--n-bootstrap", type=int, default=200)
    sddr.add_argument("--ns-error-mode", choices=("formula6", "conservative"), default="formula6")
    sddr.add_argument("--seed", type=int, default=0)
    sddr.add_argument("--output", required=True)
    sddr.set_defaults(func=_analyze_sddr)

    report = subparsers.add_parser(
        "analyze-model-comparison",
        help="ln BF error budget, model probabilities, structural masses and claim criteria",
    )
    _add_results_arguments(report, required=False)
    report.add_argument("--evidence", help="JSON {model_hash: evidence summary} instead of --results")
    report.add_argument("--penalty", type=float, help="axis-complexity model-prior penalty")
    report.add_argument("--model-prior-json", help="frozen model-prior configuration JSON")
    report.add_argument("--campaign", help="production campaign JSON (its model prior is used)")
    report.add_argument("--mc-dir", help="directory of ModelMCWeights npz files (analyze-edge-mc-error)")
    report.add_argument("--sddr", help="analyze-sddr output")
    report.add_argument("--prior-sensitivity", help="analyze-prior-sensitivity output")
    report.add_argument("--null-calibration", action="append",
                        help="JSON {label, statistic, observed, exceedances, replays} (or {nulls: [...]}); repeatable")
    report.add_argument("--stress-summary", action="append", help="event-stress suite summary; repeatable")
    report.add_argument("--nearby-summary", action="append", help="nearby-baseline suite summary; repeatable")
    report.add_argument("--loo", help="run-psis-loo-influence output")
    report.add_argument("--gates", help="JSON {model_hash: production numerical gates passed}")
    report.add_argument("--ns-error-mode", choices=("formula6", "conservative"), default="formula6")
    report.add_argument("--kappa-hat", type=float)
    report.add_argument("--alpha", type=float, default=0.01, help="pre-registered null-calibration level")
    report.add_argument("--lambdas", default="0,ln2,ln4")
    report.add_argument("--n-propagation", type=int, default=4000)
    report.add_argument("--seed", type=int, default=0)
    report.add_argument("--output", required=True)
    report.add_argument("--markdown")
    report.set_defaults(func=_analyze_model_comparison)

    loo = subparsers.add_parser(
        "run-psis-loo-influence",
        help="PSIS leave-one-out influence of every event on evidences and edge Bayes factors",
    )
    _add_data_arguments(loo)
    _add_results_arguments(loo)
    loo.add_argument("--k-threshold", type=float, default=0.7)
    loo.add_argument("--min-loo-ess", type=float, default=100.0)
    loo.add_argument("--max-complement-mass", type=float, default=0.01)
    loo.add_argument("--force-event", action="append", help="always flag this event (e.g. F0 zero support)")
    loo.add_argument("--penalty", type=float, help="axis-complexity penalty for LOO model probabilities")
    loo.add_argument("--batch-size", type=int, default=64)
    loo.add_argument("--write-exact-refit-config", help="write an event-stress suite for flagged events")
    loo.add_argument("--refit-stop-fidelity", choices=("F3", "F4"), default="F3")
    loo.add_argument("--output", required=True)
    loo.set_defaults(func=_run_psis_loo_influence)
