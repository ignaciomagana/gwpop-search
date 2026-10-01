#!/usr/bin/env python
"""End-to-end CPU smoke test of the v2 pipeline on a tiny MOCK catalog.

    JAX_PLATFORMS=cpu python scripts/v2_smoke_e2e.py --output-dir <staging dir>

Runs, through the package CLI exactly as production would (only the numbers
are shrunk):

1. mock catalog (``scripts/v2_smoke_mock.py``): few events, thinned injections,
   in the canonical sky-marginal gwcat-v2 basis;
2. enumeration: the v2 depth-1 graph (``scripts/write_v2_model_graph.py``,
   20 nodes, 19 edges) with the G12 model + data-support checks on the mock;
3. dataset manifest and production-campaign freeze (``freeze-dataset``,
   ``freeze-production-campaign --require-root-profile gwtc5-v2``) with the v2
   fidelity configuration (taper at sigma^2 = 1 inside the likelihood) and a
   SMOKE-sized nested sampler (small nlive);
4. F0 and one short F3 dynesty evaluation per model for the root R0 and one
   atom (C2, ln-width linear in q) via ``run-fidelity-evaluation``, and the
   same two F3 runs under the taper-at-2 sensitivity configuration (D3);
5. analysis: ``collect-v2-evaluations`` (D1 gates + taper mass),
   ``analyze-edge-mc-error``, ``analyze-sddr``, ``analyze-prior-sensitivity``,
   ``run-psis-loo-influence``, ``analyze-model-comparison``, ``run-ppc``,
   ``reweight-data-variants`` and ``write-v2-alt-root-config`` (D5 config only);
6. ``v2-claim-table`` (D1-D6).

Mock data only; no real-data nested sampling. The smoke numerics are NOT the
v2 production numerics (nlive, repeats and the Kish-ESS floor are shrunk) and
no scientific conclusion follows from the claim table it writes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CHILD_ATOM = "C2"


def _run(log, *argv, cwd=None) -> str:
    start = time.perf_counter()
    cmd = [sys.executable, *map(str, argv)]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=os.environ.copy())
    elapsed = time.perf_counter() - start
    log.append({"cmd": cmd[1:], "returncode": proc.returncode, "seconds": round(elapsed, 2)})
    tail = proc.stdout[-4000:]
    print(f"$ {' '.join(map(str, argv))[:220]}\n  -> rc={proc.returncode} ({elapsed:.1f} s)")
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-6000:])
        raise SystemExit(f"step failed: {argv[:3]}")
    return tail


def _cli(log, *argv) -> str:
    return _run(log, "-m", "gwpop_search.cli", *argv)


def _freeze_campaign(log, freeze: Path, graph: Path, fidelity: Path, campaign: Path, campaign_id: str) -> None:
    _cli(log, "freeze-production-campaign", "--manifest", freeze / "manifest.json", "--graph", graph,
         "--fidelity-config", fidelity, "--campaign-id", campaign_id, "--beam-width", 2,
         "--model-prior", "axis-complexity", "--model-prior-penalty", 0.6931471805599453,
         "--exploration-quota", 0, "--scheduler-seed", 1, "--root-seed", 20260930,
         "--max-gpu-hours", 1, "--max-f3-models", 2, "--max-f4-models", 1, "--max-null-replays", 1,
         "--artifact-root", "artifacts", "--state-database", "state.sqlite",
         "--require-root-profile", "gwtc5-v2", "--output", campaign)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-events", type=int, default=10)
    parser.add_argument("--n-pe", type=int, default=256)
    parser.add_argument("--n-draw", type=int, default=100_000)
    parser.add_argument("--nlive", type=int, default=48)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--ppc-draws", type=int, default=1000)  # D6: n * alpha / 2 >= 5
    parser.add_argument("--ignore-current-commit", action="store_true",
                        help="development only: accept a dirty tree / a commit other than the frozen one")
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    if "frozen" in out.parts:
        raise SystemExit("refusing to write under a frozen/ directory")
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    ignore = ["--ignore-current-commit"] if args.ignore_current_commit else []

    # 1. mock -------------------------------------------------------------
    mock = out / "mock"
    _run(log, HERE / "v2_smoke_mock.py", "--output-dir", mock, "--n-events", args.n_events,
         "--n-pe", args.n_pe, "--n-draw", args.n_draw, "--seed", 20260930)
    pe, sel = mock / "pe.h5", mock / "selection.h5"

    # 2. enumeration --------------------------------------------------------
    graph = out / "model_graph_v2_smoke.json"
    _run(log, HERE / "write_v2_model_graph.py", "--output", graph,
         "--canonical-pe", pe, "--canonical-selection", sel)
    payload = json.loads(graph.read_text())
    root = payload["root_hash"]
    child_mutation = payload["metadata"]["atoms"][CHILD_ATOM]["mutation_id"]
    child = next(e["child_hash"] for e in payload["edges"]
                 if e["parent_hash"] == root and e["mutation_id"] == child_mutation)
    assert len(payload["nodes"]) == 20 and len(payload["edges"]) == 19
    assert payload["metadata"]["g12_data_support"]["pass"] is True

    # 3. freezes (staging only) ----------------------------------------------
    freeze = out / "freeze"
    freeze.mkdir(exist_ok=True)
    (freeze / "event_selection.json").write_text(json.dumps({"mock": True, "rule": "snr_proxy > 10"}))
    (freeze / "waveform_policy.json").write_text(json.dumps({"mock": True}))
    _cli(log, "freeze-dataset", "--pe", pe, "--selection", sel, "--dataset-id", "v2-smoke-mock",
         "--event-selection-json", freeze / "event_selection.json",
         "--waveform-policy-json", freeze / "waveform_policy.json", "--output", freeze / "manifest.json")
    fidelity = freeze / "fidelity_v2_SMOKE.json"
    fidelity2 = freeze / "fidelity_v2_taper2_SMOKE.json"
    smoke_config = (
        "import sys, dataclasses as d\n"
        "from gwpop_search.inference.v2_numerics import v2_fidelity_run_config\n"
        "from gwpop_search.inference.fidelity import save_fidelity_run_config\n"
        "c = v2_fidelity_run_config(float(sys.argv[2]))\n"
        f"dy = d.replace(c.f3_evidence.dynesty, nlive={args.nlive}, maxcall={args.maxcall!r}, num_posterior_samples=1000)\n"
        "c = d.replace(c, f0=d.replace(c.f0, prior_draws=256),\n"
        "    f3_evidence=d.replace(c.f3_evidence, dynesty=dy),\n"
        "    f4_evidence=d.replace(c.f4_evidence, dynesty=dy),\n"
        "    f3_criteria=d.replace(c.f3_criteria, min_kish_ess_per_run=50.0),\n"
        "    f4_criteria=d.replace(c.f4_criteria, min_kish_ess_per_run=50.0))\n"
        "assert c.hbi.variance_taper is not None and c.hbi.variance_taper.threshold == float(sys.argv[2])\n"
        "save_fidelity_run_config(sys.argv[1], c)\n"
    )
    _run(log, "-c", smoke_config, fidelity, 1.0)
    _run(log, "-c", smoke_config, fidelity2, 2.0)
    campaign, campaign2 = freeze / "campaign.json", freeze / "campaign_taper2.json"
    for cfg, camp, cid in ((fidelity, campaign, "v2-smoke"), (fidelity2, campaign2, "v2-smoke-taper2")):
        _freeze_campaign(log, freeze, graph, cfg, camp, cid)

    # 4. evaluations -------------------------------------------------------
    runs, runs2 = out / "runs", out / "runs_taper2"
    for camp, root_dir, rungs in ((campaign, runs, ("F0", "F3")), (campaign2, runs2, ("F3",))):
        common = ["--manifest", freeze / "manifest.json", "--graph", graph, "--campaign", camp,
                  "--base-dir", freeze, "--output-root", root_dir, "--model-hash", root,
                  "--model-hash", child, *ignore]
        for rung in rungs:
            _cli(log, "run-fidelity-evaluation", "--fidelity", rung, *common)
    f3 = runs / "F3"

    # 5. analysis ------------------------------------------------------------
    ana = out / "analysis"
    ana.mkdir(exist_ok=True)
    data = ["--manifest", freeze / "manifest.json", "--base-dir", freeze]
    res = ["--graph", graph, "--results", f3]
    _cli(log, "collect-v2-evaluations", "--evaluations", f3, "--gates-output", ana / "gates.json",
         "--taper-mass-output", ana / "taper_mass.json")
    _cli(log, "analyze-edge-mc-error", *data, *res, "--n-draws", 512, "--bootstrap-replicates", 16,
         "--bootstrap-draws", 512, "--min-ess", 20, "--output-dir", ana / "mc")
    _cli(log, "analyze-sddr", *res, "--n-bootstrap", 50, "--output", ana / "sddr.json")
    _cli(log, "analyze-prior-sensitivity", *res, "--output", ana / "prior_sensitivity.json")
    _cli(log, "run-psis-loo-influence", *data, *res, "--min-loo-ess", 20, "--output", ana / "loo.json")
    _cli(log, "analyze-model-comparison", *res, "--campaign", campaign, "--mc-dir", ana / "mc" / "mc_weights",
         "--sddr", ana / "sddr.json", "--prior-sensitivity", ana / "prior_sensitivity.json",
         "--loo", ana / "loo.json", "--gates", ana / "gates.json", "--n-propagation", 1000,
         "--output", ana / "model_comparison.json", "--markdown", ana / "model_comparison.md")
    _cli(log, "run-ppc", *data, *res, "--n-draws", args.ppc_draws, "--no-draws", "--output", ana / "ppc.json")
    events = json.loads((mock / "mock_truth.json").read_text())
    variants = {"variants": [{"variant_id": "drop2", "drop_events": ["MOCK000", "MOCK001"],
                              "description": "smoke: drop two events (stands in for the 249-event variant)"}]}
    (ana / "variants.json").write_text(json.dumps(variants))
    _cli(log, "reweight-data-variants", *data, *res, "--variants", ana / "variants.json",
         "--output", ana / "data_variants.json")
    _cli(log, "write-v2-alt-root-config", "--scenario-id", "v2-D5-A1-smoke", "--alt-root", "A1",
         "--candidate", child_mutation, "--output", ana / "d5_A1_config.json")

    # 6. claim table ---------------------------------------------------------
    _cli(log, "v2-claim-table", "--report", ana / "model_comparison.json", "--graph", graph,
         "--sddr", ana / "sddr.json", "--prior-sensitivity", ana / "prior_sensitivity.json",
         "--ppc", ana / "ppc.json", "--loo", ana / "loo.json", "--evaluations", f3,
         "--taper2-evaluations", runs2 / "F3",
         "--output", out / "claims_v2_SMOKE.json", "--markdown", out / "claims_v2_SMOKE.md")

    def _digest(e):
        d = e["diagnostics"]
        return {
            "passed": d["passed"],
            "log_evidence": d["evidence"]["log_evidence_mean"],
            "log_evidence_error": d["evidence"]["conservative_error"],
            "taper": (d.get("taper") or {}).get("pooled"),
            "v2_data_support_pass": (d.get("v2_data_support") or {}).get("pass"),
            "elapsed_seconds": e["elapsed_seconds"],
        }

    evaluations = {h: _digest(json.loads((f3 / h / "evaluation.json").read_text())) for h in (root, child)}
    evaluations2 = {h: _digest(json.loads((runs2 / "F3" / h / "evaluation.json").read_text()))
                    for h in (root, child)}
    summary = {
        "format_version": "gwpop-search-v2-smoke-e2e-1.0",
        "status": "SMOKE (mock data, shrunk numerics; no scientific content)",
        "root_hash": root, "child_hash": child, "child_atom": CHILD_ATOM, "child_mutation": child_mutation,
        "mock": events,
        "evaluations": evaluations,
        "evaluations_taper2": evaluations2,
        "claims": json.loads((out / "claims_v2_SMOKE.json").read_text()).get("edges"),
        "steps": log,
    }
    (out / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "root_hash", "child_hash", "evaluations",
                                             "evaluations_taper2")}, indent=2))


if __name__ == "__main__":
    main()
