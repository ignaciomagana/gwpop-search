#!/usr/bin/env python
"""Write the DRAFT v2 (GWTC-5 atom search) model graph.

    python scripts/write_v2_model_graph.py --output <dir>/model_graph_v2.json
    python scripts/write_v2_model_graph.py --output ... --depth2-passing C2 S2 C4 \
        [--depth2-d2-passing C2 S2 C4 C6] [--depth2-score C2=19.5 --depth2-score C4=8.2 ...]

Without ``--depth2-passing`` the graph is the depth-1 graph (root R0 + the 18
atoms: 19 nodes, 18 edges; the Student-t S4 was dropped on 2026-10-02 and is
listed under ``metadata.dropped_atoms`` only); the depth-2 slots are conditional on the depth-1
results and are only added once the passing edges are known. The JSON is
``load_model_graph`` compatible and carries a ``metadata`` block (status DRAFT,
atom ids, draft priors with their decision status, the operator decisions of
2026-10-01, the pivot / kappa(m1) conventions, depth-2 rule, alternative
roots) and the G12 model checks. It refuses to write under a ``frozen``
directory.

Hashes: since the operator decisions of 2026-10-01 (C3/C4 pivot z = 0.5; Z2 and
root A2 in the local mass function convention) the hashes of C3, C4, Z2 (= A2)
and of every model built on them differ from the graphs written by fd73da8
(the pilot code); R0 and the other 16 depth-1 nodes are unchanged
(``metadata.operator_decisions.hash_changes``). The pilot (b) C4 runs used the
z = 0 pivot (a reparameterisation; the pilot tests the machinery).

Since the operator decisions of 2026-10-02 the chi_eff - q atoms C1 / C2 pivot
at q = 0.7, so their hashes (and those of every model built on them) differ
from afd5f53 and earlier (``metadata.operator_decisions.later_decisions
["2026-10-02"].hash_changes``); the C1 / C2 node metadata carry the pre-declared
reported quantities (C2: sigma(q = 0.7) headline; slope and sigma(1) secondary,
with the PE-resolution limitation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gwpop_search.grammar import load_model_graph
from gwpop_search.grammar.v2 import (
    enumerate_v2_depth1,
    extend_graph_with_depth2,
    plan_depth2,
    v2_graph_payload,
    v2_model_graph_checks,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth2-passing", nargs="*", default=None,
                        help="atom ids passing D1 + D2 at depth 1 (adds the depth-2 slots)")
    parser.add_argument("--depth2-d2-passing", nargs="*", default=None,
                        help="atom ids passing D2 at depth 1 (default: --depth2-passing); every pair of "
                        "its chi_eff atoms is a mandatory attribution pair")
    parser.add_argument("--depth2-score", action="append", default=[], metavar="ATOM=LOWER",
                        help="depth-1 D2 lower bound ln BF - 2 sigma_total - |bias| of an atom; ranks the "
                        "chi_eff atoms (pairs with the strongest first); repeatable")
    parser.add_argument("--pe", default=None, help="gwcat PE export: its z_max attr enters the G12 check")
    parser.add_argument("--selection", default=None,
                        help="gwcat selection export: its z_max attr enters the G12 check")
    parser.add_argument("--canonical-pe", default=None,
                        help="canonical PE HDF5 (adapter output): every node's support is checked "
                        "against it (G12 model + data; with --canonical-selection)")
    parser.add_argument("--canonical-selection", default=None,
                        help="canonical selection HDF5 (adapter output)")
    parser.add_argument("--v2-policy", default=None,
                        help="v2 data policy JSON: its z_max enters the data-support check")
    args = parser.parse_args(argv)
    if (args.canonical_pe is None) != (args.canonical_selection is None):
        raise SystemExit("pass both --canonical-pe and --canonical-selection, or neither")
    export_zmax = {}
    for name, path in (("pe", args.pe), ("selection", args.selection)):
        if path:
            import h5py

            with h5py.File(path, "r") as handle:  # attrs only, read-only
                export_zmax[name] = float(handle.attrs["z_max"])
    out = Path(args.output).resolve()
    if "frozen" in out.parts:
        raise SystemExit("refusing to write a DRAFT graph under a frozen/ directory")
    graph = enumerate_v2_depth1()
    plan = None
    if args.depth2_passing is not None:
        scores = {}
        for item in args.depth2_score:
            atom, sep, value = str(item).partition("=")
            if not sep:
                raise SystemExit(f"--depth2-score expects ATOM=LOWER; got {item!r}")
            scores[atom] = float(value)
        plan = plan_depth2(args.depth2_passing, chieff_d2_passing=args.depth2_d2_passing, scores=scores)
        graph = extend_graph_with_depth2(graph, plan)
    payload = v2_graph_payload(graph, depth2=plan)
    payload["metadata"]["g12_model_checks"] = v2_model_graph_checks(graph, export_zmax=export_zmax)
    payload["metadata"]["g12_model_checks"]["export_files"] = {
        name: str(Path(path).resolve()) for name, path in (("pe", args.pe), ("selection", args.selection)) if path
    }
    if args.canonical_pe:
        import jax

        jax.config.update("jax_enable_x64", True)
        from gwpop_search.data import PosteriorCatalog, SelectionCatalog
        from gwpop_search.models.data_support import v2_data_support_report

        policy = None
        if args.v2_policy:
            from gwpop_search.data.v2_policy import GwcatV2DataPolicy

            policy = GwcatV2DataPolicy.from_json(args.v2_policy)
        posterior = PosteriorCatalog.from_hdf5(args.canonical_pe)
        selection = SelectionCatalog.from_hdf5(args.canonical_selection)
        reports = [v2_data_support_report(node, posterior, selection, policy=policy) for node in graph.nodes]
        payload["metadata"]["g12_data_support"] = {
            "pass": all(r["pass"] for r in reports),
            "files": {"pe": str(Path(args.canonical_pe).resolve()),
                      "selection": str(Path(args.canonical_selection).resolve()),
                      "v2_policy": None if not args.v2_policy else str(Path(args.v2_policy).resolve())},
            "models": {r["model_hash"]: {"pass": r["pass"], "checks": r["checks"], "reported": r["reported"]}
                       for r in reports},
        }
    try:
        from gwpop_search.inference.numpyro import _code_identity

        payload["metadata"]["code_identity"] = _code_identity()
    except Exception as exc:  # pragma: no cover - provenance only
        payload["metadata"]["code_identity"] = {"error": repr(exc)}
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    out.write_text(text)
    loaded = load_model_graph(out)  # verifies every stored hash and edge
    print(json.dumps({
        "output": str(out),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "nodes": len(loaded.nodes),
        "edges": len(loaded.edges),
        "root_hash": loaded.root_hash,
        "g12_model_checks_pass": payload["metadata"]["g12_model_checks"]["pass"],
        "g12_data_support_pass": (payload["metadata"].get("g12_data_support") or {}).get("pass"),
    }, indent=2))


if __name__ == "__main__":
    main()
