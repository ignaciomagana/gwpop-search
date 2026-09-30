#!/usr/bin/env python
"""Write the DRAFT v2 (GWTC-5 atom search) model graph.

    python scripts/write_v2_model_graph.py --output <dir>/model_graph_v2.json
    python scripts/write_v2_model_graph.py --output ... --depth2-passing C2 S2 C4

Without ``--depth2-passing`` the graph is the depth-1 graph (root R0 + the 19
atoms: 20 nodes, 19 edges); the depth-2 slots are conditional on the depth-1
results and are only added once the passing edges are known. The JSON is
``load_model_graph`` compatible and carries a ``metadata`` block (status DRAFT,
atom ids, draft priors, depth-2 rule, alternative roots) and the G12 model
checks. It refuses to write under a ``frozen`` directory.
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
    parser.add_argument("--pe", default=None, help="gwcat PE export: its z_max attr enters the G12 check")
    parser.add_argument("--selection", default=None,
                        help="gwcat selection export: its z_max attr enters the G12 check")
    args = parser.parse_args(argv)
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
        plan = plan_depth2(args.depth2_passing)
        graph = extend_graph_with_depth2(graph, plan)
    payload = v2_graph_payload(graph, depth2=plan)
    payload["metadata"]["g12_model_checks"] = v2_model_graph_checks(graph, export_zmax=export_zmax)
    payload["metadata"]["g12_model_checks"]["export_files"] = {
        name: str(Path(path).resolve()) for name, path in (("pe", args.pe), ("selection", args.selection)) if path
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
    }, indent=2))


if __name__ == "__main__":
    main()
