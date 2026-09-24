#!/usr/bin/env python
"""Write the chi_eff follow-up model specs on top of the frozen GWTC-5 root.

Usage::

    python scripts/write_followup_specs.py \
        --graph /hildafs/projects/phy220048p/magana/gwpop-search-data/frozen/gwtc5-bbh-v1a/model_graph.json \
        --output followup_specs

The frozen graph is only read.
"""

from __future__ import annotations

import argparse
import json

from gwpop_search.grammar.followup import load_frozen_root, write_followup_specs

DEFAULT_GRAPH = (
    "/hildafs/projects/phy220048p/magana/gwpop-search-data/frozen/"
    "gwtc5-bbh-v1a/model_graph.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--output", default="followup_specs")
    args = parser.parse_args()
    manifest = write_followup_specs(load_frozen_root(args.graph), args.output)
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
