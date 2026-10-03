#!/usr/bin/env python
"""Gate b-0(i), the noisy-PE ensemble check, on an already built v2 mock (read-only).

    python scripts/v2_mock_ensemble_gate.py --mock-dir <mock> --output <json>
        [--n-catalogs 30] [--n-pe 2048] [--workers 8]

Operator decision 2026-10-02: K catalogs from the mock's truth and generator
(its calibration.json: a generator-1.0 mock is compared with 1.0 catalogs),
each with noisy PE, are analysed with the mock's selection set; the realised
catalog is a *tail catalog* if its joint conditional MLE of the chi_eff
hyperparameters is more than 2.5 ensemble sd from the ensemble median in any
parameter. See ``gwpop_search.validation.v2_mock.noisy_pe_ensemble_gate``.

Nothing is written inside the mock directory; the output must lie elsewhere
(and not under ``frozen/`` or ``v2r2``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mock-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-catalogs", type=int, default=30)
    parser.add_argument("--n-pe", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1, help="threads for the per-catalog PE and fits")
    args = parser.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    from gwpop_search.validation.v2_mock import ensemble_gate_for_mock, refuse_protected

    out = Path(args.output).resolve()
    mock = Path(args.mock_dir).resolve()
    refuse_protected(out)
    if mock in out.parents:
        raise SystemExit(f"refusing to write inside the mock directory: {out}")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    result = ensemble_gate_for_mock(mock, n_catalogs=args.n_catalogs, n_pe=args.n_pe, workers=args.workers,
                                    log=log)
    result["wall_time_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tail_catalog": result["tail_catalog"], "unbiased": result["unbiased"]["pass"],
                      "position": {k: round(v["n_sd_from_median"], 3) for k, v in result["position"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
