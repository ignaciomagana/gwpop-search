#!/usr/bin/env python
"""DAG-consistent v2 closure / null mock (PILOT_PLAN section (b)).

    python scripts/v2_closure_mock.py --output-dir <dir> [--kind closure_widthq|null]
        [--n-events 259] [--n-pe 8192] [--target-found 1500000] [--seed <declared>]
        [--delta-pe] [--calibration <dir>/calibration.json] [--no-gates]

Writes ``<dir>/pe.h5`` and ``<dir>/selection.h5`` in the canonical
sky-marginal gwcat-v2 basis, plus ``mock_truth.json`` (truth hyperparameters
and per-event truths), ``calibration.json``, ``gates.json`` (generator gates
b-0: delta-PE C2-slope ensemble machinery check and the realised catalog's
representativeness, draw-support and PE-box coverage, G12, and sigma^2_lnL at the
truth against the v2 sharp cut) and
``mock_manifest.json`` (truths, seeds, sizes, DAG description, sha256 of every
file).

Truths: R0 at the LVK GWTC-5 BP2P fiducial medians (PILOT_PLAN reference
table) with chi_eff mu = 0.03 and
* ``closure_widthq`` (C2): ln sigma linear in q, sigma(1) = 0.05, slope -2.2;
* ``null`` (R0): constant sigma = 0.09 (the detected RMS of the closure width).

DAG (mock-data-dag): each system has one run label, projection and noise
realisation; detection is rho_obs > 10 on the data, identical for injections
and events; the events are a p_pop/p_draw-weighted resample of detected
systems; the PE shares the detection datum rho_obs and has widths derived from
it only. The mock detection is calibrated per run to the canonical selection
(median z, detection share) and the PE widths to the real catalog. See
``gwpop_search.validation.v2_mock`` for the derivations.

Refuses to write under a ``frozen/`` or ``v2r2`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", choices=("closure_widthq", "null"), default="closure_widthq")
    parser.add_argument("--n-events", type=int, default=259)
    parser.add_argument("--n-pe", type=int, default=8192)
    parser.add_argument("--target-found", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, required=True,
                        help="master seed; production seeds are pre-declared in staging/v2/mocks/SEED_DECLARATION.json")
    parser.add_argument("--delta-pe", action="store_true",
                        help="delta-function PE at the truth (one sample per event): gate (i) input")
    parser.add_argument("--calibration", default=None,
                        help="reuse a calibration.json (else recalibrate; deterministic, fixed seed)")
    parser.add_argument("--n-pool", type=int, default=None, help="event-pool draws (default n_draw / 2)")
    parser.add_argument("--ensemble", type=int, default=200,
                        help="number of delta-PE catalogs in the gate-(i) ensemble")
    parser.add_argument("--no-gates", action="store_true")
    args = parser.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    from gwpop_search.validation.v2_mock import Calibration, build_mock, refuse_protected

    out = Path(args.output_dir).resolve()
    refuse_protected(out)
    cal = None
    if args.calibration:
        cal = Calibration.from_dict(json.loads(Path(args.calibration).read_text()))
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    manifest = build_mock(out, kind=args.kind, n_events=args.n_events, n_pe=args.n_pe,
                          target_found=args.target_found, seed=args.seed, delta_pe=args.delta_pe,
                          calibration=cal, calibration_source=args.calibration,
                          n_pool=args.n_pool, run_gates=not args.no_gates,
                          ensemble=args.ensemble, log=log)
    print(json.dumps({k: manifest[k] for k in ("kind", "delta_pe", "sizes", "gates_b0_pass", "files_sha256")},
                     indent=2, sort_keys=True))
    if manifest.get("gates_b0_pass") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
