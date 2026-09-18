# Phase 3 synthetic recovery runbook

Phase 3 is accepted only after the conventional baseline model recovers from
multiple independent closed synthetic catalogs under the same standardized HBI
engine that will later score GWTC-5 models.

## Purpose

This campaign exercises, end to end:

1. normalized source population components;
2. source-to-detector-frame cosmology/Jacobian transform;
3. PE importance reweighting;
4. raw-draw selection normalization;
5. JAX likelihood compilation;
6. NumPyro NUTS;
7. deterministic data/sampler/chain seeds;
8. chain-granularity checkpoint/restart;
9. multi-catalog numerical acceptance and recovery summaries.

It is not a detector realism study and is not used for an astrophysical claim.

## Recommended Phase-3 command

Use the campaign entry point, not a hand-written collection of single runs:

```bash
gwpop-search synthetic-campaign \
  --root runs/phase3-recovery/default \
  --n-runs 4 \
  --root-seed 20260917 \
  --n-events 48 \
  --pe-samples 256 \
  --n-injections 20000 \
  --num-warmup 1000 \
  --num-samples 1000 \
  --num-chains 4 \
  --target-accept 0.9 \
  --selection-chunk-size 4096 \
  --no-progress
```

The root seed deterministically generates independent data and sampler seeds for
all campaign members. Re-running the identical command resumes completed chains
and completed runs. Changing the campaign plan causes the manifest check to fail
rather than silently mixing configurations.

A single recovery is still available for debugging:

```bash
gwpop-search synthetic-recovery \
  --run-dir runs/phase3-recovery/debug \
  --data-seed 20260917 \
  --sampler-seed 20260918
```

## Outputs

```text
<root>/
    campaign_plan.json
    campaign_summary.json
    run_000/
        chains/
            manifest.json
            chain_000.npz
            chain_000.npz.json
            ...
        posterior.npz
        posterior.npz.json
        recovery_summary.json
    run_001/
    ...
```

Each recovery summary contains the injected truth, posterior 5/50/95 percent
quantiles, divergence count, split R-hat/effective-sample-size summaries,
catalog/selection sizes, and HBI importance diagnostics at both the injected
truth and posterior median.

The campaign summary applies the declared numerical gates to every run and
reports ensemble central-90%-interval coverage fractions and standardized
offsets. With only a few catalogs, those coverage numbers are diagnostics, not
a calibrated coverage measurement. The code deliberately does not turn them
into an automatic pass/fail threshold.

For the mandatory interruption/resume check, fingerprint completed chain
payloads plus metadata before and after resubmitting the identical campaign:

```bash
gwpop-search fingerprint-recovery-checkpoint \
  --run-dir runs/phase3-recovery/default/run_000 \
  > fingerprints_before.json

# interrupt/re-submit the exact same campaign, then:
gwpop-search fingerprint-recovery-checkpoint \
  --run-dir runs/phase3-recovery/default/run_000 \
  > fingerprints_after.json
```

Every key present before the resume must retain exactly the same fingerprint.
Newly completed chains may add keys.

To reassess already completed runs without launching inference:

```bash
gwpop-search assess-synthetic-campaign \
  --root runs/phase3-recovery/default \
  --min-runs 4
```

## H100 / Slurm

A site-neutral template is provided at:

```text
scripts/slurm/phase3_synthetic_h100.sbatch.example
```

The template does not guess a partition, account, CUDA module, or JAX wheel. It
refuses to start unless JAX sees a GPU.

Example:

```bash
export GWPOP_ENV=/path/to/venv/bin/activate
export GWPOP_RUN_DIR=/persistent/path/gwpop-phase3
sbatch scripts/slurm/phase3_synthetic_h100.sbatch.example
```

## Phase-3 acceptance gate

Before calling Phase 3 accepted:

- run at least four independent catalogs with at least four NUTS chains each;
- require the numerical campaign gate to pass for every accepted run;
- inspect posterior recovery across the ensemble rather than requiring every
  truth to fall inside one arbitrary interval in every catalog;
- interrupt and resume at least one campaign and verify completed chain
  fingerprints are unchanged;
- investigate any systematic standardized offset or repeated importance-sampling
  pathology before production.

The repository may contain staged Phase-4--10 software while this scientific
gate remains open. Staged downstream code is not evidence that Phase 3 passed.


## H100 handover

The authoritative end-to-end execution order, including the manual Phase-3
decision, Phase-7 scout validation, real-data freeze, production search,
robustness suites, and exact-search null calibration is:

```text
docs/CLAUDE_H100_VALIDATION_HANDOVER.md
docs/H100_VALIDATION_REPORT_TEMPLATE.md
```
