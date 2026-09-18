# Phase 3 synthetic recovery runbook

Phase 3 is accepted only after the conventional baseline model recovers from a
closed synthetic catalog under the same standardized HBI engine that will later
score GWTC-5 models.

## Purpose

This campaign exercises, end to end:

1. normalized source population components;
2. source-to-detector-frame cosmology/Jacobian transform;
3. PE importance reweighting;
4. raw-draw selection normalization;
5. JAX likelihood compilation;
6. NumPyro NUTS;
7. deterministic chain seeds;
8. chain-granularity checkpoint/restart.

It is not a detector realism study and is not used for an astrophysical claim.

## Local command

With an inference-capable environment:

```bash
gwpop-search synthetic-recovery \
  --run-dir runs/phase3-recovery/seed-20260917 \
  --data-seed 20260917 \
  --sampler-seed 20260918 \
  --n-events 48 \
  --pe-samples 256 \
  --n-injections 20000 \
  --num-warmup 1000 \
  --num-samples 1000 \
  --num-chains 4 \
  --selection-chunk-size 4096
```

Re-running the identical command resumes completed chains. Changing the model,
prior, HBI configuration, basis identity, event/sample counts, or sampler
configuration causes the resume manifest check to fail rather than silently
mixing campaigns.

## Outputs

```text
<run-dir>/
    chains/
        manifest.json
        chain_000.npz
        chain_000.npz.json
        ...
    posterior.npz
    posterior.npz.json
    recovery_summary.json
```

`recovery_summary.json` contains the injected truth, posterior 5/50/95 percent
quantiles, a truth-in-central-90%-interval indicator, divergence count, split
R-hat/effective-sample-size summaries, catalog/selection sizes, and HBI
importance diagnostics at both the injected truth and posterior median. The
latter include minimum event ESS, selection ESS, maximum normalized weights,
and the shape-likelihood Monte-Carlo variance estimate.

The indicator is a diagnostic for one catalog, not a calibrated coverage claim.
Coverage requires repeated independent synthetic catalogs.

## H100 / Slurm

A site-neutral template is provided at:

```
scripts/slurm/phase3_synthetic_h100.sbatch.example
```

The template deliberately does not guess a partition, account, CUDA module, or
JAX wheel. Those are machine-specific. It does refuse to start the campaign
unless `jax.devices()` contains a GPU.

Example:

```bash
export GWPOP_ENV=/path/to/venv/bin/activate
export GWPOP_RUN_DIR=/persistent/path/gwpop-phase3/seed-20260917
sbatch scripts/slurm/phase3_synthetic_h100.sbatch.example
```

## Phase-3 acceptance runs

Before Phase 4 begins:

- run the default catalog with at least 4 NUTS chains;
- require no unresolved numerical pathologies (divergences, poor split R-hat,
  low MCMC effective sample size, catastrophic PE/selection importance ESS, or
  excessive likelihood Monte-Carlo variance);
- repeat with multiple independent data/sampler seeds;
- verify posterior recovery is statistically consistent across the ensemble;
- verify an interrupted campaign resumes without changing completed chains.

A single catalog with every truth inside a chosen credible interval is not, by
itself, a coverage calibration.
