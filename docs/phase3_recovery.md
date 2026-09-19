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

## Selection injections (survey v2)

`--injection-draw` (and `SyntheticSurveyConfig.injection_draw`) chooses how the
raw-draw selection campaign is generated. The likelihood, the population model,
the detection rule (deterministic chirp-mass-scaled reach) and the raw-draw
semantics `A = T/N_draw * sum_detected p_pop/p_draw` with the true `N_draw` are
identical in both modes.

- `uniform_detector_box` (default; unchanged): uniform in `m1_detector`, `q`,
  `luminosity_distance`, `chi_eff`, isotropic sky. Default manifests, campaign
  plans and data are byte-identical to the pre-v2 code.
- `population_proxy`: injections are drawn from the baseline population at the
  proxy hyperparameters `DEFAULT_POPULATION_PROXY_HYPERPARAMETERS` and the stored
  `log_draw_density` is exactly `GwcatChiEffBBHModel()(samples, proxy)` (64-bit;
  it includes the source-to-detector Jacobian and `1/(4 pi)`). The config
  records the resolved proxy, and a proxy that does not cover
  `BASELINE_SYNTHETIC_PRIORS` is rejected.

Proxy: `alpha=1.25, mmin=2, mmax=120, peak_fraction=0.35, peak_mu=35,
peak_sigma=4, beta_q=2, kappa=-1.5, chi_mu=0.05, chi_sigma=0.25`.

- Support. `mmin`/`mmax` are the hyperprior bounds, so the proxy covers
  `m1_source in [mmin, mmax]` and `q in [max(q_floor, mmin/m1), 1]` for every
  hyperprior point, and so every model in the enumerated graph, which shares
  those mass priors. Redshift (same `zmax`), `chi_eff` and sky have full
  support for any proxy. The estimator is therefore unbiased at every trial
  point, not just near the truth.
- Efficiency. Only detected systems enter the integral, and detection favours
  heavy, nearby, near-equal-mass binaries. The proxy is therefore a
  detection-tilted version of the truth: a shallower mass slope, a larger
  35 Msun peak, `q` pushed toward 1 and redshift weighted to low z. `beta_q=2`
  also samples the `q -> 1` pile-up of any target whose `mmin > 2`, where
  `p(q|m1)` diverges as `m1 -> mmin`. `chi_eff` does not affect detection. The
  proxy keeps the truth mean and is wider than the truth width, so the chi
  weights stay bounded over the whole prior. The chi efficiency factor
  `1/E[w_chi^2]` is 0.93 at the truth and at least 0.023 anywhere in the
  hyperprior box; a 0.20 proxy width would give 1.0 at the truth but about
  4e-4 near `chi_sigma=0.5`. The values were chosen by Monte-Carlo scans of the selection
  ESS per draw at the truth, at the prior median, over posterior-like clouds
  and over random prior draws. With 400k draws, ESS/N_draw is 0.21 at the
  truth. Over 300 random hyperprior draws its median is 6e-3, 89% of draws
  have ESS/N_draw >= 1e-3 and all have ESS/N_draw >= 1e-4. The provenance is
  in `gwpop-search-data/design/phase3_synthetic_survey_v2_proxy_design/`.
- Validation (`tests/test_synthetic_population_proxy.py`): the stored density
  equals the model at the proxy. Population draws at 48 hyperprior points,
  including all mass-bound corners, never land where the proxy density is zero.
  Importance weights over all proxy draws average to 1 within MC error. The
  estimated detectable fraction agrees with brute-force population draws at the
  truth and at the prior median. KS tests check the proxy sampler against the
  model's own z, m1, q|m1 and chi_eff factors; they reject sampler offsets of
  0.1 in kappa, 0.03 in alpha or 0.1 in beta_q. The only approximation is the
  redshift sampler's linear-CDF inversion. For the default proxy it biases
  importance weights by at most 1.6e-7 over kappa in [-6, 12] (computed
  deterministically).

Measured at the Phase-3 truth and at the hyperprior median. The table uses the
four campaign catalogs (data seeds 1685370682, 1653124771, 610249415,
200179658; 48 events x 256 PE samples) and the NumPy reference; each entry is
the range over catalogs. The full table is
`gwpop-search-data/design/phase3_synthetic_survey_v2_measurements.json`.

| injection draw | N_draw | n_det | selection ESS (truth) | ESS/n_det (truth) | Var(log L) truth | Var(log L) prior median |
|---|---:|---:|---:|---:|---:|---:|
| uniform_detector_box | 20000 | 15186 | 190-320 | 0.013-0.021 | 7.20-12.11 | 25.85-31.70 |
| uniform_detector_box | 50000 | 37936 | 570-788 | 0.015-0.021 | 2.97-4.10 | 11.29-12.94 |
| uniform_detector_box | 100000 | 75865 | 567-1421 | 0.007-0.019 | 1.71-4.13 | 5.98-11.84 |
| uniform_detector_box | 200000 | 151678 | 811-2842 | 0.005-0.019 | 0.91-2.92 | 3.33-4.36 |
| population_proxy | 20000 | 10180 | 2656-4646 | 0.263-0.452 | 0.49-0.86 | 1.61-12.51 |
| population_proxy | 30000 | 15401 | 4325-6546 | 0.281-0.427 | 0.37-0.56 | 1.24-2.16 |
| population_proxy | 40000 | 20415 | 8778-9292 | 0.426-0.453 | 0.30-0.31 | 1.01-1.97 |
| population_proxy | 50000 | 25426 | 10326-12137 | 0.406-0.478 | 0.25-0.28 | 0.89-1.82 |
| population_proxy | 75000 | 38147 | 15095-16674 | 0.396-0.436 | 0.20-0.23 | 0.72-1.73 |
| population_proxy | 100000 | 50963 | 21354-22852 | 0.419-0.450 | 0.18-0.19 | 0.64-1.59 |
| population_proxy | 200000 | 101974 | 34084-43307 | 0.334-0.426 | 0.14-0.16 | 0.52-1.45 |

`Var(log L) = sum_i (1/ESS_i - 1/n_i) + N_events^2 (1/ESS_sel - 1/N_draw)`.
The PE term is 0.095-0.109 at the truth (minimum event ESS 56-72) and
0.39-1.32 at the prior median, where `mmin=6` cuts into the PE clouds of the
lightest events. In catalog 610249415 an event with true `m1_source=5.2` is
almost entirely outside that support. The PE term sets a floor that no
injection count removes.

Recommendation: `--injection-draw population_proxy`. The smallest measured
`N_draw` with `Var(log L) <= 0.3` at the truth for all four catalogs is
50000 (40000 fails at 0.310). Use 100000 (about 51k stored rows) for margin.
The selection weights have a log-divergent second moment because the pairing
density diverges at `m1 -> mmin` for any target with `mmin > 2`. The realized
selection ESS of a single injection set therefore scatters between catalogs:
34k-43k at 200000 draws and 2.7k-4.6k at 20000. At 100000 the selection term at
the truth (0.08) is below the PE term (0.10).

Caveats for the gate at posterior points:

- At each catalog's joint maximum-likelihood point, which has narrow
  `peak_sigma` (1.0-2.2) and `chi_sigma` (0.09-0.12), the 256-sample PE term
  alone is 0.51-0.91. The total is 0.67-1.23 with 100000 proxy injections and
  0.59-1.09 with 200000. With 1024 PE samples per event the PE term drops to
  0.14-0.25 there, and with 4096 to 0.04-0.06.
- The synthetic PE is centred on the true parameters and detection cuts on
  true parameters. The survey is therefore not DAG-consistent in the sense of
  Essick & Fishbach (2023), and width parameters come out biased low: the
  conditional `chi_sigma` profiles of the four catalogs peak at 0.09-0.12
  (truth 0.20). This is independent of the injection draw.

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
