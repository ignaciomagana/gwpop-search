# Phase 3 synthetic recovery runbook

Phase 3 is accepted only after the conventional baseline model recovers from
multiple independent closed synthetic catalogs under the same standardized HBI
engine that will later score GWTC-5 models.

> **Status (2026-09-19).** The NumPyro-NUTS campaign described in the later
> sections was rejected: NUTS froze against the hard support edges of the
> likelihood, and the uniform-box injections gave a selection Monte-Carlo
> variance of 7-12 at the truth. Its code, commands and formats (`...-1.0`) are
> kept only to reproduce that record. The Phase-3 re-run uses dynesty nested
> sampling for posteriors and evidences on the survey-v2 catalogs:
> "Phase 3 v2 on dynesty" directly below.

## Phase 3 v2 on dynesty

Module `gwpop_search.inference.phase3_ns` (campaign) and
`gwpop_search.inference.phase3_evidence` (Phase 3c). The likelihood, the
population model, the hyperpriors and the grammar are unchanged; every fit is
`run_dynesty_population` of the standardized HBI shape likelihood.

### Plan (`gwpop-search-phase3-campaign-plan-2.0`)

| item | v2 default |
|---|---|
| catalogs `n_runs` | 4, seed pairs `recovery_seed_pairs(n_runs, 20260917)` exactly as v1 (data seeds 1685370682, 1653124771, 610249415, 200179658) |
| repeats per catalog `R` | 4, seeds `sha256(sampler_seed:dynesty-repeat:r)` |
| survey | `noisy_observation`, `population_proxy` (eps 0.3), 48 events, 1024 PE samples/event, 100000 injections |
| model / priors / truth | `GwcatChiEffBBHModel`, `BASELINE_SYNTHETIC_PRIORS`, `DEFAULT_BASELINE_HYPERPARAMETERS` |
| dynesty | static `NestedSampler`, `bound="multi"`, `sample="rslice"`, `slices = 2*(3+ndim) = 26`, `nlive=1000`, `dlogz=0.1`, `batch_size=64` (queue size and device batch), other fields dynesty defaults, no budget |
| HBI | `HBIConfig(selection_chunk_size=None)` (one selection chunk) |
| diagnostics | 512 pooled importance draws, R-hat chains of `min(2000, floor(min Kish))` draws |
| criteria | `NSRecoveryAcceptanceCriteria` (below) |

The plan pins every `DynestyConfig` field except `checkpoint_every` (I/O
cadence only). A root whose `campaign_plan.json` differs is refused before any
computation, which also rejects a v1 (NUTS) root.

### Commands

```bash
gwpop-search synthetic-campaign-ns --root <ROOT>          # run / resume / reuse
gwpop-search assess-synthetic-campaign-ns --root <ROOT>   # re-assess with the plan's criteria
gwpop-search fingerprint-ns-run --run-dir <ROOT> --output before.json
gwpop-search fingerprint-ns-run --run-dir <ROOT> --compare-to before.json   # exit 1 on any change
gwpop-search synthetic-evidence-check --root <ROOT_3C>    # Phase 3c
```

All options default to the plan above (`--nlive`, `--sample`, `--bound`,
`--slices` [default `2*(3+ndim)` of each model], `--dlogz`, `--batch-size`,
`--maxiter`, `--maxcall`, `--num-posterior-samples`, `--checkpoint-every`
[300 s], `--n-events`, `--pe-samples`, `--n-injections`, `--injection-draw`,
`--observation-model`, `--importance-draws`, `--rhat-draws-per-run`,
`--root-seed`; the campaign also takes `--n-runs`, `--repeats`,
`--min-runs`, `--min-repeats`).

### Per-catalog summary (`gwpop-search-phase3-recovery-2.0`)

- Pooled posterior = equal-weight mixture of the `R` separately normalized
  runs (each run's `num_posterior_samples` equal-weight draws, concatenated;
  never `merge_runs`/`jitter_run`/`resample_run`): 5/50/95% quantiles, mean,
  std, truth-in-90%-interval, standardized offset `(median - truth)/std`;
  `posterior_pooled.npz`.
- Truth-rank diagnostics (`pe_truth_quantiles` per PE coordinate with a KS
  p-value; uniformity is expected for chi_eff only).
- Per run: `logz`, `logzerr`, information, `niter`, `ncall`, efficiency, Kish
  ESS, elapsed time, likelihood evaluations, selection-unsupported
  evaluations, dlogz termination, `sqrt(H/nlive)`.
- Evidence repeats: mean, std (ddof 1), maximum reported error, conservative
  error `max(std, max logzerr)`, all pairwise `|dlnZ|/sqrt(err_a^2+err_b^2)`.
- Cross-run rank-normalized split R-hat per parameter (runs as chains;
  identical to `arviz.rhat(method="rank")`).
- Importance diagnostics with the backend's jitted diagnostics under the runs'
  own HBI configuration (checked against every run's likelihood identity) at
  the pooled posterior-median point, as quantiles over 512 seeded pooled draws
  (`inverted_cdf`), and at the truth (reported only).

### Acceptance criteria v2 (`NSRecoveryAcceptanceCriteria`, format `gwpop-search-ns-acceptance-criteria-2.0`)

Per catalog, every check must pass:

| check | limit |
|---|---|
| repeats | >= 4 |
| max cross-run R-hat over parameters | <= 1.01 |
| min Kish ESS per run | >= 1000 |
| evidence repeat std | <= 0.2 |
| max reported logzerr | <= 0.2 |
| max pairwise repeat z | <= 3.0 |
| NaN/+inf evaluations | 0 (structural: the backend raises) |
| selection-unsupported evaluations | 0 |
| every run terminated by `dlogz` | yes |
| min event ESS | >= 20 at the point, the draw median and the draw q10 |
| selection ESS | >= 200 at the point, the draw median and the draw q10 |
| max event weight | <= 0.25 at the point, the draw median and the draw q90 |
| max selection weight | <= 0.10 at the point, the draw median and the draw q90 |
| Var(log L) | <= 1.0 at the point, the draw median and the draw q90 |

Campaign (`gwpop-search-phase3-campaign-2.0`): `phase3_numerical_gate_passed`
iff at least `min_runs = 4` catalogs exist and every catalog passes. Ensemble
central-90% coverage fractions and median standardized offsets are reported,
not thresholded. `NSRecoveryAcceptanceCriteria.f3_level()` holds the F3 gates
of the dynesty ladder (R-hat 1.05, Kish 500, evidence 0.5/0.5, z 3.5,
importance 10/100/0.35/0.15/2.0).

### Layout

```text
<ROOT>/campaign_plan.json
<ROOT>/campaign_summary.json
<ROOT>/run_000/recovery_summary.json
<ROOT>/run_000/posterior_pooled.npz
<ROOT>/run_000/repeats/repeat_000/{manifest.json, checkpoint.pkl, result.npz, result.npz.json}
...
```

### Resume integrity

Every repeat is resumable on its own: `checkpoint.pkl` (written atomically by
dynesty every `--checkpoint-every` seconds and at the end) lets a killed job
continue the identical trajectory, and a completed repeat (`result.npz` +
sidecar) is reused without rewriting any file. The manifest pins the data
digests, priors, HBI and dynesty configuration, code (including uncommitted
edits) and runtime, so a resume refuses anything else.
`tests/test_phase3_ns.py` SIGKILLs a campaign process mid-way through a repeat
and verifies that the resumed campaign reproduces the uninterrupted control bit
for bit while the completed repeat's fingerprints stay unchanged.

Exercise on H100: fingerprint, kill the job with `SIGKILL` while a repeat is
running (after its first checkpoint), re-submit the identical command, then
compare:

```bash
gwpop-search fingerprint-ns-run --run-dir <ROOT> --output fp_before.json
# kill -9 <pid>; re-run the identical synthetic-campaign-ns command
gwpop-search fingerprint-ns-run --run-dir <ROOT> --compare-to fp_before.json --output fp_after.json
```

### Phase 3c: Bayes-factor sanity check (`gwpop-search-phase3-evidence-check-1.0`)

Two exactly nested atoms: `chieff.mean.linear_q` (`chi_mu_q_slope` ~
U(-0.6, 0.6)) and `pairing.beta.linear_m1` (`beta_q_m1_slope` ~ U(-0.3, 0.3)).

- (a) Null: on each of the 4 campaign catalogs (same seeds and survey), F3-style
  evidence (`R = 2`, the campaign's dynesty settings, `slices = 2*(3+ndim)`:
  26 for the root, 28 for each child) of the declarative root
  `compile_model_spec(baseline_model_spec())` and of both children.
  `ln BF(child/root) = mean lnZ_child - mean lnZ_root`,
  `sigma = sqrt(cons_child^2 + cons_root^2)` with `cons = max(repeat std, max
  logzerr)`, and the Savage-Dickey cross-check `ln pi(0) - ln p(0|D)` (NS-weighted
  Gaussian KDE of the pooled child posterior of the slope; 0 is interior, the
  Verdinelli-Wasserman factor is 1).
- (b) Injected: one catalog per atom drawn from the child model with the slope
  at 0.9 x its upper prior bound (0.54, 0.27), all else at the root truth, by the
  grammar-matched structured generator (`scouts.synthetic`), which applies the
  survey's `noisy_observation` model (detection on observed data, PE drawn given
  the observation). `tests/test_phase3_evidence_check.py` verifies that the
  draws follow the compiled child densities exactly.
- Before any fit the declarative root and `GwcatChiEffBBHModel` are compared on
  every catalog's PE and selection samples at the truth and 16 prior draws
  (same support, `|delta log p| <= 1e-10`; measured 1.7e-13 on a v2 catalog).

Pre-declared engineering pass rule (`EvidenceCheckPassRule`, recorded in the
plan and the summary): (1) every fit passes the F3-level gates; (2)
`ln BF(child/root) < 3` in all 8 null cases; (3) `ln BF(child/root) > 0` in both
injected cases; (4) `|ln BF - ln BF_SDDR| <= max(0.3, 2 sigma)` in every null
case.

Injection strengths. With 48 events the atoms are only moderately detectable:
the conditional profile likelihood of the slope (other hyperparameters at the
truth, the partner intercept profiled) on nine v2 catalogs per atom gave slope
widths of about 0.18 (`chi_mu_q_slope`) and 0.07-0.09 (`beta_q_m1_slope`), and
profile approximations of `ln BF` of 0.5-5.0 at 0.55 and 0.5-9.7 at
0.25-0.3 (all positive). The marginal evidence is lower than the profile
approximation, so (3) is a weak power check with a few-percent chance of a
non-detection by noise, not a calibration.

Layout: `<ROOT_3C>/evidence_check_plan.json`,
`<ROOT_3C>/evidence_check_summary.json`,
`<ROOT_3C>/null/catalog_XXX/{root,<atom>}/{fit_summary.json,repeat_YYY/}`,
`<ROOT_3C>/injected/<atom>/{root,<atom>}/...`. Resumable like the campaign.

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

This command reproduces the legacy survey (truth-centered PE, uniform-box
injections) that the 2026-09-19 H100 campaign used. The Phase-3 re-run should
use the survey-v2 options described in "Synthetic survey v2" below
(`--observation-model noisy_observation --injection-draw population_proxy`).

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

## Synthetic survey v2

Two independent options change how the closed survey is generated. Neither
changes the likelihood, the population model, the priors or the gates, and at
their defaults (`truth_centered`, `uniform_detector_box`) data, manifests and
campaign plans are byte-identical to the pre-v2 code
(`tests/test_synthetic_legacy_golden.py` pins the legacy draws computed at
e3c5b01).

### Observation model (`--observation-model`)

- `truth_centered` (default, legacy): detection cuts on the true parameters and
  each event's PE is a truncated normal centred on its true parameters. This is
  not a data-generating process (Essick & Fishbach 2023, arXiv:2310.02017):
  detection is not a function of data, and the truth sits at the same quantile
  of every event's PE. Width hyperparameters come out biased low: for a
  400-event catalog the conditional `chi_sigma` profile peaks at 0.15-0.16
  (truth 0.20), close to `sqrt(0.20^2 - 0.12^2) = 0.16`.
- `noisy_observation` (DAG-consistent): each event and each injection gets one
  noise realisation of its observed data
  `(ln m1_detector, q, ln d_L, chi_eff) + N(0, diag(s^2))` with
  `s = (pe_m1_fractional_sigma, pe_q_sigma, pe_d_l_fractional_sigma,
  pe_chi_eff_sigma)`; the sky is observed exactly. Detection is the same
  chirp-mass-scaled reach evaluated on the observed chirp mass and distance
  (`observed_detection_mask`), for events and injections alike. PE samples are
  exact posterior draws given the observed data under the stored uniform
  detector-box prior (`ln m1 ~ N(x_m + s_m^2, s_m)` and
  `ln d_L ~ N(x_d + s_d^2, s_d)` carry the Jacobian of the uniform-in-m1 and
  uniform-in-d_L prior), so the stored `log_ref_density` is the true PE prior.
  The box's mass edge covers every hyperprior population
  (`1.1 x 120 x (1 + zmax)`), independent of the truth. Injections store their
  true parameters, so `A` estimates `P(detected data | theta)` averaged over
  the population, which is the selection term of this data model.

Validation (`tests/test_synthetic_observation_model.py`): PE draws match the
analytic posterior (KS, including near the box edges); every event's data pass
the detection rule while some true parameters lie beyond the reach (Malmquist);
the injection estimate of the detectable fraction matches brute-force noisy
detection of direct population draws for both injection draws (and differs from
the truth-side fraction); the truth's rank in each event's chi_eff PE is
uniform (KS p > 1e-3 for 300 events; the legacy PE gives p < 1e-10, the gate
`pe_truth_quantiles` implements); the 400-event `chi_sigma` profile peaks
within 0.02 of the truth (legacy: <= 0.17).

The likelihood-level check is in "Measurements" below.

### Selection injections (`--injection-draw`)

- `uniform_detector_box` (default, legacy): uniform in `m1_detector`, `q`,
  `luminosity_distance`, `chi_eff`, isotropic sky.
- `population_proxy`: injections are drawn from the baseline population at the
  proxy hyperparameters `DEFAULT_POPULATION_PROXY_HYPERPARAMETERS` (`alpha=1.25,
  mmin=2, mmax=120, peak_fraction=0.35, peak_mu=35, peak_sigma=4, beta_q=2,
  kappa=-1.5, chi_mu=0.05, chi_sigma=0.25`), mixed with a defensive pairing
  component: for `m1_source < 12`, a fraction `eps = 0.3`
  (`injection_draw_defensive_fraction`) of the draws replace q by `1 - u` with
  u log-uniform on `[1e-6 u_max, u_max]`, `u_max = 1 - max(q_floor, 2/m1)`. The
  stored `log_draw_density` is exactly that mixture: the baseline's own
  normalized detector-basis density at the proxy (64-bit, with the
  source-to-detector Jacobian and `1/(4 pi)`) with `p_q(q|m1)` replaced by
  `(1 - eps) p_q + eps g`. `eps = 0` is exactly the pure proxy. The density is
  evaluated with the baseline class carrying the caller's model context
  (cosmology, zmax, q_floor), because the sampler always draws the baseline
  family.

Why the defensive component: for any target with `mmin > 2` the pairing density
`q^beta / Z(m1)` diverges like `1/(1 - mmin/m1)` as `m1 -> mmin`, which no
single population proxy follows. Pure-proxy weights therefore have a Pareto-2
tail and an infinite second moment: a single row can carry 12% of the
selection sum and move `48 log A` by 6 nats (1 in 100 sets of 1e5 draws at the
truth; up to 5.5% of sets have a row above 1% of the sum at `mmin = 9.8`). A
density that is log-uniform in `1 - q` is the one that bounds the pairing ratio
for every support width at once; the ratio is at most `ln(1e6)/eps = 46`
(`test_defensive_component_bounds_pairing_corner_weights`). The values were
chosen by a replicate scan (400 independent 1e5-draw sets, 45 target points,
9 variants; `design/phase3_synthetic_survey_v2_review_fixes/`): `eps = 0.3`
removed every set with a row above 1% of the sum at the truth, the prior median
and `mmin` in {2.5, 7.5, 9.8} without lowering the selection ESS.

Support and guards:
- `mmin`/`mmax` of the proxy are the hyperprior bounds, so it covers every
  population of every model in the grammar (all mass families live on
  `[mmin, mmax]`, all pairing families on `q >= max(q_floor, mmin/m1)`). The
  config rejects a proxy that does not cover `BASELINE_SYNTHETIC_PRIORS` and a
  synthetic truth outside the proxy.
- `require_population_proxy_coverage(selection, priors=..., hyperparameters=...)`
  must be called by every consumer with what it actually evaluates. The
  recovery driver, the fidelity evaluator (every model and fidelity), evidence
  repeats, null replays (all graph nodes, before any compute) and the HSGP
  scouts (their fixed base point) call it. A dynesty driver must call it too.
- The resolved proxy is stored read-only (`FrozenHyperparameters`); the config
  is hashable and picklable.
- `redshift_sampling_grid >= 4096` is required: the linear-CDF redshift sampler
  then biases the weights by at most 7e-7 over the kappa prior (1.6e-7 at the
  default 8192; 2e-4 at 256).
- A pre-defensive `population_proxy` manifest (no
  `injection_draw_defensive_fraction`) is rejected rather than reinterpreted.
- Survey-v2 options enter the synthetic-null `dataset_identity`
  (`baseline-null:<seed>:survey-<hash>`); legacy identities are unchanged.
  `frozen_selection_resample` nulls reject survey-v2 options.
- CLI: `--n-injections` defaults to 20000 for the box (unchanged) and 100000
  for `population_proxy`, with a warning below 100000.

### Measurements

All numbers: NumPy reference, CPU, code 4ec73ae, the four campaign catalogs
(data seeds 1685370682, 1653124771, 610249415, 200179658; 48 events), at the
Phase-3 truth and at the hyperprior median; ranges are over catalogs. Full
data: `gwpop-search-data/design/phase3_synthetic_survey_v2_measurements.json`
(format 1.1; the pre-review table of commit 3e48a49 is archived as
`phase3_synthetic_survey_v2_proxy_design/phase3_synthetic_survey_v2_measurements_3e48a49.json`).
Scripts: `gwpop-search-data/design/phase3_synthetic_survey_v2_review_fixes/`.
`Var(log L) = sum_i (1/ESS_i - 1/n_i) + 48^2 (1/ESS_sel - 1/N_draw)` (PE term
plus selection term, plug-in from one realisation).

Single realisation, 256 PE samples per event:

| observation | injection draw | eps | N_draw | n_det | ESS/n_det (truth) | selection term (truth) | Var(log L) truth | Var(log L) prior median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noisy_observation | population_proxy | 0.3 | 20000 | 10123 | 0.244-0.381 | 0.478-0.818 | 0.65-0.97 | 2.27-inf |
| noisy_observation | population_proxy | 0.3 | 50000 | 25230 | 0.252-0.405 | 0.180-0.316 | 0.33-0.50 | 1.50-inf |
| noisy_observation | population_proxy | 0.3 | 100000 | 50614 | 0.340-0.389 | 0.094-0.111 | 0.24-0.39 | 1.21-inf |
| noisy_observation | population_proxy | 0.3 | 200000 | 101177 | 0.337-0.394 | 0.046-0.056 | 0.20-0.33 | 1.08-inf |
| noisy_observation | population_proxy | 0 | 100000 | 50463 | 0.276-0.401 | 0.091-0.142 | 0.24-0.42 | 1.21-inf |
| noisy_observation | uniform_detector_box | | 20000 | 15693 | 0.012-0.018 | 8.12-12.29 | 8.29-12.57 | 28.55-inf |
| noisy_observation | uniform_detector_box | | 200000 | 156989 | 0.002-0.011 | 1.36-9.63 | 1.64-9.80 | 4.20-inf |
| truth_centered | population_proxy | 0.3 | 50000 | 25489 | 0.363-0.430 | 0.165-0.203 | 0.26-0.31 | 0.89-1.82 |
| truth_centered | population_proxy | 0.3 | 100000 | 51104 | 0.300-0.388 | 0.094-0.127 | 0.19-0.23 | 0.64-1.58 |
| truth_centered | population_proxy | 0 | 100000 | 50963 | 0.419-0.450 | 0.078-0.085 | 0.18-0.19 | 0.64-1.59 |
| truth_centered | uniform_detector_box | | 20000 | 15186 | 0.013-0.021 | 7.09-12.00 | 7.20-12.11 | 25.85-31.70 |
| truth_centered | uniform_detector_box | | 200000 | 151678 | 0.005-0.019 | 0.80-2.83 | 0.91-2.92 | 3.33-4.36 |

(`truth_centered` with `eps = 0` reproduces the pre-review pure-proxy table
exactly; the JSON has every N_draw.) The PE term at the truth is 0.095-0.109
under `truth_centered` and 0.147-0.281 under `noisy_observation` (minimum event
ESS 8.4-52 of 256): once the PE is centred on noisy data, events near the
population edges keep only part of their PE inside the support. At the
hyperprior median (`mmin = 6`) catalog 610249415 has an event with no PE
sample inside the support under `noisy_observation`, so `log L = -inf` there;
that is a property of the point, not of the injections. The uniform box has
the same pairing-corner tail as the pure proxy (ESS as low as 239 at 200000
draws).

Replicate study of the selection term: 200 independent injection sets per
cell, `population_proxy`, noisy detection unless noted. "Replicate" is
`48^2 Var(log A)` across sets; the plug-in is the median single-set estimate.
The selection term does not depend on the catalog.

| observation | eps | N_draw | replicate (truth) | median plug-in (truth) | sets with a row > 1% of the sum (truth) | largest one-row shift of 48 log A (truth) | replicate (median) | largest one-row shift (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| noisy_observation | 0 | 50000 | 0.218 | 0.189 | 3.5% | 2.35 | 0.460 | 1.00 |
| noisy_observation | 0 | 100000 | 0.151 | 0.100 | 2.0% | 2.51 | 0.264 | 1.49 |
| noisy_observation | 0 | 200000 | 0.065 | 0.051 | 1.0% | 0.65 | 0.126 | 0.69 |
| noisy_observation | 0.3 | 20000 | 0.537 | 0.486 | 5.0% | 0.69 | 1.375 | 0.41 |
| noisy_observation | 0.3 | 50000 | 0.203 | 0.200 | 0 | 0.42 | 0.565 | 0.24 |
| noisy_observation | 0.3 | 75000 | 0.116 | 0.132 | 0 | 0.28 | 0.320 | 0.25 |
| noisy_observation | 0.3 | 100000 | 0.084 | 0.100 | 0 | 0.34 | 0.286 | 0.19 |
| noisy_observation | 0.3 | 200000 | 0.064 | 0.051 | 0 | 0.14 | 0.151 | 0.13 |
| truth_centered | 0 | 100000 | 0.093 | 0.089 | 1.5% | 0.56 | 0.315 | 2.41 |
| truth_centered | 0.3 | 100000 | 0.082 | 0.090 | 0 | 0.20 | 0.220 | 0.12 |

With the defensive component the replicate variance agrees with the plug-in
estimate to within 25% and no set has a row carrying more than 1% of the
selection sum from 50000 draws up; the pure proxy's replicate variance exceeds
its plug-in by up to 50% and single rows shift `48 log A` by up to 2.5 nats.

PE samples per event under `noisy_observation` (`population_proxy`,
`eps = 0.3`):

| PE samples | N_draw | PE term (truth) | min event ESS (truth) | Var(log L) truth | Var(log L) prior median | min event ESS (median) |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 50000 | 0.147-0.281 | 8.4-52.0 | 0.33-0.50 | 1.50-inf | 0.0-8.1 |
| 256 | 100000 | 0.147-0.281 | 8.4-52.0 | 0.24-0.39 | 1.21-inf | 0.0-8.1 |
| 256 | 200000 | 0.147-0.281 | 8.4-52.0 | 0.20-0.33 | 1.08-inf | 0.0-8.1 |
| 512 | 100000 | 0.080-0.261 | 5.6-78.3 | 0.18-0.36 | 0.87-1.89 | 1.0-8.3 |
| 1024 | 50000 | 0.038-0.071 | 30.1-227.3 | 0.22-0.25 | 0.80-1.08 | 5.1-17.7 |
| 1024 | 100000 | 0.038-0.071 | 30.1-227.3 | 0.13-0.17 | 0.53-0.83 | 5.1-17.7 |
| 1024 | 200000 | 0.038-0.071 | 30.1-227.3 | 0.09-0.12 | 0.40-0.70 | 5.1-17.7 |

DAG check: conditional likelihood profiles of each hyperparameter (others at
the truth) for three 600-event catalogs (1024 PE samples, 200000 proxy draws).
Entries are the quadratic peak and its offset from the truth in units of the
profile's curvature width.

| seed | observation | alpha | peak_fraction | peak_mu | peak_sigma | beta_q | kappa | chi_mu | chi_sigma |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | truth_centered | 3.17 (+2.0) | 0.109 (+0.9) | 34.0 (-3.0) | 1.71 (-6.0) | 0.70 (-2.1) | 2.28 (+1.9) | 0.038 (-1.2) | 0.148 (-7.3) |
| 12 | truth_centered | 3.08 (+1.0) | 0.109 (+0.8) | 33.5 (-4.4) | 1.25 (-5.3) | 0.73 (-1.9) | 1.92 (-0.5) | 0.043 (-0.7) | 0.157 (-6.1) |
| 13 | truth_centered | 3.17 (+2.1) | 0.100 (-0.0) | 33.7 (-3.9) | 0.77 (-6.2) | 0.89 (-0.7) | 1.88 (-0.8) | 0.050 (-0.0) | 0.160 (-5.6) |
| 11 | noisy_observation | 3.02 (+0.2) | 0.117 (+1.5) | 35.5 (+1.5) | 3.94 (-0.2) | 1.08 (+0.6) | 2.30 (+2.0) | 0.038 (-1.3) | 0.198 (-0.2) |
| 12 | noisy_observation | 3.01 (+0.1) | 0.101 (+0.1) | 35.1 (+0.1) | 3.84 (-0.4) | 0.89 (-0.8) | 1.97 (-0.2) | 0.047 (-0.3) | 0.190 (-1.3) |
| 13 | noisy_observation | 3.02 (+0.2) | 0.089 (-1.2) | 34.5 (-1.3) | 4.07 (+0.2) | 1.21 (+1.3) | 2.03 (+0.2) | 0.041 (-0.9) | 0.198 (-0.2) |

Truth: alpha 3, peak_fraction 0.10, peak_mu 35, peak_sigma 4, beta_q 1,
kappa 2, chi_mu 0.05, chi_sigma 0.20. `truth_centered` biases the widths by 5-7
sigma (chi_sigma 0.15-0.16; peak_sigma peaks at 1.75 or at the lowest grid
point, 1.5, so its quadratic peak is an extrapolation) and peak_mu by 3-4
sigma; under `noisy_observation` every pull lies within 2 sigma (rms 0.9). The edge
parameters are non-regular (their Monte-Carlo profiles step as the edge
crosses PE samples): under `noisy_observation` the truth lies 1.3-3.2 nats
below the conditional maximum in mmin (maxima 5.12-5.21) and 0.4-1.7 nats in
mmax (under `truth_centered`: 1.0-2.4 and 0.1-0.2). The recovery ensemble
should check their coverage.

### Recommendation

For the Phase-3 re-run use `--observation-model noisy_observation
--injection-draw population_proxy` (default proxy and `eps = 0.3`) with
`--pe-samples 1024` and `--n-injections 100000` (the `population_proxy`
default).

- Rule applied: the largest over the four catalogs of [PE term at the truth +
  replicate selection term] must be at most 0.3, and no replicate set may have
  a row above 1% of the selection sum. With 1024 PE samples the smallest
  passing N_draw is 50000 (0.27); 100000 gives 0.15 and keeps the prior-median
  total at 0.53-0.83. With 256 or 512 PE samples no injection count passes:
  the PE term alone reaches 0.28 (256 samples, catalog 610249415) and 0.26
  (512 samples, catalog 200179658); the PE weights of events near the
  population edges are heavy-tailed, so this term scatters between PE draws.
- `truth_centered` data are not suitable for recovery or null calibration
  whatever the injection draw: they bias the widths by 5-7 sigma.
- The legacy defaults stay available only to reproduce existing campaigns.

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
