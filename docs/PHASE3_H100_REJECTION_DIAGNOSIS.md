# Phase-3 H100 recovery — REJECTED (2026-09-19): diagnosis for redesign

Status: **Phase 3 is REJECTED** for the configuration in
`docs/CLAUDE_H100_VALIDATION_HANDOVER.md` §3 (operator decision, 2026-09-19).
No real-data canonicalization, freeze or production has been run. This document
is the complete diagnosis handed to the planning agent. **Nothing in the model,
likelihood, priors, thresholds, sampler configuration or campaign plan has been
changed**; every fix below is a candidate for the planning agent / operator to
decide.

## 1. Execution record

- Code: handoff revision `100f44c27fdd15ecd1ce0da9250e940007322e23` (clean tree;
  code checkpoint `6b05f76`, 283 tests).
- Host: PSC `miko`, 1× NVIDIA H100 NVL 94 GB, driver 545.23.08 (CUDA 12.3).
- Environment: Python 3.12.14, jax/jaxlib 0.11.2 (`jax[cuda12]`), NumPyro 0.21.0,
  JAXNS 2.6.9 — every Python dependency pinned to the versions resolved by the
  green CI run. `JAX_ENABLE_X64=true`.
- Site runtime setting: `XLA_FLAGS=--xla_gpu_enable_command_buffer=`. Without it
  XLA CUDA-graph command buffers corrupt the host heap on this older driver
  (full test suite aborted: `malloc_printerr` inside `xla::gpu::GpuExecutable`
  construction after a NUTS run followed by another compile). Execution-only; no
  numerical effect. Full suite with it: **283 passed** on the H100 and on CPU.
- Command: exactly the handover §3 campaign (`--n-runs 4 --root-seed 20260917
  --n-events 48 --pe-samples 256 --n-injections 20000 --num-warmup 1000
  --num-samples 1000 --num-chains 4 --target-accept 0.9
  --selection-chunk-size 4096 --no-progress`).
- Seed pairs (`campaign_plan.json`, sha256 `cbdca92c…0566`):
  run 0 (1685370682, 533246956), run 1 (1653124771, 1151310806),
  run 2 (610249415, 2024837906), run 3 (200179658, 80079290).
- Timeline (EDT): launched 00:29:08; `run_000/chain_000` checkpoint 02:57:05
  (**2.46 h for one chain**); planned SIGKILL 04:11:05 during `chain_001`;
  identical relaunch 04:12:10; operator-ordered stop 05:56:14. Runs 1–3 never
  started.

## 2. Observation — run 0 chain 0 is frozen

`chain_000` (chain seed 1773505056), 1000 warmup + 1000 samples:

- **1000/1000 post-warmup iterations at the maximum tree depth (1023 leapfrog
  steps)**; mean acceptance 0.961; 0 divergences.
- The chain does not move. Posterior std relative to the prior width is ≲ 1e-3
  for every parameter:

  | parameter | truth | chain mean | chain std |
  | --- | ---: | ---: | ---: |
  | alpha | 3.0 | 2.8233 | < 1e-4 |
  | mmin | 5.0 | 5.8235 | 1e-4 |
  | mmax | 90.0 | 64.741 | 0.002 |
  | peak_fraction | 0.10 | 0.3307 | < 1e-4 |
  | peak_mu | 35.0 | 32.641 | 0.0006 |
  | peak_sigma | 4.0 | 3.727 | 0.0007 |
  | beta_q | 1.0 | 0.9967 | 0.0004 |
  | kappa | 2.0 | 1.3093 | 0.0003 |
  | chi_mu | 0.05 | 0.0434 | < 1e-4 |
  | chi_sigma | 0.20 | 0.0833 | < 1e-4 |

With a frozen chain the R-hat ≤ 1.01 / MCMC-ESS ≥ 200 gate is unattainable
regardless of the other chains.

## 3. Environment ruled out

At the frozen point the JAX log-likelihood on CPU is 179.7289618605451 and the
NumPy reference backend gives 179.7289618605451. At the injected truth: JAX-CPU
175.39623739303832, NumPy reference 175.39623739303835, JAX-GPU (H100)
175.39623739303835. The pathology is a property of the model/likelihood/sampler
configuration, not of the GPU, the JAX version or the XLA flag.

## 4. Diagnosis A — hard-edge likelihood cliffs freeze NUTS

Every population component uses **hard support edges**: `m1 ∈ [mmin, mmax]`
(power law + peak, broken power law, two-peak) and `m2 = q·m1 ≥ mmin`
(power-law q, truncated-Gaussian q). There is no smoothing anywhere in
`models/components.py`. The Monte-Carlo likelihood

```text
log L = sum_i log( (1/n) sum_j p_pop(theta_ij) / pi_ij ) - N log A
```

is therefore **piecewise smooth with a jump every time an edge crosses a PE
sample or a detected injection**. The jump size is `log(1 - w)` where `w` is the
crossing sample's normalized weight within its sum. HMC gradients do not see
the jumps.

At the frozen point (`logs/phase3/diag/stuck_point_run0_chain0_cpu.json`):

- event 2's PE sum is carried by one sample: event ESS **1.09**, max normalized
  weight **0.958** (sample #205, m2_source = 5.8236028 Msun; 158 of the event's
  256 samples have m2 < 6);
- the chain sits at mmin = 5.82350, just below that sample. The smooth gradient
  `d log L / d mmin = +6179` pushes mmin up into the edge, and crossing it
  deletes the sample: **log L drops by 4.33 nats**. One-dimensional scans show
  jumps up to 4.25 nats within 1e-5 Msun in mmin (median step 0.008), and small
  0.002–0.007-nat jumps along mmax;
- other gates also fail there: min event ESS 1.09, selection ESS 107,
  Var(log L) 22.96.

The chain is pinned against a cliff. Dual averaging collapses the step size,
the diagonal mass matrix adapts to the resulting tiny variance, and every
trajectory runs 1023 accurate but negligible steps: high acceptance, no motion.

Cliff-proneness beyond this one point:

- the NUTS initial point (`init_to_median` = prior medians: mmin 6, alpha 4,
  peak_fraction 0.25, beta_q 4, …) already has min event ESS 20.0, max event
  weight 0.17, selection ESS 88.8 and Var(log L) 26.3;
- over 300 uniform draws from the Phase-3 hyperprior on the run-0 catalog (209 with finite log L): the
  minimum event ESS is below 2 in 52% of draws, below 5 in 69% and below 20 in 85%; the largest
  single-sample weight exceeds 0.5 in 62% (median minimum event ESS 1.87). Cliff-prone regions are the
  generic situation across the prior, not an unlucky corner.

## 5. Diagnosis B — selection Monte-Carlo variance is infeasible even at the truth

Independently of the sampler, the recommended configuration cannot meet
`max_shape_log_likelihood_variance = 1.0` (nor, for run 1, `min_selection_ess =
200`) **at the injected truth**:

| run | data seed | n_det / n_draw | min event ESS | selection ESS | Var(log L) at truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1685370682 | 15123 / 20000 | 63.9 | 319.7 | 7.197 |
| 1 | 1653124771 | 15192 / 20000 | 71.5 | **190.1** | 12.112 |
| 2 | 610249415 | 15231 / 20000 | 57.3 | 234.5 | 9.804 |
| 3 | 200179658 | 15197 / 20000 | 56.1 | 279.8 | 8.223 |

The variance is dominated by the selection term `N² (1/ESS_sel − 1/N_draw)`
with N = 48 events. Selection ESS is only **1.2–2.1 % of the detected
injections** because the synthetic injections are drawn *uniformly* in the
detector-frame box (m1_det up to ~400 Msun, q, d_L up to d_L(z = 2.5), χ_eff);
~76 % of them are "detected" by the chirp-mass reach rule, while the population
(PL+peak, alpha = 3, mmin = 5, mmax = 90, low z) occupies a small corner of
that box.

Scaling on the run-0 catalog at the truth (same events; more injections):

| n_injections | n_det | selection ESS | ESS / n_det | Var(log L) |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 15,123 | 320 | 2.1 % | 7.20 |
| 100,000 | 75,839 | 1,284 | 1.7 % | 1.88 |
| 400,000 | 303,805 | 3,742 | 1.2 % | **0.72** |

Meeting Var ≤ 1 requires selection ESS ≳ N² ≈ 2,300 at the posterior median,
i.e. ~400k uniform injections for this survey (with no margin for the
posterior median sitting away from the truth, e.g. the 88.8 selection ESS at
the prior median).

## 6. Cost

Clean per-evaluation timing on the H100 (recommended configuration, run-0
data): 4.0–4.1 ms per likelihood value+gradient (100-call averages, idle GPU). A chain at saturated
tree depth costs 2000 × 1023 × that ≈ 2.5 h (observed: 8,873 s including data
generation and compilation), so the full 16-chain campaign would have taken
≈ 40 H100-hours. A healthy chain at typical tree depth 4–6 would take minutes.

## 7. Checkpoint/resume integrity — passed

The kill/relaunch exercise worked as designed. After the SIGKILL during
`chain_001` and the identical relaunch, the resumed process re-validated the
manifest, loaded `chain_000` from its checkpoint (py-spy locals:
`results=[chain_000]`, `chain_index=1`, `expected_seed=219485684`) and advanced
to chain 1. `chain_000.npz` (`f2c30a6f…abeb1`) and `chain_000.npz.json`
(`aa322a77…a429a`) are byte-identical before and after. The campaign was then
stopped by operator decision before a resumed chain completed.

## 8. Relevance to production

The same hard edges exist in every production grammar component, so the
real-data NUTS fidelities are exposed to the same cliffs: F1 (128 PE
samples/event → larger per-sample weights → bigger jumps), F2 and F4 (full
data). Real GWTC-5 events with low secondary masses sit near `m2 = mmin`. The
evidence stage (JAXNS) is not gradient-based and does not freeze the same way,
but its MC-variance and ESS gates are exposed to the same edge-crossing
discontinuities.

## 9. Candidate directions for the planning agent (not decided, not implemented)

A. Selection precision (Diagnosis B)
- raise `--n-injections` (≥ 400k for Var ≤ 1 at truth; more for margin), and/or
- draw synthetic injections from a broad, population-shaped distribution with a
  known draw density (e.g. power law in m1_source, cosmological z) so that
  ESS/n_det rises from ~1–2 % to tens of percent, and/or
- revisit N = 48 against the N² scaling of the selection variance.

B. Hard-edge cliffs (Diagnosis A)
- smooth the support edges (e.g. a low-mass taper on m1 and on m2 = q·m1, and a
  high-mass taper), which makes the MC likelihood continuous;
- more PE samples per event reduce the per-sample weight and hence the jump
  size, but do not remove the discontinuity;
- sampler knobs (init strategy, longer warmup, dense mass, tree depth) do not
  remove the discontinuities; `init_to_median` starts in a cliff-prone region.

C. Gate semantics
- the gate evaluates importance diagnostics at the posterior median; in the
  selection-dominated regime Var ≤ 1 ⇔ ESS_sel ≳ N². Tree-depth saturation is not
  itself a gate check (the frozen chain would still fail R-hat/ESS, but only
  after ~40 H100-hours); a max-tree-depth-fraction diagnostic would make such a
  failure visible early.

## 10. Decision after this diagnosis (2026-09-19)

The operator then delegated planning authority to the H100 orchestrator (Claude) with the instruction to
drop HMC/NUTS in favour of dynesty nested sampling for posteriors and evidences, and to validate the
model-comparison mathematics. The redesign being implemented is: a dynesty 3.1 backend with GPU-batched
likelihood evaluation (nested sampling needs no gradients, so the hard-edge cliffs cannot freeze it; the
cliffs remain a Monte-Carlo-precision concern handled by the importance gates), and a Phase-3 synthetic
survey whose injections are drawn from a broad population proxy with an exact draw density (addressing
Diagnosis B). Phase 3 will be re-run under the unchanged numerical intent before any real-data work.

## 11. Artifacts

Under `/hildafs/projects/phy220048p/magana/gwpop-search-data` (H100 host):

- `runs/phase3-recovery/default/` — plan (`cbdca92c…`), `run_000/chains/`
  (`chain_000.npz` `f2c30a6f…`, `.json` `aa322a77…`, `manifest.json` `c309510e…`);
- `logs/phase3/` — attempt logs, watcher log, fingerprints before/after
  (identical), py-spy locals of the resumed process;
- `logs/phase3/diag/` — frozen-point parity/scan JSON (`291ce74c…`), truth-point
  diagnostics for all runs, scaling/cliff/timing JSON;
- `scripts/phase3_diag_stuck_point.py`, `scripts/phase3_diag_scaling.py`,
  `scripts/phase3_interrupt_and_resume.sh`;
- `report/H100_VALIDATION_REPORT.md` — running audit record.
