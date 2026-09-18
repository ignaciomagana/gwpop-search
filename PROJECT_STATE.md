# Project state / durable handoff

Last updated: 2026-09-17

## Current phase

**Phase 3 - IN PROGRESS.**

Phases 0-2 are complete. Phase 3 now has a conventional normalized BBH model,
an explicit source-to-gwcat density transform, a working NumPyro/JAX inference
path, a closed synthetic PE+selection dataset, checkpoint/resume, CI, and an
H100/Slurm recovery entry point.

The remaining Phase-3 gate is a statistically meaningful multi-chain,
multi-seed synthetic recovery campaign on the H100. **Do not begin Phase 4
(model grammar/search) until that recovery gate passes.**

No production GWTC-5 population inference has been run from this repository.

Package version remains `0.2.0` until Phase 3 is complete.

## Project intent

Build `gwpop-search`: a standardized HBI package plus model-graph search
system for GW population structure, initially GWTC-5 BBHs. The eventual
production workflow will be handed to Fable on H100-class compute.

## Frozen architectural decisions

- The repo owns its own HBI implementation.
- Search/model-selection logic sits above HBI and cannot modify the production
  likelihood during a search.
- Population models will be declarative serialized specifications.
- Model-graph edges represent exactly one controlled mutation.
- Final scientific model scoring uses evidence + explicit model priors, not the
  largest raw Bayes factor found by a broad search.
- Search-level claims are ultimately calibrated by replaying the complete
  search on null catalogs.
- Flexible spline/GP/HSGP models are scouts for interpretable descendants.
- Deterministic finite model search comes before agents.
- Production data provenance/manifests remain intentionally deferred until the
  internal data/HBI/model/search interfaces are validated.
- PE priors and selection draw/reference densities are adapter inputs. HBI
  never reconstructs them from filenames or release assumptions.
- PE and selection products carry an exact `CoordinateBasis` identity.
- Selection inputs have distinct `raw_draw` and `estimator_ready` semantics.
- Generic inference code never floors a zero denominator density.
- A population model returns the complete normalized density in the declared
  data basis and owns every coordinate transform/Jacobian needed to get there.
- Phase-3 baseline inference uses the gwcat-v2 `chieff` space. The final
  production spin space remains open.
- Phase-4 search remains blocked until Phase-3 synthetic recovery is accepted.

## gwcat read-only reference

```text
repository: ignaciomagana/gwcat
branch: master
commit: 8f9e2f12b499a6b2bf16ed938f66d020b12c44c2
```

Do not modify gwcat as part of this project.

Important contract carried into `gwpop-search`:

- fitted core: `m1det, q, dL, ra, dec`;
- sky is part of the explicit density measure;
- source masses/redshift/derived quantities may be advisory columns;
- exported `p_pe` is the PE denominator;
- exported `pdraw` is estimator-ready and already carries its documented
  campaign/exposure convention;
- component-spin exports may carry `chi_eff` as a derived column.

## Phase 1 - canonical data layer

Implemented:

- `src/gwpop_search/data/schema.py`
- `src/gwpop_search/data/posterior.py`
- `src/gwpop_search/data/selection.py`
- `src/gwpop_search/data/pair.py`
- `src/gwpop_search/data/fixtures.py`
- `src/gwpop_search/data/adapters/gwcat_v2.py`

The gwcat-v2 basis constructor is now also exposed as
`gwcat_v2_basis_for_spin` so synthetic and real adapter products literally use
the same basis definition.

Historical Phase-1 local reference run:

```text
Python 3.13.5
16 passed
```

## Phase 2 - standardized HBI

Implemented:

- NumPy reference event importance reweighting;
- raw multi-campaign selection;
- estimator-ready gwcat selection path;
- rate-marginalized shape likelihood;
- explicit-rate Poisson point-process likelihood;
- event/selection importance ESS and maximum-weight diagnostics;
- likelihood Monte-Carlo variance diagnostics;
- selection chunking;
- differentiable JAX likelihood with NumPy parity.

Pinned raw campaign convention:

```text
A_k = T_k / N_draw,k * sum_detected p_pop / p_draw,k
A   = sum_k A_k
```

Pinned estimator-ready convention:

```text
A = sum p_pop / pdraw
```

with no second `ndraw`, observing-time, or campaign-mixture factor.

Likelihoods:

```text
shape:   log L = sum_i log ell_i - N log A
Poisson: log L = sum_i log ell_i + N log R - R A
```

## Phase 3 implementation landed

### Population components

`src/gwpop_search/models/components.py`

Normalized JAX components:

- power law;
- broken power law;
- power law + truncated Gaussian peak;
- conditional `q^beta` with the secondary-mass lower bound;
- redshift/rate density proportional to
  `dVc/dz * (1+z)^(kappa-1)`;
- truncated-Gaussian `chi_eff`.

The masked support algebra is explicitly autodiff-safe. CI exposed and fixed an
important bug here: undefined arithmetic inside masked-out support branches can
poison gradients even when the final density is `-inf`. The components now
use finite surrogate arithmetic internally and apply exact support masks last.

### Fixed cosmology and source-to-data transform

`src/gwpop_search/models/cosmology.py`

`FlatLambdaCDM` supplies fixed-cosmology `dL(z)`, `z(dL)`, `ddL/dz`, and
`dVc/dz`.

`src/gwpop_search/models/baseline.py`

`GwcatChiEffBBHModel` is normalized in source coordinates

```text
(m1_source, q, z, chi_eff)
```

and evaluates the gwcat detector-frame density

```text
(m1_detector, q, dL, dOmega, chi_eff)
```

with the explicit transformation

```text
p_det = p_src / [(1+z) * (ddL/dz)] * 1/(4*pi).
```

The model derives `z` from `dL` and `m1_source=m1_detector/(1+z)`; it does
not define the density using advisory source-frame columns.

### NumPyro inference

`src/gwpop_search/inference/priors.py`

Contains serializable `PriorSpec` objects and
`BASELINE_SYNTHETIC_PRIORS`. These priors are for Phase-3 recovery only and
are not the final GWTC-5 prior contract.

`src/gwpop_search/inference/numpyro.py`

Provides:

- lazy NumPyro dependency loading;
- explicit `NUTSConfig`;
- current default initialization at the prior median;
- shape-likelihood NumPyro model using the common JAX HBI engine;
- chain-grouped samples and HMC extra fields;
- deterministic per-chain seeds;
- chain-granularity checkpoint/resume;
- manifest mismatch protection;
- package version + git commit in the resume identity.

The median initializer is a reproducibility choice, not a workaround for an
invalid model: after the autodiff masking fix, the actual baseline NumPyro
smoke passes even with the older random initializer.

### Closed synthetic recovery dataset

`src/gwpop_search/inference/synthetic.py`

Produces:

- baseline-population detected event truths;
- PE posterior samples under an explicit detector-basis reference prior;
- one raw-draw selection campaign under an explicit detector-basis draw
  density;
- exact gwcat-v2 `chieff` coordinate-basis identity;
- a deterministic chirp-mass-scaled synthetic detection reach.

This is a validation survey, not a detector-realism model.

### Recovery campaign / H100 path

`src/gwpop_search/inference/recovery.py`

`gwpop-search synthetic-recovery` generates the mock and runs/resumes NUTS.

Outputs include:

- combined posterior;
- injected truth and 5/50/95 posterior summaries;
- divergences;
- split R-hat;
- MCMC effective sample size;
- PE event importance ESS;
- selection ESS;
- maximum normalized weights;
- shape-likelihood Monte-Carlo variance;
- the same HBI diagnostics at the injected truth and posterior median.

Runbook:

`docs/phase3_recovery.md`

Site-neutral H100 Slurm example:

`scripts/slurm/phase3_synthetic_h100.sbatch.example`

The template refuses to launch unless JAX sees a GPU and intentionally does
not guess site-specific partition/account/CUDA/JAX setup.

## Integrated CI status

GitHub Actions now installs `.[dev]`, including NumPyro, and runs the full
repository test suite on Python 3.12 with JAX x64 enabled.

Authoritative green checkpoints during Phase 3:

- autodiff-safe baseline + actual 10-parameter NumPyro smoke:
  **55 passed**;
- direct actual-baseline HBI value/gradient test:
  **56 passed**;
- recovery chain diagnostics:
  **57 passed**.

The newest recovery-importance diagnostic test was added after those green
checkpoints; consult the latest Actions run before quoting a newer total.

Tests now cover the Phase-1 data layer, Phase-2 NumPy/JAX HBI, component
normalization/Jacobians, actual baseline finite gradients, NumPyro execution,
checkpoint contracts, synthetic data generation, CLI parsing, and recovery
diagnostic utilities.

## Phase 3 remaining acceptance gate

Before Phase 4:

1. run the default synthetic campaign on H100 with at least 4 chains;
2. inspect split R-hat, MCMC ESS, divergences, event PE ESS, selection ESS,
   max weights, and likelihood Monte-Carlo variance;
3. repeat across multiple independent data seeds and sampler seeds;
4. verify recovery is statistically consistent across the ensemble, not just
   one fortunate catalog;
5. interrupt/resume at least one campaign and confirm completed chain artifacts
   are unchanged.

Do not declare coverage because every truth happens to land in one catalog's
credible interval.

The real-data baseline parity check is intentionally deferred to the later
GWTC-5 production freeze because production data manifests, event cuts,
waveform policy, and selection campaigns were explicitly deferred by project
design.

## Immediate next action

Hand the current Phase-3 recovery command to the H100/Fable environment and
execute the multi-seed recovery matrix described in
`docs/phase3_recovery.md`.

**Do not start Phase 4 yet.**

## Questions deliberately left open

- final production spin space;
- exact gwcat export/parameter space for production;
- GWTC-5 BBH event cut and waveform policy;
- final reference baseline family/priors for GWTC-5 parity;
- evidence estimator;
- model-prior hyperparameters;
- PostgreSQL vs SQLite for distributed production state;
- null-catalog PE approximation for search calibration.

## Production/H100 handoff requirements still outstanding

Before the later full-search Fable handoff:

- target CUDA/JAX environment/bootstrap instructions;
- frozen production dataset manifest and hashes;
- data validation command;
- real-data baseline reproduction command;
- search launch/resume command;
- production Slurm templates;
- deterministic seed policy;
- database/artifact locations;
- full search checkpoint/restart;
- final production acceptance tests.

## Rule for future ChatGPT/Codex/Fable work

Read this file, `SPEC.md`, `ROADMAP.md`, `docs/data_contract.md`,
`docs/hbi_contract.md`, and `docs/phase3_recovery.md` before modifying the
architecture. Update this file whenever a phase completes or a scientific or
software contract changes.
