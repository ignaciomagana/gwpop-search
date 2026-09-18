# Project state / durable handoff

Last updated: 2026-09-17

## Current phase

**Phase 2 â COMPLETE.**

The repository now contains the canonical data layer from Phase 1 and a standardized NumPy/JAX HBI likelihood from Phase 2. No GWTC-5 population inference has been run from this repository yet.

Package version: `0.2.0`.

## Project intent

Build `gwpop-search`: a standardized HBI package plus model-graph search system for GW population structure, initially GWTC-5 BBHs. The end state is suitable for handing to a high-compute worker on an H100 machine for production inference.

## Frozen architectural decisions

- The repo owns its own HBI implementation; it does not merely orchestrate an external population package.
- Search/model-selection logic sits above the HBI layer and cannot modify the production likelihood during a search.
- Population models will be declarative serialized specifications.
- Model-graph edges will represent exactly one controlled mutation.
- Final scientific model scoring will use evidence + explicit model priors, not the largest raw BF found by a broad search.
- Search-level claims will ultimately be calibrated by replaying the complete search on null catalogs.
- Flexible spline/GP/HSGP models are scouts for interpretable parametric descendants.
- Deterministic finite model search comes before agents.
- Data provenance/manifests remain intentionally deferred until the data/HBI/model interfaces are validated.
- PE priors and selection draw/reference densities are adapter inputs. HBI code never reconstructs them from filenames or release assumptions.
- PE and selection products carry an exact `CoordinateBasis` identity and must match before inference.
- Selection inputs have two distinct modes: `raw_draw` and `estimator_ready`.
- Generic inference code never floors a zero denominator density.
- A model callable returns the complete normalized population log density in the declared data basis. If a source-frame model needs advisory coordinates to transform into that basis, it must explicitly declare `required_fields` and own the Jacobian.

## gwcat reference inspected

Read-only reference:

```text
repository: ignaciomagana/gwcat
branch: master
commit: 8f9e2f12b499a6b2bf16ed938f66d020b12c44c2
```

Do not modify gwcat as part of this project.

Important gwcat contract carried into this project:

- current fitted core: `m1det, q, dL, ra, dec`;
- sky contributes the explicit density factor on both PE and selection sides;
- source masses, redshift, `m2det`, and derived spin quantities are advisory unless in the fitted space;
- gwcat `p_pe` is consumed directly as the PE denominator;
- gwcat `pdraw` is an estimator-ready denominator and already carries the documented multi-campaign mixture/exposure convention;
- component-spin exports may carry `chi_eff` as a derived/advisory column.

## Phase 1 implementation

Data layer:

- `src/gwpop_search/data/schema.py`
- `src/gwpop_search/data/posterior.py`
- `src/gwpop_search/data/selection.py`
- `src/gwpop_search/data/pair.py`
- `src/gwpop_search/data/fixtures.py`
- `src/gwpop_search/data/adapters/gwcat_v2.py`

Phase-1 local reference run after the final sky-basis correction:

```text
Python 3.13.5
16 passed
```

## Phase 2 implementation

New HBI modules:

- `src/gwpop_search/hbi/types.py`
  - explicit `HBIConfig` and `RateTreatment`
  - event, campaign, selection, catalog, and variance result types
- `src/gwpop_search/hbi/common.py`
  - population-density validation
  - event/selection importance ESS
  - maximum normalized weight
  - delta-method Monte-Carlo log-integral variance
  - model `required_fields` contract
  - raw-vs-estimator-ready selection normalization
- `src/gwpop_search/hbi/numpy_backend.py`
  - event importance-reweighting likelihood
  - raw multi-campaign selection estimator
  - estimator-ready selection path
  - shape/rate-marginalized likelihood
  - explicit-rate Poisson point-process likelihood
  - per-campaign diagnostics
  - selection chunking
- `src/gwpop_search/hbi/jax_backend.py`
  - padded/masked ragged PE representation
  - fixed-size selection chunks
  - `lax.scan` log-space selection accumulation
  - differentiable shape and Poisson likelihood builders
- `src/gwpop_search/hbi/__init__.py`
  - public NumPy API and lazy JAX builders

### Selection convention now pinned in HBI

For raw campaign `k`:

```text
A_k = T_k / N_draw,k * sum_detected p_pop / p_draw,k
A = sum_k A_k
```

Setting `HBIConfig.raw_selection_use_observing_time=False` explicitly changes the campaign factor to `1/N_draw,k`.

For `estimator_ready` selection products, including the current gwcat adapter:

```text
A = sum p_pop / pdraw
```

with no second `ndraw`, observing-time, or campaign-mixture factor.

### Likelihoods

Shape:

```text
log L = sum_i log ell_i - N log A
```

Poisson with explicit rate `R`:

```text
log L = sum_i log ell_i + N log R - R A
```

### Phase 2 tests

Added:

- `tests/test_hbi_numpy.py`
- `tests/test_hbi_jax.py`

The targeted Phase-2 local reference run used Python 3.13.5 and reported:

```text
14 passed in 6.04s
```

It covers analytic event reweighting, raw and estimator-ready selection normalization, shape/PPP formulas, permutation and chunk invariance, support failures, likelihood-variance diagnostics, toy recovery, NumPy/JAX equality, and JAX differentiation.

Important bookkeeping note: the Phase-1 `16 passed` and Phase-2 `14 passed` results were obtained in separate local reference checkouts. Do not report them as a single combined 30-test suite until a full repository checkout/CI run executes all tests together.

No GitHub Actions CI workflow has been added yet.

## Immediate next phase

**Phase 3: baseline normalized BBH population components + NumPyro inference.**

Implement in this order:

1. population-model interface wrapping normalized source-population densities into the adapter density basis;
2. explicit detector/source-frame + cosmology/Jacobian transform layer needed for the gwcat basis;
3. normalized primary-mass baseline (power law / broken power law / peak components as required by the chosen baseline);
4. normalized `q` conditional;
5. simple redshift/rate evolution;
6. normalized `chi_eff` baseline and one declared treatment of the remaining spin coordinates;
7. composed baseline BBH model with explicit hyperpriors;
8. numerical normalization tests across random hyperparameters;
9. NumPyro NUTS wrapper over the JAX HBI likelihood;
10. synthetic population + PE/selection recovery;
11. posterior serialization/checkpoint-restart path;
12. first Slurm/H100 run configuration, still using synthetic data.

Do not begin the declarative model grammar or automated model search before a conventional baseline model recovers from simulation under this HBI layer.

## Questions deliberately left open

- Exact production spin space: component vs projected alternatives.
- Exact gwcat export version/parameter-space selected for production.
- Final GWTC-5 BBH event cut and waveform policy.
- Exact baseline family used for the first GWTC-5 reproduction.
- Evidence estimator.
- Model-prior hyperparameters.
- PostgreSQL vs SQLite for distributed production state.
- Exact null-catalog PE approximation for large calibration campaigns.

## Production/H100 handoff requirement

Before handing to Fable for the full search, the repository must contain:

- environment/bootstrap instructions for the target CUDA/JAX stack;
- frozen dataset manifest and hashes;
- one command to validate data;
- one command to reproduce the baseline;
- one command to launch/resume the search;
- Slurm templates;
- deterministic seed policy;
- database/artifact locations;
- automatic checkpoint/restart;
- final acceptance tests.

## Rule for future ChatGPT/Codex/Fable work

Read this file, `SPEC.md`, `ROADMAP.md`, `docs/data_contract.md`, and `docs/hbi_contract.md` before modifying the architecture. Update this file whenever a phase completes or a scientific/software contract changes.
