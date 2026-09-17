# Project state / durable handoff

Last updated: 2026-09-17

## Current phase

**Phase 1 — COMPLETE.**

The repository now has canonical PE/selection data containers, explicit density-basis contracts, synthetic fixtures, and a tested gwcat-v2 adapter boundary. No GWTC-5 population inference has been run from this repository.

## Project intent

Build `gwpop-search`: a standardized HBI package plus model-graph search system for GW population structure, initially GWTC-5 BBHs. The end state is suitable for handing to a high-compute worker on an H100 machine for production inference.

## Decisions already made

- The repo owns its own HBI implementation; it will not merely orchestrate an external population package.
- Search/model-selection logic is above the HBI layer and cannot modify the production likelihood during a search.
- Population models are declarative serialized specifications.
- Model-graph edges represent exactly one controlled mutation.
- Scientific model scoring eventually uses evidence + explicit model priors, not the largest raw BF found by a broad search.
- Search-level claims must ultimately be calibrated by replaying the complete search on null catalogs.
- Flexible spline/GP/HSGP models are scouts that can suggest interpretable parametric descendants.
- Build deterministic finite model search before adding agents.
- Data provenance/manifests remain intentionally deferred until the internal data/HBI interfaces are working.
- PE prior and selection draw/reference densities are adapter inputs. HBI code must never reconstruct them from filenames or release assumptions.
- PE and selection products carry an exact `CoordinateBasis` identity and must match before inference.
- Selection inputs have two distinct modes: `raw_draw` and `estimator_ready`. The HBI engine must dispatch on the mode and cannot silently apply both normalization conventions.
- Generic inference code never floors a zero denominator density.

## gwcat reference inspected

Read-only reference:

```
repository: ignaciomagana/gwcat
branch: master
commit: 8f9e2f12b499a6b2bf16ed938f66d020b12c44c2
```

Do not modify gwcat as part of this project.

Important contract learned from that revision:

- the current registered fitted core is `m1det, q, dL, ra, dec`;
- sky contributes the same explicit isotropic density factor on PE and selection sides;
- source masses, redshift, m2det, and derived spin quantities are advisory unless they belong to the chosen fitted space;
- gwcat `p_pe` is consumed directly as the PE denominator;
- gwcat `pdraw` is an estimator-ready denominator and already carries the documented multi-campaign mixture/exposure convention;
- the component-spin space carries `chi_eff` only as a derived/advisory column.

The Phase-1 adapter therefore preserves the gwcat exported density basis rather than converting it to source-frame coordinates. Source-population-to-export-basis transforms belong in the HBI/model layer and must carry explicit Jacobians.

## Phase 1 implementation

New package modules:

- `src/gwpop_search/data/schema.py`
  - `CoordinateBasis`
  - basis hashing / exact pair compatibility
  - denominator-density validation
  - typed contract errors
- `src/gwpop_search/data/posterior.py`
  - ragged `PosteriorCatalog`
  - concatenated columns + offsets
  - per-event availability
  - finite `log_ref_density`
  - internal HDF5 round trip
- `src/gwpop_search/data/selection.py`
  - `SelectionCatalog`
  - `Campaign`
  - `SelectionMode.RAW_DRAW`
  - `SelectionMode.ESTIMATOR_READY`
  - internal HDF5 round trip
- `src/gwpop_search/data/pair.py`
  - PE/selection cross-validation
- `src/gwpop_search/data/fixtures.py`
  - deterministic ragged PE + two-campaign raw-draw fixtures
- `src/gwpop_search/data/adapters/gwcat_v2.py`
  - gwcat PE 2.0/2.1 adapter
  - gwcat selection 2.0/2.1 adapter
  - supported spaces: chieff, chieff_chip, component

Package version is now `0.1.0`.

## Phase 1 tests

Added:

- `tests/test_data_schema.py`
- `tests/test_data_containers.py`
- `tests/test_gwcat_v2_adapter.py`
- updated package smoke test

Local reference run after the final sky-basis correction:

```
Python 3.13.5
16 passed
```

The gwcat adapter parity test checks that adaptation preserves

```
sum p_pop / pdraw
```

and explicitly checks that the adapter does not divide by `ndraw` again.

No CI workflow has been added yet; the passing test count above is the local Phase-1 acceptance run.

## Immediate next phase

**Phase 2: standardized HBI engine.**

Implement in this order:

1. NumPy reference event-reweighting likelihood with per-event contributions.
2. NumPy raw-draw selection estimator with per-campaign accounting.
3. NumPy estimator-ready selection path for gwcat products.
4. shape/rate-marginalized catalog likelihood.
5. explicit Poisson point-process rate likelihood.
6. event importance ESS + max-weight diagnostics.
7. selection ESS + per-campaign diagnostics.
8. likelihood-variance diagnostics.
9. JAX implementation with parity against NumPy.
10. selection chunking with chunk-size invariance tests.

Phase 2 acceptance remains:

- analytic/toy likelihood checks;
- NumPy/JAX equality;
- permutation/chunking invariance;
- deliberate reference-density/support failures caught;
- simulation recovery before real GWTC-5 inference.

Do not start the model grammar or baseline population-family expansion before this likelihood layer passes.

## Questions deliberately left open

- Exact production spin space: component vs projected alternatives.
- Exact gwcat export version/parameter-space selected for production.
- Final GWTC-5 BBH event cut and waveform policy.
- Evidence estimator.
- Model-prior hyperparameters.
- PostgreSQL vs SQLite for distributed production state.
- Exact null-catalog PE approximation for large calibration campaigns.

## Production/H100 handoff requirement

Before handing to Fable, the repository must contain:

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
