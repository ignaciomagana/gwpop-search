# gwpop-search

Systematic gravitational-wave population-model search with a standardized hierarchical Bayesian inference (HBI) engine.

The project separates four concerns:

1. **data contract** — posterior-event samples and detected selection injections enter through stable, validated interfaces;
2. **HBI engine** — one implementation of the hierarchical likelihood, selection integral, diagnostics, and later evidence calculations;
3. **model grammar** — population models are declarative objects connected by controlled one-mutation edges;
4. **search layer** — deterministic enumeration first, autonomous proposal/scheduling only after the inference engine is validated.

The immediate target is GWTC-5 BBH population inference, but the core package contains no GWTC-release-specific ingestion logic.

## Status

**Phase 1 complete: canonical data containers and gwcat-v2 adapter.**

Implemented:

- typed density-coordinate identities with strict PE/selection basis matching;
- ragged `PosteriorCatalog` using concatenated sample columns + offsets;
- per-event parameter availability checks;
- explicit finite log PE reference densities;
- campaign-aware `SelectionCatalog`;
- distinct `raw_draw` and `estimator_ready` selection modes;
- internal HDF5 round-trip schemas for PE and selection containers;
- deterministic synthetic fixtures;
- read-only adapter for validated gwcat v2 PE/selection exports;
- tests that preserve gwcat `p_pe`/`pdraw` denominators without rebuilding priors or double-applying `ndraw`.

The gwcat adapter represents the density measure actually covered by current gwcat parameter spaces: detector-frame primary mass, mass ratio, luminosity distance, sky, and the selected spin coordinates. Source masses, redshift, and derived spin quantities remain advisory sample columns until an explicit population-density transform is implemented.

## Next

**Phase 2: standardized HBI engine.**

The next implementation is the trusted event reweighting + selection likelihood, first in NumPy as a reference and then in JAX, with event/selection ESS and likelihood-variance diagnostics. Population-model families still come later.

See:

- `SPEC.md` — scientific and software specification.
- `ROADMAP.md` — phased build plan and acceptance criteria.
- `PROJECT_STATE.md` — durable handoff / build memory.
- `docs/data_contract.md` — implemented PE and selection interfaces.
- `docs/hbi_contract.md` — Phase-2 HBI contract.

## Design rule

Agents and search algorithms may choose **which legal model to evaluate next**. They do not get to modify the likelihood, data transformations, priors, or validation rules during a production search.
