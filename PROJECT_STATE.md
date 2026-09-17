# Project state / durable handoff

Last updated: 2026-09-17

## Current phase

**Phase 0 — COMPLETE.**

The repository is initialized, installable, and carries the scientific/software contracts needed to begin implementation. No scientific inference has been run from this repository.

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
- Data provenance/manifests are intentionally deferred until the internal data/HBI interfaces are working.
- Input layout should be gwcat-like for both PE and selection products: explicit schemas, concatenated/ragged sample handling where useful, strict required-parameter checks, and explicit reference-density semantics.
- `gwcat` was inspected read-only on 2026-09-17. Do not modify it as part of this project.

## gwcat reference inspected

Read-only reference:

```
repository: ignaciomagana/gwcat
branch: master
commit: 8f9e2f12b499a6b2bf16ed938f66d020b12c44c2
```

Useful conventions observed there:

- PE store uses concatenated 1-D sample columns with an integer offsets index.
- It stores the union of parameters and an event-by-parameter availability mask rather than shrinking to the intersection.
- Exports declare required parameters and fail loudly when an event cannot supply them.
- Selection is a first-class object (`SelectionSet` / `CombinedSelectionSet`) rather than an anonymous array.
- Multi-campaign selection normalization and `pdraw` semantics are explicit.
- PE `p_pe` and selection `pdraw` encode reference-density/Jacobian choices; downstream code must not silently apply those factors twice.
- Spin basis and projection/reference assumptions are explicit and validated.
- v2 export products pair PE and selection schemas and cross-check compatibility.

For `gwpop-search`, prefer consuming a stable gwcat export/adapter contract over copying release-specific ingestion logic.

## Phase 0 files

- `README.md`
- `SPEC.md`
- `ROADMAP.md`
- `PROJECT_STATE.md`
- `docs/data_contract.md`
- `docs/hbi_contract.md`
- `pyproject.toml`
- `src/gwpop_search/__init__.py`
- `src/gwpop_search/cli.py`
- `tests/test_package.py`

## Immediate next phase

**Phase 1: canonical internal data containers and fake fixtures.**

Suggested commit sequence:

1. `data/schema.py`: coordinate-space metadata, required-field validation, basis identity.
2. `data/posterior.py`: ragged `PosteriorCatalog` with offsets and `log_ref_density`.
3. `data/selection.py`: campaign-aware `SelectionCatalog` with raw-draw vs estimator-ready semantics.
4. synthetic fixture writers/readers and failure-mode tests.
5. adapter boundary for gwcat v2 products, with parity tests but no production data manifest yet.

Only after these pass should Phase 2 implement the likelihood.

## Questions deliberately left open

- Exact canonical spin basis for the first production sweep: chi_eff-only vs chi_eff+chi_p vs component spins.
- Exact gwcat export version/parameter-space selected for production.
- Final GWTC-5 BBH event cut and waveform policy.
- Exact multi-campaign exposure convention used by the canonical selection adapter; this must be pinned by a parity test, not memory.
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

Read this file, `SPEC.md`, and `ROADMAP.md` before modifying the architecture. Update this file whenever a phase completes or a scientific/software contract changes.
