# Build roadmap

The build is deliberately phased. A phase is complete only when its acceptance tests pass; later phases must not silently change earlier scientific contracts.

## Phase 0 — architecture and durable project state

**Goal:** freeze boundaries before implementation.

Deliverables:

- repository/package structure;
- scientific/software specification;
- PE and selection data contracts;
- HBI likelihood contract;
- persistent `PROJECT_STATE.md`;
- minimal packaging scaffold.

Acceptance:

- no GWTC-specific logic in the HBI interface;
- explicit coordinate/reference-density contracts;
- clear separation between deterministic inference and agents;
- subsequent work can resume from the repository alone.

## Phase 1 — canonical data containers and synthetic fixtures

**Goal:** make the HBI engine independent of release formats.

Implement:

- `PosteriorCatalog` interface;
- `SelectionCatalog` and per-campaign metadata interface;
- ragged event samples via concatenated columns + offsets;
- parameter availability checks;
- canonical log-reference-density fields;
- tiny synthetic PE and injection fixtures;
- validation of support, finite weights, offsets, dimensions.

A gwcat adapter should be added here, initially for the selected stable export contract rather than raw PESummary/LVK files.

Acceptance:

- fake catalogs round-trip;
- missing required coordinates fail loudly;
- PE and selection basis mismatch fails before inference;
- no release-name heuristics in HBI code.

## Phase 2 — standardized HBI engine

**Goal:** one trusted population likelihood.

Implement:

- event importance-reweighting likelihood;
- selection estimator;
- rate-marginalized shape likelihood;
- explicit Poisson-rate likelihood;
- JAX vectorization/JIT path;
- chunking for large injections;
- event ESS and selection ESS;
- likelihood-variance diagnostics;
- per-event log-likelihood contributions;
- deterministic CPU tests against direct numerical calculations.

Acceptance:

- analytic/toy populations recover known hyperparameters;
- JAX and NumPy reference implementations agree;
- likelihood is invariant to harmless sample ordering/chunking;
- deliberate support/reference-density errors are caught.

## Phase 3 — baseline population components

**Goal:** reproduce a small conventional BBH analysis without search.

Implement normalized components for:

- power-law / broken-power-law mass distributions;
- one/two mass peaks as needed;
- power-law q;
- simple redshift evolution;
- truncated chi_eff Gaussian;
- simple chi_p or component-spin option;
- mixture composition.

Inference backend:

- NumPyro NUTS first;
- posterior serialization;
- restartable Slurm/H100 configuration.

Acceptance:

- simulation recovery;
- baseline run stable under seeds;
- selected reference baseline approximately reproduces the declared comparison analysis when the same data/cuts/priors are used.

## Phase 4 — declarative model grammar

**Goal:** candidate models are data, not handwritten run scripts.

Implement:

- model-spec schema;
- canonical serialization and hashing;
- prior schema;
- component registry;
- typed mutation registry;
- graph parent/child relation;
- static validation of incompatible combinations.

Acceptance:

- same canonical model -> same hash;
- one mutation -> exactly one structural change;
- illegal/unidentifiable combinations reject before sampling;
- a fixed list of ~20--40 models can be enumerated deterministically.

## Phase 5 — evidence and deterministic model graph

**Goal:** score the first finite search space.

Implement:

- common evidence backend interface;
- chosen production evidence estimator;
- repeat-estimate uncertainty;
- pairwise edge comparisons;
- model priors;
- posterior model probabilities;
- graph/report generation.

Acceptance:

- evidence estimator validated on problems with known evidence;
- repeated estimates agree within declared tolerance;
- first finite model DAG produced end-to-end.

## Phase 6 — multi-fidelity scheduler

**Goal:** stop spending production compute on dead branches.

Implement:

- F0/F1/F2/F3/F4 run classes;
- promotion/pruning policy;
- compute-cost accounting;
- deterministic beam search;
- exploration quota.

Acceptance:

- scheduler replay is deterministic given state/seeds;
- screening decisions never overwrite production results;
- promotion history is queryable.

## Phase 7 — flexible structure scouts

**Goal:** use flexible models to propose interpretable additions to the grammar.

Implement:

- spline/HSGP residual interface;
- residual summaries;
- candidate structural dependency extraction;
- human-reviewable grammar extension workflow.

Acceptance:

- injected correlation structures are recovered by the scout;
- null mocks do not automatically become permanent grammar additions;
- compiled parametric descendants can be independently fit.

## Phase 8 — search calibration and adversarial validation

**Goal:** quantify the look-elsewhere effect of the entire search.

Implement:

- population catalog simulator;
- selection + measurement/PE approximation pipeline;
- full search replay on nulls;
- holdout event prediction;
- leave-one-out/loud-event stress tests;
- alternative nearby baselines.

Acceptance:

- calibrated null distribution of search-level statistics;
- known injected structure recovered at expected frequency;
- false discoveries measured under declared nulls.

## Phase 9 — agents / swarm

**Goal:** automate experiment selection, not mathematical truth.

Roles:

- search proposer;
- numerical diagnostic critic;
- residual/structure proposer;
- adversarial validator;
- synthesis/report agent.

Agents interact only through typed tasks/results and the durable database. They cannot patch the production likelihood during a search.

Acceptance:

- disabling all agents leaves a valid deterministic pipeline;
- every agent action is reproducible as a recorded proposed task;
- no untracked mutation enters a production comparison.

## Phase 10 — GWTC-5 production / H100 handoff

Freeze:

- data provenance and manifests;
- event selection;
- PE sample-set/waveform policy;
- O3/O4 selection campaigns;
- production priors;
- model prior;
- evidence configuration;
- H100 Slurm/JAX environment;
- search budget.

Then hand the repository to the production inference worker (Fable) with a single executable runbook and machine-readable campaign configuration.

## Commit discipline

At the end of every phase:

1. update `PROJECT_STATE.md`;
2. record decisions that future phases must not silently revisit;
3. list unresolved scientific choices explicitly;
4. leave the working tree runnable/tests passing;
5. commit before beginning the next phase.
