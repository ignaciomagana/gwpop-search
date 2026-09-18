# Project state / durable handoff

Last updated: 2026-09-17

## HARD ACCEPTANCE GATE — DO NOT OVERRIDE

**Phase 3 is NOT accepted until the H100 multi-seed recovery campaign is
actually executed and reviewed.**

Implementation, CI, synthetic smoke tests, downstream Phase-4--10 staging, or
successful local/unit tests do **not** satisfy this gate. Future ChatGPT,
Codex, Fable, and human work must continue to report Phase 3 as open until the
H100 campaign results have been inspected against the acceptance criteria below
and the durable state is explicitly updated after that review.

## Scientific acceptance state

**Phase 3 remains the open scientific gate.**

Phases 0--2 are accepted. The software for Phases 4--10 has now been staged,
integrated, and extensively CI-tested so development can continue without
waiting for H100 access, but those later phases are **not declared
scientifically accepted out of order**.

The remaining Phase-3 requirement is a statistically meaningful multi-chain,
multi-seed synthetic recovery campaign on H100, followed by review of numerical
diagnostics, ensemble recovery, and at least one checkpoint/resume integrity
test.

No production GWTC-5 population inference has been run from this repository.

Package version remains `0.2.0` until the Phase-3 acceptance gate is closed.

## Current software checkpoint

Authoritative fully green integrated checkpoint:

~~~text
commit: 001025b4fb76e5dc8d7225316ccad21e7272d2b4
tests:  143 passed
CI:     GitHub Actions / Python 3.12 / JAX x64
~~~

Commits after that checkpoint add the explicit freeze CLI and freeze-builder
tests. Consult current Actions before quoting a newer test total.

## Project intent

Build `gwpop-search`: a standardized HBI package plus deterministic
model-graph search system for GW population structure, initially GWTC-5 BBHs.
An optional agent/swarm layer sits above the deterministic core. Production
inference will be handed to Fable on H100-class compute after the scientific
and data freezes are accepted.

## Frozen architectural decisions

- The repository owns one standardized HBI implementation.
- Search/model-selection logic cannot modify the production likelihood.
- Population models are declarative serialized specifications.
- The model hash includes the scientific structure **and hyperpriors**.
- Model-graph edges represent exactly one controlled structural mutation.
- PE priors and selection draw/reference densities are explicit adapter inputs.
- PE and selection products carry an exact `CoordinateBasis` identity.
- Selection inputs retain distinct `raw_draw` and `estimator_ready`
  semantics.
- Generic inference code never floors a zero denominator density.
- A population model owns every transform/Jacobian required to return the
  complete normalized density in the declared data basis.
- Phase-3 baseline inference uses the gwcat-v2 `chieff` space.
- F1/F2 scores are **compute-allocation statistics only** and are never reported
  as Bayes factors.
- Scientific model comparison uses evidence plus an explicit model prior.
- Posterior model probabilities over a declared graph are written only when
  every graph node has proper F3/F4 evidence.
- Search-level null calibration replays the same frozen search procedure; it is
  not a replacement for structural model priors.
- Flexible HSGP/spline models are scouts for interpretable parametric
  descendants, not automatic discoveries.
- Agents may propose typed tasks but cannot inject code, arbitrary mutations, or
  likelihood changes.
- The deterministic pipeline remains valid with agents disabled.
- Production data and campaign freezes are content/hash addressed and pinned to
  an exact git revision.

## gwcat read-only reference

~~~text
repository: ignaciomagana/gwcat
branch: master
commit: 8f9e2f12b499a6b2bf16ed938f66d020b12c44c2
~~~

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
- `src/gwpop_search/data/thinning.py`

The gwcat-v2 basis constructor is exposed as `gwcat_v2_basis_for_spin`.

Screening reductions are explicit:

- PE samples are randomly thinned per event;
- selection rows are stratified by campaign;
- selection thinning carries an inclusion-probability/Horvitz--Thompson
  correction by shifting the stored draw density by `log(f)`.

Reduced catalogs are for F0/F1 screening only.

## Phase 2 - standardized HBI

Implemented:

- NumPy reference event importance reweighting;
- raw multi-campaign selection;
- estimator-ready gwcat selection;
- rate-marginalized shape likelihood;
- explicit-rate Poisson point-process likelihood;
- differentiable JAX likelihood with NumPy parity;
- selection chunking;
- event/selection ESS;
- maximum normalized importance weights;
- likelihood Monte-Carlo variance diagnostics.

Pinned raw campaign convention:

~~~text
A_k = T_k / N_draw,k * sum_detected p_pop / p_draw,k
A   = sum_k A_k
~~~

Pinned estimator-ready convention:

~~~text
A = sum p_pop / pdraw
~~~

with no second `ndraw`, observing-time, or campaign-mixture factor.

Likelihoods:

~~~text
shape:   log L = sum_i log ell_i - N log A
Poisson: log L = sum_i log ell_i + N log R - R A
~~~

## Phase 3 - implemented, acceptance run pending

Implemented:

- conventional normalized BBH baseline;
- fixed flat-LambdaCDM source-to-gwcat transform;
- explicit detector-frame Jacobian;
- NumPyro NUTS;
- deterministic chain seeds;
- chain-granularity checkpoint/resume;
- code-pinned manifests;
- closed synthetic PE + raw-selection survey;
- multi-seed recovery campaign controller;
- numerical campaign gate;
- ensemble recovery/coverage summaries;
- checkpoint fingerprints;
- H100 Slurm entry point.

Recommended command:

~~~bash
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
~~~

See `docs/phase3_recovery.md` and
`scripts/slurm/phase3_synthetic_h100.sbatch.example`.

### Phase-3 acceptance gate

Before declaring Phase 3 accepted:

1. run at least four independent synthetic catalogs with at least four chains;
2. require no unresolved R-hat, MCMC ESS, divergence, PE/selection ESS,
   maximum-weight, or likelihood-variance pathology;
3. inspect recovery across the ensemble rather than one catalog;
4. inspect standardized offsets for repeated bias;
5. interrupt/resume at least one campaign and confirm completed chain
   fingerprints are unchanged.

A few truth-in-credible-interval indicators are diagnostics, not a calibrated
coverage measurement.

## Phase 4 - declarative model grammar (staged)

Implemented:

- canonical `ModelSpec`;
- canonical JSON/YAML serialization;
- SHA-256 scientific model identity;
- hyperprior schema;
- component registry;
- typed one-axis mutation registry;
- deterministic breadth-first graph enumeration;
- verified graph loader;
- static component/option validation;
- compilation of legal specs back to normalized JAX gwcat-basis densities.

Initial compiled families include:

- power law, PL+peak, broken power law, PL+two peaks;
- power-law q and truncated-Gaussian q;
- constant/linear/logistic q-mass structure where defined;
- truncated-Gaussian `chi_eff` with selected mean/width dependencies;
- two-component `chi_eff` mixture;
- power-law and Madau--Dickinson-like redshift evolution.

Underspecified generic latent-mixture mutations were deliberately excluded from
the first graph.

## Phase 5 - evidence and model scoring (staged)

Implemented:

- JAXNS evidence through NumPyro's official nested-sampling bridge;
- evidence result serialization;
- repeated evidence estimates;
- reported JAXNS uncertainty and between-repeat scatter;
- evidence cache manifests pinned to model, data identity, basis, events,
  counts, HBI config, evidence config, seed, and code;
- explicit model-prior interface;
- uniform and structural-complexity model priors;
- edge log Bayes factors and posterior odds;
- posterior model probabilities;
- posterior mass for structural axes.

The evidence backend has an analytic known-evidence CI test.

Important: a production `scored_graph.json` is emitted only with complete
F3/F4 evidence coverage over the declared graph.

## Phase 6 - deterministic multi-fidelity search (staged)

Implemented:

- F0--F4 fidelity enum/contracts;
- deterministic beam scheduling;
- deterministic exploration quota;
- diagnostic veto;
- append-only SQLite state;
- idempotent promotion history;
- deterministic seeds;
- resumable end-to-end executor;
- per-fidelity model-count budgets;
- durable cumulative compute budget.

Concrete evaluator:

- F0: reduced-data HBI sanity;
- F1: short reduced-data NUTS, allocation-only BIC-like score;
- F2: full-data NUTS, allocation-only BIC-like score;
- F3: repeated full-data JAXNS evidence;
- F4: strict full-data NUTS + repeated evidence.

All stages use the same population compiler and standardized HBI likelihood.

## Phase 7 - flexible scouts (partially staged)

Implemented:

- HSGP Laplacian basis utilities;
- squared-exponential spectral weights;
- tensor basis construction;
- weighted residual-dependence summaries;
- typed `StructureProposal`;
- mapping only to already registered legal mutations;
- injected-dependence and null-proposal tests.

Still outstanding before Phase 7 acceptance:

- full flexible residual HBI inference model;
- end-to-end injected-correlation recovery with selection/PE;
- production scout diagnostics and compiled-descendant comparison.

## Phase 8 - search calibration/adversarial validation (partially staged)

Implemented:

- deterministic event folds;
- held-out detected-event predictive score using `ell_i/A`;
- search-level null replay storage/calibration;
- finite-sample corrected empirical tail probabilities;
- exact-search baseline-null bridge:
  baseline population -> synthetic PE/selection -> same F0--F4 search ->
  maximum encountered BF/posterior-odds statistic.

Still outstanding before Phase 8 acceptance:

- large null replay campaign;
- structured-injection recovery-frequency campaign;
- production leave-one-out/loud-event stress orchestration;
- nearby-baseline production suite.

## Phase 9 - optional agents (staged safety boundary)

Implemented:

- typed roles/tasks;
- inert JSON-like proposal payloads;
- registered-model/mutation/validation checks;
- deterministic task IDs;
- append-only proposal/decision audit log;
- agents-disabled path;
- rejection of arbitrary executable/code payload fields.

No agent is allowed to patch the production likelihood or create an
unregistered scientific mutation.

The deterministic pipeline is the source of truth. Agent-provider orchestration
can be added later without changing the scientific core.

## Phase 10 - production freeze/H100 handoff machinery (staged)

Implemented:

- SHA-256 dataset manifest;
- explicit event-selection and waveform-policy metadata;
- PE/selection artifact checksum and size verification;
- canonical model-graph hash/root;
- versioned production campaign schema (`1.1`);
- full frozen F0--F4 numerical config;
- frozen model prior, scheduler, seed policy, compute budget;
- artifact/state locations;
- exact git-commit binding;
- freeze validation CLI;
- deterministic production runner;
- H100 Slurm template;
- Fable runbook;
- explicit dataset/campaign freeze builders.

Operator flow:

~~~bash
gwpop-search freeze-dataset ...
gwpop-search write-default-fidelity-config --output fidelity.json
# review/edit fidelity.json
gwpop-search freeze-production-campaign ...
gwpop-search validate-production-freeze ...
gwpop-search run-production-search ...
~~~

The final command is also the resume command.

See `docs/fable_h100_handoff.md` and
`scripts/slurm/production_search_h100.sbatch.example`.

### Still scientifically deferred for actual GWTC-5 production

- final spin coordinate space;
- exact GWTC-5 BBH event cut;
- waveform/sample-set policy;
- exact O3/O4 selection products;
- production population hyperpriors;
- structural model-prior hyperparameter(s);
- search budget;
- final frozen data manifest and hashes.

Those choices must be made explicitly at the production freeze. The code must
not infer them from filenames or release conventions.

## Immediate next actions

1. Run/accept the Phase-3 H100 multi-seed recovery campaign.
2. Finish the Phase-7 full HSGP residual inference scout.
3. Add production stress-suite orchestration for Phase 8.
4. Once the real GWTC-5/gwcat PE+selection products and scientific policies are
   frozen, create `dataset_manifest.json`, reviewed `fidelity.json`,
   `model_graph.json`, and `campaign.json`.
5. Validate and hand the exact campaign/commit to Fable.

## Rule for future ChatGPT/Codex/Fable work

Read this file, `SPEC.md`, `ROADMAP.md`, `docs/data_contract.md`,
`docs/hbi_contract.md`, `docs/phase3_recovery.md`, and
`docs/fable_h100_handoff.md` before modifying architecture.

Do not silently change an accepted scientific contract. Update this file
whenever a contract changes, a phase acceptance gate closes, or a major staged
subsystem becomes runnable.
