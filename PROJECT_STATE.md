# Project state / durable handoff

Last updated: 2026-09-18

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
commit: e6d8aef86eac2b37478d1488a53f33d6d1ea5c7c
tests:  272 passed
CI:     GitHub Actions / Python 3.12 / JAX x64
~~~

Commits after that checkpoint are handover/documentation synchronization unless
this file is updated again. Consult current Actions before quoting a newer
total.

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
- checkpoint fingerprints plus `fingerprint-recovery-checkpoint` CLI;
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

## Phase 7 - flexible scouts (software staged; H100 acceptance pending)

Implemented:

- conditionally normalized HSGP residual population models;
- Laplacian/tensor basis and squared-exponential spectral weights;
- full standardized-HBI/selection-aware NumPyro scout inference;
- versioned scout configurations and strict resume manifests;
- MCMC + PE/selection importance diagnostics and proposal firewall;
- posterior conditional-moment summaries;
- typed `StructureProposal` objects mapped only to registered legal mutations;
- exact descendant `ModelSpec` compilation;
- explicit human accept/reject review records;
- provenance-pinned export of baseline hyperparameters from valid F3/F4 fits;
- independent full-HBI F3 parent/child evidence comparison after acceptance;
- grammar-matched structured synthetic injections for every supported linear
  scout dependence versus `m1_source`, `q`, and `z`;
- injection strengths constrained to the registered child-prior support;
- resumable multi-seed structured-scout campaigns and null/off-target controls;
- frozen eight-seed engineering gate (100% numerical pass, >=75% recovery of
  reachable strong injections, <=25% off-target-run fraction, <=25% null
  proposal fraction);
- deterministic 88-run H100 matrix covering q(m1) and chi_eff dependence on
  m1, q, and z;
- exact injected-descendant confirmation on the same structured PE/selection
  catalog through the normal review/materialization path plus independent F3
  parent/child evidence.

Still outstanding before Phase 7 scientific acceptance:

- run and review the declared multi-seed H100 structured-injection/null matrix;
- verify each production-enabled scout axis recovers declared strong injections
  without unacceptable null/off-target proposal behavior;
- verify at least one accepted injected proposal is independently refit and
  evidence-compared successfully using
  `confirm-structured-scout-descendant`.

No scout result automatically changes the production graph.

## Phase 8 - search calibration/adversarial validation (software staged; H100 acceptance pending)

Implemented:

- deterministic event folds and detected-event predictive density `ell_i/A`;
- resumable K-fold holdout campaigns that refit each selected model on training
  folds with strict full-data NUTS diagnostics before scoring held-out events;
- canonical event subsetting/drop operations for ragged PE catalogs;
- versioned leave-one-out/custom/loud-event stress suites;
- deterministic reruns from explicit nearby baseline `ModelSpec` roots;
- within-dataset edge-Bayes-factor and mutation-support comparisons;
- search-level null replay storage and finite-sample empirical tail calibration;
- frozen exact-null campaign configuration with explicit null population truth;
- production exact-null mode that resamples detected truths from the frozen
  estimator-ready production selection with relative weight
  `p_pop(theta|Lambda_null)/pdraw(theta)`;
- null-selection preflight reporting resampling ESS and maximum discrete weight;
- default resampling ESS floor of 200 and independent 12 GPU-hour per-null
  ceiling frozen in exact-null config v1.3;
- exact null replay of the **same final scientific procedure** as observed data:
  adaptive F0--F4 search plus full valid evidence completion;
- observed-state calibration blocked until the observed graph has complete valid
  evidence;
- structured-scout injection/null campaigns described in Phase 7.

Still outstanding before Phase 8 scientific acceptance:

- execute the predeclared H100 structured-injection/null campaigns;
- execute the selected production holdout/event-drop/nearby-baseline suites;
- execute the frozen exact-search null campaign and inspect the empirical
  maximum-BF/posterior-odds distribution;
- document the measured false-proposal/search-tail behavior rather than
  substituting nominal single-comparison thresholds.

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

- audited/idempotent gwcat-v2 -> canonical PE/selection HDF5 conversion with
  explicit spin-basis requirement and source/output SHA-256 report;
- SHA-256 dataset manifest;
- explicit event-selection and waveform-policy metadata;
- PE/selection artifact checksum and size verification;
- canonical model-graph hash/root plus graph inspection/exact model extraction;
- versioned production campaign schema (`1.1`);
- full frozen F0--F4 numerical config;
- frozen model prior, scheduler, seed policy, compute budget;
- artifact/state locations;
- exact git-commit binding;
- freeze validation CLI;
- deterministic production runner;
- explicit full-graph evidence-completion stage required before normalized model
  probabilities;
- invalid F3/F4 evaluations excluded from scientific evidence;
- H100 Slurm template runs adaptive search then evidence completion;
- production/Fable runbook;
- final-form Claude H100 validation/computation execution contract;
- H100 validation report/artifact template;
- deterministic HSGP validation shell matrix + Slurm wrapper;
- explicit dataset/campaign freeze builders.

Operator flow:

~~~bash
gwpop-search canonicalize-gwcat-v2 ...
gwpop-search freeze-dataset ...
gwpop-search write-default-fidelity-config --output fidelity.json
# review/edit fidelity.json
gwpop-search freeze-production-campaign ...
gwpop-search validate-production-freeze ...
gwpop-search run-production-search ...
gwpop-search complete-production-evidence ...
~~~

Both search and evidence-completion commands are resumable. A campaign whose
frozen `max_f3_models` is smaller than the graph node count is intentionally
discovery-only and cannot produce a normalized posterior over the full graph.

See `docs/fable_h100_handoff.md`,
`docs/CLAUDE_H100_VALIDATION_HANDOVER.md`, and
`scripts/slurm/production_search_h100.sbatch.example`.

### Still scientifically deferred for actual GWTC-5 production

These are human scientific/data freezes, not missing software:

- final spin coordinate space (initial handover assumes reviewed `chieff`);
- exact GWTC-5 BBH event cut;
- waveform/sample-set policy;
- exact O3/O4 selection products;
- production population hyperpriors;
- structural model-prior hyperparameter(s);
- search budget;
- final frozen data manifest and hashes.

Claude is explicitly instructed not to infer these from filenames or to change
them after seeing validation/search results.

Those choices must be made explicitly at the production freeze. The code must
not infer them from filenames or release conventions.

## Immediate next actions

1. Require a final green Actions run after the handover/state documentation
   synchronization.
2. Hand the exact green revision and
   `docs/CLAUDE_H100_VALIDATION_HANDOVER.md` to Claude on H100.
3. Run and **manually review** Phase 3. Phase 3 remains open until that review
   is explicitly recorded.
4. Run the frozen 88-run Phase-7 scout matrix and one injected descendant F3
   confirmation.
5. Only after Phase 3 acceptance, canonicalize/freeze the reviewed real data and
   human-approved GWTC-5 scientific policies.
6. Run production adaptive search + mandatory evidence completion.
7. Run real-data scouts, holdout, event-drop, nearby-baseline, and v1.3
   frozen-selection exact-null validation; preserve every failed run and fill
   `docs/H100_VALIDATION_REPORT_TEMPLATE.md`.

## Rule for future ChatGPT/Codex/Fable work

Read this file, `SPEC.md`, `ROADMAP.md`, `docs/data_contract.md`,
`docs/hbi_contract.md`, `docs/phase3_recovery.md`, and
`docs/fable_h100_handoff.md` before modifying architecture.

Do not silently change an accepted scientific contract. Update this file
whenever a contract changes, a phase acceptance gate closes, or a major staged
subsystem becomes runnable.
