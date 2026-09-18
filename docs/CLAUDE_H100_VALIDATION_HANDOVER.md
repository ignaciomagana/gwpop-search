# Claude H100 validation and computation handover

Status: **living execution contract — build phase**
Last updated: 2026-09-18

This file is the final handover target for Claude on the H100 machine. It is
being written while the repository is completed. Treat requirements marked
**HARD GATE** as scientific constraints, not suggestions.

## 0. Role and operating rule

Claude is the execution/review operator for the already specified
`gwpop-search` scientific stack. Do not redesign the likelihood, silently
change priors, invent model mutations, change data semantics, or reinterpret a
failed diagnostic as acceptable.

Always begin by reading:

- `PROJECT_STATE.md`
- `SPEC.md`
- `ROADMAP.md`
- `docs/data_contract.md`
- `docs/hbi_contract.md`
- `docs/phase3_recovery.md`
- `docs/fable_h100_handoff.md`
- this file

Check out the exact handover commit and keep the working tree clean during
scientific runs. Record `git rev-parse HEAD`, CUDA/JAX/NumPyro/JAXNS versions,
GPU model, and Slurm allocation in the run notes.

## 1. HARD GATE: Phase 3

**Phase 3 is NOT accepted until the H100 multi-seed recovery campaign is
actually executed and reviewed.**

Software implementation, CI, smoke tests, later-phase code, or a single good
synthetic recovery do not close this gate.

The minimum acceptance campaign is the reviewed command in
`docs/phase3_recovery.md`, currently equivalent to:

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

Before acceptance, Claude must inspect—not merely print—the following for every
seed and for the ensemble:

1. R-hat and MCMC ESS for every inferred hyperparameter.
2. Divergences and any tree-depth/pathology indicators available in artifacts.
3. Per-event importance ESS and maximum normalized event weights.
4. Selection ESS, maximum selection weight, and likelihood Monte-Carlo
   variance diagnostics.
5. Posterior recovery against truth for every hyperparameter.
6. Standardized truth offsets across seeds, looking for repeated directional
   bias rather than treating one truth-in-CI flag as calibration.
7. At least one explicit interrupted/resumed campaign with verification that
   already-completed chain fingerprints are unchanged.

Do not set Phase 3 accepted from an automated boolean alone. The final decision
requires human review of the artifacts and an explicit update to
`PROJECT_STATE.md`.

If Phase 3 fails, stop production work. Diagnose and fix the scientific or
numerical issue, rerun the affected validation, and do not weaken thresholds
post hoc merely to obtain a pass.

## 2. Current software checkpoint

At repository commit `58e90e9ba471916b2663513555024527c286a0e4`,
GitHub Actions reported:

```text
219 passed
Python 3.12
JAX x64
```

The final handover commit will supersede this checkpoint. Claude must verify CI
for the exact handover commit before H100 work begins.

## 3. Scientific contracts that must not change during execution

- One standardized HBI likelihood is used for every population model.
- gwcat exported `p_pe` is the PE denominator.
- gwcat exported `pdraw` is estimator-ready; do not divide by `ndraw`,
  observing time, or campaign fractions a second time.
- Phase-3/initial production fitted density space is gwcat-v2 `chieff` unless
  the final scientific freeze explicitly says otherwise.
- Population models return the full normalized density in the declared data
  measure, including source/detector/cosmology Jacobians.
- F1/F2 values are allocation statistics only, never Bayes factors.
- Model comparison uses F3/F4 evidence plus the frozen structural model prior.
- A normalized model posterior over the declared graph is allowed only after
  valid F3/F4 evidence exists for every graph node.
- Search-level null calibration must replay the same final scientific
  procedure as the observed analysis: adaptive search **plus evidence
  completion**.
- HSGP scouts can propose only already registered interpretable descendants.
  They never auto-promote structure.
- A scout descendant must be explicitly reviewed, materialized, and then
  independently refit/evidence-compared under full HBI.
- Agents are optional and cannot change production likelihood/data/priors.

## 4. H100 execution stages

The intended execution order is:

### Stage A — repository/environment validation

1. Checkout exact handover commit.
2. Install the GPU-capable environment described in
   `docs/fable_h100_handoff.md`.
3. Set `JAX_ENABLE_X64=true`.
4. Verify JAX sees the H100.
5. Run the full test suite.
6. Record environment/version information.

### Stage B — Phase-3 acceptance campaign

Run the multi-seed recovery campaign, perform the manual review in Section 1,
exercise checkpoint/resume, and write a concise validation report into the
campaign directory. Do not proceed to production if this stage is not accepted.

### Stage C — real-data canonicalization and freeze

Use the reviewed gwcat-v2 PE and selection exports. Canonicalize them with an
explicit density/spin space; for the initial `chieff` analysis:

```bash
gwpop-search canonicalize-gwcat-v2 \
  --pe-export "$GWPOP_GWCAT_PE" \
  --selection-export "$GWPOP_GWCAT_SELECTION" \
  --spin-basis chieff \
  --output-dir "$GWPOP_CANONICAL_DATA"
```

Read `$GWPOP_CANONICAL_DATA/canonicalization_report.json`. Verify the source
hashes correspond to the reviewed gwcat products, the event list is the
intended list, `selection_mode` is `estimator_ready`, the basis identity is
the expected `chieff` basis, and the reported PE/selection counts are
plausible before freezing anything.

Then freeze:

- event-selection metadata;
- waveform/sample-set policy;
- canonical PE and selection checksums;
- model graph;
- F0--F4 numerical settings;
- production hyperpriors;
- model prior;
- scheduler and seeds;
- GPU/model/null budgets;
- exact git commit.

Do not infer any of these choices from filenames.

### Stage D — frozen production search

Validate the production freeze, run/resume the adaptive deterministic search,
then run `complete-production-evidence` on the same state. A production job is
not scientifically complete merely because adaptive search exits successfully.

### Stage E — flexible scouts

Run only the reviewed HSGP scout configurations. Any proposed structure must
pass numerical diagnostics, be explicitly accepted/rejected, and an accepted
child must be independently F3-refit against its parent. Keep the scout
posterior and descendant comparison as separate artifacts.

### Stage F — adversarial/search validation

Run the frozen validation suite:

- structured scout injections and null controls;
- event-drop/leave-one-out stress;
- explicitly named loud-event removals;
- nearby-baseline reruns;
- exact full-procedure null-search calibration.

Interpret search-level significance from the complete replay distribution, not
from the largest uncorrected Bayes factor found during model mining.

### Stage G — final result package

Produce a final machine-readable and human-readable validation package
containing:

- exact code/data/config hashes;
- Phase-3 review and decision;
- production search/evidence coverage;
- scored graph/model probabilities;
- edge Bayes factors/posterior odds;
- numerical diagnostics;
- scout reviews and independent descendant comparisons;
- stress-suite summaries;
- exact-search null calibration;
- failures/exclusions with reasons;
- command log and Slurm job IDs.

Do not delete failed runs. Preserve them as part of the audit trail.

## 5. Stop conditions

Stop and report rather than improvising if any of the following occurs:

- Phase-3 H100 recovery fails review.
- frozen file/hash validation fails;
- checked-out commit does not match the campaign;
- a required graph model lacks valid evidence after completion;
- the frozen F3/model/GPU budget is insufficient for full evidence coverage;
- an F3/F4 numerical diagnostic fails;
- exact null replay cannot reproduce the full observed procedure;
- a scout proposal is not covered by a registered legal mutation;
- an accepted scout child cannot be independently refit cleanly;
- checkpoint identity conflicts with current inputs/configuration.

## 6. Build-time TODOs before this document becomes final

These items are being closed by ChatGPT before handover:

- [x] first-class gwcat-v2 -> canonical HDF5 conversion CLI;
- [x] canonicalization validation/report artifact;
- [ ] refresh `PROJECT_STATE.md` for the current Phase-7/8/10 software;
- [x] ensure exact-null full-procedure tests are green after the v1.1 contract;
- [x] ensure scout review/refit tests are green;
- [ ] add final reviewed H100 validation command matrix;
- [ ] add final artifact checklist and decision template;
- [ ] pin final handover commit and CI test count.

Until every build-time TODO above is resolved, this document is not the final
handover.
