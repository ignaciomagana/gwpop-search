# Claude H100 validation and computation handover

Status: **H100 execution candidate — Phase 3 still scientifically OPEN**
Last updated: 2026-09-18

This is the execution contract for the H100 worker. The repository defines the
scientific machinery; Claude executes it, reviews numerical artifacts, and
records decisions. Claude must not silently redesign the likelihood, data
semantics, hyperpriors, model graph, model prior, numerical thresholds, or
search procedure.

The completed review must use
`docs/H100_VALIDATION_REPORT_TEMPLATE.md`.

## 0. Non-negotiable order

Execute in this order:

1. exact repository/environment verification;
2. Phase-3 multi-seed H100 recovery and manual review;
3. Phase-7 structured-scout H100 engineering validation;
4. canonicalize and review the real gwcat-v2 data;
5. freeze the GWTC-5 dataset/model/numerical/search configuration;
6. adaptive production search;
7. mandatory full-graph evidence completion;
8. real-data HSGP scouts and explicit proposal review;
9. held-out/event-drop/nearby-baseline robustness;
10. exact full-search null calibration using the frozen production selection;
11. final validation package and human decision.

**Do not proceed to step 4 if Phase 3 is rejected.**

Always read first:

- `PROJECT_STATE.md`
- `SPEC.md`
- `ROADMAP.md`
- `docs/data_contract.md`
- `docs/hbi_contract.md`
- `docs/phase3_recovery.md`
- `docs/fable_h100_handoff.md`
- `docs/H100_VALIDATION_REPORT_TEMPLATE.md`
- this file

## 1. Scientific contracts that must not change

- The repository owns one standardized HBI likelihood.
- Search/scout/agent code cannot change the likelihood.
- gwcat `p_pe` is the complete PE denominator in the declared basis.
- gwcat `pdraw` is estimator-ready. Never divide by `ndraw`, observing time,
  or campaign fractions a second time.
- The initial production density space is gwcat-v2 `chieff` unless the human
  scientific freeze explicitly selects another already-supported space.
- Population models return a fully normalized density in the data measure,
  including source/detector/cosmology Jacobians.
- F1/F2 scores allocate compute only. They are not Bayes factors.
- Scientific model comparison uses F3/F4 evidence plus the frozen model prior.
- A normalized posterior over the declared graph is allowed only when every
  graph node has valid F3/F4 evidence.
- Adaptive search is followed by explicit evidence completion.
- Search-level null calibration replays the same adaptive search **plus evidence
  completion**.
- HSGP scouts propose registered interpretable descendants only.
- Scouts never auto-modify the frozen graph.
- Every real-data scout proposal is explicitly accepted/rejected.
- Every accepted real-data scout child is independently F3-refit against its
  parent.
- Failed numerical evaluations remain failed; do not weaken thresholds after
  seeing a result.
- Failed/abandoned artifacts stay in the audit trail.

## 2. Environment and repository verification

The exact execution commit is the green handover `main` revision supplied to
Claude. Record it in the validation report before running anything.

Example:

```bash
git status --porcelain
git rev-parse HEAD
export GWPOP_GIT_COMMIT="$(git rev-parse HEAD)"
export JAX_ENABLE_X64=true
```

The working tree must be clean.

Install the repository only after the site's CUDA-enabled JAX installation is
correct. The evidence extra installs NumPyro/JAXNS-side dependencies:

```bash
pip install -e '.[evidence]'
```

Do not replace a working CUDA JAX installation with a CPU wheel.

Verify the environment:

```bash
python - <<'PY'
import platform
import jax
import numpyro
print("Python:", platform.python_version())
print("JAX:", jax.__version__)
print("NumPyro:", numpyro.__version__)
print("x64:", bool(jax.config.x64_enabled))
print("devices:", jax.devices())
if not bool(jax.config.x64_enabled):
    raise SystemExit("JAX x64 is disabled")
if not any(d.platform == "gpu" for d in jax.devices()):
    raise SystemExit("No JAX GPU visible")
PY

pytest -q
```

Copy the report template into persistent run storage and fill it as work
proceeds.

## 3. HARD GATE — Phase-3 H100 recovery

Phase 3 is **not accepted** because code exists or CI is green. It closes only
after the H100 multi-seed campaign is run and manually reviewed.

Recommended campaign:

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

Equivalent site-neutral Slurm template:

```text
scripts/slurm/phase3_synthetic_h100.sbatch.example
```

### 3.1 Required checkpoint/resume exercise

At least once, stop a Phase-3 allocation after one or more chain checkpoints
exist but before the campaign is complete.

Record fingerprints:

```bash
gwpop-search fingerprint-recovery-checkpoint \
  --run-dir runs/phase3-recovery/default/run_000 \
  > phase3_fingerprints_before.json
```

Re-submit the **identical** campaign command. After it finishes:

```bash
gwpop-search fingerprint-recovery-checkpoint \
  --run-dir runs/phase3-recovery/default/run_000 \
  > phase3_fingerprints_after.json

python - <<'PY'
import json
before = json.load(open("phase3_fingerprints_before.json"))
after = json.load(open("phase3_fingerprints_after.json"))
changed = {
    key: (value, after.get(key))
    for key, value in before.items()
    if after.get(key) != value
}
if changed:
    raise SystemExit(f"completed checkpoint fingerprints changed: {changed}")
print("All previously completed chain fingerprints are unchanged.")
PY
```

### 3.2 Manual Phase-3 review

Run:

```bash
gwpop-search assess-synthetic-campaign \
  --root runs/phase3-recovery/default \
  --min-runs 4
```

For every seed inspect:

- every hyperparameter R-hat and MCMC ESS;
- divergences/tree-depth pathology;
- minimum event ESS;
- selection ESS;
- maximum normalized event/selection weights;
- shape-log-likelihood Monte-Carlo variance;
- posterior versus injected truth.

Across seeds inspect standardized truth offsets for repeated same-sign bias.
Truth-in-one-credible-interval flags are diagnostics, not a calibrated coverage
test.

**Human decision required:** ACCEPT or REJECT Phase 3.

If rejected, stop. Diagnose/fix/re-run; do not proceed to real GWTC-5 work.

## 4. Phase-7 structured HSGP validation on H100

This stage validates the scout as an engineering discovery mechanism before it
is allowed to suggest real-data descendants.

The gate is frozen in code before execution:

- at least 8 independent catalogs per matrix member;
- every run numerically valid;
- reachable strong injection proposed in at least 75% of runs;
- off-target proposals in at most 25% of runs;
- null controls produce any proposal in at most 25% of runs.

These are engineering thresholds, **not** GWTC-5 significance thresholds.

### 4.1 Frozen 88-run matrix

Use:

```text
scripts/h100/structured_scout_validation_matrix.sh.example
scripts/slurm/scout_validation_h100.sbatch.example
```

The fixed matrix is:

| scout | injection | strength | runs | root seed |
| --- | --- | ---: | ---: | ---: |
| q given m1 | null | 0 | 8 | 20261001 |
| q given m1 | pairing.beta.linear_m1 | +0.225 | 8 | 20261002 |
| chi_eff given m1 | null | 0 | 8 | 20261011 |
| chi_eff given m1 | chieff.mean.linear_m1 | +0.015 | 8 | 20261012 |
| chi_eff given m1 | chieff.width.linear_m1 | +0.0375 | 8 | 20261013 |
| chi_eff given q | null | 0 | 8 | 20261021 |
| chi_eff given q | chieff.mean.linear_q | +0.45 | 8 | 20261022 |
| chi_eff given q | chieff.width.linear_q | +1.5 | 8 | 20261023 |
| chi_eff given z | null | 0 | 8 | 20261031 |
| chi_eff given z | chieff.mean.linear_z | +0.30 | 8 | 20261032 |
| chi_eff given z | chieff.width.linear_z | +0.75 | 8 | 20261033 |

Every strength lies inside the registered child hyperprior. Do not alter these
numbers after seeing results.

Example submission:

```bash
export GWPOP_ENV=/path/to/environment/activate
export GWPOP_SCOUT_VALIDATION_ROOT=/persistent/gwpop/scout-validation
sbatch scripts/slurm/scout_validation_h100.sbatch.example
```

Review every `campaign_summary.json`. Every production-enabled scout surface
must pass its null and injected cases.

### 4.2 Independent injected-descendant F3 confirmation

Write one explicit F3 configuration for this engineering confirmation:

```bash
gwpop-search write-default-fidelity-config \
  --output "$GWPOP_SCOUT_VALIDATION_ROOT/fidelity.json"
```

After the `chieff_q_mean` campaign passes, independently confirm one of its
successful injected proposals:

```bash
gwpop-search confirm-structured-scout-descendant \
  --campaign-root "$GWPOP_SCOUT_VALIDATION_ROOT/chieff_q_mean" \
  --fidelity-config "$GWPOP_SCOUT_VALIDATION_ROOT/fidelity.json" \
  --output-root "$GWPOP_SCOUT_VALIDATION_ROOT/chieff_q_mean_confirmation"
```

The command deterministically chooses the lowest-index numerically valid run
that proposed the injected mutation unless `--run-index` is explicitly
supplied. It writes the engineering review, exact parent/child specs, independent
F3 evaluations, and `confirmation_summary.json`.

The command refuses to run unless the full structured-scout campaign passed its
frozen engineering gate. Require both parent and child F3 evaluations to pass
numerical diagnostics and require
`log BF(child/parent) > 0` for
`engineering_confirmation_passed=true`.

Record the child/parent log Bayes factor. This validates the injected discovery
and review/refit mechanism; it is not an astrophysical claim.

If the scout matrix or confirmation fails, do not use the affected scout surface
for production interpretation.

## 5. Canonicalize the reviewed real data

### 5.1 Human scientific inputs required

Before this stage the human freeze must specify:

- exact GWTC-5 BBH event cut;
- exact PE sample/waveform policy;
- exact reviewed gwcat-v2 PE export;
- exact reviewed gwcat-v2 O3/O4 selection export;
- chosen spin density space;
- event-selection metadata;
- waveform/sample-set metadata.

Claude must not infer these choices from filenames.

For the initial `chieff` analysis:

```bash
export GWPOP_GWCAT_PE=/reviewed/gwcat_pe.h5
export GWPOP_GWCAT_SELECTION=/reviewed/gwcat_selection.h5
export GWPOP_FREEZE_DIR=/persistent/gwpop/frozen
export GWPOP_CANONICAL_DATA="$GWPOP_FREEZE_DIR/canonical"

gwpop-search canonicalize-gwcat-v2 \
  --pe-export "$GWPOP_GWCAT_PE" \
  --selection-export "$GWPOP_GWCAT_SELECTION" \
  --spin-basis chieff \
  --output-dir "$GWPOP_CANONICAL_DATA"
```

Read `canonicalization_report.json`. Verify:

- source hashes match the reviewed gwcat products;
- basis is the intended gwcat-v2 `chieff` measure;
- event ordering/list is correct;
- PE and selected-injection counts are plausible;
- selection mode is `estimator_ready`;
- denominator contract says exported `p_pe` and `pdraw` are authoritative.

Partial/conflicting re-canonicalization must fail rather than overwrite.

## 6. Freeze GWTC-5 science and computation

Create reviewed JSON metadata first, e.g. `event_selection.json` and
`waveform_policy.json`. These files must contain the actual approved choices,
not generic placeholders.

Freeze the data:

```bash
gwpop-search freeze-dataset \
  --pe "$GWPOP_CANONICAL_DATA/pe.h5" \
  --selection "$GWPOP_CANONICAL_DATA/selection.h5" \
  --dataset-id gwtc5-bbh-v1 \
  --event-selection-json "$GWPOP_FREEZE_DIR/event_selection.json" \
  --waveform-policy-json "$GWPOP_FREEZE_DIR/waveform_policy.json" \
  --output "$GWPOP_FREEZE_DIR/dataset_manifest.json"
```

Write/review the numerical ladder:

```bash
gwpop-search write-default-fidelity-config \
  --output "$GWPOP_FREEZE_DIR/fidelity.json"
```

Generate the finite graph only after its depth/model cap is approved:

```bash
gwpop-search enumerate-models \
  --output "$GWPOP_FREEZE_DIR/model_graph.json" \
  --max-depth <REVIEWED_DEPTH> \
  --max-models <REVIEWED_MODEL_COUNT>
```

Inspect it:

```bash
gwpop-search inspect-model-graph \
  --graph "$GWPOP_FREEZE_DIR/model_graph.json" \
  > "$GWPOP_FREEZE_DIR/model_graph_inventory.json"
```

Review every structural axis/hyperprior represented by the graph.

Freeze the production campaign only after the following are explicitly approved:

- structural model prior and any complexity penalty;
- beam width/exploration quota;
- scheduler/root seeds;
- production GPU-hour budget;
- F3/F4 model limits;
- null-replay budget;
- persistent artifact/state paths.

Example interface only:

```bash
gwpop-search freeze-production-campaign \
  --manifest "$GWPOP_FREEZE_DIR/dataset_manifest.json" \
  --graph "$GWPOP_FREEZE_DIR/model_graph.json" \
  --fidelity-config "$GWPOP_FREEZE_DIR/fidelity.json" \
  --campaign-id gwtc5-bbh-search-v1 \
  --model-prior axis-complexity \
  --model-prior-penalty <REVIEWED_PENALTY> \
  --beam-width <REVIEWED_BEAM_WIDTH> \
  --exploration-quota <REVIEWED_EXPLORATION_QUOTA> \
  --scheduler-seed <REVIEWED_SCHEDULER_SEED> \
  --root-seed <REVIEWED_ROOT_SEED> \
  --max-gpu-hours <REVIEWED_GPU_HOURS> \
  --max-f3-models <AT_LEAST_GRAPH_NODE_COUNT_FOR_FULL_POSTERIOR> \
  --max-f4-models <REVIEWED_F4_LIMIT> \
  --max-null-replays <AT_LEAST_100_FOR_DECLARED_NULL_CAMPAIGN> \
  --artifact-root runs/gwtc5-bbh-search-v1 \
  --state-database runs/gwtc5-bbh-search-v1/state.sqlite \
  --output "$GWPOP_FREEZE_DIR/campaign.json"
```

If `max_f3_models` is smaller than the graph node count, the campaign is
discovery-only and cannot produce a normalized full-graph posterior.

## 7. Validate and run production

Set:

```bash
export GWPOP_MANIFEST="$GWPOP_FREEZE_DIR/dataset_manifest.json"
export GWPOP_GRAPH="$GWPOP_FREEZE_DIR/model_graph.json"
export GWPOP_CAMPAIGN="$GWPOP_FREEZE_DIR/campaign.json"
export GWPOP_DATA_BASE="$GWPOP_FREEZE_DIR"
export GWPOP_WORK_DIR=/persistent/gwpop/production
```

Validate:

```bash
gwpop-search validate-production-freeze \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --base-dir "$GWPOP_DATA_BASE"
```

Do not use `--ignore-current-commit` for production.

Run/resume adaptive F0--F4 search:

```bash
gwpop-search run-production-search \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --base-dir "$GWPOP_DATA_BASE" \
  --work-dir "$GWPOP_WORK_DIR"
```

Then mandatory evidence completion:

```bash
gwpop-search complete-production-evidence \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --base-dir "$GWPOP_DATA_BASE" \
  --work-dir "$GWPOP_WORK_DIR"
```

Equivalent batch entry point:

```text
scripts/slurm/production_search_h100.sbatch.example
```

Do not report full model probabilities unless
`evidence_completion_summary.json` says full graph evidence is available and
`scored_graph.json` exists.

A failed F3/F4 numerical gate is not discarded to make the graph complete.

## 8. Real-data HSGP scouts

Only scout surfaces that passed Section 4 may be used.

### 8.1 Freeze the scout baseline from a valid full-data fit

Obtain the graph root from `model_graph_inventory.json` and materialize it:

```bash
export GWPOP_ROOT_HASH=<ROOT_HASH>

gwpop-search extract-model \
  --graph "$GWPOP_GRAPH" \
  --model-hash "$GWPOP_ROOT_HASH" \
  --output "$GWPOP_FREEZE_DIR/root_model.json"
```

Use a numerically valid root-model F4 evaluation if available; otherwise use a
valid F3 evaluation. Inspect the evaluation diagnostics first.

Then:

```bash
gwpop-search export-scout-baseline \
  --evaluation <VALID_ROOT_F3_OR_F4_EVALUATION_JSON> \
  --model "$GWPOP_FREEZE_DIR/root_model.json" \
  --output "$GWPOP_FREEZE_DIR/root_hyperparameters.json"
```

Keep the `.provenance.json` sidecar.

### 8.2 Run the four production scouts

Write the same four surface configs used in validation:

```bash
mkdir -p "$GWPOP_FREEZE_DIR/scouts"

gwpop-search write-default-scout-config \
  --target q --covariate m1_source \
  --output "$GWPOP_FREEZE_DIR/scouts/q_m1.json"

gwpop-search write-default-scout-config \
  --target chi_eff --covariate m1_source \
  --output "$GWPOP_FREEZE_DIR/scouts/chieff_m1.json"

gwpop-search write-default-scout-config \
  --target chi_eff --covariate q \
  --output "$GWPOP_FREEZE_DIR/scouts/chieff_q.json"

gwpop-search write-default-scout-config \
  --target chi_eff --covariate z \
  --output "$GWPOP_FREEZE_DIR/scouts/chieff_z.json"
```

Predeclared production scout seeds:

- q|m1: 20261201
- chi_eff|m1: 20261202
- chi_eff|q: 20261203
- chi_eff|z: 20261204

Example:

```bash
gwpop-search run-hsgp-scout \
  --manifest "$GWPOP_MANIFEST" \
  --base-dir "$GWPOP_DATA_BASE" \
  --base-model "$GWPOP_FREEZE_DIR/root_model.json" \
  --base-hyperparameters-json "$GWPOP_FREEZE_DIR/root_hyperparameters.json" \
  --scout-config "$GWPOP_FREEZE_DIR/scouts/chieff_q.json" \
  --run-dir "$GWPOP_WORK_DIR/validation/scouts/chieff_q" \
  --seed 20261203
```

Run the analogous three other surfaces.

### 8.3 Explicit real-data review boundary

For every validated proposal in a scout summary, create a review record.

Rejected example:

```bash
gwpop-search review-scout-proposal \
  --scout-summary <SCOUT_SUMMARY> \
  --parent-model "$GWPOP_FREEZE_DIR/root_model.json" \
  --proposal-id <PROPOSAL_ID> \
  --decision rejected \
  --note "<REVIEW NOTE>" \
  --review-output <REVIEW_JSON>
```

Accepted example:

```bash
gwpop-search review-scout-proposal \
  --scout-summary <SCOUT_SUMMARY> \
  --parent-model "$GWPOP_FREEZE_DIR/root_model.json" \
  --proposal-id <PROPOSAL_ID> \
  --decision accepted \
  --note "<REVIEW NOTE>" \
  --review-output <REVIEW_JSON> \
  --child-output <CHILD_MODEL_JSON>
```

Every accepted child must then be independently F3-refit:

```bash
gwpop-search compare-scout-descendant \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --review <REVIEW_JSON> \
  --parent-model "$GWPOP_FREEZE_DIR/root_model.json" \
  --child-model <CHILD_MODEL_JSON> \
  --root <COMPARISON_ROOT> \
  --base-dir "$GWPOP_DATA_BASE"
```

A scout proposal alone is never a discovery.

## 9. Held-out predictive validation

Before running holdout, record the exact model hashes in the validation report.
At minimum include:

- the graph root;
- every model that will carry a central structural claim.

Use five deterministic folds:

```bash
gwpop-search run-holdout-validation \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --model-hash "$GWPOP_ROOT_HASH" \
  --model-hash <CENTRAL_CLAIM_MODEL_HASH> \
  --n-folds 5 \
  --fold-seed 20261211 \
  --root "$GWPOP_WORK_DIR/validation/holdout" \
  --base-dir "$GWPOP_DATA_BASE"
```

Every fold uses strict production F4 NUTS diagnostics before its held-out
`ell_i/A` score is admitted. A model total is blocked if any fold fails.

Held-out totals are predictive diagnostics, not Bayes factors.

## 10. Event-drop robustness

### 10.1 Full leave-one-out influence scan

Run every event through F2 only. This is deliberately an influence scan, not an
evidence claim:

```bash
gwpop-search write-loo-stress-config \
  --manifest "$GWPOP_MANIFEST" \
  --stop-fidelity F2 \
  --max-gpu-hours-per-scenario 6 \
  --output "$GWPOP_FREEZE_DIR/loo_f2.json"

gwpop-search run-event-stress-suite \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --stress-config "$GWPOP_FREEZE_DIR/loo_f2.json" \
  --root "$GWPOP_WORK_DIR/validation/loo_f2" \
  --base-dir "$GWPOP_DATA_BASE" \
  --work-dir "$GWPOP_WORK_DIR" \
  --reference-state-database <PRODUCTION_STATE_SQLITE>
```

Do not call F2 shifts Bayes factors.

### 10.2 Predeclared loud/custom removals

Before inspecting the stress results, record the loud/custom event list and
rationale in the validation report. The list may come from catalog-level
properties such as SNR/localization, not from searching for the subset that
maximizes or erases a result.

Create explicit F3 scenarios with `write-event-drop-stress-config` and run
them with `run-event-stress-suite`.

F3 stress search is adaptive. Compare only evidence edges present in both the
reference and stressed state. Do not renormalize an incomplete stressed graph.

## 11. Nearby-baseline robustness

Predeclare a small set of physically nearby alternative roots before running
this suite. Recommended one-axis roots from the registered grammar are:

- `mass.family.broken_powerlaw`;
- `pairing.family.truncated_gaussian_q`;
- `redshift.family.madau_dickinson`.

Materialize the exact root specs from the frozen graph by hash using
`inspect-model-graph` + `extract-model`; do not hand-edit them.

For each scenario:

```bash
gwpop-search write-nearby-baseline-config \
  --scenario-id <SCENARIO_ID> \
  --root-model <EXACT_ROOT_MODEL_JSON> \
  --max-depth 1 \
  --max-models <REVIEWED_LOCAL_GRAPH_SIZE> \
  --stop-fidelity F3 \
  --max-gpu-hours-per-scenario <REVIEWED_CAP> \
  --max-f3-models <REVIEWED_LOCAL_F3_LIMIT> \
  --max-f4-models <REVIEWED_LOCAL_F4_LIMIT> \
  --output <NEARBY_CONFIG_JSON>
```

Then run the suite:

```bash
gwpop-search run-nearby-baseline-suite \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --nearby-config <NEARBY_CONFIG_JSON> \
  --root "$GWPOP_WORK_DIR/validation/nearby" \
  --base-dir "$GWPOP_DATA_BASE" \
  --work-dir "$GWPOP_WORK_DIR" \
  --reference-state-database <PRODUCTION_STATE_SQLITE>
```

Nearby-baseline searches are adaptive. Interpret mutation-support comparisons
only where evidence is actually present; a local `scored_graph.json` is valid
only if that local graph has complete evidence coverage.

## 12. Exact full-search null calibration

This is the search-level look-elsewhere calibration.

Production mode is `frozen_selection_resample`, not the simple Phase-3
synthetic detector survey.

### 12.1 Null population truth

Use the same valid root-model F3/F4 posterior median exported in Section 8:

```text
$GWPOP_FREEZE_DIR/root_hyperparameters.json
```

This choice is frozen before null results are seen.

### 12.2 Preflight real selection support

Run:

```bash
gwpop-search diagnose-frozen-selection-null \
  --manifest "$GWPOP_MANIFEST" \
  --base-dir "$GWPOP_DATA_BASE" \
  --model "$GWPOP_FREEZE_DIR/root_model.json" \
  --hyperparameters-json "$GWPOP_FREEZE_DIR/root_hyperparameters.json" \
  > "$GWPOP_FREEZE_DIR/null_resampling_preflight.json"
```

Record:

- resampling ESS;
- maximum single selected-injection probability;
- number of positive-weight selected injections.

The production null configuration requires resampling ESS >= 200 by default.
If the preflight fails this scientific support gate, stop and improve the
selection-injection support. Do not lower the threshold after seeing the
result.

### 12.3 Freeze the null campaign

Predeclared production null campaign:

- 100 exact search replays;
- root seed 20261101;
- observed event count read from the frozen manifest;
- 256 PE samples/event in the declared PE approximation;
- frozen production selection reused exactly;
- resampling ESS gate 200;
- 12 H100-equivalent GPU-hour ceiling per null replay;
- full adaptive F0--F4 search;
- mandatory full-graph F3/F4 evidence completion.

Write:

```bash
gwpop-search write-null-calibration-config \
  --n-nulls 100 \
  --root-seed 20261101 \
  --manifest "$GWPOP_MANIFEST" \
  --data-mode frozen_selection_resample \
  --pe-samples 256 \
  --truth-hyperparameters-json "$GWPOP_FREEZE_DIR/root_hyperparameters.json" \
  --min-resampling-ess 200 \
  --max-gpu-hours-per-null 12 \
  --output "$GWPOP_FREEZE_DIR/exact_nulls.json"
```

The production campaign must have `max_null_replays >= 100` and
`max_f3_models >= number_of_graph_nodes`.

Prepare the immutable exact-null plan once:

```bash
export GWPOP_NULL_CONFIG="$GWPOP_FREEZE_DIR/exact_nulls.json"
export GWPOP_NULL_ROOT="$GWPOP_WORK_DIR/validation/exact_nulls"

gwpop-search prepare-null-search-calibration \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --null-config "$GWPOP_NULL_CONFIG" \
  --root "$GWPOP_NULL_ROOT" \
  --base-dir "$GWPOP_DATA_BASE"
```

For the declared 100-null / two-H100 campaign, submit the array-safe template:

```bash
export GWPOP_ENV=/path/to/environment/activate
sbatch scripts/slurm/exact_nulls_h100_array.sbatch.example
```

The template uses `--array=0-99%2`: each task owns exactly one deterministic
null index and its own `searches/null_XXXXX/{state.sqlite,artifacts}` subtree.
Do not submit duplicate tasks for the same index concurrently.

After the array is complete, finalize/calibrate:

```bash
gwpop-search finalize-null-search-calibration \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --null-config "$GWPOP_NULL_CONFIG" \
  --root "$GWPOP_NULL_ROOT" \
  --base-dir "$GWPOP_DATA_BASE" \
  --work-dir "$GWPOP_WORK_DIR" \
  --observed-state-database <PRODUCTION_STATE_SQLITE>
```

Finalization refuses to proceed until every declared `null_XXXXX.json` exists
with its deterministic index/seed identity. The older
`run-null-search-calibration` command remains a serial convenience wrapper
over the same prepare/index/finalize primitives and is appropriate only for
small/debug campaigns.

Each null draws detected truths from the frozen selected injections with
relative probability

```text
p_pop(theta | Lambda_null) / pdraw(theta)
```

and reuses the exact frozen production selection catalog in HBI.

The PE around each resampled detected truth is a declared approximation. Treat
the resulting empirical search tail as calibration under this complete declared
null procedure, not as an exact raw-strain frequentist false-alarm probability.

With 100 nulls the minimum finite-sample corrected tail probability is
`1 / 101`. If the observed statistic lies at that floor, report it as
**resolution-limited**. Do not extrapolate a smaller p-value and do not
adaptively add nulls after seeing the tail without defining a new reviewed
campaign.

## 13. Final result package

Use `docs/H100_VALIDATION_REPORT_TEMPLATE.md`.

The package must contain or reference:

- exact git commit and clean-tree record;
- software/GPU/CUDA/JAX/NumPyro/JAXNS versions;
- full test-suite result;
- Phase-3 plans, chain checkpoints, posterior/recovery summaries, and
  before/after fingerprint files;
- explicit human Phase-3 ACCEPT/REJECT decision;
- all 88 structured-scout plans/summaries;
- injected-descendant F3 confirmation artifacts;
- gwcat source hashes and canonicalization report;
- dataset manifest;
- event-selection/waveform policy JSON;
- fidelity config;
- model graph and inventory;
- production campaign;
- production state SQLite;
- adaptive search summary;
- evidence coverage/completion summaries;
- full scored graph if evidence-complete;
- production scout configs/posteriors/summaries/review records/children/F3
  comparisons;
- holdout manifest/fold/model summaries;
- LOO and explicit event-drop configs/summaries;
- nearby-baseline plans/summaries;
- null resampling preflight;
- exact-null plan, Slurm array job/task IDs, every replay result/state, missing-index check, and calibrated summary;
- all Slurm logs/job IDs;
- every failed/abandoned run with reason.

## 14. Stop conditions

Stop and report rather than improvising if:

- Phase 3 fails manual H100 review;
- an earlier completed checkpoint changes on resume;
- a frozen hash/size/commit validation fails;
- the real-data basis/denominator semantics differ from the reviewed contract;
- production graph evidence cannot be completed within the frozen budget;
- an F3/F4 numerical gate fails;
- an HSGP surface fails its structured-validation gate;
- a production scout proposal is not a registered legal mutation;
- an accepted production scout child fails independent F3 confirmation;
- holdout folds fail strict NUTS/importance diagnostics;
- null selection resampling ESS is below the frozen threshold;
- an exact null cannot complete the same search + evidence-completion procedure;
- the per-null compute ceiling is exhausted;
- checkpoint/resume identity conflicts occur.

Do not fix a scientific failure by relaxing a threshold after the result is
known.

## 15. Current software checkpoint and handover rule

The exact final handover commit/test count is recorded in `PROJECT_STATE.md`
after the last code/documentation commit is green.

Claude must use the final handover revision supplied by the human operator.
If GitHub Actions capacity has been restored, verify the exact revision in CI.
If it has not, run the full `pytest -q` suite on the H100 host before any
scientific computation and record that result in the validation report.

As of 2026-09-18 the repository's included GitHub Actions allowance is exhausted
(3000/3000 minutes used). Workflow runs after the last real test checkpoint may
show red immediately without starting pytest. Do not interpret those
quota-blocked runs as software failures.

At the time this execution contract was rewritten, the most recent completed
integrated checkpoint was:

```text
commit: baf4726eda052a86b3445969e5fcc037994210d6
tests:  268 passed
CI:     GitHub Actions / Python 3.12 / JAX x64
```

Newer handover/confirmation commits supersede that checkpoint once their Actions
runs are green.
