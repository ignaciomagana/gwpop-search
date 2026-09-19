# gwpop-search

Systematic gravitational-wave population-model search with a standardized
hierarchical Bayesian inference (HBI) engine.

The project separates:

1. **data contract** - validated PE and selection products;
2. **HBI engine** - one event/selection/rate implementation for every model;
3. **model grammar** - declarative population models with controlled mutations;
4. **deterministic search** - F0--F4 inference/evidence with durable state;
5. **validation** - holdout prediction, synthetic recovery, null-search replay;
6. **optional agents** - typed experiment proposals above the deterministic core.

The immediate target is GWTC-5 BBH population inference. Release-specific raw
ingestion does not live inside the HBI core.

## Scientific status

**Phase 3 remains the open acceptance gate.**

Phases 0--2 are accepted. Phases 4--10 have substantial staged implementation
and CI coverage, but are not being declared scientifically accepted out of
order. No production GWTC-5 inference has been run from this repository.

The remaining Phase-3 task is the H100 multi-seed synthetic recovery campaign:

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

See docs/phase3_recovery.md.

## Implemented search stack

The current repository includes:

- canonical gwcat-style PE and selection containers plus audited/idempotent
  gwcat-v2 canonicalization with source/output hashes;
- raw-draw and estimator-ready selection semantics;
- NumPy and differentiable JAX HBI backends;
- NumPyro NUTS with chain-granularity resume;
- normalized declarative BBH population components;
- canonical model specifications/hashes and typed graph mutations;
- deterministic finite model-graph enumeration;
- JAXNS evidence with repeat uncertainty;
- explicit structural model priors and edge comparisons;
- F0--F4 deterministic search with diagnostics, budgets, and SQLite state;
- inclusion-probability-corrected Monte-Carlo screening reductions;
- optional conditionally normalized HSGP residual scouting with typed
  interpretable descendants;
- targeted scout validation/independent descendant confirmation machinery,
  including a comprehensive 88-run matrix if that optional extension is
  scientifically invoked;
- held-out detected-event prediction;
- event-drop and nearby-baseline search stress suites;
- exact-search null replay calibration, with production nulls resampled from
  the frozen estimator-ready selection and an explicit resampling-ESS gate;
- typed non-executable agent proposal contracts;
- frozen production manifests/configuration and an H100/Fable runner.

The **finite atomic model graph is the primary scientific analysis**. HSGP is a
later optional residual scout, not a prerequisite for running or interpreting
the atom search.

F1/F2 screen values are compute-allocation statistics only. They are never
reported as Bayes factors. Posterior model probabilities are written only when
every node in the declared graph has proper F3/F4 evidence.

## Production freeze flow

Once the Phase-3 gate and GWTC-5 scientific/data choices are accepted, freeze
the production inputs explicitly.

Starting from reviewed gwcat-v2 exports, canonicalize first:

~~~bash
gwpop-search canonicalize-gwcat-v2 \
  --pe-export /reviewed/gwcat_pe.h5 \
  --selection-export /reviewed/gwcat_selection.h5 \
  --spin-basis chieff \
  --output-dir frozen/canonical
~~~

Review `frozen/canonical/canonicalization_report.json`, then create the
dataset manifest:

~~~bash
gwpop-search freeze-dataset \
  --pe frozen/canonical/pe.h5 \
  --selection frozen/canonical/selection.h5 \
  --dataset-id gwtc5-bbh-v1 \
  --event-selection-json event_selection.json \
  --waveform-policy-json waveform_policy.json \
  --output frozen/dataset_manifest.json
~~~

Write the full default F0--F4 numerical configuration, then review/edit it:

~~~bash
gwpop-search write-default-fidelity-config \
  --output frozen/fidelity.json
~~~

Serialize the reviewed model graph and freeze the production campaign:

~~~bash
gwpop-search enumerate-models \
  --output frozen/model_graph.json \
  --max-depth 2 \
  --max-models 40

gwpop-search freeze-production-campaign \
  --manifest frozen/dataset_manifest.json \
  --graph frozen/model_graph.json \
  --fidelity-config frozen/fidelity.json \
  --campaign-id gwtc5-bbh-search-v1 \
  --model-prior axis-complexity \
  --model-prior-penalty 0.6931471805599453 \
  --beam-width 8 \
  --exploration-quota 2 \
  --scheduler-seed 20260917 \
  --root-seed 20260917 \
  --max-gpu-hours 1000 \
  --max-f3-models 40 \
  --max-f4-models 8 \
  --max-null-replays 200 \
  --artifact-root runs/gwtc5-bbh-search-v1 \
  --state-database runs/gwtc5-bbh-search-v1/state.sqlite \
  --output frozen/campaign.json
~~~

The numeric values above are an example invocation, not frozen production
science choices. Review them before creating the real campaign.

Validate:

~~~bash
gwpop-search validate-production-freeze \
  --manifest frozen/dataset_manifest.json \
  --graph frozen/model_graph.json \
  --campaign frozen/campaign.json \
  --base-dir frozen
~~~

Run or resume the adaptive search:

~~~bash
gwpop-search run-production-search \
  --manifest frozen/dataset_manifest.json \
  --graph frozen/model_graph.json \
  --campaign frozen/campaign.json \
  --base-dir frozen \
  --work-dir /persistent/gwpop
~~~

Screening prioritizes expensive evidence work; it does not define the final
Bayesian model posterior. If evidence coverage is incomplete, finish valid F3
evidence over every declared node before using posterior model probabilities:

~~~bash
gwpop-search complete-production-evidence \
  --manifest frozen/dataset_manifest.json \
  --graph frozen/model_graph.json \
  --campaign frozen/campaign.json \
  --base-dir frozen \
  --work-dir /persistent/gwpop
~~~

For a full model posterior, the frozen max-f3-models budget must be at least the
number of nodes in the declared graph. A smaller value intentionally defines a
discovery-only campaign; evidence completion will refuse to bypass that budget.

The complete validation sequence is in
`docs/CLAUDE_H100_VALIDATION_HANDOVER.md`; use
`docs/H100_VALIDATION_REPORT_TEMPLATE.md` as the audit record.

The production-only runbook remains `docs/fable_h100_handoff.md`, with
`scripts/slurm/production_search_h100.sbatch.example` as the site-neutral
batch template.

## Key documents

- SPEC.md - scientific/software architecture.
- ROADMAP.md - phased build and acceptance criteria.
- PROJECT_STATE.md - authoritative durable build state.
- docs/data_contract.md - PE/selection contract.
- docs/hbi_contract.md - standardized likelihood contract.
- docs/phase3_recovery.md - current Phase-3 H100 recovery runbook.
- docs/fable_h100_handoff.md - production freeze and H100/Fable handoff.
- docs/CLAUDE_H100_VALIDATION_HANDOVER.md - authoritative end-to-end H100
  execution/validation contract.
- docs/H100_VALIDATION_REPORT_TEMPLATE.md - required H100 review/audit record.

## Design rule

Search algorithms and agents may choose **which legal model or validation to
evaluate next**. They do not get to modify the production likelihood, data
transformations, scientific priors, or validation rules during a frozen
production search.
