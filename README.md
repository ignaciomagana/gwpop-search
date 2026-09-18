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

- canonical gwcat-style PE and selection containers;
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
- HSGP scout basis/residual proposal infrastructure;
- held-out detected-event prediction;
- exact-search null replay infrastructure;
- typed non-executable agent proposal contracts;
- frozen production manifests/configuration and an H100/Fable runner.

F1/F2 screen values are compute-allocation statistics only. They are never
reported as Bayes factors. Posterior model probabilities are written only when
every node in the declared graph has proper F3/F4 evidence.

## Production freeze flow

Once the Phase-3 gate and GWTC-5 scientific/data choices are accepted, freeze
the production inputs explicitly.

Create a dataset manifest from the canonical HDF5s:

~~~bash
gwpop-search freeze-dataset \
  --pe /path/to/pe.h5 \
  --selection /path/to/selection.h5 \
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
  --max-f3-models 20 \
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

Run or resume:

~~~bash
gwpop-search run-production-search \
  --manifest frozen/dataset_manifest.json \
  --graph frozen/model_graph.json \
  --campaign frozen/campaign.json \
  --base-dir frozen \
  --work-dir /persistent/gwpop
~~~

See docs/fable_h100_handoff.md and
scripts/slurm/production_search_h100.sbatch.example.

## Key documents

- SPEC.md - scientific/software architecture.
- ROADMAP.md - phased build and acceptance criteria.
- PROJECT_STATE.md - authoritative durable build state.
- docs/data_contract.md - PE/selection contract.
- docs/hbi_contract.md - standardized likelihood contract.
- docs/phase3_recovery.md - current Phase-3 H100 recovery runbook.
- docs/fable_h100_handoff.md - production freeze and H100/Fable handoff.

## Design rule

Search algorithms and agents may choose **which legal model or validation to
evaluate next**. They do not get to modify the production likelihood, data
transformations, scientific priors, or validation rules during a frozen
production search.
