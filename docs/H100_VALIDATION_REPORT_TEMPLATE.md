# H100 validation report template

Use one copy of this file for the actual H100 campaign. Do not replace failed
entries with reruns silently; append the rerun and explain why it was required.

## 1. Run identity

- repository commit:
- `GWPOP_GIT_COMMIT`:
- working tree clean: yes / no
- host/cluster:
- Slurm job IDs:
- GPU model/count:
- CUDA version:
- Python:
- JAX:
- jaxlib:
- NumPyro:
- JAXNS:
- `JAX_ENABLE_X64`:
- full test-suite result:
- operator:
- review date:

## 2. Phase-3 H100 recovery — HARD GATE

Campaign root:

Command:

Checkpoint/resume exercise:
- interrupted run:
- completed chain fingerprint(s) before interruption:
- completed chain fingerprint(s) after resume:
- unchanged: yes / no

Per-seed review:

| run | data seed | sampler seed | numerical gate | max R-hat | min MCMC ESS | divergences | min event ESS | selection ESS | max event weight | max selection weight | Var(log L) | notes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | | | | | | | | | | | | |
| 1 | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | |

For every baseline hyperparameter, inspect the four posterior summaries and the
ensemble standardized truth offsets. Record repeated same-sign offsets or
boundary behavior here:

- `alpha`:
- `mmin`:
- `mmax`:
- `peak_fraction`:
- `peak_mu`:
- `peak_sigma`:
- `beta_q`:
- `kappa`:
- `chi_mu`:
- `chi_sigma`:

**Human Phase-3 decision:** ACCEPT / REJECT

Reason:

If REJECT, stop the GWTC-5 production workflow.

## 3. Canonical production data review

gwcat PE export:
- path:
- SHA-256:
- format version:
- writer commit:
- spin basis:

gwcat selection export:
- path:
- SHA-256:
- format version:
- writer commit:
- spin basis:
- `pdraw_state`:

Canonicalization report:
- path:
- source hashes match reviewed products: yes / no
- canonical PE SHA-256:
- canonical selection SHA-256:
- basis identity:
- event count:
- PE sample count:
- selected-injection count:
- selection mode is `estimator_ready`: yes / no
- event list reviewed: yes / no
- denominator contract reviewed: yes / no

## 4. Frozen scientific configuration

Dataset manifest hash:

Event-selection policy:

Waveform/sample-set policy:

Model graph:
- graph hash:
- root hash:
- number of nodes:
- maximum depth:
- graph inventory reviewed: yes / no

Fidelity configuration hash/file:

Production campaign hash:

Production hyperpriors reviewed: yes / no

Model prior:
- type:
- complexity penalty if applicable:

Scheduler:
- beam width:
- exploration quota:
- scheduler seed:

Seed root:

Budget:
- max production GPU hours:
- max F3 models:
- max F4 models:
- max null replays:

Freeze validation passed: yes / no

## 5. Production search and evidence completion

Adaptive search:
- state database:
- artifact root:
- completed fidelity:
- cumulative compute cost:
- diagnostic failures:

Evidence completion:
- graph models:
- valid evidence initially:
- newly evaluated:
- blocked invalid:
- valid evidence finally:
- complete graph evidence coverage: yes / no

A normalized full-graph posterior may be reported only if coverage is complete.

Scored-graph outputs:
- `scored_graph.json`:
- highest posterior-mass structures:
- edge Bayes factors worth follow-up:
- numerical caveats:

Do not describe F1/F2 allocation scores as Bayes factors.

## 6. Phase-7 HSGP validation matrix

The engineering gate is frozen before H100 execution:
- minimum 8 independent runs;
- every run numerically valid;
- reachable strong injection proposed in at least 75% of runs;
- off-target proposals in at most 25% of runs;
- null controls produce any proposal in at most 25% of runs.

For each matrix member record `campaign_summary.json`:

| scout surface | injected mutation | strength | n runs | numerical pass | expected-proposal fraction | off-target-run fraction | any-proposal fraction | engineering gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q(m1) | null | 0 | 8 | | n/a | | | |
| q(m1) | pairing.beta.linear_m1 | +0.225 | 8 | | | | | |
| chi_eff(m1) | null | 0 | 8 | | n/a | | | |
| chi_eff(m1) | chieff.mean.linear_m1 | +0.015 | 8 | | | | | |
| chi_eff(m1) | chieff.width.linear_m1 | +0.0375 | 8 | | | | | |
| chi_eff(q) | null | 0 | 8 | | n/a | | | |
| chi_eff(q) | chieff.mean.linear_q | +0.45 | 8 | | | | | |
| chi_eff(q) | chieff.width.linear_q | +1.5 | 8 | | | | | |
| chi_eff(z) | null | 0 | 8 | | n/a | | | |
| chi_eff(z) | chieff.mean.linear_z | +0.30 | 8 | | | | | |
| chi_eff(z) | chieff.width.linear_z | +0.75 | 8 | | | | | |

**Phase-7 software-validation decision:** ACCEPT / REJECT

At least one accepted injected proposal must then be materialized through the
review API and independently F3 evidence-compared to its parent. Record:
- proposal ID:
- review record:
- parent hash:
- child hash:
- independent comparison path:
- campaign engineering gate passed before confirmation: yes / no
- both numerically valid: yes / no
- log BF child/parent:
- log BF child/parent > 0: yes / no
- `engineering_confirmation_passed`: yes / no

This injected-descendant comparison validates the injected discovery and
review/refit mechanism; it is not a GWTC-5 discovery claim.

## 7. Production HSGP scouts

Base model hash:

Source F3/F4 evaluation:

`export-scout-baseline` provenance sidecar:

Run only scout surfaces that passed Section 6. For each:
- scout config:
- run directory:
- seed:
- numerical diagnostics:
- raw proposals:
- validated proposals:

For every proposal, create an explicit ACCEPTED or REJECTED review record.
Accepted children must be independently F3-refit against the parent before any
scientific interpretation.

## 8. Held-out predictive validation

Predeclare the model hashes to validate before running K-fold fits. At minimum
include the graph root and any model that will carry a central structural
claim.

- K:
- fold seed:
- selected model hashes:
- all folds numerically valid for each reported model: yes / no

| model hash | scored events | total held-out log predictive | failed folds | notes |
| --- | ---: | ---: | ---: | --- |
| | | | | |

These totals are posterior-predictive validation scores, not Bayes factors.

## 9. Event-drop / loud-event stress

Leave-one-out suite:
- config:
- stop fidelity:
- scenarios complete:
- material structural changes:

Explicit loud-event/custom drops:
- predeclared events:
- rationale:
- result:

Do not choose event removals after looking for the subset that erases or
maximizes a result.

## 10. Nearby-baseline robustness

Predeclared alternative root models:

| scenario | root hash | graph size | stop fidelity | comparison summary |
| --- | --- | ---: | --- | --- |
| | | | | |

Record whether the central structural conclusion depends on the reference
baseline rather than only whether one alternative has a larger evidence.

## 11. Exact search-level null calibration

Production null mode must be `frozen_selection_resample` unless this report
explicitly documents an engineering-only synthetic-survey run.

Null config:
- n nulls:
- root seed:
- event count (must equal observed):
- PE samples/event:
- minimum selection-resampling ESS:
- measured selection-resampling ESS:
- maximum resampling probability:
- per-null GPU-hour ceiling:
- exact-null plan path/hash:
- Slurm array job ID:
- array shape/concurrency:
- failed/requeued task IDs:
- missing indices before finalization:
- observed production state database:

Every null must replay:
1. the frozen deterministic adaptive F0--F4 procedure; and
2. full valid evidence completion over the declared graph.

The production H100 workflow should use the prepared plan plus unique indexed
array tasks. Finalization must refuse if any declared index is missing or has
the wrong deterministic seed identity.

Calibration:
- nulls completed:
- median maximum log BF:
- q90 maximum log BF:
- q99 maximum log BF:
- observed maximum log BF:
- finite-sample corrected upper-tail probability:
- minimum resolvable tail probability:
- corresponding posterior-odds calibration:

A tail probability at the finite-sample floor is reported as resolution-limited;
do not extrapolate it to a smaller p-value.

## 12. Final decision table

| gate | status | blocking issue / note |
| --- | --- | --- |
| exact code/environment | | |
| Phase 3 H100 recovery | | |
| canonical data audit | | |
| production freeze | | |
| production numerical diagnostics | | |
| complete graph evidence | | |
| Phase-7 scout validation | | |
| production scout review/refit | | |
| held-out prediction | | |
| event-drop robustness | | |
| nearby-baseline robustness | | |
| exact-search null calibration | | |

## 13. Final artifact inventory

Record paths and SHA-256 hashes where applicable:

- environment/version log;
- Phase-3 campaign plan/summary and per-run recovery summaries;
- canonicalization report;
- dataset manifest;
- fidelity config;
- model graph;
- production campaign;
- production SQLite state;
- search execution summary;
- evidence coverage/completion summaries;
- scored graph;
- scout configs, posteriors, summaries, review records, child specs, independent
  comparisons;
- structured-scout campaign plans/summaries;
- holdout manifest/fold/model summaries;
- event-drop and nearby-baseline configs/summaries;
- exact-null plan, per-null replay records, and exact-null summary;
- Slurm logs/job IDs;
- this completed report.

## 14. Unresolved failures

List every failed or abandoned run and the reason it is not used. Do not delete
failed artifacts from the audit trail.

## 15. Final human review

Reviewer:

Date:

Phase 3 accepted: yes / no

Production analysis accepted for scientific interpretation: yes / no

Notes:
