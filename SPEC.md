# gwpop-search specification

## 1. Scientific objective

Build a reproducible search engine for compact-binary population structure. The first application is the GWTC-5 BBH population. The system should explore a large but explicitly constrained family of population models, quantify support for controlled model mutations, and calibrate the search procedure against null catalogs.

The project is not an LLM wrapper around independent population runs. It is a population-inference package with an optional agent/search layer above it.

## 2. Non-negotiable invariants

1. **One HBI implementation.** Every candidate model is scored by the same likelihood and selection code.
2. **Declarative models.** Production search models are serialized specifications. Search agents may choose legal specifications; they may not inject arbitrary likelihood code.
3. **One mutation per graph edge.** Model comparisons remain interpretable.
4. **Reference densities are explicit inputs.** PE priors and injection draw densities are loaded through the data contract and never guessed from filenames or release names inside the HBI engine.
5. **Coordinate basis is explicit.** Population density, PE reference density, selection draw density, and Jacobians must refer to the same declared basis.
6. **No hidden model mining.** The final search works with model priors / posterior model probabilities and null-search calibration, not the largest Bayes factor found after an unrecorded search.
7. **Diagnostics can veto a run.** Numerical validity is separate from scientific preference.
8. **Deterministic core first.** The model grammar, enumeration, HBI, and validation must work without any LLM agent.
9. **Search is replayable.** Given the same data snapshot, seed policy, grammar, scheduler version, and budget, the search history can be reproduced.
10. **No production claim from screening fidelity.** Cheap runs only allocate compute.

## 3. Package architecture

```
src/gwpop_search/
    data/           canonical PE / selection interfaces and adapters
    hbi/            event likelihood, selection, rate treatment, diagnostics
    models/         population components and composition
    grammar/        legal model schema and one-edge mutations
    inference/      NUTS / SMC / evidence backend interfaces
    search/         deterministic enumeration, beam search, later MCTS
    validation/     ESS, convergence, PPC, holdout, injection recovery
    nulls/          generate/replay full searches under declared nulls
    store/          result database and artifact manifests
    cli/            command-line entry points
    agents/         optional proposers/critics; never part of core likelihood
```

Phase 0 creates only the package boundary and contracts. Later phases fill these modules.

## 4. Canonical population coordinates

The initial BBH internal coordinate set is

```
(m1_source, q, z, chi_eff, chi_p)
```

with optional extensions for component spins and additional observables.

Detector-frame PE products can be consumed through an adapter, but the adapter must declare every transformation and Jacobian. The HBI code itself should not contain GWTC-release-specific transformations.

A model evaluates

```
log_prob(theta, hyperparameters, context) -> array
```

and exposes its normalized support. Component distributions are composable but the fully composed model owns normalization.

## 5. Standardized HBI contract

For event i with posterior samples theta_ij drawn under reference prior pi_i,

```
log L_i(Lambda)
  = logsumexp_j[
        log p_pop(theta_ij | Lambda)
        - log pi_i(theta_ij)
    ] - log n_i.
```

The selection estimator is computed from the declared selection product in the same coordinate basis. Its exact multi-campaign normalization is part of the data/HBI interface and will be parity-tested against the gwcat convention before GWTC-5 production use.

The engine will support at least two rate treatments:

- **shape / rate-marginalized likelihood**, with a selection term proportional to `-N log alpha(Lambda)` under the declared rate prior;
- **Poisson point-process likelihood**, with an explicit rate parameter and observing-time exposure.

The rate treatment must be explicit in the run specification.

All numerics are performed in log space except where a tested reference-product convention requires otherwise.

## 6. Model grammar

A model specification contains independent blocks such as:

- primary-mass distribution;
- mass-ratio conditional;
- effective-spin distribution;
- precession/component-spin distribution;
- redshift/rate evolution;
- mixture structure;
- conditional-dependence edges.

Example:

```yaml
mass:
  family: pl_peak

pairing:
  family: powerlaw_q
  beta:
    dependence: logistic_m1

chieff:
  family: truncated_gaussian
  mean:
    dependence: constant
  width:
    dependence: linear_q

redshift:
  family: powerlaw

mixture:
  components: 1
```

The canonicalized specification is hashed. The hash identifies the scientific model independently of sampler settings.

Mutations are typed operations such as:

```
change_mass_family
add_mass_feature
add_q_m1_dependence
add_chieff_mean_m1
add_chieff_mean_q
add_chieff_width_q
add_chieff_width_z
split_spin_component
add_mass_redshift_dependence
```

A graph edge applies exactly one mutation.

## 7. Model prior and scientific score

The search does not optimize raw maximum Bayes factor over an ever-growing model list.

For model M,

```
log p(M | D) = log Z_M + log p(M) + constant.
```

The model prior penalizes unnecessary structure and is itself versioned. Candidate choices include priors on number of dependence edges, mixture components, and flexible degrees of freedom.

Posterior probabilities of structural statements can then be marginalized over functional forms, e.g. the posterior mass of all models containing a q <- m1 dependence.

## 8. Multi-fidelity search

Planned evaluation levels:

- **F0 sanity:** model construction, normalization/support tests, tiny synthetic likelihood.
- **F1 screen:** reduced posterior/injection samples and short inference; never quoted scientifically.
- **F2 inference:** full events with adequate selection samples and standard diagnostics.
- **F3 evidence:** common evidence backend and repeated evidence estimates.
- **F4 production:** strict diagnostics, robustness runs, holdout prediction, and null-search calibration.

Promotion rules are deterministic and stored.

Initial scheduler: deterministic enumeration / beam search. MCTS or agent-guided proposal comes only after deterministic search is validated.

## 9. Flexible models as scouts

Flexible residual models (spline / GP / HSGP) are primarily reconnaissance tools.

The intended loop is:

```
parametric baseline
    -> flexible residual
    -> identify posterior residual structure
    -> compile an interpretable grammar mutation
    -> refit with standard HBI
    -> compare / validate
```

A flexible residual by itself is not automatically a claimed astrophysical feature.

## 10. Validation firewall

A promoted scientific feature must eventually survive:

- event reweighting ESS requirements;
- selection ESS / likelihood-variance requirements;
- sampler convergence and repeated seeds;
- prior-width sensitivity where relevant;
- alternative nearby baseline models;
- posterior-predictive checks;
- held-out predictive checks;
- injection recovery;
- replay of the complete search on null catalogs.

The null replay targets the statistic induced by the *search procedure*, for example the distribution of the maximum evidence improvement encountered under a null population.

## 11. Result store

The durable state will record at minimum:

```
model_hash
parent_hash
mutation
model_spec
model_prior_version
dataset_id
hbi_version
git_commit
run_config
seed
status
posterior artifact
logZ and uncertainty
event ESS summary
selection ESS summary
convergence diagnostics
validation outcomes
compute usage
```

The database is the source of truth for the swarm. Agent chat context is not.

## 12. GWTC-5 first target

The first real-data milestone is deliberately small: a frozen BBH dataset and approximately 20--40 controlled models around a reproduced baseline. The purpose is to validate the model DAG, likelihood parity, selection treatment, diagnostics, evidence machinery, and result database before autonomous exploration.

## 13. Deferred until later

The following are intentionally not fixed in Phase 0:

- final GWTC-5 event cut;
- exact public release manifest and checksums;
- production PE waveform policy;
- production O3/O4 injection manifest;
- final evidence backend;
- H100-specific JAX installation;
- database deployment choice;
- agent model/provider.

Data provenance and release manifests will be added after the internal schema and HBI engine are tested, per project plan.
