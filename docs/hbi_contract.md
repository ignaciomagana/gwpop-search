# Standardized HBI contract

The purpose of this module is to make all population models share one hierarchical likelihood. Model code provides population densities; it does not reimplement event reweighting, selection effects, rate treatment, or diagnostics.

## 1. Inputs

The HBI engine consumes:

```
PosteriorCatalog
SelectionCatalog
PopulationModel
RunSpec
```

A `PopulationModel` supplies a normalized population density in the same declared coordinate basis as the data adapters:

```python
model.log_prob(theta, hyperparameters, context) -> log_density
```

A model may be factored internally, but the HBI engine only sees the final log density and support.

## 2. Event likelihood

For event i with posterior samples `theta_ij` drawn under reference density `pi_i`,

```
w_ij(Lambda) = p_pop(theta_ij | Lambda) / pi_i(theta_ij)
```

and

```
log ell_i(Lambda)
  = logsumexp_j(log w_ij) - log n_i.
```

The implementation must expose the per-event contributions `log ell_i`; only their sum enters the catalog term.

No implicit renormalization of individual events is allowed beyond the Monte Carlo average above.

## 3. PE effective sample size

For each event,

```
ESS_i = (sum_j w_ij)^2 / sum_j w_ij^2.
```

The implementation should also expose at least:

- `ESS_i / n_i`;
- max normalized importance weight;
- number/fraction of non-finite or zero-support weights;
- optionally Pareto-k diagnostics later.

A model evaluation may be numerically valid but scientifically rejected by the run-quality policy if event ESS is too low.

## 4. Selection estimator

The selection layer evaluates the model-dependent detectable fraction/exposure `alpha(Lambda)` from the `SelectionCatalog`.

There are two allowed input modes.

### 4.1 Raw-draw mode

For campaign k with `N_draw,k` generated injections and retained/detected injections `theta_ak` drawn from `p_draw,k`,

```
alpha_k(Lambda)
  ~= (1 / N_draw,k)
     sum_{a in detected,k}
       p_pop(theta_ak | Lambda) / p_draw,k(theta_ak)
```

up to the explicitly declared exposure convention.

Campaign combination and observing-time factors are part of the `SelectionEstimatorSpec`; they are not guessed from array lengths.

### 4.2 Estimator-ready mode

An adapter may provide a validated denominator/normalization that already encodes the campaign mixture/exposure convention. In this mode the adapter supplies the complete semantics and the HBI estimator applies exactly the documented Monte Carlo sum.

This is the intended path for a stable gwcat export if parity testing confirms it.

The two modes must have distinct types or enum values. Accidentally applying an `N_draw` factor twice must be impossible through the public API.

## 5. Selection ESS

For selection importance weights `u_a`,

```
ESS_sel = (sum_a u_a)^2 / sum_a u_a^2.
```

For multiple campaigns, report both per-campaign and combined diagnostics.

The engine should expose the quantities required for a Talbot/Golomb-style likelihood-variance diagnostic rather than only a scalar pass/fail flag.

## 6. Shape / rate-marginalized likelihood

For N observed events, a common shape likelihood has the form

```
log L_shape(Lambda)
  = sum_i log ell_i(Lambda)
    - N log alpha(Lambda)
    + C
```

for the declared rate prior/convention.

The exact constant and rate-prior assumptions are irrelevant for posterior sampling only when they are truly parameter independent, but they matter for evidence. Therefore the rate convention must be part of `RunSpec` and the evidence implementation must retain all model-dependent normalization terms.

## 7. Poisson point-process likelihood

With explicit rate parameter R and exposure `mu(Lambda, R)`,

```
log L_PPP
  = -mu(Lambda, R)
    + sum_i log [ R * ell_i(Lambda) ]
    + declared constants.
```

The implementation should factor rate and shape cleanly enough to analytically marginalize R where justified.

No model component is allowed to decide its own rate convention.

## 8. Log-space numerics

Use:

- `logsumexp` for all sample sums;
- explicit `-inf` for genuine zero support;
- no artificial density floors in denominators;
- stable chunked accumulation for large selection sets.

Chunking must be mathematically invariant: changing chunk size cannot change the result beyond floating-point tolerance.

## 9. JAX design

Production implementation target:

- JAX arrays;
- jitted population density evaluation;
- vectorized event likelihood where shapes permit;
- ragged events handled by offsets, padding/masking, or segmented reductions chosen from benchmarks;
- selection injections evaluated in chunks to fit accelerator memory;
- no Python callbacks inside the compiled inner likelihood.

A simple NumPy reference implementation should remain in tests as an oracle.

## 10. Diagnostics object

Every likelihood evaluation/run summary should be able to produce structured diagnostics:

```
event_ess
event_ess_fraction
event_max_weight_fraction
selection_ess
selection_ess_by_campaign
selection_weight_variance
log_likelihood_variance_estimate
n_invalid_weights
support_failures
```

Sampler diagnostics are separate:

```
rhat
bulk_ess
tail_ess
divergences
tree_depth
acceptance
```

Do not conflate importance-sampling ESS with MCMC ESS.

## 11. Model normalization

Each population component must be normalized over its declared support or carry an explicit normalization term.

Tests should include numerical integration of each component at randomly selected hyperparameters.

Truncation bounds that vary with another coordinate (e.g. q support conditional on m1) must be part of the density, not an after-the-fact sample cut.

## 12. Hyperpriors

Hyperpriors belong to the model/run specification and are evaluated separately from the catalog likelihood:

```
log posterior
  = log likelihood
    + log hyperprior.
```

Evidence calculations must use the exact declared hyperprior and its normalized density.

The model prior `p(M)` is a different object again: it weights model structures after/with their evidences.

## 13. Evidence contract

A backend must return at least

```
logZ
logZ_uncertainty
backend
backend_version
seed
settings
diagnostics
```

All production model comparisons use one declared evidence backend/config family unless an explicit cross-backend validation is being performed.

NUTS posterior samples alone are not treated as an evidence estimate.

## 14. Reproducibility

A production run identity includes:

```
model_hash
dataset_id
run_spec_hash
code commit
seed
backend version
```

Changing sampler settings should create a new run but not a new scientific model hash.

## 15. Minimum tests before real data

Before loading GWTC-5, the engine must pass:

1. one-dimensional analytic event-reweighting tests;
2. analytic/trivial selection integrals;
3. multi-campaign selection parity fixture;
4. NumPy vs JAX likelihood equality;
5. sample-order permutation invariance;
6. chunk-size invariance;
7. deliberate zero-reference-density failure;
8. recovery of known hyperparameters from simulated catalogs;
9. stable behavior as PE/selection sample counts increase.
