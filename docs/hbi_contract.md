# Standardized HBI contract

This is the implemented contract for the common hierarchical likelihood used by every population model. Model code supplies a population density; it does not reimplement event reweighting, selection effects, rate treatment, or numerical diagnostics.

## Inputs

The HBI engine consumes a `PosteriorCatalog`, a `SelectionCatalog`, a population log-density callable, hyperparameters, and an `HBIConfig`.

A population callable has the form

```python
log_density(samples, hyperparameters) -> log p_pop
```

and must return the complete normalized population density in the **declared PE/selection density basis**. Plain callables receive the independent basis coordinates. A model object may declare `required_fields` when an explicit coordinate transformation/Jacobian also needs advisory columns; that declaration must still include every independent basis coordinate.

The HBI layer never reconstructs a PE prior, injection draw density, or coordinate Jacobian.

## Event likelihood

For event `i`, with posterior samples `theta_ij` and adapter-supplied reference density `pi_i`,

```text
log w_ij = log p_pop(theta_ij | Lambda) - log pi_i(theta_ij)

log ell_i = logsumexp_j(log w_ij) - log n_i
```

`evaluate_events` returns every `log ell_i` separately plus importance diagnostics. Genuine zero population support is represented by `-inf`; NaN and `+inf` population densities are hard errors.

## Selection estimator

There are two deliberately distinct contracts.

### raw_draw

For campaign `k`, with `N_draw,k` generated draws, observing time `T_k`, and retained/detected rows,

```text
A_k(Lambda)
  = T_k / N_draw,k
    * sum_a p_pop(theta_ak | Lambda) / p_draw,k(theta_ak)
```

and the combined exposure is

```text
A(Lambda) = sum_k A_k(Lambda).
```

`HBIConfig.raw_selection_use_observing_time=False` explicitly changes the campaign factor to `1/N_draw,k`; it is never inferred from missing metadata.

### estimator_ready

For an adapter product whose denominator already encodes the complete campaign/exposure convention (the current gwcat-v2 path),

```text
A(Lambda)
  = sum_a p_pop(theta_a | Lambda) / pdraw(theta_a).
```

There is **no second `1/ndraw`, observing-time, or campaign-mixture factor**. Phase 1 separately pins this adapter convention.

## Shape likelihood

The implemented shape likelihood is

```text
log L_shape(Lambda)
  = sum_i log ell_i(Lambda)
    - N log A(Lambda)
```

up to model-independent constants. The rate treatment is explicit in `HBIConfig`; a model component cannot choose it.

## Poisson point-process likelihood

For an explicit positive rate `R`,

```text
log L_PPP(Lambda, R)
  = sum_i log ell_i(Lambda)
    + N log R
    - R A(Lambda).
```

This keeps rate and shape separate so later inference code can sample or analytically marginalize the rate under a declared prior.

## Importance diagnostics

For retained importance weights `w_j`, the HBI layer reports

```text
ESS = (sum w)^2 / sum w^2
ESS / N_draw
max normalized weight
number of zero-support weights
```

and a delta-method Monte-Carlo variance estimate for the log integral,

```text
Var[log I_hat] ~= 1/ESS - 1/N_draw.
```

For event PE integrals, `N_draw` is the number of posterior samples for that event. For raw selection campaigns it is the actual generated injection count, so undetected injections enter as exact zero weights without being stored.

For multiple raw campaigns, the combined log-exposure variance is propagated from independent campaign estimates using squared exposure fractions. The shape-likelihood diagnostic is

```text
Var[log L_shape]
  ~= sum_i Var[log ell_i]
     + N^2 Var[log A].
```

This is a diagnostic quantity, not a hard acceptance threshold yet. Production thresholds are deferred to the validation phase.

## NumPy reference backend

`gwpop_search.hbi.numpy_backend` is the correctness oracle. It provides:

- `evaluate_events`;
- `evaluate_selection`;
- `evaluate_catalog_terms`;
- `shape_log_likelihood`;
- `poisson_log_likelihood`;
- `catalog_log_likelihood`.

Selection population densities may be evaluated in chunks. Chunk size is required to be numerically invariant within floating-point tolerance.

## JAX backend

`gwpop_search.hbi.jax_backend` provides differentiable builders for the inner production likelihood:

- ragged PE events are padded/masked once outside the jitted function;
- selection samples are padded into fixed-size chunks;
- chunk accumulation uses `lax.scan` and log-space `logaddexp`;
- population density evaluation remains inside JAX;
- shape and Poisson likelihoods are differentiable with respect to hyperparameters.

The JAX backend is an optional dependency and is exposed lazily through the public `hbi` package so the NumPy reference path does not require JAX.

## Numerical rules

- All Monte-Carlo sums use log-space `logsumexp`.
- Denominator densities are validated by the data layer and are never numerically floored.
- `-inf` population density is valid zero support.
- NaN and `+inf` population density are errors in the NumPy reference path.
- A selection estimator with no finite population support is an explicit failure, not a large negative finite likelihood.
- Sample ordering and selection chunking must not change the likelihood beyond floating-point tolerance.

## Phase-2 validation

The Phase-2 reference tests cover:

1. analytic constant event reweighting;
2. raw multi-campaign selection normalization with `T_k/N_draw,k`;
3. explicit no-time raw selection convention;
4. estimator-ready selection without a second `ndraw` factor;
5. shape and Poisson formulas;
6. selection chunk-size invariance;
7. PE/selection permutation invariance;
8. NaN/+inf rejection and valid `-inf` population support;
9. event/selection likelihood-variance diagnostics;
10. toy hyperparameter recovery;
11. NumPy/JAX full-likelihood equality;
12. NumPy/JAX event and selection term equality;
13. JAX chunk-size invariance;
14. JAX differentiability with `jax.grad`.

The local Phase-2 reference run on 2026-09-17 used Python 3.13.5 and reported `14 passed` in approximately six seconds.

## Still outside Phase 2

The HBI engine does **not** yet define astrophysical population families, hyperpriors, NumPyro sampling, evidence estimation, model priors, production ESS/variance veto thresholds, or GWTC-5 data manifests. Those belong to later phases and must not leak into this common likelihood layer.
