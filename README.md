# gwpop-search

Systematic gravitational-wave population-model search with a standardized hierarchical Bayesian inference (HBI) engine.

The project separates four concerns:

1. **data contract** â posterior-event samples and detected selection injections enter through stable, validated interfaces;
2. **HBI engine** â one implementation of the hierarchical likelihood, selection integral, diagnostics, and later evidence calculations;
3. **model grammar** â population models are declarative objects connected by controlled one-mutation edges;
4. **search layer** â deterministic enumeration first, autonomous proposal/scheduling only after the inference engine is validated.

The immediate target is GWTC-5 BBH population inference, but the core package contains no GWTC-release-specific ingestion logic.

## Status

**Phase 2 complete: standardized HBI engine.**

Implemented so far:

- typed PE/selection density-coordinate contracts and a gwcat-v2 adapter;
- ragged posterior catalogs and campaign-aware selection catalogs;
- a NumPy/SciPy reference event-reweighting likelihood;
- raw multi-campaign selection exposures using explicit `T_k/N_draw,k` normalization;
- estimator-ready selection for gwcat `pdraw` products without double normalization;
- rate-marginalized shape and explicit-rate Poisson point-process likelihoods;
- event and selection importance ESS, max-weight, and Monte-Carlo variance diagnostics;
- chunk-invariant selection evaluation;
- a differentiable JAX likelihood with NumPy parity and fixed-shape selection chunking.

The population-density API is deliberately narrow: a candidate model returns the complete normalized log density in the declared data basis. Models that transform a source-frame population into that basis must own the transformation and Jacobian explicitly; the HBI engine does not guess them.

## Next

**Phase 3: baseline BBH population components + NumPyro inference.**

The next milestone is one conventional normalized BBH model, simulation recovery, and a restartable NUTS run path before any model-graph search is introduced.

See:

- `SPEC.md` â scientific and software specification.
- `ROADMAP.md` â phased build plan and acceptance criteria.
- `PROJECT_STATE.md` â durable handoff / build state.
- `docs/data_contract.md` â implemented PE and selection interfaces.
- `docs/hbi_contract.md` â implemented standardized HBI contract.

## Design rule

Agents and search algorithms may choose **which legal model to evaluate next**. They do not get to modify the likelihood, data transformations, priors, or validation rules during a production search.
