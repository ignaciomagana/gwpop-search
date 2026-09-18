# gwpop-search

Systematic gravitational-wave population-model search with a standardized
hierarchical Bayesian inference (HBI) engine.

The project separates four concerns:

1. **data contract** - validated PE and selection products;
2. **HBI engine** - one event/selection/rate implementation for every model;
3. **model grammar** - declarative population models with controlled mutations;
4. **search layer** - deterministic search first, agents only after validation.

The immediate target is GWTC-5 BBH population inference. Release-specific raw
ingestion does not live inside the HBI core.

## Status

**Phase 3 in progress: conventional BBH baseline + NumPyro recovery.**

Phases 0-2 are complete. Current Phase-3 implementation includes:

- normalized power-law, broken-power-law, PL+peak, conditional-q, redshift,
  and truncated-`chi_eff` components;
- fixed flat-LambdaCDM transforms;
- explicit source-to-gwcat detector-frame Jacobian;
- `GwcatChiEffBBHModel`;
- serializable synthetic hyperpriors;
- NumPyro NUTS over the common JAX HBI likelihood;
- deterministic chain seeds and code-pinned checkpoint/resume manifests;
- a closed PE+raw-selection synthetic recovery dataset in the exact gwcat-v2
  `chieff` basis;
- GitHub Actions integration tests including the actual 10-parameter baseline
  through NumPyro;
- recovery summaries with posterior truth checks, split R-hat, MCMC ESS,
  divergences, PE/selection importance diagnostics, and likelihood variance;
- a one-command synthetic campaign and H100/Slurm example.

Run the Phase-3 recovery campaign with:

```bash
gwpop-search synthetic-recovery \
  --run-dir runs/phase3-recovery/seed-20260917 \
  --data-seed 20260917 \
  --sampler-seed 20260918
```

See `docs/phase3_recovery.md` for the full run contract.

## Current gate

Phase 4 is intentionally blocked until the baseline passes a multi-chain,
multi-seed synthetic recovery campaign on H100. A short NumPyro smoke test is
not a scientific recovery test.

Real GWTC-5 baseline parity is deferred to the later production freeze, where
the event cut, waveform policy, PE products, selection campaigns, and data
manifests will be frozen together.

## Key documents

- `SPEC.md` - scientific/software architecture.
- `ROADMAP.md` - phased build and acceptance criteria.
- `PROJECT_STATE.md` - authoritative durable build state.
- `docs/data_contract.md` - PE/selection contract.
- `docs/hbi_contract.md` - standardized likelihood contract.
- `docs/phase3_recovery.md` - current H100 recovery runbook.

## Design rule

Agents/search algorithms may choose **which legal model to evaluate next**.
They do not get to modify the likelihood, data transformations, priors, or
validation rules during a production search.
