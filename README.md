# gwpop-search

Systematic gravitational-wave population-model search with a standardized hierarchical Bayesian inference (HBI) engine.

The project separates four concerns:

1. **data contract** — posterior-event samples and detected selection injections enter through stable, validated interfaces;
2. **HBI engine** — one implementation of the hierarchical likelihood, selection integral, diagnostics, and later evidence calculations;
3. **model grammar** — population models are declarative objects connected by controlled one-mutation edges;
4. **search layer** — deterministic enumeration first, autonomous proposal/scheduling only after the inference engine is validated.

The immediate target is GWTC-5 BBH population inference, but the core package should not contain GWTC-5-specific release logic.

## Status

**Phase 0: architecture/specification.**

No production inference is implemented yet. The repository currently defines the contracts, package layout, build phases, and persistent project state that subsequent work must follow.

See:

- `SPEC.md` — scientific and software specification.
- `ROADMAP.md` — phased build plan and acceptance criteria.
- `PROJECT_STATE.md` — durable handoff / build memory.
- `docs/data_contract.md` — intended PE and selection interfaces, informed by `gwcat`.
- `docs/hbi_contract.md` — standardized HBI likelihood and diagnostics contract.

## Design rule

Agents and search algorithms may choose **which legal model to evaluate next**. They do not get to modify the likelihood, data transformations, priors, or validation rules during a production search.

That separation is the scientific firewall between automated model exploration and the inference machinery used to score a model.
