# Fable / H100 production handoff

This document is the production execution contract for gwpop-search. It is
intentionally separate from the Phase-3 synthetic recovery gate.

## Preconditions

Do not launch GWTC-5 production inference until:

1. the Phase-3 multi-seed synthetic recovery campaign has been reviewed and
   accepted;
2. the production PE and selection HDF5 products have been frozen;
3. dataset_manifest.json has been created from those exact files;
4. the finite model graph has been serialized to model_graph.json;
5. campaign.json has been frozen against the dataset-manifest hash, graph
   hash/root, exact git commit, model prior, F0--F4 numerical configuration,
   scheduler, seed policy, compute budget, artifact root, and state database.

The current repository contains the machinery for this freeze. The actual
GWTC-5 provenance/event/waveform choices remain a separate scientific freeze.

## Environment

Create the GPU-capable environment before entering the batch job. The evidence
extra installs NumPyro and JAXNS dependencies, but the site-appropriate CUDA JAX
wheel must be chosen for the machine.

Example after the correct JAX/CUDA installation is already available:

~~~bash
pip install -e '.[evidence]'
~~~

Production requires JAX x64:

~~~bash
export JAX_ENABLE_X64=true
~~~

The exact code revision is part of the campaign identity. Check out that commit
and expose it to checkpoint manifests:

~~~bash
git checkout <campaign git_commit>
export GWPOP_GIT_COMMIT="$(git rev-parse HEAD)"
~~~

Do not use --ignore-current-commit for production.

## Frozen input validation

Assume:

~~~bash
export GWPOP_MANIFEST=/path/to/dataset_manifest.json
export GWPOP_GRAPH=/path/to/model_graph.json
export GWPOP_CAMPAIGN=/path/to/campaign.json
export GWPOP_DATA_BASE=/path/containing/the/frozen/hdf5/files
export GWPOP_WORK_DIR=/persistent/path/gwpop-production
~~~

Validate before launching inference:

~~~bash
gwpop-search validate-production-freeze \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --base-dir "$GWPOP_DATA_BASE"
~~~

Validation fails if any frozen PE/selection checksum or size changes, the graph
hash/root differs, the campaign points to a different dataset manifest, or the
checked-out git revision does not match.

## Run or resume the deterministic search

~~~bash
gwpop-search run-production-search \
  --manifest "$GWPOP_MANIFEST" \
  --graph "$GWPOP_GRAPH" \
  --campaign "$GWPOP_CAMPAIGN" \
  --base-dir "$GWPOP_DATA_BASE" \
  --work-dir "$GWPOP_WORK_DIR"
~~~

The same command is the resume command.

Completed NUTS chains, evidence repeats, model evaluations, and scheduler
history are reused only when their frozen identities match. A conflicting
resume fails rather than overwriting the old scientific state.

## Fidelity semantics

- **F0**: finite normalized-HBI sanity check at the hyperprior center using a
  small deterministic Monte-Carlo reduction.
- **F1**: short NUTS on reduced PE and inclusion-probability-corrected selection
  samples. Its BIC-like score is for compute allocation only.
- **F2**: full-data NUTS with stronger convergence/importance diagnostics. Its
  BIC-like score is still allocation-only.
- **F3**: repeated full-data JAXNS evidence.
- **F4**: strict full-data NUTS plus repeated higher-accuracy evidence.

No F1/F2 screening value is a Bayes factor or a reported scientific result.

## Model probabilities

scored_graph.json is produced only if every node in the declared graph has a
proper F3/F4 evidence estimate. If screening left some nodes without evidence,
the runner writes evidence_coverage.json and refuses to renormalize the
surviving subset as if it were the declared full model space.

This is separate from search-level null calibration. Adaptive/broader searches
must still be replayed under the declared null procedure before a search-level
claim.

## Durable outputs

The campaign JSON chooses the persistent artifact root and SQLite state path.
A typical tree is:

~~~text
<work-dir>/
  runs/production-v1/
    state.sqlite
    F0/<model-hash>/evaluation.json
    F1/<model-hash>/
      nuts/
      posterior.npz
      evaluation.json
    F2/<model-hash>/
    F3/<model-hash>/
      evidence/
      evaluation.json
    F4/<model-hash>/
      nuts/
      evidence/
      posterior.npz
      evaluation.json
    search_execution_summary.json
    evidence_coverage.json
    scored_graph.json              # only with complete evidence coverage
    production_run_summary.json
~~~

The SQLite store records immutable model specs, evaluations, and promotion
history. Exact scheduler replay is idempotent.

## Compute limits

The production campaign freezes:

- maximum cumulative single-GPU evaluation hours;
- maximum number of F3 models;
- maximum number of F4 models;
- null-replay budget.

A model-count limit is checked before launching that fidelity. Completed work is
never discarded when a compute limit is hit.

## Slurm

Use:

~~~text
scripts/slurm/production_search_h100.sbatch.example
~~~

The template is site-neutral and does not guess partition, account, CUDA
modules, or filesystem layout. It refuses to proceed unless JAX sees a GPU.

The initial production runner is deliberately single-process with durable
SQLite state. Parallel/distributed workers can be added later without changing
the scientific model/HBI/evidence contracts.
