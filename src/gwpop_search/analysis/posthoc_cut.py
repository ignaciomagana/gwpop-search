"""Post-hoc D2 cut bracketing: ``P(sigma^2 <= c)`` for an existing evaluation.

Under the v2 sharp variance cut (likelihood ``x 1[sigma^2 <= c]``, the LVK
form) the evidence at a tighter cut ``c' < c`` is exactly
``Z(c') = Z(c) P_post(sigma^2 <= c' | cut c)``. Evaluations written after the
operator decision of 2026-10-01 record these fractions themselves
(``diagnostics.taper.pooled.posterior_mass_below``,
``NumericalCriteria.posterior_mass_below_cuts``); this module recomputes them
for an evaluation that finished without the field.

The recomputation is the evaluator's own taper diagnostic
(:func:`gwpop_search.inference.dynesty_backend.posterior_taper_mass`): the
run directory's dynesty results are loaded, ``sigma^2`` is re-evaluated at
every weighted dead + live point with the run's own data, model and HBI
configuration (the likelihood identity stored with each run is verified, and
the re-evaluated tapered log-likelihood must reproduce the sampled one), and
the importance-weighted fractions are summarised with their Kish-ESS
binomial errors. ``max_points_per_run`` evaluates a systematic-resampling
subset on a slow device; the errors then include both sampling stages.

Nothing in the run directory is modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ._common import AnalysisInputError, json_ready
from .claims_v2 import mass_below_entry

POSTHOC_MASS_BELOW_FORMAT = "gwpop-search-v2-posterior-mass-below-1.0"


def evaluation_results(run_dir: str | Path):
    """``(evaluation payload, [DynestyResult per repeat], ModelSpec)`` of a run directory."""
    from gwpop_search.grammar import ModelSpec
    from gwpop_search.inference.dynesty_backend import load_dynesty_result

    run_dir = Path(run_dir)
    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.is_file():
        raise AnalysisInputError(f"{run_dir}: no evaluation.json")
    payload = json.loads(evaluation_path.read_text())
    if payload.get("fidelity") not in ("F3", "F4"):
        raise AnalysisInputError(f"{evaluation_path}: not an evidence-rung evaluation ({payload.get('fidelity')})")
    evidence = run_dir / str(((payload.get("diagnostics") or {}).get("nested_sampling") or {}).get(
        "evidence_dir", "evidence"))
    spec = ModelSpec.from_dict(json.loads((evidence / "model_spec.json").read_text()))
    if spec.model_hash != payload["model_hash"]:
        raise AnalysisInputError(f"{evidence}/model_spec.json is not model {payload['model_hash']}")
    manifest = json.loads((evidence / "manifest.json").read_text())
    repeats = int((manifest.get("campaign_config") or {}).get("repeats", 0))
    paths = sorted(evidence.glob("repeat_*/result.npz"))
    if repeats < 1 or len(paths) != repeats:
        raise AnalysisInputError(f"{evidence}: {len(paths)} completed repeats, manifest declares {repeats}")
    if manifest.get("dataset_identity") != payload.get("dataset_identity"):
        raise AnalysisInputError(f"{evidence}: evidence and evaluation dataset identities differ")
    return payload, [load_dynesty_result(path) for path in paths], spec


def posthoc_taper_block(
    results,
    posterior,
    selection,
    spec,
    hbi_config,
    *,
    cuts: Sequence[float],
    max_points_per_run: int | None = None,
    seed: int = 0,
    batch_size: int = 64,
) -> dict:
    """The evaluator's taper block with ``posterior_mass_below`` at ``cuts`` (threshold added)."""
    from gwpop_search.inference.dynesty_backend import build_batched_log_likelihood, posterior_taper_mass
    from gwpop_search.models import compile_model_spec

    taper = getattr(hbi_config, "variance_taper", None)
    if taper is None:
        raise AnalysisInputError("the evaluation's HBI configuration has no variance taper")
    cuts = tuple(sorted({*(float(c) for c in cuts), float(taper.threshold)}))
    model = compile_model_spec(spec)
    loglike = build_batched_log_likelihood(
        posterior, selection, model, tuple(results[0].names), hbi_config=hbi_config, batch_size=batch_size
    )
    return posterior_taper_mass(
        results, loglike, cuts=cuts, max_points_per_run=max_points_per_run, subsample_seed=seed
    )


def posthoc_mass_below(
    run_dir: str | Path,
    posterior,
    selection,
    fidelity_config,
    *,
    dataset_identity: str,
    cuts: Sequence[float],
    max_points_per_run: int | None = None,
    seed: int = 0,
    batch_size: int = 64,
    band_mass_atol: float = 1.0e-9,
) -> dict:
    """Recompute ``P(sigma^2 <= c)`` for the evaluation in ``run_dir``.

    ``fidelity_config`` (the campaign's :class:`FidelityRunConfig`) must hash
    to the evaluation's ``fidelity_config_sha256`` and ``dataset_identity``
    must be its dataset (the frozen manifest hash); the likelihood identity of
    every run is verified by :func:`posterior_taper_mass`. Returns ``{"model_hash",
    "fidelity", "evaluation", "entry" (:func:`claims_v2.mass_below_entry`, the
    D2 input), "taper" (the full block), "consistency"}``; ``consistency``
    compares the recomputed near-cut band mass with the evaluation's stored
    one (equal on the full point set; a subsample estimate otherwise).
    """
    from gwpop_search.inference.fidelity import fidelity_config_sha256

    run_dir = Path(run_dir)
    payload, results, spec = evaluation_results(run_dir)
    if payload.get("dataset_identity") != dataset_identity:
        raise AnalysisInputError(
            f"{run_dir}: evaluation dataset {payload.get('dataset_identity')} is not {dataset_identity}"
        )
    sha = fidelity_config_sha256(fidelity_config)
    if payload.get("fidelity_config_sha256") != sha:
        raise AnalysisInputError(
            f"{run_dir}: evaluation fidelity configuration {payload.get('fidelity_config_sha256')} "
            f"is not the supplied one ({sha})"
        )
    block = posthoc_taper_block(
        results, posterior, selection, spec, fidelity_config.hbi, cuts=cuts,
        max_points_per_run=max_points_per_run, seed=seed, batch_size=batch_size,
    )
    stored = (((payload.get("diagnostics") or {}).get("taper") or {}).get("pooled") or {})
    stored_band = stored.get("posterior_mass_in_taper_region")
    new_band = block["pooled"]["posterior_mass_in_taper_region"]
    subsampled = "subsample" in block["pooled"]
    consistency = {
        "stored_near_cut_band_mass": stored_band,
        "recomputed_near_cut_band_mass": new_band,
        "subsampled": subsampled,
        "reproduces_sampled_log_likelihood": block["reproduces_sampled_log_likelihood"],
        "max_relative_log_likelihood_mismatch": block["max_relative_log_likelihood_mismatch"],
    }
    if stored_band is not None and not subsampled:
        consistency["band_mass_matches"] = bool(abs(float(stored_band) - float(new_band)) <= band_mass_atol)
    return json_ready({
        "model_hash": payload["model_hash"],
        "fidelity": payload["fidelity"],
        "evaluation": str(run_dir / "evaluation.json"),
        "entry": mass_below_entry(block["pooled"], source=run_dir / "evaluation.json"),
        "taper": block,
        "consistency": consistency,
    })


def posthoc_report(rows: Sequence[Mapping], *, cuts: Sequence[float], provenance: Mapping | None = None) -> dict:
    """Versioned output: ``models`` = ``{model_hash: entry}`` (``v2-claim-table --mass-below``)."""
    models = {}
    for row in rows:
        if row["model_hash"] in models:
            raise AnalysisInputError(f"model {row['model_hash']} appears twice")
        models[row["model_hash"]] = row["entry"]
    return json_ready({
        "format_version": POSTHOC_MASS_BELOW_FORMAT,
        "definition": (
            "P(sigma^2 <= c | sharp cut at the evaluation's threshold) over the importance-weighted "
            "dead + live points, error sqrt(p (1 - p) / n_eff) with n_eff the Kish ESS; "
            "ln Z(c') = ln Z(threshold) + ln P(sigma^2 <= c')"
        ),
        "cuts_requested": [float(c) for c in cuts],
        "models": models,
        "evaluations": list(rows),
        "provenance": dict(provenance or {}),
    })


def read_mass_below(payload: Mapping) -> dict[str, dict]:
    """``{model_hash: entry}`` from a :func:`posthoc_report` payload."""
    if payload.get("format_version") != POSTHOC_MASS_BELOW_FORMAT:
        raise AnalysisInputError(
            f"not a {POSTHOC_MASS_BELOW_FORMAT} file (format {payload.get('format_version')!r})"
        )
    return {str(k): dict(v) for k, v in (payload.get("models") or {}).items()}
