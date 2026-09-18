"""Export a frozen scout baseline from a valid full-data F3/F4 fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gwpop_search.grammar import ModelSpec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median_from_evaluation(payload: dict[str, object]) -> dict[str, float]:
    fidelity = str(payload.get("fidelity", ""))
    diagnostics = dict(payload.get("diagnostics", {}))
    if not bool(diagnostics.get("passed", False)):
        raise ValueError("evaluation did not pass its numerical diagnostics")

    if fidelity == "F3":
        median = diagnostics.get("posterior_median")
    elif fidelity == "F4":
        nuts = dict(diagnostics.get("nuts", {}))
        if not bool(nuts.get("passed", False)):
            raise ValueError("F4 NUTS diagnostics did not pass")
        median = nuts.get("posterior_median")
    else:
        raise ValueError("scout baseline must come from a valid F3 or F4 fit")

    if not isinstance(median, dict):
        raise ValueError("evaluation does not contain a posterior median")
    result = {str(name): float(value) for name, value in median.items()}
    return result


def export_scout_baseline_hyperparameters(
    evaluation_path: str | Path,
    model: ModelSpec,
    output_path: str | Path,
) -> dict[str, object]:
    """Write flat median hyperparameters plus an immutable provenance sidecar."""
    evaluation_path = Path(evaluation_path).resolve()
    output_path = Path(output_path).resolve()
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)
    if output_path.exists() or output_path.with_suffix(
        output_path.suffix + ".provenance.json"
    ).exists():
        raise ValueError("scout baseline output already exists")

    payload = json.loads(evaluation_path.read_text())
    if str(payload.get("model_hash", "")) != model.model_hash:
        raise ValueError("evaluation model hash does not match supplied model spec")

    median = _median_from_evaluation(payload)
    expected = set(model.priors)
    actual = set(median)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "posterior median parameter set does not match model priors: "
            f"missing={missing}, extra={extra}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(median, sort_keys=True, indent=2))

    provenance = {
        "format_version": "gwpop-search-scout-baseline-1.0",
        "model_hash": model.model_hash,
        "fidelity": str(payload["fidelity"]),
        "dataset_identity": str(payload.get("dataset_identity", "")),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": _sha256_file(evaluation_path),
        "hyperparameters_path": str(output_path),
        "hyperparameters_sha256": _sha256_file(output_path),
        "source": (
            "f4_nuts_posterior_median"
            if str(payload["fidelity"]) == "F4"
            else "f3_nested_sampling_posterior_median"
        ),
    }
    provenance_path = output_path.with_suffix(
        output_path.suffix + ".provenance.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2)
    )
    return provenance
