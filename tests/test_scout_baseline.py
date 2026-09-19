import json

import pytest

from gwpop_search.grammar import baseline_model_spec
from gwpop_search.scouts import export_scout_baseline_hyperparameters


def _median(model, offset=0.0):
    result = {}
    for index, name in enumerate(model.priors):
        result[name] = float(index + 1) + float(offset)
    return result


def _write_eval(
    path,
    model,
    *,
    fidelity="F3",
    passed=True,
    median=None,
    model_hash=None,
    format_version="gwpop-search-fidelity-evaluation-2.0",
):
    median = _median(model) if median is None else median
    diagnostics = {
        "passed": passed,
        "checks": [],
        "evidence": {"log_evidence_mean": -10.0, "conservative_error": 0.1},
        "posterior": {"median": median},
        "posterior_median": median,
    }
    payload = {
        "format_version": format_version,
        "model_hash": model.model_hash if model_hash is None else model_hash,
        "fidelity": fidelity,
        "dataset_identity": "dataset-hash",
        "sampler_backend": {"name": "dynesty", "version": "3.1.0"},
        "diagnostics": diagnostics,
    }
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "fidelity,source",
    [
        ("F3", "f3_dynesty_posterior_median"),
        ("F4", "f4_dynesty_posterior_median"),
    ],
)
def test_export_scout_baseline_from_valid_full_fit(
    tmp_path,
    fidelity,
    source,
):
    model = baseline_model_spec()
    evaluation = tmp_path / "evaluation.json"
    output = tmp_path / "baseline_hp.json"
    _write_eval(evaluation, model, fidelity=fidelity)

    provenance = export_scout_baseline_hyperparameters(
        evaluation,
        model,
        output,
    )

    assert provenance["model_hash"] == model.model_hash
    assert provenance["format_version"] == "gwpop-search-scout-baseline-1.1"
    assert provenance["fidelity"] == fidelity
    assert provenance["source"] == source
    assert provenance["sampler_backend"] == {"name": "dynesty", "version": "3.1.0"}
    assert provenance["dataset_identity"] == "dataset-hash"
    assert provenance["evaluation_sha256"]
    assert provenance["hyperparameters_sha256"]
    assert json.loads(output.read_text()) == _median(model)
    sidecar = json.loads(
        output.with_suffix(".json.provenance.json").read_text()
    )
    assert sidecar == provenance


def test_export_scout_baseline_rejects_non_full_fidelity(tmp_path):
    model = baseline_model_spec()
    evaluation = tmp_path / "evaluation.json"
    _write_eval(evaluation, model, fidelity="F2")

    with pytest.raises(ValueError, match="valid F3 or F4"):
        export_scout_baseline_hyperparameters(
            evaluation,
            model,
            tmp_path / "baseline_hp.json",
        )


def test_export_scout_baseline_rejects_failed_diagnostics(tmp_path):
    model = baseline_model_spec()
    evaluation = tmp_path / "evaluation.json"
    _write_eval(evaluation, model, fidelity="F3", passed=False)

    with pytest.raises(ValueError, match="did not pass"):
        export_scout_baseline_hyperparameters(
            evaluation,
            model,
            tmp_path / "baseline_hp.json",
        )


def test_export_scout_baseline_rejects_model_hash_mismatch(tmp_path):
    model = baseline_model_spec()
    evaluation = tmp_path / "evaluation.json"
    _write_eval(
        evaluation,
        model,
        fidelity="F3",
        model_hash="not-the-model",
    )

    with pytest.raises(ValueError, match="model hash"):
        export_scout_baseline_hyperparameters(
            evaluation,
            model,
            tmp_path / "baseline_hp.json",
        )


def test_export_scout_baseline_rejects_parameter_set_mismatch(tmp_path):
    model = baseline_model_spec()
    median = _median(model)
    median.pop(next(iter(median)))
    evaluation = tmp_path / "evaluation.json"
    _write_eval(evaluation, model, fidelity="F3", median=median)

    with pytest.raises(ValueError, match="parameter set"):
        export_scout_baseline_hyperparameters(
            evaluation,
            model,
            tmp_path / "baseline_hp.json",
        )


def test_export_scout_baseline_refuses_overwrite(tmp_path):
    model = baseline_model_spec()
    evaluation = tmp_path / "evaluation.json"
    output = tmp_path / "baseline_hp.json"
    _write_eval(evaluation, model, fidelity="F3")
    export_scout_baseline_hyperparameters(evaluation, model, output)

    with pytest.raises(ValueError, match="already exists"):
        export_scout_baseline_hyperparameters(evaluation, model, output)


def test_export_scout_baseline_refuses_nuts_era_evaluations(tmp_path):
    model = baseline_model_spec()
    evaluation = tmp_path / "evaluation.json"
    _write_eval(
        evaluation,
        model,
        fidelity="F4",
        format_version="gwpop-search-fidelity-evaluation-1.0",
    )
    with pytest.raises(ValueError, match="evaluation-2.0"):
        export_scout_baseline_hyperparameters(
            evaluation,
            model,
            tmp_path / "baseline_hp.json",
        )
