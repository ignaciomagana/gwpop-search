"""End-to-end v2 analysis commands on saved dynesty results of a tiny synthetic campaign:
``run-ppc``, ``reweight-data-variants`` and ``v2-claim-table`` (fed by the real
``analyze-*`` outputs, so the claim table is exercised on the tools' own schema)."""

import json
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from test_analysis_cli import _run_cli, tiny_campaign  # noqa: E402,F401  (module-scoped fixture)

from gwpop_search.analysis.ppc import PREDECLARED_STATISTICS  # noqa: E402


def test_v2_commands_end_to_end(tiny_campaign, monkeypatch):  # noqa: F811
    root, pe_path, sel_path, graph_path, parent, child = tiny_campaign
    data = ["--pe", str(pe_path), "--selection", str(sel_path)]
    common = ["--graph", str(graph_path), "--results", str(root / "runs")]

    # nothing in these commands may start a sampler
    from gwpop_search.inference import dynesty_backend

    def _forbidden(*args, **kwargs):
        raise AssertionError("a v2 analysis command started a dynesty run")

    monkeypatch.setattr(dynesty_backend, "run_dynesty", _forbidden)

    # -- run-ppc: source-frame observables derived with the model's cosmology
    _run_cli(["run-ppc", *data, *common, "--n-draws", "500", "--batch-size", "50", "--output", str(root / "ppc.json")])
    ppc = json.loads((root / "ppc.json").read_text())
    assert set(ppc["models"]) == {parent.model_hash, child.model_hash}
    one = ppc["models"][child.model_hash]
    assert one["identity_verified"] is True
    assert sorted(one["derived_columns"]["pe"]) == ["m1_source", "z"]
    assert set(one["statistics"]) == set(PREDECLARED_STATISTICS)
    assert one["status"] in {"pass", "fail", "unreliable"}
    assert len(one["draws"]["t_obs"]["ks_m1"]) == 500

    # -- reweight-data-variants: event drop + row subset; flag only, never a rerun
    from gwpop_search.data import SelectionCatalog

    n_sel = SelectionCatalog.from_hdf5(sel_path).n_selected
    np.save(root / "mask.npy", np.arange(n_sel) % 4 != 0)
    (root / "variants.json").write_text(json.dumps({
        "variants": [
            {"variant_id": "o3o4b_249", "drop_events": ["SYNTH_0000"], "selection_row_mask": "mask.npy"},
            {"variant_id": "snr9", "selection": str(sel_path)},
        ],
        "edges": [{"parent_hash": parent.model_hash, "child_hash": child.model_hash, "log_bayes_factor": 0.5}],
    }))
    _run_cli(["reweight-data-variants", *data, *common, "--variants", str(root / "variants.json"),
              "--output", str(root / "variants_out.json")])
    out = json.loads((root / "variants_out.json").read_text())
    assert out["reruns_launched"] == 0 and "never launches" in out["rerun_policy"]
    row = out["models"][parent.model_hash]["o3o4b_249"]
    assert row["n_events_variant"] == 5 and row["derivation"]["selection"] == "row_subset"
    # the tiny campaign has far fewer than 1000 effective points: always flagged
    assert row["rerun_recommended"] and "ess_below_min" in row["rerun_reasons"]
    same = out["models"][parent.model_hash]["snr9"]
    assert same["delta_log_evidence"] == pytest.approx(0.0, abs=1e-12)
    assert len(out["edges"]) == 2 and out["rerun_recommended_for"]

    # -- the analyze-* chain and the v2 claim table
    _run_cli(["analyze-edge-mc-error", *data, *common, "--n-draws", "200", "--output-dir", str(root / "mc2")])
    _run_cli(["analyze-prior-sensitivity", *common, "--output", str(root / "prior2.json")])
    _run_cli(["analyze-sddr", *common, "--n-bootstrap", "20", "--output", str(root / "sddr2.json")])
    _run_cli(["analyze-model-comparison", *common, "--penalty", str(math.log(2.0)),
              "--mc-dir", str(root / "mc2" / "mc_weights"), "--sddr", str(root / "sddr2.json"),
              "--prior-sensitivity", str(root / "prior2.json"), "--n-propagation", "200",
              "--output", str(root / "report2.json")])
    report = json.loads((root / "report2.json").read_text())
    lnbf = report["edges"][0]["log_bayes_factor"]
    (root / "alt.json").write_text(json.dumps({"edges": {"chieff.mean.linear_m1": lnbf}, "inapplicable": []}))
    (root / "taper2.json").write_text(json.dumps({"rows": [
        {"parent_hash": parent.model_hash, "child_hash": child.model_hash, "log_bayes_factor": lnbf, "valid": True}
    ]}))
    (root / "labels.json").write_text(json.dumps({"chieff.mean.linear_m1": "C5"}))
    _run_cli(["v2-claim-table", "--report", str(root / "report2.json"), "--graph", str(graph_path),
              "--sddr", str(root / "sddr2.json"), "--prior-sensitivity", str(root / "prior2.json"),
              "--taper2", str(root / "taper2.json"), "--alt-root", f"A1={root / 'alt.json'}",
              "--alt-root", f"A2={root / 'alt.json'}", "--ppc", str(root / "ppc.json"),
              "--atom-labels", str(root / "labels.json"),
              "--output", str(root / "claims.json"), "--markdown", str(root / "claims.md")])
    table = json.loads((root / "claims.json").read_text())
    assert table["n_atoms_tried"] == 1
    row = table["edges"][0]
    assert row["atom_label"] == "C5" and row["edge_key"] == "chieff.mean.linear_m1"
    assert row["D5_alternative_roots"]["status"] == "pass"
    assert row["D6_ppc"]["claimed_model"] == (child.model_hash if lnbf > 0 else parent.model_hash)
    # a |ln BF| of a few tenths on 6 events is neither supported nor disfavoured
    assert row["label"] == "INCONCLUSIVE"
    assert row["D2_strength"]["status"] in {"fail", "incomplete"}
    assert "`C5`" in (root / "claims.md").read_text()
