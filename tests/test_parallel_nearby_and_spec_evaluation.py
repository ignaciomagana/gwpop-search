"""Per-model nearby-baseline evaluation and single-spec evaluation are identities.

``run_nearby_model_evaluation`` pre-computes one F3 evaluation of one
nearby-baseline scenario into the suite's own artifact tree, so the suite must
reload it (no dynesty run started) and end up with exactly the serial tree and
summary. ``run_model_spec_evaluation`` evaluates a declared spec at the frozen
campaign's settings; for the spec of a graph model it must be the very
evaluation ``run-fidelity-evaluation`` (and the production search) performs:
same hash, seed, dataset identity, run fingerprint and ln Z.
"""

import hashlib
import json
import shutil
from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from gwpop_search.cli import build_parser  # noqa: E402
from gwpop_search.grammar import (  # noqa: E402
    DEFAULT_MUTATIONS,
    apply_mutation,
    baseline_model_spec,
    enumerate_model_graph,
    load_model_spec,
)
from gwpop_search.grammar.io import save_model_graph  # noqa: E402
from gwpop_search.inference import DynestyConfig  # noqa: E402
from gwpop_search.inference.evidence_campaign import (  # noqa: E402
    EvidenceCampaignConfig,
)
from gwpop_search.inference.fidelity import (  # noqa: E402
    F0SanityConfig,
    FidelityRunConfig,
    NumericalCriteria,
)
from gwpop_search.inference.synthetic import (  # noqa: E402
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.production import (  # noqa: E402
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    build_dataset_manifest_from_canonical_files,
    installed_sampler_backend,
    model_graph_hash,
    run_model_spec_evaluation,
    save_dataset_manifest,
    save_production_campaign,
    with_prior,
)
from gwpop_search.search import Fidelity, SchedulerConfig, evaluation_seed  # noqa: E402
from gwpop_search.store import ResultStore  # noqa: E402
from gwpop_search.validation import (  # noqa: E402
    NearbyBaselineConfig,
    NearbyBaselineScenario,
    NearbyBaselineSuiteSpec,
    nearby_baseline_seed,
    nearby_dataset_identity,
    nearby_scenario_graph,
    nearby_scenario_model_run_dir,
    nearby_scenario_models,
    run_nearby_baseline_suite,
    run_nearby_model_evaluation,
    save_nearby_baseline_suite_spec,
)

# ``execution.total_compute_cost`` sums evaluation wall-clock hours: the only
# field the reloading suite may (and should) change.
_NEARBY_TIMING_FIELDS = (("execution", "total_compute_cost"),)


def _without(payload, paths):
    payload = json.loads(json.dumps(payload))
    for path in paths:
        node = payload
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return payload


def _lenient_criteria():
    return NumericalCriteria(
        max_cross_run_r_hat=100.0,
        min_kish_ess_per_run=1.0,
        max_evidence_error=100.0,
        max_evidence_repeat_std=100.0,
        max_repeat_consistency_z=100.0,
        min_event_ess=1e-9,
        min_selection_ess=1e-9,
        max_event_weight_fraction=1.0,
        max_selection_weight_fraction=1.0,
        max_shape_log_likelihood_variance=1e12,
    )


def _fidelity_config():
    # The same small-but-converged configuration as
    # tests/test_parallel_model_evaluation.py.
    dynesty = DynestyConfig(
        nlive=32,
        bound="multi",
        sample="rslice",
        slices=1,
        dlogz=10.0,
        batch_size=64,
        num_posterior_samples=200,
    )
    evidence = EvidenceCampaignConfig(
        repeats=2,
        dynesty=dynesty,
        slices_multiplier=None,
    )
    return FidelityRunConfig(
        f0=F0SanityConfig(
            prior_draws=64,
            batch_size=16,
            parity_pe_samples_per_event=8,
            parity_selected_per_campaign=64,
            parity_points=2,
        ),
        f3_evidence=evidence,
        f4_evidence=evidence,
        f3_criteria=_lenient_criteria(),
        f4_criteria=_lenient_criteria(),
        posterior_draws=128,
        rhat_draws_per_run=100,
        diagnostics_batch_size=16,
    )


def _graph():
    return enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=2,
    )


def _campaign(graph):
    return ProductionCampaignConfig(
        campaign_id="parallel-nearby-spec-eval-test",
        dataset_manifest_hash="a" * 64,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="b" * 40,
        model_prior={"version": "axis-complexity-v1", "penalty_per_axis": 0.5},
        fidelity=_fidelity_config(),
        scheduler=SchedulerConfig(beam_width=4, exploration_quota=0),
        seed_policy=SeedPolicy(root_seed=31),
        budget=SearchBudget(
            max_gpu_hours=50.0,
            max_f3_models=4,
            max_f4_models=2,
            max_null_replays=8,
        ),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )


def _survey():
    return SyntheticSurveyConfig(
        n_events=4,
        posterior_samples_per_event=8,
        n_injections=300,
        population_batch_size=128,
        redshift_sampling_grid=512,
        observation_model="noisy_observation",
    )


@pytest.fixture(scope="module")
def dataset():
    return generate_baseline_synthetic_dataset(seed=77, config=_survey())


def _mutation(mutation_id):
    return next(item for item in DEFAULT_MUTATIONS if item.mutation_id == mutation_id)


def _nearby_root():
    # An alternative root one mutation away from the frozen root, as the
    # campaign's pre-registered nearby baselines are.
    return apply_mutation(baseline_model_spec(), _mutation("mass.family.powerlaw"))


def _suite(*scenario_ids):
    return NearbyBaselineSuiteSpec(
        scenarios=tuple(
            NearbyBaselineScenario(
                scenario_id,
                _nearby_root(),
                max_depth=1,
                max_models=2,
                note="test alternative root",
            )
            for scenario_id in scenario_ids
        ),
        config=NearbyBaselineConfig(
            max_gpu_hours_per_scenario=25.0,
            max_f3_models=4,
            max_f4_models=2,
        ),
    )


def _result_files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("result.npz"))
    }


def _result_mtimes(root):
    return {
        str(path.relative_to(root)): path.stat().st_mtime_ns
        for path in sorted(root.rglob("result.npz"))
    }


def _count_dynesty_runs(monkeypatch):
    from gwpop_search.inference import dynesty_backend

    started = []
    real = dynesty_backend.run_dynesty

    def counting(*args, **kwargs):
        started.append(kwargs.get("identity"))
        return real(*args, **kwargs)

    monkeypatch.setattr(dynesty_backend, "run_dynesty", counting)
    return started


def _stripped_suite(summary):
    payload = json.loads(json.dumps(summary))
    for scenario in payload["scenarios"]:
        scenario["execution"].pop("total_compute_cost", None)
    return payload


# ---------------------------------------------------------------------------
# (a) nearby-baseline suite
# ---------------------------------------------------------------------------


def test_nearby_scenario_models_are_the_models_the_suite_evaluates(
    tmp_path,
    dataset,
):
    graph = _graph()
    campaign = _campaign(graph)
    suite = _suite("nearby_powerlaw_mass")
    scenario = suite.scenarios[0]

    listing = nearby_scenario_models(scenario, suite.config)
    scenario_graph = nearby_scenario_graph(scenario)
    hashes = [row["model_hash"] for row in listing["models"]]
    assert hashes == sorted(model.model_hash for model in scenario_graph.nodes)
    assert listing["root_model_hash"] == scenario.root_spec.model_hash
    assert listing["n_graph_models"] == 2
    assert listing["f3_budget_fits_all_models"] is True
    assert sum(row["is_root"] for row in listing["models"]) == 1

    run_nearby_baseline_suite(
        tmp_path,
        dataset.posterior,
        dataset.selection,
        graph,
        campaign,
        base_dataset_identity="c" * 64,
        suite=suite,
    )
    store = ResultStore(tmp_path / scenario.scenario_id / "state.sqlite")
    f3 = [row["model_hash"] for row in store.evaluations(fidelity="F3")]
    f0 = [row["model_hash"] for row in store.evaluations(fidelity="F0")]
    # The suite runs F0 on every model first, then F3 on every F0 pass.
    assert sorted(f0) == hashes
    assert sorted(f3) == hashes
    seeds = {
        row["model_hash"]: int(row["seed"])
        for row in store.evaluations(fidelity="F3")
    }
    root_seed = nearby_baseline_seed(
        campaign.seed_policy.root_seed,
        scenario.scenario_id,
    )
    assert seeds == {
        model_hash: evaluation_seed(root_seed, model_hash, Fidelity.F3_EVIDENCE)
        for model_hash in hashes
    }


def test_per_model_nearby_evaluation_reproduces_the_serial_suite(
    monkeypatch,
    tmp_path,
    dataset,
):
    graph = _graph()
    campaign = _campaign(graph)
    suite = _suite("nearby_powerlaw_mass", "nearby_powerlaw_mass_b")
    identity = "c" * 64

    serial_root = tmp_path / "serial"
    serial = run_nearby_baseline_suite(
        serial_root,
        dataset.posterior,
        dataset.selection,
        graph,
        campaign,
        base_dataset_identity=identity,
        suite=suite,
    )

    parallel_root = tmp_path / "parallel"
    n_precomputed = 0
    for scenario in suite.scenarios:
        for row in nearby_scenario_models(scenario, suite.config)["models"]:
            payload = run_nearby_model_evaluation(
                parallel_root,
                dataset.posterior,
                dataset.selection,
                campaign,
                base_dataset_identity=identity,
                scenario=scenario,
                model_hash=row["model_hash"],
            )
            n_precomputed += 1
            assert payload["state_database_written"] is False
            assert payload["fidelity"] == "F3"
            assert payload["scenario_id"] == scenario.scenario_id
            assert payload["dataset_identity"] == nearby_dataset_identity(
                identity,
                scenario,
            )
            assert payload["run_dir"] == str(
                nearby_scenario_model_run_dir(
                    parallel_root,
                    scenario.scenario_id,
                    row["model_hash"],
                    Fidelity.F3_EVIDENCE,
                )
            )
    assert list(parallel_root.rglob("state.sqlite")) == []
    # The plan freezes the whole suite (and the reference graph): suite only.
    assert not (parallel_root / "nearby_baseline_plan.json").exists()
    # F0 stays with the suite.
    assert list(parallel_root.rglob("F0")) == []

    pre_mtimes = _result_mtimes(parallel_root)
    assert len(pre_mtimes) == 2 * n_precomputed  # repeats=2 per model

    started = _count_dynesty_runs(monkeypatch)
    parallel = run_nearby_baseline_suite(
        parallel_root,
        dataset.posterior,
        dataset.selection,
        graph,
        campaign,
        base_dataset_identity=identity,
        suite=suite,
    )

    # The suite reloaded every dynesty run; it sampled nothing.
    assert started == []
    assert _result_mtimes(parallel_root) == pre_mtimes
    # Every result.npz is byte-identical to the serial tree's.
    assert _result_files(parallel_root) == _result_files(serial_root)

    assert _stripped_suite(parallel) == _stripped_suite(serial)
    for scenario in suite.scenarios:
        serial_summary = json.loads(
            (serial_root / scenario.scenario_id / "nearby_baseline_summary.json").read_text()
        )
        parallel_summary = json.loads(
            (parallel_root / scenario.scenario_id / "nearby_baseline_summary.json").read_text()
        )
        assert _without(parallel_summary, _NEARBY_TIMING_FIELDS) == _without(
            serial_summary,
            _NEARBY_TIMING_FIELDS,
        )
    assert json.loads(
        (parallel_root / "nearby_baseline_plan.json").read_text()
    ) == json.loads((serial_root / "nearby_baseline_plan.json").read_text())


def test_a_misplaced_nearby_evaluation_is_refused_not_silently_reused(
    tmp_path,
    dataset,
):
    """Same root spec, different scenario: seed and dataset identity differ."""
    graph = _graph()
    campaign = _campaign(graph)
    wrong_suite = _suite("nearby_wrong")
    right_suite = _suite("nearby_right")
    wrong = wrong_suite.scenarios[0]
    right = right_suite.scenarios[0]
    model_hash = wrong.root_spec.model_hash
    root = tmp_path / "root"
    run_nearby_model_evaluation(
        root,
        dataset.posterior,
        dataset.selection,
        campaign,
        base_dataset_identity="c" * 64,
        scenario=wrong,
        model_hash=model_hash,
    )
    shutil.copytree(
        nearby_scenario_model_run_dir(root, wrong.scenario_id, model_hash, "F3"),
        nearby_scenario_model_run_dir(root, right.scenario_id, model_hash, "F3"),
    )
    with pytest.raises(ValueError, match="mismatch|does not match"):
        run_nearby_baseline_suite(
            root,
            dataset.posterior,
            dataset.selection,
            graph,
            campaign,
            base_dataset_identity="c" * 64,
            suite=right_suite,
        )

    with pytest.raises(ValueError, match="not in the graph of nearby scenario"):
        run_nearby_model_evaluation(
            root,
            dataset.posterior,
            dataset.selection,
            campaign,
            base_dataset_identity="c" * 64,
            scenario=right,
            # The frozen root is not in the alternative-root neighbourhood.
            model_hash=graph.root_hash,
        )


# ---------------------------------------------------------------------------
# CLI plumbing on a frozen dataset on disk
# ---------------------------------------------------------------------------


def _run_cli(argv, capsys):
    args = build_parser().parse_args(argv)
    capsys.readouterr()
    args.func(args)
    return capsys.readouterr().out


@pytest.fixture()
def frozen(tmp_path, dataset):
    """A frozen dataset, graph and campaign on disk, as the CLI requires."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset.posterior.to_hdf5(data_dir / "pe.h5")
    dataset.selection.to_hdf5(data_dir / "selection.h5")
    manifest = build_dataset_manifest_from_canonical_files(
        data_dir / "pe.h5",
        data_dir / "selection.h5",
        dataset_id="parallel-nearby-spec-eval",
        event_selection={"far_threshold_per_year": 1.0},
        waveform_policy={"pe_release": "test"},
        stored_pe_path="pe.h5",
        stored_selection_path="selection.h5",
    )
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest_path, manifest)
    graph = _graph()
    graph_path = tmp_path / "graph.json"
    save_model_graph(graph_path, graph)
    campaign = replace(
        _campaign(graph),
        dataset_manifest_hash=manifest.manifest_hash,
        sampler_backend=installed_sampler_backend(),
    )
    campaign_path = tmp_path / "campaign.json"
    save_production_campaign(campaign_path, campaign)
    common = [
        "--manifest", str(manifest_path),
        "--graph", str(graph_path),
        "--campaign", str(campaign_path),
        "--base-dir", str(data_dir),
        "--ignore-current-commit",
    ]
    return {
        "manifest": manifest,
        "graph": graph,
        "graph_path": graph_path,
        "campaign": campaign,
        "common": common,
    }


def test_nearby_cli_lists_and_precomputes_what_the_suite_reloads(
    monkeypatch,
    tmp_path,
    capsys,
    frozen,
):
    suite = _suite("nearby_powerlaw_mass")
    config_path = tmp_path / "nearby.json"
    save_nearby_baseline_suite_spec(config_path, suite)

    out = _run_cli(
        ["list-nearby-scenario-models", "--nearby-config", str(config_path)],
        capsys,
    )
    hashes = out.split()
    assert hashes == [
        row["model_hash"]
        for row in nearby_scenario_models(suite.scenarios[0])["models"]
    ]
    listing = json.loads(
        _run_cli(
            [
                "list-nearby-scenario-models",
                "--nearby-config", str(config_path),
                "--scenario-id", "nearby_powerlaw_mass",
                "--json",
            ],
            capsys,
        )
    )
    assert listing["scenarios"][0]["max_f3_models"] == 4

    root = tmp_path / "suite"
    for model_hash in hashes:
        payload = json.loads(
            _run_cli(
                [
                    "run-nearby-model-evaluation",
                    *frozen["common"],
                    "--nearby-config", str(config_path),
                    "--scenario-id", "nearby_powerlaw_mass",
                    "--model-hash", model_hash,
                    "--root", str(root),
                ],
                capsys,
            )
        )
        assert payload["model_hash"] == model_hash
        assert payload["dataset_identity"] == nearby_dataset_identity(
            frozen["manifest"].manifest_hash,
            suite.scenarios[0],
        )

    started = _count_dynesty_runs(monkeypatch)
    summary = json.loads(
        _run_cli(
            [
                "run-nearby-baseline-suite",
                *frozen["common"],
                "--nearby-config", str(config_path),
                "--work-dir", str(tmp_path),
                "--root", str(root),
            ],
            capsys,
        )
    )
    assert started == []
    assert summary["scenarios"][0]["n_models_with_evidence"] == len(hashes)


# ---------------------------------------------------------------------------
# (b) one declared model spec at the frozen campaign's settings
# ---------------------------------------------------------------------------


def _evaluation_core(path):
    """``evaluation.json`` without its wall-clock fields."""
    payload = json.loads(path.read_text())
    payload.pop("elapsed_seconds")
    return payload["diagnostics"]["evidence"], {
        key: payload[key]
        for key in (
            "model_hash",
            "fidelity",
            "dataset_identity",
            "fidelity_config_sha256",
            "screen_value",
        )
    }, payload["diagnostics"]["passed"], [
        (item["name"], item.get("passed"), item.get("value"))
        for item in payload["diagnostics"]["checks"]
    ]


def test_spec_of_a_graph_model_reproduces_its_fidelity_evaluation(
    monkeypatch,
    tmp_path,
    capsys,
    frozen,
):
    graph = frozen["graph"]
    campaign = frozen["campaign"]
    model_hash = max(model.model_hash for model in graph.nodes)

    reference_root = tmp_path / "fidelity"
    reference = json.loads(
        _run_cli(
            [
                "run-fidelity-evaluation",
                *frozen["common"],
                "--fidelity", "F3",
                "--model-hash", model_hash,
                "--output-root", str(reference_root),
            ],
            capsys,
        )
    )["evaluations"][0]

    spec_path = tmp_path / "spec.json"
    exported = json.loads(
        _run_cli(
            [
                "export-model-spec",
                "--graph", str(frozen["graph_path"]),
                "--model-hash", model_hash,
                "--output", str(spec_path),
            ],
            capsys,
        )
    )
    assert exported["model_hash"] == model_hash
    assert exported["derived_from"]["priors"] == {}
    assert load_model_spec(spec_path) == graph.by_hash[model_hash]

    spec_root = tmp_path / "spec_eval"
    payload = json.loads(
        _run_cli(
            [
                "run-model-spec-evaluation",
                *frozen["common"],
                "--model-spec", str(spec_path),
                "--root", str(spec_root),
            ],
            capsys,
        )
    )
    # Same model hash, seed (= the production seed) and dataset identity.
    assert payload["model_hash"] == model_hash
    assert payload["seed"] == reference["seed"] == evaluation_seed(
        campaign.seed_policy.root_seed,
        model_hash,
        Fidelity.F3_EVIDENCE,
    )
    assert payload["dataset_identity"] == frozen["manifest"].manifest_hash
    assert payload["graph"] == {
        "root_hash": graph.root_hash,
        "model_in_graph": True,
        "depth": graph.depths[model_hash],
    }
    assert payload["spec_file"]["sha256"] == hashlib.sha256(
        spec_path.read_bytes()
    ).hexdigest()
    assert payload["code"]["source_sha256"]
    assert payload["diagnostics_pass"] == reference["diagnostics_pass"]
    assert payload["screen_value"] == reference["screen_value"]
    assert payload["log_evidence_mean"] == reference["screen_value"]

    reference_dir = reference_root / "F3" / model_hash
    spec_dir = spec_root / "F3" / model_hash
    # Same run fingerprint (the evidence manifests pin data, seed, config,
    # code and runtime) and byte-identical dynesty results.
    for relative in ("evidence/manifest.json", "evidence/model_spec.json"):
        assert (spec_dir / relative).read_bytes() == (
            reference_dir / relative
        ).read_bytes()
    for repeat in ("repeat_000", "repeat_001"):
        assert (spec_dir / "evidence" / repeat / "manifest.json").read_bytes() == (
            reference_dir / "evidence" / repeat / "manifest.json"
        ).read_bytes()
    assert _result_files(spec_dir) == _result_files(reference_dir)
    assert _evaluation_core(spec_dir / "evaluation.json") == _evaluation_core(
        reference_dir / "evaluation.json"
    )
    # The run directory holds exactly what the production evaluator writes;
    # the provenance lives next to it.
    assert sorted(
        str(p.relative_to(spec_dir)) for p in spec_dir.rglob("*")
    ) == sorted(str(p.relative_to(reference_dir)) for p in reference_dir.rglob("*"))
    provenance = spec_root / "F3" / f"{model_hash}.spec_evaluation.json"
    assert json.loads(provenance.read_text()) == payload

    # Evaluating the spec into the fidelity-evaluation tree reloads it: the
    # fingerprint is identical, so nothing is resampled.
    started = _count_dynesty_runs(monkeypatch)
    again = json.loads(
        _run_cli(
            [
                "run-model-spec-evaluation",
                *frozen["common"],
                "--model-spec", str(spec_path),
                "--root", str(reference_root),
            ],
            capsys,
        )
    )
    assert started == []
    assert again["log_evidence_mean"] == payload["log_evidence_mean"]


def test_widened_prior_spec_is_a_new_model_and_dry_run_writes_nothing(
    tmp_path,
    capsys,
    frozen,
    dataset,
):
    graph = frozen["graph"]
    campaign = frozen["campaign"]
    parent = graph.by_hash[graph.root_hash]
    name = min(
        key
        for key, prior in parent.priors.items()
        if prior.family == "uniform"
    )
    low = parent.priors[name].parameters["low"]
    high = parent.priors[name].parameters["high"]
    width = high - low

    spec_path = tmp_path / "widened.json"
    exported = json.loads(
        _run_cli(
            [
                "export-model-spec",
                "--graph", str(frozen["graph_path"]),
                "--model-hash", graph.root_hash,
                "--set-prior", name, "uniform", str(low - width), str(high + width),
                "--output", str(spec_path),
            ],
            capsys,
        )
    )
    widened = load_model_spec(spec_path)
    assert widened.model_hash == exported["model_hash"] != parent.model_hash
    assert widened == with_prior(
        parent,
        name,
        "uniform",
        {"low": low - width, "high": high + width},
    )
    assert list(exported["derived_from"]["priors"]) == [name]
    assert exported["derived_from"]["blocks"] == {}
    with pytest.raises(ValueError, match="is not declared"):
        with_prior(parent, name + "_typo", "uniform", {"low": 0.0, "high": 1.0})

    root = tmp_path / "dry"
    payload = json.loads(
        _run_cli(
            [
                "run-model-spec-evaluation",
                *frozen["common"],
                "--model-spec", str(spec_path),
                "--derived-from", graph.root_hash,
                "--root", str(root),
                "--dry-run",
            ],
            capsys,
        )
    )
    assert payload["dry_run"] is True
    assert payload["graph"]["model_in_graph"] is False
    assert payload["seed"] == evaluation_seed(
        campaign.seed_policy.root_seed,
        widened.model_hash,
        Fidelity.F3_EVIDENCE,
    )
    assert list(payload["derived_from"]["priors"]) == [name]
    assert not root.exists()

    # The production artifact root is never a spec-evaluation root.
    production = tmp_path / "production_artifacts"
    with pytest.raises(ValueError, match="production artifact root"):
        run_model_spec_evaluation(
            production,
            dataset.posterior,
            dataset.selection,
            replace(campaign, artifact_root=str(production.resolve())),
            widened,
            dataset_identity=frozen["manifest"].manifest_hash,
            dry_run=True,
        )
