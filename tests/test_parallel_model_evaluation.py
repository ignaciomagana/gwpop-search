"""Pre-computing one model's evidence must reproduce the serial replay/suite exactly.

``run_exact_null_index`` and ``run_event_drop_stress_suite`` evaluate every
graph model serially in one process. ``run_null_model_evaluation`` and
``run_stress_model_evaluation`` run exactly one of those evaluations ahead of
time, into the same artifact tree. That is execution-only plumbing, so the
tree the replay/suite ends up with, and the summary it writes, must be the ones
it would have written on its own -- and the second stage must reload the
dynesty runs instead of resampling them.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from gwpop_search.data import (  # noqa: E402
    Campaign,
    SelectionCatalog,
    SelectionMode,
)
from gwpop_search.grammar import (  # noqa: E402
    baseline_model_spec,
    enumerate_model_graph,
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
from gwpop_search.nulls import (  # noqa: E402
    ExactNullCampaignConfig,
    prepare_exact_null_campaign,
    run_exact_null_index,
    run_null_model_evaluation,
    save_exact_null_campaign_config,
)
from gwpop_search.production import (  # noqa: E402
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    build_dataset_manifest_from_canonical_files,
    installed_sampler_backend,
    model_graph_hash,
    save_dataset_manifest,
    save_production_campaign,
)
from gwpop_search.search import SchedulerConfig  # noqa: E402
from gwpop_search.validation import (  # noqa: E402
    EventDropScenario,
    EventStressConfig,
    run_event_drop_stress_suite,
    run_stress_model_evaluation,
)

# Wall-clock fields. The whole point of the parallel path is that the second
# stage spends less time than the serial one: it reloads finished dynesty runs
# instead of sampling them. Both are sums over ``EvaluationRecord.compute_cost``
# (= evaluation wall-clock hours), so they are the only fields that may differ
# and they carry no scientific content.
_TIMING_FIELDS = (
    ("metadata", "execution", "total_compute_cost"),
    ("metadata", "evidence_completion", "total_compute_cost"),
)
_STRESS_TIMING_FIELDS = (("execution", "total_compute_cost"),)


def _without(payload, paths):
    """A deep copy of ``payload`` with each ``path`` of keys removed."""
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
    # Small but genuinely converged: the F3 gates require every run to stop
    # on dlogz, so a maxiter/maxcall-truncated sampler cannot stand in here.
    # bound/sample are pinned by the ladder (multi/rslice); ``slices=1`` with
    # no per-model multiplier is the cheapest legal trajectory.
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
        campaign_id="parallel-model-eval-test",
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


def _null_config(n_nulls=1):
    return ExactNullCampaignConfig(
        n_nulls=n_nulls,
        root_seed=404,
        survey=_survey(),
        data_mode="synthetic_survey",
        pe_scale_policy="declared_fixed",
        max_gpu_hours_per_null=25.0,
    )


def _result_files(root):
    """``{path relative to root: bytes}`` of every dynesty ``result.npz``.

    ``result.npz`` holds only the sampler's arrays (samples, log weights, log
    likelihoods, log volumes, equal-weight draws); wall-clock time lives in the
    ``.json`` sidecar. ``np.savez_compressed`` stamps no timestamps, so equal
    numerics give equal bytes.
    """
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
    """Record every nested-sampling run started from here on."""
    from gwpop_search.inference import dynesty_backend

    started = []
    real = dynesty_backend.run_dynesty

    def counting(*args, **kwargs):
        started.append(kwargs.get("identity"))
        return real(*args, **kwargs)

    monkeypatch.setattr(dynesty_backend, "run_dynesty", counting)
    return started


# ---------------------------------------------------------------------------
# (a) exact-null replay
# ---------------------------------------------------------------------------


def test_per_model_null_evaluation_reproduces_the_serial_replay(
    monkeypatch,
    tmp_path,
):
    graph = _graph()
    campaign = _campaign(graph)
    config = _null_config()

    serial_root = tmp_path / "serial"
    prepare_exact_null_campaign(serial_root, graph, campaign, config)
    serial = run_exact_null_index(
        serial_root,
        graph,
        campaign,
        config,
        null_index=0,
    )

    parallel_root = tmp_path / "parallel"
    prepare_exact_null_campaign(parallel_root, graph, campaign, config)
    for model in graph.nodes:
        payload = run_null_model_evaluation(
            parallel_root,
            graph,
            campaign,
            config,
            null_index=0,
            model_hash=model.model_hash,
        )
        assert payload["state_database_written"] is False
        assert payload["fidelity"] == "F3"
        assert payload["null_index"] == 0
        assert payload["model_hash"] == model.model_hash
    # Same contract as run-fidelity-evaluation: no production state is written.
    assert list(parallel_root.rglob("state.sqlite")) == []

    pre_stage = _result_files(parallel_root)
    assert len(pre_stage) == 2 * len(graph.nodes)  # repeats=2 per model
    pre_mtimes = _result_mtimes(parallel_root)

    started = _count_dynesty_runs(monkeypatch)

    # A resubmitted array task reloads the finished runs too: the per-model
    # command is idempotent and cannot disturb the tree.
    run_null_model_evaluation(
        parallel_root,
        graph,
        campaign,
        config,
        null_index=0,
        model_hash=graph.root_hash,
    )
    assert started == []
    assert _result_mtimes(parallel_root) == pre_mtimes

    parallel = run_exact_null_index(
        parallel_root,
        graph,
        campaign,
        config,
        null_index=0,
    )

    # (iii) the replay reloaded every dynesty run; it sampled nothing.
    assert started == []
    assert _result_mtimes(parallel_root) == pre_mtimes

    # (i) every result.npz is byte-identical to the serial tree's.
    assert _result_files(parallel_root) == _result_files(serial_root)

    # (ii) the replay summary agrees once wall-clock fields are removed.
    serial_payload = json.loads((serial_root / "null_00000.json").read_text())
    parallel_payload = json.loads(
        (parallel_root / "null_00000.json").read_text()
    )
    assert _without(parallel_payload, _TIMING_FIELDS) == _without(
        serial_payload,
        _TIMING_FIELDS,
    )
    assert parallel.max_log_bayes_factor == serial.max_log_bayes_factor
    assert parallel.max_log_posterior_odds == serial.max_log_posterior_odds
    assert parallel.best_model_hash == serial.best_model_hash
    assert parallel.n_models_evaluated == len(graph.nodes)


def test_per_model_null_evaluation_requires_the_plan_and_needs_no_f0(
    tmp_path,
):
    """The frozen plan gates it; F0 stays with the replay, which runs it later."""
    graph = _graph()
    campaign = _campaign(graph)
    config = _null_config()

    root = tmp_path / "root"
    with pytest.raises(ValueError, match="prepare-null-search-calibration"):
        run_null_model_evaluation(
            root,
            graph,
            campaign,
            config,
            null_index=0,
            model_hash=graph.root_hash,
        )

    prepare_exact_null_campaign(root, graph, campaign, config)
    run_null_model_evaluation(
        root,
        graph,
        campaign,
        config,
        null_index=0,
        model_hash=graph.root_hash,
    )
    artifacts = root / "searches" / "null_00000" / "artifacts"
    assert (artifacts / "F3" / graph.root_hash / "evaluation.json").is_file()
    assert not (artifacts / "F0").exists()

    with pytest.raises(ValueError, match="unknown graph model hash"):
        run_null_model_evaluation(
            root,
            graph,
            campaign,
            config,
            null_index=0,
            model_hash="f" * 64,
        )
    with pytest.raises(ValueError, match="outside"):
        run_null_model_evaluation(
            root,
            graph,
            campaign,
            config,
            null_index=1,
            model_hash=graph.root_hash,
        )


# ---------------------------------------------------------------------------
# (b) event-drop stress suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stress_dataset():
    return generate_baseline_synthetic_dataset(seed=77, config=_survey())


def _scenarios():
    return (
        EventDropScenario(
            "loo_000_SYNTH_0000",
            ("SYNTH_0000",),
            category="leave_one_out",
            note="leave out SYNTH_0000",
        ),
    )


def _stress_config():
    return EventStressConfig(
        max_gpu_hours_per_scenario=25.0,
        max_f3_models=4,
        max_f4_models=2,
    )


def test_per_model_stress_evaluation_reproduces_the_serial_suite(
    monkeypatch,
    tmp_path,
    stress_dataset,
):
    graph = _graph()
    campaign = _campaign(graph)
    scenarios = _scenarios()
    config = _stress_config()
    identity = "c" * 64

    serial_root = tmp_path / "serial"
    serial = run_event_drop_stress_suite(
        serial_root,
        stress_dataset.posterior,
        stress_dataset.selection,
        graph,
        campaign,
        base_dataset_identity=identity,
        scenarios=scenarios,
        config=config,
    )

    parallel_root = tmp_path / "parallel"
    for model in graph.nodes:
        payload = run_stress_model_evaluation(
            parallel_root,
            stress_dataset.posterior,
            stress_dataset.selection,
            graph,
            campaign,
            base_dataset_identity=identity,
            scenario=scenarios[0],
            model_hash=model.model_hash,
        )
        assert payload["state_database_written"] is False
        assert payload["fidelity"] == "F3"
        assert payload["scenario_id"] == scenarios[0].scenario_id
    assert list(parallel_root.rglob("state.sqlite")) == []
    # The plan freezes the whole scenario list, so only the suite may write it.
    assert not (parallel_root / "stress_plan.json").exists()

    pre_mtimes = _result_mtimes(parallel_root)
    assert len(pre_mtimes) == 2 * len(graph.nodes)

    started = _count_dynesty_runs(monkeypatch)
    parallel = run_event_drop_stress_suite(
        parallel_root,
        stress_dataset.posterior,
        stress_dataset.selection,
        graph,
        campaign,
        base_dataset_identity=identity,
        scenarios=scenarios,
        config=config,
    )

    assert started == []
    assert _result_mtimes(parallel_root) == pre_mtimes
    assert _result_files(parallel_root) == _result_files(serial_root)

    assert _stripped_suite(parallel) == _stripped_suite(serial)
    for scenario in scenarios:
        serial_summary = json.loads(
            (serial_root / scenario.scenario_id / "stress_summary.json").read_text()
        )
        parallel_summary = json.loads(
            (parallel_root / scenario.scenario_id / "stress_summary.json").read_text()
        )
        assert _without(parallel_summary, _STRESS_TIMING_FIELDS) == _without(
            serial_summary,
            _STRESS_TIMING_FIELDS,
        )
        assert parallel_summary["dataset_identity"] == serial_summary[
            "dataset_identity"
        ]


def _stripped_suite(summary):
    payload = json.loads(json.dumps(summary))
    for scenario in payload["scenarios"]:
        scenario["execution"].pop("total_compute_cost", None)
    return payload


# ---------------------------------------------------------------------------
# (c) provenance still protects the tree
# ---------------------------------------------------------------------------


def test_a_misplaced_null_evaluation_is_refused_not_silently_reused(tmp_path):
    """Evidence computed for one null index cannot pass as another's.

    The per-model command derives the run directory from ``--null-index``, so
    the only way to get a foreign run into a replay's tree is to put it there.
    When that happens the evidence manifest (dataset identity, seed) must
    refuse it instead of letting the replay score the wrong catalog.
    """
    graph = _graph()
    campaign = _campaign(graph)
    config = _null_config(n_nulls=2)

    root = tmp_path / "root"
    prepare_exact_null_campaign(root, graph, campaign, config)
    run_null_model_evaluation(
        root,
        graph,
        campaign,
        config,
        null_index=1,
        model_hash=graph.root_hash,
    )
    searches = root / "searches"
    shutil.copytree(
        searches / "null_00001" / "artifacts" / "F3" / graph.root_hash,
        searches / "null_00000" / "artifacts" / "F3" / graph.root_hash,
    )

    with pytest.raises(ValueError, match="mismatch|does not match"):
        run_exact_null_index(
            root,
            graph,
            campaign,
            config,
            null_index=0,
        )


def test_a_misplaced_stress_evaluation_is_refused_not_silently_reused(
    tmp_path,
    stress_dataset,
):
    graph = _graph()
    campaign = _campaign(graph)
    wrong, right = (
        EventDropScenario(
            "loo_000_SYNTH_0000",
            ("SYNTH_0000",),
            category="leave_one_out",
            note="leave out SYNTH_0000",
        ),
        EventDropScenario(
            "loo_001_SYNTH_0001",
            ("SYNTH_0001",),
            category="leave_one_out",
            note="leave out SYNTH_0001",
        ),
    )
    identity = "c" * 64
    root = tmp_path / "root"
    run_stress_model_evaluation(
        root,
        stress_dataset.posterior,
        stress_dataset.selection,
        graph,
        campaign,
        base_dataset_identity=identity,
        scenario=wrong,
        model_hash=graph.root_hash,
    )
    shutil.copytree(
        root / wrong.scenario_id / "artifacts" / "F3" / graph.root_hash,
        root / right.scenario_id / "artifacts" / "F3" / graph.root_hash,
    )

    with pytest.raises(ValueError, match="mismatch|does not match"):
        run_event_drop_stress_suite(
            root,
            stress_dataset.posterior,
            stress_dataset.selection,
            graph,
            campaign,
            base_dataset_identity=identity,
            scenarios=(right,),
            config=_stress_config(),
        )


# ---------------------------------------------------------------------------
# (e) production null mode, across real processes
# ---------------------------------------------------------------------------
#
# Production calibrates with data_mode="frozen_selection_resample", where the
# null catalog is resampled from the frozen selection rather than drawn from a
# synthetic survey, and every per-model worker regenerates that catalog in its
# own interpreter. The tests above cover one process and the synthetic mode, so
# this one drives the real CLI through subprocess.run: if catalog regeneration
# or the dataset identity were not reproducible across interpreters, the replay
# would resample instead of reload, or the evidence manifest would refuse the
# run outright.


def _estimator_ready(raw):
    """The frozen production selection as a single estimator-ready campaign.

    Frozen-selection nulls require an ``estimator_ready`` denominator; this is
    the same reduction ``tests/test_frozen_selection_null.py`` uses.
    """
    return SelectionCatalog(
        samples=raw.samples,
        log_draw_density=raw.log_draw_density,
        campaign_id=np.asarray(["combined"] * raw.n_selected),
        campaigns=(
            Campaign(
                "combined",
                n_draw=sum(item.n_draw for item in raw.campaigns),
                observing_time_yr=sum(
                    item.observing_time_yr for item in raw.campaigns
                ),
            ),
        ),
        basis=raw.basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics="test estimator-ready denominator",
        metadata={"fixture": "parallel-model-evaluation"},
    )


def _frozen_production_catalogs():
    dataset = generate_baseline_synthetic_dataset(
        seed=91,
        config=SyntheticSurveyConfig(
            n_events=4,
            posterior_samples_per_event=8,
            n_injections=1200,
            population_batch_size=128,
            redshift_sampling_grid=512,
            observation_model="noisy_observation",
        ),
    )
    return dataset.posterior, _estimator_ready(dataset.selection)


def _frozen_null_config(posterior):
    return ExactNullCampaignConfig(
        n_nulls=1,
        root_seed=404,
        survey=SyntheticSurveyConfig(
            n_events=int(posterior.n_events),
            # match_observed pins this to the frozen catalog's sample count.
            posterior_samples_per_event=8,
            population_batch_size=128,
            redshift_sampling_grid=512,
            observation_model="noisy_observation",
        ),
        data_mode="frozen_selection_resample",
        pe_scale_policy="match_observed",
        # A toy frozen selection supports a resampling ESS of ~13; the gate is
        # lowered to match the fixture, not the science.
        min_resampling_ess=8.0,
        min_resampling_ess_per_event=2.0,
        max_gpu_hours_per_null=25.0,
    )


def _cli_env():
    """Parent environment, with the package importable and JAX pinned to CPU.

    The dynesty manifest pins the runtime (JAX backend, device, XLA flags,
    x64), so the worker must agree with the parent or the replay would refuse
    to reload its runs.
    """
    import gwpop_search

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "true"
    source_root = str(Path(gwpop_search.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        source_root if not existing else f"{source_root}{os.pathsep}{existing}"
    )
    return env


def test_frozen_selection_null_per_model_evaluation_in_separate_processes(
    monkeypatch,
    tmp_path,
):
    if jax.default_backend() != "cpu":
        pytest.skip(
            "cross-process run identity needs the parent on CPU as well, "
            "because the dynesty manifest pins the JAX device"
        )

    posterior, selection = _frozen_production_catalogs()
    graph = _graph()
    config = _frozen_null_config(posterior)

    # A frozen dataset on disk, because the CLI validates the production
    # freeze and loads the catalogs from the manifest.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    posterior.to_hdf5(data_dir / "pe.h5")
    selection.to_hdf5(data_dir / "selection.h5")
    manifest = build_dataset_manifest_from_canonical_files(
        data_dir / "pe.h5",
        data_dir / "selection.h5",
        dataset_id="parallel-model-evaluation-frozen-null",
        event_selection={"far_threshold_per_year": 1.0},
        waveform_policy={"pe_release": "test"},
        stored_pe_path="pe.h5",
        stored_selection_path="selection.h5",
    )
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest_path, manifest)

    graph_path = tmp_path / "graph.json"
    save_model_graph(graph_path, graph)

    campaign = replace(
        _campaign(graph),
        dataset_manifest_hash=manifest.manifest_hash,
        sampler_backend=installed_sampler_backend(),
    )
    campaign_path = tmp_path / "campaign.json"
    save_production_campaign(campaign_path, campaign)

    null_config_path = tmp_path / "null_config.json"
    save_exact_null_campaign_config(null_config_path, config)

    frozen = {
        "production_posterior": posterior,
        "production_selection": selection,
        "production_dataset_identity": manifest.manifest_hash,
    }

    serial_root = tmp_path / "serial"
    prepare_exact_null_campaign(serial_root, graph, campaign, config, **frozen)
    serial = run_exact_null_index(
        serial_root,
        graph,
        campaign,
        config,
        null_index=0,
        **frozen,
    )

    parallel_root = tmp_path / "parallel"
    prepare_exact_null_campaign(parallel_root, graph, campaign, config, **frozen)

    env = _cli_env()
    for model in graph.nodes:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "gwpop_search.cli",
                "run-null-model-evaluation",
                "--manifest", str(manifest_path),
                "--graph", str(graph_path),
                "--campaign", str(campaign_path),
                "--null-config", str(null_config_path),
                "--root", str(parallel_root),
                "--null-index", "0",
                "--model-hash", model.model_hash,
                "--base-dir", str(data_dir),
                # The fixture campaign pins a placeholder commit, as every
                # other CLI-level test's does.
                "--ignore-current-commit",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        payload = json.loads(completed.stdout)
        assert payload["state_database_written"] is False
        assert payload["null_index"] == 0
        assert payload["model_hash"] == model.model_hash
        assert payload["fidelity"] == "F3"
        assert payload["dataset_identity"].startswith(
            f"frozen-selection-null:{manifest.manifest_hash}:"
        )
        assert payload["dataset_identity"].endswith(":pe-match_observed")
    assert list(parallel_root.rglob("state.sqlite")) == []

    pre_mtimes = _result_mtimes(parallel_root)
    assert len(pre_mtimes) == 2 * len(graph.nodes)  # repeats=2 per model

    started = _count_dynesty_runs(monkeypatch)
    parallel = run_exact_null_index(
        parallel_root,
        graph,
        campaign,
        config,
        null_index=0,
        **frozen,
    )

    # The replay reloaded every run the separate processes produced.
    assert started == []
    assert _result_mtimes(parallel_root) == pre_mtimes
    assert _result_files(parallel_root) == _result_files(serial_root)

    serial_payload = json.loads((serial_root / "null_00000.json").read_text())
    parallel_payload = json.loads(
        (parallel_root / "null_00000.json").read_text()
    )
    assert _without(parallel_payload, _TIMING_FIELDS) == _without(
        serial_payload,
        _TIMING_FIELDS,
    )
    assert parallel.max_log_bayes_factor == serial.max_log_bayes_factor
    assert parallel.max_log_posterior_odds == serial.max_log_posterior_odds
    assert (
        parallel_payload["metadata"]["null_data_mode"]
        == "frozen_selection_resample"
    )
