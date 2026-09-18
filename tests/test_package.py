from gwpop_search import __version__
from gwpop_search.cli import build_parser


def test_version_is_exposed():
    assert __version__ == "0.2.0"


def test_cli_builds():
    parser = build_parser()
    assert parser.prog == "gwpop-search"


def test_synthetic_recovery_cli_parses_campaign_configuration():
    args = build_parser().parse_args(
        [
            "synthetic-recovery",
            "--run-dir",
            "runs/test",
            "--data-seed",
            "11",
            "--sampler-seed",
            "12",
            "--n-events",
            "24",
            "--num-chains",
            "2",
            "--selection-chunk-size",
            "1024",
        ]
    )
    assert args.command == "synthetic-recovery"
    assert args.run_dir == "runs/test"
    assert args.data_seed == 11
    assert args.sampler_seed == 12
    assert args.n_events == 24
    assert args.num_chains == 2
    assert args.selection_chunk_size == 1024
    assert callable(args.func)



def test_synthetic_campaign_cli_parses_matrix_configuration():
    args = build_parser().parse_args(
        [
            "synthetic-campaign",
            "--root",
            "runs/campaign",
            "--n-runs",
            "6",
            "--root-seed",
            "99",
            "--num-chains",
            "4",
        ]
    )
    assert args.command == "synthetic-campaign"
    assert args.root == "runs/campaign"
    assert args.n_runs == 6
    assert args.root_seed == 99
    assert args.num_chains == 4
    assert callable(args.func)


def test_assess_synthetic_campaign_cli_parses():
    args = build_parser().parse_args(
        [
            "assess-synthetic-campaign",
            "--root",
            "runs/campaign",
            "--min-runs",
            "8",
        ]
    )
    assert args.command == "assess-synthetic-campaign"
    assert args.min_runs == 8
    assert callable(args.func)



def test_enumerate_models_cli_parses():
    args = build_parser().parse_args(
        [
            "enumerate-models",
            "--output",
            "graph.json",
            "--max-depth",
            "2",
            "--max-models",
            "32",
        ]
    )
    assert args.command == "enumerate-models"
    assert args.output == "graph.json"
    assert args.max_models == 32
    assert callable(args.func)


def test_validate_model_cli_parses():
    args = build_parser().parse_args(
        ["validate-model", "--spec", "model.yaml"]
    )
    assert args.command == "validate-model"
    assert args.spec == "model.yaml"
    assert callable(args.func)



def test_run_production_search_cli_parses():
    args = build_parser().parse_args(
        [
            "run-production-search",
            "--manifest",
            "dataset_manifest.json",
            "--graph",
            "model_graph.json",
            "--campaign",
            "campaign.json",
            "--base-dir",
            "data",
            "--work-dir",
            "work",
        ]
    )
    assert args.command == "run-production-search"
    assert args.manifest == "dataset_manifest.json"
    assert args.graph == "model_graph.json"
    assert args.campaign == "campaign.json"
    assert args.base_dir == "data"
    assert args.work_dir == "work"
    assert callable(args.func)



def test_freeze_dataset_cli_parses():
    args = build_parser().parse_args(
        [
            "freeze-dataset",
            "--pe",
            "pe.h5",
            "--selection",
            "selection.h5",
            "--dataset-id",
            "gwtc5-bbh-v1",
            "--event-selection-json",
            "event_selection.json",
            "--waveform-policy-json",
            "waveform_policy.json",
            "--output",
            "dataset_manifest.json",
        ]
    )
    assert args.command == "freeze-dataset"
    assert args.pe == "pe.h5"
    assert args.selection == "selection.h5"
    assert args.dataset_id == "gwtc5-bbh-v1"
    assert args.output == "dataset_manifest.json"
    assert callable(args.func)


def test_write_default_fidelity_config_cli_parses():
    args = build_parser().parse_args(
        [
            "write-default-fidelity-config",
            "--output",
            "fidelity.json",
        ]
    )
    assert args.command == "write-default-fidelity-config"
    assert args.output == "fidelity.json"
    assert callable(args.func)


def test_freeze_production_campaign_cli_parses():
    args = build_parser().parse_args(
        [
            "freeze-production-campaign",
            "--manifest",
            "dataset_manifest.json",
            "--graph",
            "model_graph.json",
            "--fidelity-config",
            "fidelity.json",
            "--campaign-id",
            "gwtc5-bbh-v1",
            "--model-prior",
            "axis-complexity",
            "--model-prior-penalty",
            "0.7",
            "--beam-width",
            "8",
            "--exploration-quota",
            "2",
            "--scheduler-seed",
            "11",
            "--root-seed",
            "12",
            "--max-gpu-hours",
            "1000",
            "--max-f3-models",
            "20",
            "--max-f4-models",
            "8",
            "--max-null-replays",
            "200",
            "--artifact-root",
            "runs/prod",
            "--state-database",
            "runs/prod/state.sqlite",
            "--output",
            "campaign.json",
        ]
    )
    assert args.command == "freeze-production-campaign"
    assert args.campaign_id == "gwtc5-bbh-v1"
    assert args.model_prior == "axis-complexity"
    assert args.model_prior_penalty == 0.7
    assert args.max_f4_models == 8
    assert callable(args.func)



def test_hsgp_scout_cli_parses():
    args = build_parser().parse_args(
        [
            "run-hsgp-scout",
            "--manifest",
            "dataset_manifest.json",
            "--base-model",
            "baseline.json",
            "--base-hyperparameters-json",
            "baseline_hp.json",
            "--scout-config",
            "scout.json",
            "--run-dir",
            "runs/scout",
            "--seed",
            "17",
        ]
    )
    assert args.command == "run-hsgp-scout"
    assert args.seed == 17
    assert callable(args.func)


def test_structured_scout_campaign_cli_parses():
    args = build_parser().parse_args(
        [
            "structured-scout-campaign",
            "--root",
            "runs/structured",
            "--mutation-id",
            "chieff.mean.linear_q",
            "--strength",
            "0.4",
            "--scout-config",
            "scout.json",
        ]
    )
    assert args.command == "structured-scout-campaign"
    assert args.mutation_id == "chieff.mean.linear_q"
    assert args.strength == 0.4
    assert callable(args.func)


def test_assess_structured_scout_campaign_cli_parses():
    args = build_parser().parse_args(
        ["assess-structured-scout-campaign", "--root", "runs/structured"]
    )
    assert args.command == "assess-structured-scout-campaign"
    assert callable(args.func)


def test_write_default_scout_config_cli_parses():
    args = build_parser().parse_args(
        [
            "write-default-scout-config",
            "--target",
            "chi_eff",
            "--covariate",
            "q",
            "--output",
            "scout.json",
        ]
    )
    assert args.command == "write-default-scout-config"
    assert args.target == "chi_eff"
    assert args.covariate == "q"
    assert callable(args.func)



def test_write_loo_stress_config_cli_parses():
    args = build_parser().parse_args(
        [
            "write-loo-stress-config",
            "--manifest",
            "dataset_manifest.json",
            "--output",
            "stress.json",
        ]
    )
    assert args.command == "write-loo-stress-config"
    assert args.stop_fidelity == "F3"
    assert callable(args.func)


def test_write_event_drop_stress_config_cli_parses():
    args = build_parser().parse_args(
        [
            "write-event-drop-stress-config",
            "--scenario-id",
            "drop_loud",
            "--drop-event",
            "GW_A",
            "--drop-event",
            "GW_B",
            "--category",
            "loud_event",
            "--output",
            "stress.json",
        ]
    )
    assert args.command == "write-event-drop-stress-config"
    assert args.drop_event == ["GW_A", "GW_B"]
    assert args.category == "loud_event"
    assert callable(args.func)


def test_run_event_stress_suite_cli_parses():
    args = build_parser().parse_args(
        [
            "run-event-stress-suite",
            "--manifest",
            "dataset_manifest.json",
            "--graph",
            "model_graph.json",
            "--campaign",
            "campaign.json",
            "--stress-config",
            "stress.json",
            "--root",
            "runs/stress",
        ]
    )
    assert args.command == "run-event-stress-suite"
    assert args.root == "runs/stress"
    assert callable(args.func)



def test_write_nearby_baseline_config_cli_parses():
    args = build_parser().parse_args(
        [
            "write-nearby-baseline-config",
            "--scenario-id",
            "broken_mass",
            "--root-model",
            "broken.json",
            "--output",
            "nearby.json",
        ]
    )
    assert args.command == "write-nearby-baseline-config"
    assert args.scenario_id == "broken_mass"
    assert args.max_depth == 1
    assert callable(args.func)


def test_run_nearby_baseline_suite_cli_parses():
    args = build_parser().parse_args(
        [
            "run-nearby-baseline-suite",
            "--manifest",
            "dataset_manifest.json",
            "--graph",
            "model_graph.json",
            "--campaign",
            "campaign.json",
            "--nearby-config",
            "nearby.json",
            "--root",
            "runs/nearby",
        ]
    )
    assert args.command == "run-nearby-baseline-suite"
    assert args.root == "runs/nearby"
    assert callable(args.func)



def test_write_null_calibration_config_cli_parses():
    args = build_parser().parse_args(
        [
            "write-null-calibration-config",
            "--n-nulls",
            "50",
            "--output",
            "nulls.json",
        ]
    )
    assert args.command == "write-null-calibration-config"
    assert args.n_nulls == 50
    assert args.stop_fidelity == "F4"
    assert callable(args.func)


def test_run_null_search_calibration_cli_parses():
    args = build_parser().parse_args(
        [
            "run-null-search-calibration",
            "--manifest",
            "dataset_manifest.json",
            "--graph",
            "model_graph.json",
            "--campaign",
            "campaign.json",
            "--null-config",
            "nulls.json",
            "--root",
            "runs/nulls",
        ]
    )
    assert args.command == "run-null-search-calibration"
    assert args.root == "runs/nulls"
    assert callable(args.func)
