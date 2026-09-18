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
