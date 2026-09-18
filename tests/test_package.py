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
