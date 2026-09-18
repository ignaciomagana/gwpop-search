from gwpop_search import __version__
from gwpop_search.cli import build_parser


def test_version_is_exposed():
    assert __version__ == "0.2.0"


def test_cli_builds():
    parser = build_parser()
    assert parser.prog == "gwpop-search"
