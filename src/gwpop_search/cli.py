"""Command-line entry point.

The CLI exists in Phase 0 so the package is installable. Scientific subcommands
are added only when their implementation phase is complete.
"""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gwpop-search",
        description="Systematic gravitational-wave population-model search.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main() -> None:
    build_parser().parse_args()


if __name__ == "__main__":
    main()
