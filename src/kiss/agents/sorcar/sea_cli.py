# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``sea`` command: ``sea lint`` (check scripts) and ``sea docs`` (render the vocabulary).

``uv run check`` runs both; see :mod:`kiss.agents.sorcar.sea_lint` and
:mod:`kiss.agents.sorcar.sea_docs` for the rules and the tables.
"""

from __future__ import annotations

import argparse
import sys

from kiss.agents.sorcar import sea_docs, sea_lint


def main(argv: list[str] | None = None) -> int:
    """Run ``sea <lint|docs> ...`` and return the sub-command's exit code.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(prog="sea", description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sea_lint.add_arguments(
        sub.add_parser("lint", help="check agent scripts against the SEA contract")
    )
    sea_docs.add_arguments(
        sub.add_parser("docs", help="regenerate the settings/kind/option/command tables")
    )
    args = parser.parse_args(argv)
    return sea_lint.run(args) if args.command == "lint" else sea_docs.run(args)


if __name__ == "__main__":
    sys.exit(main())
