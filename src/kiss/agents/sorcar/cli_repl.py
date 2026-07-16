# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_repl`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_repl as _impl
from kiss.ui.cli.cli_repl import (  # noqa: F401
    _TASK_DESC_WIDTH,
    SLASH_COMMANDS,
    CliCompleter,
    _load_history_lines,
    _print_help,
    _read_line,
    _read_line_ptk,
    _save_history_lines,
    build_help_text,
    picker_ordered_models,
    readline,
)

sys.modules[__name__] = _impl
