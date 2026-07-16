# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_line_continuation`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_line_continuation as _impl
from kiss.ui.cli.cli_line_continuation import (  # noqa: F401
    ends_with_line_continuation,
    read_continuations,
)

sys.modules[__name__] = _impl
