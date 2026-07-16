# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_prompt`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_prompt as _impl
from kiss.ui.cli.cli_prompt import (  # noqa: F401
    _AT_RE,
    _MODEL_CMD_RE,
    _MODIFY_OTHER_KEYS_ENTER,
    PtkCompleter,
    PtkLineReader,
    _prompt_continuation,
    _unmap_enter_aliases,
)

sys.modules[__name__] = _impl
