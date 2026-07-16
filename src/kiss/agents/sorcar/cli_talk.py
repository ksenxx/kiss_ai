# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_talk`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_talk as _impl
from kiss.ui.cli.cli_talk import (  # noqa: F401
    TalkPlayer,
    player_command,
    reset_shared_player_for_tests,
    say_command,
    shared_player,
)

sys.modules[__name__] = _impl
