# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_voice`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_voice as _impl
from kiss.ui.cli.cli_voice import (  # noqa: F401
    LISTENING_TEXT,
    TRANSCRIBING_TEXT,
    VoiceListener,
    VoicePump,
    VoiceSession,
    listener_command,
    read_voice_line_plain,
    start_voice,
    start_voice_anchored,
)

sys.modules[__name__] = _impl
